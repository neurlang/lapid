//! MoE matmul (`Kernel::MatMulId`) and the gated delta rule (`Kernel::GatedDeltaNet`).
//!
//! MatMulId reuses the matmul templates in their MATMUL_ID mode:
//! - t <= GEMV_MAX_M (decode): one GEMV threadgroup row per (t, r) pair reading its expert
//!   (`Key::GemvId`), out-of-range ids raise the status flag;
//! - otherwise two stages: `kernels/moe_route.metal` buckets the pairs by expert into scratch,
//!   then a grouped GEMM (`Key::GemmId`, grid z = expert) reads each expert's weights once per
//!   64 routed rows (grid y sized for 4x the average rows per expert; threadgroups loop).
//!
//! GatedDeltaNet: `kernels/gated_delta_net.metal`, one simdgroup per state column.

use std::sync::Arc;

use lapid_compiler::{GatedDeltaNetDims, GatedDeltaNetSpec, MatMulIdDims, MatMulIdSpec, MatMulSpec};
use lapid_tensor::{DType, Error, Result};

use super::index::STATUS_DECL;
use super::matmul::{GEMV_MAX_M, TILE, gemv_cols, quant_scales};
use super::msl::bindings;
use super::template::{self, DELTA_NET, MOE_ROUTE};
use super::{Bind, Key, Params, Plan, Stage, bad, check_plain, dim};

/// Largest expert count of the route stage (threadgroup counters).
pub(crate) const MAX_EXPERTS: usize = 4096;
const ROUTE_THREADS: u32 = 1024;
/// Largest head key dim of the delta-net kernel (8 registers per lane).
pub(crate) const DELTA_MAX_DK: usize = 256;

/// The matmul spec the MATMUL_ID templates are rendered from.
pub(crate) fn mat_spec(s: &MatMulIdSpec) -> Arc<MatMulSpec> {
    Arc::new(MatMulSpec {
        a_dtype: s.x_dtype,
        b_dtype: s.w_dtype,
        b_transposed: true,
        extra_dtypes: vec![],
        out_dtype: s.out_dtype,
        epilogue: None,
    })
}

pub(super) fn plan_matmul_id(s: &Arc<MatMulIdSpec>, d: &MatMulIdDims) -> Result<Plan> {
    check_plain(s.x_dtype, "matmul_id x")?;
    check_plain(s.out_dtype, "matmul_id output")?;
    if s.x_dtype == DType::I32 || s.out_dtype == DType::I32 || s.w_dtype == DType::I32 {
        return Err(Error::new("unsupported_dtype", "metal matmul_id: integer x / w / output"));
    }
    let (expert_stride, scales) = if s.w_dtype.is_quant() {
        let f = s.w_dtype.quant_format().ok_or_else(|| bad("matmul_id: quantized dtype without a format"))?;
        let scales = quant_scales(s.w_dtype, d.n, d.k)?;
        (lapid_quant::layout(f, d.n, d.k).total, scales)
    } else {
        (d.n.checked_mul(d.k).ok_or_else(|| bad("matmul_id: size overflow"))?, 0)
    };
    let pairs = d.t.checked_mul(d.r).ok_or_else(|| bad("matmul_id: size overflow"))?;
    // Expert byte / element offsets must fit the 32-bit index math.
    u32::try_from(d.e.saturating_mul(expert_stride)).map_err(|_| Error::new("too_large", "matmul_id: expert stack exceeds 32-bit indexing"))?;
    let mut p = Params::default();
    // Same layout as the matmul params (m n k, A, B, scales, out), then R, E, stride, cap.
    p.u(1)?.u(d.n)?.u(d.k)?;
    p.u(0)?.u(d.k)?.u(d.k)?;
    p.u(0)?.u(0)?.u(d.k)?;
    p.u(scales)?;
    p.u(0)?.u(d.n)?.u(d.n)?;
    p.u(d.r)?.u(d.e)?.u(expert_stride)?.u(pairs)?;
    let ms = mat_spec(s);
    if pairs == 0 || d.n == 0 {
        return Ok(Plan { stages: vec![], scratch: 0 });
    }
    if d.t <= GEMV_MAX_M {
        let stage = Stage {
            key: Key::GemvId(ms),
            params: p.0,
            groups: [dim(d.n.div_ceil(gemv_cols(s.w_dtype)))?, 1, dim(pairs)?],
            threads: 128,
            binds: vec![Bind::Input(0), Bind::Input(1), Bind::Input(2), Bind::Output, Bind::Status],
        };
        return Ok(Plan { stages: vec![stage], scratch: 0 });
    }
    if d.e > MAX_EXPERTS {
        return Err(Error::new("unsupported", format!("metal matmul_id: {} experts > {MAX_EXPERTS}", d.e)));
    }
    let mut rp = Params::default();
    rp.u(pairs)?.u(d.e)?.u(pairs)?;
    let route = Stage {
        key: Key::MoeRoute,
        params: rp.0,
        groups: [1, 1, 1],
        threads: ROUTE_THREADS,
        binds: vec![Bind::Input(2), Bind::Scratch, Bind::Status],
    };
    let gemm = Stage {
        key: Key::GemmId(ms),
        params: p.0,
        groups: [dim(d.n.div_ceil(TILE))?, dim(id_row_groups(pairs, d.e))?, dim(d.e)?],
        threads: 128,
        binds: vec![Bind::Input(0), Bind::Input(1), Bind::Output, Bind::Scratch],
    };
    let scratch = (d.e + d.e * pairs) * 4;
    Ok(Plan { stages: vec![route, gemm], scratch })
}

/// Row-tile threadgroups per expert of the routed GEMM: 4x the average tile count (each
/// threadgroup loops over the expert's tiles with this stride), so few empty threadgroups
/// launch when the pairs spread over many experts; an expert with more rows loops.
fn id_row_groups(pairs: usize, e: usize) -> usize {
    let avg = pairs.div_ceil(e.max(1));
    (4 * avg).div_ceil(TILE).clamp(1, pairs.div_ceil(TILE).max(1))
}

pub(super) fn source_route() -> String {
    let binds = bindings(&[("ids".into(), DType::I32)], None, &["device uint* route", STATUS_DECL]);
    template::module(MOE_ROUTE, &[("BINDINGS", &binds), ("MAX_E", &MAX_EXPERTS.to_string())], &[])
}

pub(super) fn plan_delta_net(s: &Arc<GatedDeltaNetSpec>, d: &GatedDeltaNetDims) -> Result<Plan> {
    check_plain(s.out_dtype, "gated_delta_net output")?;
    if d.dk > DELTA_MAX_DK {
        return Err(Error::new("unsupported", format!("metal gated_delta_net: dk = {} > {DELTA_MAX_DK}", d.dk)));
    }
    let mut p = Params::default();
    p.u(d.t)?.u(d.h)?.u(d.dk)?.u(d.dv)?.bits(d.scale);
    // Index math: the largest element index is (b * h * (t + dk)) * dv.
    let total = d.b.checked_mul(d.h).and_then(|x| x.checked_mul(d.t + d.dk)).and_then(|x| x.checked_mul(d.dv.max(d.dk)));
    total.and_then(|x| u32::try_from(x).ok()).ok_or_else(|| Error::new("too_large", "gated_delta_net exceeds 32-bit indexing"))?;
    let dkc = d.dk.div_ceil(32).max(1) as u32;
    let empty = d.b == 0 || d.h == 0 || d.dv == 0;
    let groups = if empty { [0; 3] } else { [dim(d.dv.div_ceil(4))?, dim(d.h)?, dim(d.b)?] };
    Ok(Plan::single(Stage {
        key: Key::DeltaNet(s.clone(), dkc),
        params: p.0,
        groups,
        threads: 128,
        binds: (0..6).map(Bind::Input).chain([Bind::Output]).collect(),
    }))
}

pub(super) fn source_delta_net(s: &GatedDeltaNetSpec, dkc: u32) -> String {
    let ins: Vec<(String, DType)> = ["q", "k", "v", "g", "beta", "state"].iter().map(|n| (n.to_string(), DType::F32)).collect();
    let binds = bindings(&ins, Some(s.out_dtype), &[]);
    template::module(DELTA_NET, &[("BINDINGS", &binds), ("DKC", &dkc.to_string())], &[])
}
