//! Index kernels: `GatherAxis` (take_along_axis, exact element copy) and `TopK` (argmax
//! fast path for k = 1, radix select + bitonic sort for k > 1).
//! Templates: `kernels/gather_axis.metal`, `kernels/argmax.metal`, `kernels/topk.metal`.

use std::sync::Arc;

use lapid_compiler::{GatherAxisDims, GatherAxisSpec, TopKDims, TopKSpec};
use lapid_tensor::{DType, Error, Result};

use super::msl::bindings;
use super::template::{self, ARGMAX, GATHER_AXIS, TOPK};
use super::{Bind, Key, Params, Stage, bad, check_plain, grid_items, grid_threads};

const GATHER_THREADS: u32 = 256;
/// Largest k of the general top-k kernel (candidates live in threadgroup memory: 8 bytes each).
pub(crate) const TOPK_MAX: usize = 1024;
const TOPK_THREADS: u32 = 512;

/// Status-flag declaration bound after the output of gather kernels.
pub(crate) const STATUS_DECL: &str = "device uint* status";

/// Raw element type of a plain dtype (copies move bits, never convert).
fn raw_ty(d: DType) -> &'static str {
    if d.size() == 2 { "ushort" } else { "uint" }
}

pub(super) fn plan_gather_axis(s: &Arc<GatherAxisSpec>, d: &GatherAxisDims) -> Result<Stage> {
    check_plain(s.dtype, "gather_axis")?;
    let total = d.outer.checked_mul(d.n).and_then(|x| x.checked_mul(d.inner)).ok_or_else(|| bad("gather_axis: size overflow"))?;
    let mut p = Params::default();
    p.u(total)?.u(d.n)?.u(d.inner)?.u(d.axis_len)?;
    p.view(&d.x, 3)?.view(&d.idx, 3)?;
    Ok(Stage {
        key: Key::GatherAxis(s.clone()),
        params: p.0,
        groups: grid_threads(total, GATHER_THREADS),
        threads: GATHER_THREADS,
        binds: vec![Bind::Input(0), Bind::Input(1), Bind::Output, Bind::Status],
    })
}

pub(super) fn source_gather_axis(s: &GatherAxisSpec) -> String {
    let t = raw_ty(s.dtype);
    let binds = format!(
        "    device const {t}* x [[buffer(1)]],\n    device const int* indices [[buffer(2)]],\n    device {t}* out [[buffer(3)]],\n    {STATUS_DECL} [[buffer(4)]],\n"
    );
    template::module(GATHER_AXIS, &[("BINDINGS", &binds), ("T", t)], &[])
}

pub(super) fn plan_topk(s: &Arc<TopKSpec>, d: &TopKDims) -> Result<Stage> {
    check_plain(s.in_dtype, "topk input")?;
    if d.k == 0 || d.k > d.cols {
        return Err(bad(format!("topk: k = {} outside 1..={}", d.k, d.cols)));
    }
    if d.k > TOPK_MAX {
        return Err(Error::new("unsupported", format!("metal topk: k = {} > {TOPK_MAX}", d.k)));
    }
    let mut p = Params::default();
    p.u(d.rows)?.u(d.cols)?.u(d.k)?;
    let k1 = d.k == 1;
    let threads = if k1 { (d.cols.div_ceil(8).next_power_of_two().clamp(32, 1024)) as u32 } else { TOPK_THREADS };
    let groups = if d.cols == 0 { [0; 3] } else { grid_items(d.rows) };
    Ok(Stage { key: Key::TopK(s.clone(), k1), params: p.0, groups, threads, binds: vec![Bind::Input(0), Bind::Output] })
}

pub(super) fn source_topk(s: &TopKSpec, k1: bool) -> String {
    let binds = bindings(&[("x".into(), s.in_dtype)], Some(DType::I32), &[]);
    let kmax = TOPK_MAX.to_string();
    template::module(if k1 { ARGMAX } else { TOPK }, &[("BINDINGS", &binds), ("KMAX", &kmax)], &[])
}
