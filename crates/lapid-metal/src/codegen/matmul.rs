//! Matrix multiplication: simdgroup-matrix GEMM (`kernels/gemm.metal`) and the small-m
//! GEMV (`kernels/gemv.metal`, `kernels/gemv_kq.metal` for K-quant B), both with the fused epilogue; B may be float, int8 / int4
//! (SoA groups of 32) or a ggml K-quant (AoS super-blocks, `kernels/kquant.metal`).

use std::sync::Arc;

use lapid_compiler::{ExprType, MatMulDims, MatMulSpec};
use lapid_tensor::{DType, Result};

use super::msl::{self, bindings};
use super::template::{self, GEMM, GEMV, GEMV_KQ};
use super::index::STATUS_DECL;
use super::{Bind, Key, Params, Stage, bad, check_plain, dim, io_binds};

/// GEMV handles up to this many rows (decoder steps / short prompts).
pub(crate) const GEMV_MAX_M: usize = 4;
pub(super) const TILE: usize = 64;
/// Output columns per GEMV threadgroup (4 simdgroups x 2; x 4 for K-quant B).
pub(super) fn gemv_cols(b: DType) -> usize {
    if kq(b).is_some() { 16 } else { 8 }
}
/// First params slot of the epilogue extra views (after m n k, A, B, scales, out view).
const EXTRA_BASE: usize = 13;

pub(super) fn plan(s: &Arc<MatMulSpec>, d: &MatMulDims) -> Result<Stage> {
    check_plain(s.a_dtype, "matmul A")?;
    check_plain(s.out_dtype, "matmul output")?;
    for &e in &s.extra_dtypes {
        check_plain(e, "matmul epilogue input")?;
    }
    if d.extra_views.len() != s.extra_dtypes.len() {
        return Err(bad(format!("matmul: {} extra views for {} extras", d.extra_views.len(), s.extra_dtypes.len())));
    }
    if s.epilogue.is_none() && !s.extra_dtypes.is_empty() {
        return Err(bad("matmul: epilogue inputs without an epilogue"));
    }
    if s.epilogue.as_ref().is_some_and(|e| e.ty() != Some(ExprType::F32)) {
        return Err(bad("matmul: the epilogue expression must be f32"));
    }
    let scales = if s.b_dtype.is_quant() {
        if !s.b_transposed || d.b.batch_stride != 0 || d.b.offset != 0 {
            return Err(bad("matmul: quantized B must be transposed, unbatched and at offset 0"));
        }
        quant_scales(s.b_dtype, d.n, d.k)?
    } else {
        0
    };
    let mut p = Params::default();
    p.u(d.m)?.u(d.n)?.u(d.k)?;
    p.u(d.a.offset)?.u(d.a.batch_stride)?.u(d.a.row_stride)?;
    p.u(d.b.offset)?.u(d.b.batch_stride)?.u(d.b.row_stride)?;
    p.u(scales)?;
    p.u(d.out.offset)?.u(d.out.batch_stride)?.u(d.out.row_stride)?;
    for v in &d.extra_views {
        p.view(v, 3)?;
    }
    let binds: Vec<Bind> = io_binds(2 + s.extra_dtypes.len());
    let empty = d.m == 0 || d.n == 0 || d.batch == 0;
    let gemv = s.b_transposed && d.m >= 1 && d.m <= GEMV_MAX_M;
    let (key, groups) = if gemv {
        (Key::Gemv(s.clone(), d.m as u32), [dim(d.n.div_ceil(gemv_cols(s.b_dtype)))?, 1, dim(d.batch)?])
    } else {
        (Key::Gemm(s.clone()), [dim(d.n.div_ceil(TILE))?, dim(d.m.div_ceil(TILE))?, dim(d.batch)?])
    };
    Ok(Stage { key, params: p.0, groups: if empty { [0; 3] } else { groups }, threads: 128, binds })
}

/// Offset (in halves) of the scales plane of a quantized `[n, k]` B (0 for the AoS K
/// formats); checks k against the format's block size.
pub(super) fn quant_scales(b: DType, n: usize, k: usize) -> Result<usize> {
    let f = b.quant_format().ok_or_else(|| bad(format!("matmul: {} is not quantized", b.name())))?;
    let be = f.block_elems();
    if !k.is_multiple_of(be) {
        return Err(bad(format!("matmul: {} B needs k % {be} == 0, got k = {k}", b.name())));
    }
    Ok(if f.is_aos() { 0 } else { lapid_quant::layout(f, n, k).scales_offset / 2 })
}

/// K-quant function suffix and block bytes (`kernels/kquant.metal`).
pub(super) fn kq(d: DType) -> Option<(&'static str, &'static str)> {
    match d {
        DType::Q4K => Some(("q4k", "144")),
        DType::Q5K => Some(("q5k", "176")),
        DType::Q6K => Some(("q6k", "210")),
        _ => None,
    }
}

/// How a matmul source is bound: plain, or the MatMulId GEMV (ids + status) / GEMM (route).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode {
    Plain,
    IdGemv,
    IdGemm,
}

/// Inputs (a, b, e0..) and the epilogue slots shared by both kernels.
fn common(s: &MatMulSpec, mode: Mode) -> (String, String, String) {
    let bname = if mode == Mode::Plain { "b" } else { "w" };
    let mut ins = vec![("a".to_string(), s.a_dtype), (bname.to_string(), s.b_dtype)];
    for (e, &dt) in s.extra_dtypes.iter().enumerate() {
        ins.push((format!("e{e}"), dt));
    }
    let binds = match mode {
        Mode::Plain => bindings(&ins, Some(s.out_dtype), &[]),
        Mode::IdGemv => {
            ins.push(("ids".to_string(), DType::I32));
            bindings(&ins, Some(s.out_dtype), &[STATUS_DECL])
        }
        Mode::IdGemm => bindings(&ins, Some(s.out_dtype), &["device const uint* route"]),
    };
    let mut loads = String::new();
    let epilogue = match &s.epilogue {
        Some(expr) => {
            if msl::uses_bits(expr, 0) {
                loads.push_str("                const uint w0 = as_type<uint>(x0);\n");
            }
            for e in 0..s.extra_dtypes.len() {
                let b = EXTRA_BASE + 4 * e;
                let index = format!("P[{b}] + bt * P[{}] + gr * P[{}] + gc * P[{}]", b + 1, b + 2, b + 3);
                loads += &msl::load_lines(expr, 1 + e, &format!("e{e}"), &index, "                ");
            }
            msl::render(expr)
        }
        None => "x0".to_string(),
    };
    (binds, loads, epilogue)
}

/// Slots and flags common to both templates: B element type, K-quant helpers, MATMUL_ID.
fn slots_flags(s: &MatMulSpec, mode: Mode) -> (Vec<(&'static str, &'static str)>, Vec<&'static str>) {
    let mut slots = vec![("BTY", msl::elem_ty(s.b_dtype))];
    let mut flags = Vec::new();
    if let Some((f, bytes)) = kq(s.b_dtype) {
        slots.push(("KQ", f));
        slots.push(("KQ_BYTES", bytes));
        flags.push("B_KQ");
    }
    if mode != Mode::Plain {
        flags.push("MATMUL_ID");
    }
    (slots, flags)
}

fn module(src: &str, s: &MatMulSpec, slots: &[(&str, &str)], flags: &[&str]) -> String {
    if kq(s.b_dtype).is_some() { template::module_kq(src, slots, flags) } else { template::module(src, slots, flags) }
}

pub(super) fn source_gemm(s: &MatMulSpec, mode: Mode) -> String {
    let (binds, loads, epilogue) = common(s, mode);
    // B tile element type: f16 for 16-bit / quantized weights (exact for f16; the dequantized
    // q * d product is rounded once), f32 otherwise (exact, but f32 x f32 simdgroup products
    // run ~10x slower than f32 x f16 on Apple GPUs).
    let tb = if s.b_dtype == DType::F16 || s.b_dtype.is_quant() { "half" } else { "float" };
    let (mut slots, mut flags) = slots_flags(s, mode);
    match s.b_dtype {
        DType::Int8 => flags.push("B_INT8"),
        DType::Int4 => flags.push("B_INT4"),
        d if d.is_quant() => {}
        _ if s.b_transposed => flags.push("B_TRANSPOSED"),
        _ => flags.push("B_PLAIN"),
    }
    slots.extend([("BINDINGS", binds.as_str()), ("TB", tb), ("EXTRA_LOADS", loads.as_str()), ("EPILOGUE", epilogue.as_str())]);
    module(GEMM, s, &slots, &flags)
}

pub(super) fn source_gemv(s: &MatMulSpec, m: u32, mode: Mode) -> String {
    let (binds, loads, epilogue) = common(s, mode);
    let mr = m.to_string();
    let (mut slots, mut flags) = slots_flags(s, mode);
    flags.push(match s.a_dtype {
        DType::F32 => "A_F32",
        DType::F16 => "A_F16",
        _ => "A_ANY",
    });
    match s.b_dtype {
        DType::Int8 => flags.push("B_INT8"),
        DType::Int4 => flags.push("B_INT4"),
        d if d.is_quant() => {}
        DType::F32 => flags.extend(["B_FLOAT", "B_F32"]),
        DType::F16 => flags.extend(["B_FLOAT", "B_F16"]),
        _ => flags.extend(["B_FLOAT", "B_ANY"]),
    }
    slots.extend([("BINDINGS", binds.as_str()), ("MR", mr.as_str()), ("EXTRA_LOADS", loads.as_str()), ("EPILOGUE", epilogue.as_str())]);
    module(if kq(s.b_dtype).is_some() { GEMV_KQ } else { GEMV }, s, &slots, &flags)
}
