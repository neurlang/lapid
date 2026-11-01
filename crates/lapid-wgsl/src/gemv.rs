//! Matrix-vector kernel for small m with transposed B (the decoder hot path); template
//! `kernels/gemv.wgsl`. Rust fills the 8-element dot product for the B dtype and the A / B
//! bindings (vec4 loads when the views are aligned).

use lapid_compiler::{MatMulDims, MatMulSpec};
use lapid_tensor::DType;

use super::matmul::{epilogue, epilogue_bindings};
use super::{common, kquant, template};
use crate::Caps;

const KERNEL: &str = include_str!("kernels/gemv.wgsl");
const KERNEL_K: &str = include_str!("kernels/gemv_k.wgsl");

/// Output columns per workgroup (one 32-lane group or subgroup per column).
pub fn gemv_cols(caps: &Caps) -> usize {
    if caps.subgroup { (256 / caps.subgroup_size.clamp(1, 256)) as usize } else { 8 }
}

/// Adjacent output columns per lane group of the (non K-quant) GEMV kernel.
pub fn gemv_nc(_spec: &MatMulSpec) -> usize {
    // 2 and 4 were measured on GB10 (LFM3 / Whisper decode shapes): no gain within noise;
    // the kernel is at ~85-90% of DRAM bandwidth plus the per-launch floor.
    1
}

/// Alignment variants: (A is f32 and readable as vec4, B is f16 readable as 8-half vectors).
pub fn vec_variants(spec: &MatMulSpec, d: &MatMulDims) -> (bool, bool) {
    let a = &d.a;
    let vec_a = spec.a_dtype == DType::F32
        && d.k.is_multiple_of(4)
        && a.offset.is_multiple_of(4)
        && (a.row_stride.is_multiple_of(4) || d.m <= 1)
        && (a.batch_stride.is_multiple_of(4) || d.batch <= 1);
    let b = &d.b;
    let vec_b = spec.b_dtype == DType::F16
        && d.k.is_multiple_of(8)
        && b.offset.is_multiple_of(8)
        && b.row_stride.is_multiple_of(8)
        && (b.batch_stride.is_multiple_of(8) || d.batch <= 1);
    (vec_a, vec_b)
}

fn a_binding(spec: &MatMulSpec, vec_a: bool, caps: &Caps) -> String {
    if vec_a {
        return "@group(0) @binding(1) var<storage, read> a4: array<vec4<f32>>;\n\
                fn lda4(e: u32) -> vec4<f32> {\n    return a4[e >> 2u];\n}\n\
                fn lda(e: u32) -> f32 {\n    return a4[e >> 2u][e & 3u];\n}\n"
            .to_string();
    }
    let mut s = common::input(1, "a", spec.a_dtype, caps);
    s.push_str(
        "fn lda4(e: u32) -> vec4<f32> {\n    return vec4<f32>(ld_a(e), ld_a(e + 1u), ld_a(e + 2u), ld_a(e + 3u));\n}\n\
         fn lda(e: u32) -> f32 {\n    return ld_a(e);\n}\n",
    );
    s
}

/// (B binding, body of `dot8`).
fn b_parts(spec: &MatMulSpec, vec_b: bool, caps: &Caps) -> (String, String) {
    match spec.b_dtype {
        DType::Int8 => (
            format!("@group(0) @binding(2) var<storage, read> b: array<u32>;\n{}", common::quant_fns("b", DType::Int8)),
            r#"    let e = j * K + k0;
    let sc = b_scale(P[10], e >> 5u);
    let w0 = vec4<f32>(unpack4xI8(b[e >> 2u]));
    let w1 = vec4<f32>(unpack4xI8(b[(e >> 2u) + 1u]));
    return (dot(w0, lda4(a_row + k0)) + dot(w1, lda4(a_row + k0 + 4u))) * sc;
"#
            .to_string(),
        ),
        DType::Int4 => (
            format!("@group(0) @binding(2) var<storage, read> b: array<u32>;\n{}", common::quant_fns("b", DType::Int4)),
            r#"    // word e / 8 of a 32-group holds elements 4w..4w+3 (low nibbles) and
    // 16+4w..16+4w+3 (high nibbles), w = word index in the group
    let e = j * K + k0;
    let sc = b_scale(P[10], e >> 5u);
    let w = b[e >> 3u];
    let kg = k0 & ~31u;
    let wo = ((k0 >> 3u) & 3u) * 4u;
    let lo = vec4<f32>(f32(w & 15u), f32((w >> 8u) & 15u), f32((w >> 16u) & 15u), f32((w >> 24u) & 15u)) - vec4<f32>(8.0);
    let hi = vec4<f32>(f32((w >> 4u) & 15u), f32((w >> 12u) & 15u), f32((w >> 20u) & 15u), f32(w >> 28u)) - vec4<f32>(8.0);
    return (dot(lo, lda4(a_row + kg + wo)) + dot(hi, lda4(a_row + kg + 16u + wo))) * sc;
"#
            .to_string(),
        ),
        _ if vec_b => (
            "@group(0) @binding(2) var<storage, read> bv: array<vec4<u32>>;\n".to_string(),
            r#"    let w = bv[(b_base + j * P[9] + k0) >> 3u];
    let b0 = vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y));
    let b1 = vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w));
    return dot(b0, lda4(a_row + k0)) + dot(b1, lda4(a_row + k0 + 4u));
"#
            .to_string(),
        ),
        d => (
            common::input(2, "b", d, caps),
            r#"    let brow = b_base + j * P[9];
    var s: f32 = 0.0;
    for (var t = 0u; t < 8u; t += 1u) {
        let k = k0 + t;
        if (k < K) {
            s = s + lda(a_row + k) * ld_b(brow + k);
        }
    }
    return s;
"#
            .to_string(),
        ),
    }
}

pub(super) fn source(spec: &MatMulSpec, vec_a: bool, vec_b: bool, caps: &Caps) -> String {
    let mut bindings = a_binding(spec, vec_a, caps);
    if kquant::is_k(spec.b_dtype) {
        bindings.push_str("@group(0) @binding(2) var<storage, read> b: array<u32>;\n");
        bindings.push_str(&kquant::fns_dot("b", spec.b_dtype));
        bindings.push_str(&epilogue_bindings(spec, caps));
        let epi = epilogue(spec, "            ");
        let bb = kquant::block_bytes(spec.b_dtype).to_string();
        return template::module(caps, KERNEL_K, &[("BINDINGS", &bindings), ("BB", &bb), ("EPILOGUE", &epi)], &[], 256, true);
    }
    let (b_bind, dot8) = b_parts(spec, vec_b, caps);
    bindings.push_str(&b_bind);
    bindings.push_str(&epilogue_bindings(spec, caps));
    let epi = epilogue(spec, "                ");
    let nc = gemv_nc(spec).to_string();
    template::module(caps, KERNEL, &[("BINDINGS", &bindings), ("DOT8", &dot8), ("NC", &nc), ("EPILOGUE", &epi)], &[], 256, true)
}
