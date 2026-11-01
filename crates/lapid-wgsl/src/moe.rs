//! `Kernel::MatMulId` (mixture-of-experts matmul, ggml MUL_MAT_ID): template
//! `kernels/matmul_id.wgsl`. Expert weights are float `[E, N, K]` or E quantized `[N, K]`
//! matrices back to back (`lapid_quant::layout(..).total` bytes each).

use std::sync::Arc;

use lapid_compiler::{MatMulIdDims, MatMulIdSpec};
use lapid_tensor::{DType, Error, Result};

use super::gather::scales_words;
use super::gemv::gemv_cols;
use super::{Key, Params, Plan, bad, check_plain, common, grid, kquant, template};
use crate::Caps;

const KERNEL: &str = include_str!("kernels/matmul_id.wgsl");

pub(super) fn plan(spec: &Arc<MatMulIdSpec>, d: &MatMulIdDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.x_dtype, "matmul_id x")?;
    check_plain(spec.out_dtype, "matmul_id output")?;
    let (so, stride) = match spec.w_dtype.quant_format() {
        Some(f) => {
            if !d.k.is_multiple_of(f.block_elems()) {
                return Err(bad(format!("matmul_id: {} experts need k % {} == 0", spec.w_dtype.name(), f.block_elems())));
            }
            let total = lapid_quant::layout(f, d.n, d.k).total;
            (scales_words(spec.w_dtype, d.n, d.k), total / 4)
        }
        None => (0, d.n.checked_mul(d.k).ok_or_else(|| bad("matmul_id: size overflow"))?),
    };
    if d.e == 0 && d.t * d.r > 0 {
        return Err(Error::new("index_out_of_range", "matmul_id: no experts"));
    }
    let ncb = d.n.div_ceil(gemv_cols(caps));
    let mut p = Params::default();
    p.u(d.t)?.u(d.r)?.u(d.n)?.u(d.k)?.u(so)?.u(stride)?.u(ncb)?.u(d.e)?;
    // expert offsets (x * stride) must fit 32-bit indexing
    p.u(d.e.checked_mul(stride).ok_or_else(|| bad("matmul_id: size overflow"))?)?;
    let items = (d.t * d.r * ncb) as u64;
    Ok(Plan { scratch: 0, next: None, key: Key::MatMulId(spec.clone()), params: p.0, groups: grid(items, caps) })
}

/// `fn wdeq(x, e)`: element `e` (row-major in the expert's [N, K]) of expert `x`.
fn w_loader(d: DType, caps: &Caps) -> String {
    match d {
        DType::Int8 => "@group(0) @binding(2) var<storage, read> w: array<u32>;\n\
            fn wdeq(x: u32, e: u32) -> f32 {\n    let wb = x * P[5];\n    \
            let q = extractBits(bitcast<i32>(w[wb + (e >> 2u)]), (e & 3u) * 8u, 8u);\n    \
            return f32(q) * unpack2x16float(w[wb + P[4] + (e >> 6u)])[(e >> 5u) & 1u];\n}\n"
            .to_string(),
        DType::Int4 => "@group(0) @binding(2) var<storage, read> w: array<u32>;\n\
            fn wdeq(x: u32, e: u32) -> f32 {\n    let wb = x * P[5];\n    \
            let r = e & 31u;\n    let byte = (e >> 5u) * 16u + (r & 15u);\n    \
            let q = f32((w[wb + (byte >> 2u)] >> ((byte & 3u) * 8u + (r >> 4u) * 4u)) & 15u) - 8.0;\n    \
            return q * unpack2x16float(w[wb + P[4] + (e >> 6u)])[(e >> 5u) & 1u];\n}\n"
            .to_string(),
        d if kquant::is_k(d) => {
            let bb = kquant::block_bytes(d);
            format!(
                "@group(0) @binding(2) var<storage, read> w: array<u32>;\n{}\
                 fn wdeq(x: u32, e: u32) -> f32 {{\n    return w_kdeq(x * P[5] * 4u + (e >> 8u) * {bb}u, e & 255u);\n}}\n",
                kquant::fns_dot("w", d)
            )
        }
        d => format!("{}fn wdeq(x: u32, e: u32) -> f32 {{\n    return ld_w(x * P[5] + e);\n}}\n", common::input(2, "w", d, caps)),
    }
}

pub(super) fn source(spec: &MatMulIdSpec, caps: &Caps) -> String {
    let mut bindings = common::input(1, "a", spec.x_dtype, caps);
    bindings.push_str(
        "fn lda4(e: u32) -> vec4<f32> {\n    return vec4<f32>(ld_a(e), ld_a(e + 1u), ld_a(e + 2u), ld_a(e + 3u));\n}\n",
    );
    bindings.push_str(&w_loader(spec.w_dtype, caps));
    bindings.push_str(&common::output(4, spec.out_dtype, caps));
    let dot = if kquant::is_k(spec.w_dtype) {
        let bb = kquant::block_bytes(spec.w_dtype);
        format!(
            "            let nb = K / 256u;\n            let row_byte = x * P[5] * 4u + j * nb * {bb}u;\n            \
             for (var idx = lane; idx < nb * 8u; idx += lanes) {{\n                let blk = idx >> 3u;\n                \
             part = part + w_kdot(row_byte + blk * {bb}u, idx & 7u, a_row + blk * 256u);\n            }}\n"
        )
    } else {
        "            for (var k = lane; k < K; k += lanes) {\n                part = part + ld_a(a_row + k) * wdeq(x, j * K + k);\n            }\n"
            .to_string()
    };
    template::module(caps, KERNEL, &[("BINDINGS", &bindings), ("DOT", &dot)], &[], 256, true)
}
