//! Linear algebra, indexing, in-place update and composite ops (lazy, see `ops`).

use engine::Tensor;

use crate::error::{LapidStatus, guard};
use crate::handle::{LapidTensor, axis_arg, put_tensor, tensor};
use crate::ops::{op1, op2};

/// a @ b with batch broadcasting: a `[..., m, k]`, b `[..., k, n]` -> `[..., m, n]`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_matmul(a: *const LapidTensor, b: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op2(a, b, out, Tensor::matmul) }
}

/// a @ b^T: a `[..., m, k]`, b `[n, k]` (or batched `[..., n, k]`) -> `[..., m, n]`. The
/// Linear-layer form (x @ W^T); b may be a quantized 2-D weight.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_matmul_t(a: *const LapidTensor, b: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op2(a, b, out, Tensor::matmul_t) }
}

/// Embedding lookup: `table` `[V, D]` (float or quantized), `indices` i32 of any shape S
/// -> `[S..., D]` f32.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_gather(
    table: *const LapidTensor,
    indices: *const LapidTensor,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    unsafe { op2(table, indices, out, Tensor::gather) }
}

/// take_along_axis: `indices` (i32) has x's rank and dims except along `axis`;
/// `out[.., j, ..] = x[.., indices[.., j, ..], ..]`. Out-of-range indices fail at eval
/// with `LAPID_ERR_INDEX_OUT_OF_RANGE`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_gather_axis(
    x: *const LapidTensor,
    axis: i64,
    indices: *const LapidTensor,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    unsafe { op2(x, indices, out, |x, i| x.gather_axis(axis_arg(axis)?, i)) }
}

/// The `k` largest values along `axis` (sorted descending, ties: lower index first):
/// `*values` in x's dtype and `*indices` i32, both with `axis` set to `k`. On error neither
/// output is written.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_topk(
    x: *const LapidTensor,
    k: usize,
    axis: i64,
    values: *mut *mut LapidTensor,
    indices: *mut *mut LapidTensor,
) -> LapidStatus {
    guard(|| {
        let x = unsafe { tensor(x)? };
        if values.is_null() || indices.is_null() {
            return Err(crate::error::invalid("values / indices output is NULL"));
        }
        let (v, i) = x.topk(k, axis_arg(axis)?)?;
        put_tensor(values, v)?;
        put_tensor(indices, i)
    })
}

/// Only the indices (i32) of `lapid_topk`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_topk_indices(
    x: *const LapidTensor,
    k: usize,
    axis: i64,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    unsafe { op1(x, out, |x| x.topk_indices(k, axis_arg(axis)?)) }
}

/// `dest` with `[start, start + src.shape[axis])` along `axis` replaced by `src` (same
/// dtype and rank). When `dest` is a realized device tensor used by nothing else in the
/// evaluated graph, the write is in place (the result aliases dest's buffer, and `dest`
/// observes the new contents after eval): the KV-cache update primitive.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_write_slice(
    dest: *const LapidTensor,
    axis: i64,
    start: usize,
    src: *const LapidTensor,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    unsafe { op2(dest, src, out, |d, s| d.write_slice(axis_arg(axis)?, start, s)) }
}

/// Softmax over the last axis.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_softmax(x: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, Tensor::softmax) }
}

/// Layer norm over the last axis: `(x - mean) / sqrt(var + eps) * weight + bias`
/// (`weight`, `bias` `[D]`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_layer_norm(
    x: *const LapidTensor,
    weight: *const LapidTensor,
    bias: *const LapidTensor,
    eps: f32,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    guard(|| {
        let (x, w, b) = unsafe { (tensor(x)?, tensor(weight)?, tensor(bias)?) };
        put_tensor(out, x.layer_norm(w, b, eps)?)
    })
}

/// RMS norm over the last axis: `x * rsqrt(mean(x^2) + eps) * weight` (`weight` `[D]`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_rms_norm(
    x: *const LapidTensor,
    weight: *const LapidTensor,
    eps: f32,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    unsafe { op2(x, weight, out, |x, w| x.rms_norm(w, eps)) }
}

/// Scaled dot-product attention: q `[B, H, Sq, D]`, k / v `[B, H, Sk, D]` (same float
/// dtype), optional additive f32 `mask` broadcastable to `[B, H, Sq, Sk]` (NULL for none).
/// `causal`: query i attends key j iff j <= i + (Sk - Sq). Output `[B, H, Sq, D]`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_attention(
    q: *const LapidTensor,
    k: *const LapidTensor,
    v: *const LapidTensor,
    mask: *const LapidTensor,
    scale: f32,
    causal: bool,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    guard(|| {
        let (q, k, v) = unsafe { (tensor(q)?, tensor(k)?, tensor(v)?) };
        let mask = if mask.is_null() { None } else { Some(unsafe { tensor(mask)? }) };
        put_tensor(out, Tensor::attention(q, k, v, mask, scale, causal)?)
    })
}

/// Mixture-of-experts matmul (ggml MUL_MAT_ID): x `[T, K]` (float), experts `w` `[E, N, K]`
/// (float or quantized, each in matmul_t form), `ids` i32 `[T, R]` -> `[T, R, N]` with
/// `out[t, r] = w[ids[t, r]] @ x[t]`. Out-of-range ids fail at eval.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_matmul_id(
    x: *const LapidTensor,
    w: *const LapidTensor,
    ids: *const LapidTensor,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    guard(|| {
        let (x, w, ids) = unsafe { (tensor(x)?, tensor(w)?, tensor(ids)?) };
        put_tensor(out, x.matmul_id(w, ids)?)
    })
}

/// Gated delta rule (ggml GATED_DELTA_NET): q, k `[B, T, H, Dk]`, v `[B, T, H, Dv]`, g (log
/// decay) and beta `[B, T, H]`, state `[B, H, Dk, Dv]`, all f32. Returns `[B, H, T + Dk, Dv]`:
/// rows `[0, T)` are the outputs, rows `[T, T + Dk)` the final state.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_gated_delta_net(
    q: *const LapidTensor,
    k: *const LapidTensor,
    v: *const LapidTensor,
    g: *const LapidTensor,
    beta: *const LapidTensor,
    state: *const LapidTensor,
    scale: f32,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    guard(|| {
        let (q, k, v) = unsafe { (tensor(q)?, tensor(k)?, tensor(v)?) };
        let (g, beta, state) = unsafe { (tensor(g)?, tensor(beta)?, tensor(state)?) };
        put_tensor(out, Tensor::gated_delta_net(q, k, v, g, beta, state, scale)?)
    })
}
