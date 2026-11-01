//! Gemma 4 decoder blocks: attention (q/k RMSNorm, weight-free v norm, scale 1.0, GQA,
//! zero-tail RoPE, KV sharing, `attention_k_eq_v`), the sparse MoE block of the 26B-A4B
//! (`Gemma4TextRouter` + `Gemma4TextExperts`) and the image-block attention mask.
//!
//! MoE (HF `Gemma4TextDecoderLayer` with `enable_moe_block`), on the residual `r`:
//! ```text
//! s = (rms(r) * router.scale * D^-0.5) Wr^T;  p = softmax(s);  (w, ids) = topk(p, k)
//! w = w / sum(w) * per_expert_scale[ids]
//! y = post_norm_2( sum_r w[r] * down_e(gelu_tanh(gate_e(x2)) * up_e(x2)) ),  x2 = pre_norm_2(r)
//! ```
//! Expert stacks: fused `gate_up` [E, 2I, D] (gate rows first) and `down` [E, D, I], run with
//! `Tensor::matmul_id`.

use lapid::{DType, Error, Result, Tensor, UnaryOp, bail};
use lapid_nn::attention::{merge_heads, repeat_kv, scaled_dot_product_attention};
use lapid_nn::weights::fake_quant_act;
use lapid_nn::{KvCache, Linear, Mask, RmsNorm, ops};

use super::rope::ZeroTailRope;

/// Largest head dim the fused attention kernels take on every backend (CUDA / Vulkan: 256).
pub(crate) const FUSED_ATTENTION_MAX_HEAD_DIM: usize = 256;

/// Attention from primitives (matmul + softmax) for head dims the fused kernels do not take
/// (Gemma 4 full-attention layers: 512). q [1, H, n, D], k / v [1, Hkv, Sk, D]; `mask` is an
/// additive [n, Sk] mask, else causal.
pub(crate) fn attention_unfused(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, mask: Option<&Tensor>) -> Result<Tensor> {
    let (h, n) = (q.shape()[1], q.shape()[2]);
    let hkv = k.shape()[1];
    let sk = k.shape()[2];
    let (k, v) = (repeat_kv(k, h / hkv.max(1))?, repeat_kv(v, h / hkv.max(1))?);
    let qf = q.cast(DType::F32)?;
    let mut s = qf.matmul_t(&k.cast(DType::F32)?)?;
    if scale != 1.0 {
        s = s.mul_scalar(f64::from(scale))?;
    }
    if let Some(m) = mask {
        s = s.add(m)?;
    } else if n > 1 {
        let off = sk - n;
        let mut m = vec![0.0f32; n * sk];
        for i in 0..n {
            for j in (i + off + 1)..sk {
                m[i * sk + j] = f32::NEG_INFINITY;
            }
        }
        s = s.add(&Tensor::from_f32(&m, &[n, sk])?)?;
    }
    s.softmax()?.matmul(&v.cast(DType::F32)?)?.cast(q.dtype())
}

/// RMSNorm without weight (Gemma 4 `v_norm`, router norm), in f32.
pub(crate) fn rms_plain(x: &Tensor, eps: f32) -> Result<Tensor> {
    let xf = x.cast(DType::F32)?;
    let inv = xf.mul(&xf)?.mean(-1, true)?.add_scalar(f64::from(eps))?.unary(UnaryOp::Rsqrt)?;
    xf.mul(&inv)
}

/// Additive [n, sk] mask of a sliding layer whose chunk holds image blocks: key j is
/// visible to query i iff `pos(i) - pos(j) < window` and (`j <= i` or both are in the same
/// block). `blocks[i]` = block id of chunk row i (-1 = text); the chunk is the last `n` keys
/// (blocks never span chunks, so earlier keys are never in a query's block).
pub(crate) fn block_sliding_mask(blocks: &[i32], sk: usize, window: Option<usize>) -> Result<Tensor> {
    let n = blocks.len();
    if sk < n {
        bail!("shape_mismatch", "gemma4moe: {sk} keys for {n} queries");
    }
    let off = sk - n;
    let mut m = vec![0.0f32; n * sk];
    for i in 0..n {
        let qpos = i + off;
        for j in 0..sk {
            let causal_ok = j <= qpos;
            let same = j >= off && blocks[i] >= 0 && blocks[j - off] == blocks[i];
            let in_window = window.is_none_or(|w| j > qpos || qpos - j < w);
            if !((causal_ok || same) && in_window) {
                m[i * sk + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_f32(&m, &[n, sk])
}

pub(crate) struct Attn {
    pub q: Linear,
    /// `None` on kv-shared layers.
    pub k: Option<Linear>,
    /// `None` on kv-shared layers and on k_eq_v full layers (V = K before k_norm).
    pub v: Option<Linear>,
    pub o: Linear,
    pub q_norm: RmsNorm,
    pub k_norm: Option<RmsNorm>,
    pub rope: ZeroTailRope,
    /// Sliding window (`None` = full attention, causal).
    pub window: Option<usize>,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    /// Layer whose K/V this layer attends to (itself unless kv-shared).
    pub source: usize,
    pub eps: f32,
}

pub(crate) type SharedKv = Vec<Option<(Tensor, Tensor)>>;

impl Attn {
    /// h [1, n, D] -> [1, n, D]. `blocks` (per chunk row, -1 = text) switches sliding layers
    /// to the bidirectional image-block mask.
    pub fn forward(&self, h: &Tensor, cache: &mut KvCache, layer: usize, shared: &mut SharedKv, blocks: Option<&[i32]>) -> Result<Tensor> {
        let n = h.shape()[1];
        let offset = cache.offset();
        let q = self.q.forward(h)?.reshape(&[1, n, self.n_head, self.head_dim])?;
        let q = self.q_norm.forward(&q)?.permute(&[0, 2, 1, 3])?;
        let q = self.rope.apply(&q, offset)?;
        let (k, v) = match (&self.k, &self.k_norm) {
            (Some(kp), Some(kn)) => {
                let kraw = kp.forward(h)?.reshape(&[1, n, self.n_kv, self.head_dim])?;
                let vraw = match &self.v {
                    Some(vp) => vp.forward(h)?.reshape(&[1, n, self.n_kv, self.head_dim])?,
                    None => kraw.clone(),
                };
                let k = self.rope.apply(&kn.forward(&kraw)?.permute(&[0, 2, 1, 3])?, offset)?;
                let v = rms_plain(&vraw, self.eps)?.permute(&[0, 2, 1, 3])?;
                let kv = cache.append(layer, &k, &v)?;
                shared[layer] = Some(kv.clone());
                kv
            }
            _ => shared
                .get(self.source)
                .cloned()
                .flatten()
                .ok_or_else(|| Error::new("internal", format!("gemma4moe: layer {layer} reads K/V of layer {} before it ran", self.source)))?,
        };
        let custom = match (blocks, self.window) {
            (Some(b), Some(w)) => Some(block_sliding_mask(b, k.shape()[2], Some(w))?),
            _ => None,
        };
        let a = if self.head_dim > FUSED_ATTENTION_MAX_HEAD_DIM && (self.window.is_none() || custom.is_some()) {
            attention_unfused(&q, &k, &v, 1.0, custom.as_ref())?
        } else {
            let mask = match (custom, self.window) {
                (Some(additive), _) => Mask::Custom { additive, causal: false },
                (None, Some(w)) => Mask::Sliding(w),
                (None, None) => Mask::Causal,
            };
            scaled_dot_product_attention(&q, &k, &v, 1.0, &mask)?
        };
        self.o.forward(&merge_heads(&a)?)
    }
}

/// Sparse expert block of one layer.
pub(crate) struct Moe {
    /// Router projection [E, D].
    pub router: Tensor,
    /// Router input scale [D] (f32).
    pub router_scale: Tensor,
    /// Per-expert output scale [E] (f32).
    pub per_expert_scale: Tensor,
    /// [E, 2I, D] (gate rows, then up rows).
    pub gate_up: Tensor,
    /// [E, D, I].
    pub down: Tensor,
    pub top_k: usize,
    pub n_experts: usize,
    /// Quantize the expert matmul inputs (fake-quant activation reference).
    pub act_quant: bool,
    pub pre_norm_2: RmsNorm,
    pub post_norm_2: RmsNorm,
    pub eps: f32,
}

impl Moe {
    /// residual [1, n, D] → post_norm_2(experts(pre_norm_2(residual))) [1, n, D].
    pub fn forward(&self, residual: &Tensor) -> Result<Tensor> {
        let (n, d) = (residual.shape()[1], residual.shape()[2]);
        let k = self.top_k;
        let r = residual.reshape(&[n, d])?;
        let rin = rms_plain(&r, self.eps)?.mul(&self.router_scale)?.mul_scalar(1.0 / (d as f64).sqrt())?;
        let probs = rin.matmul_t(&self.router)?.cast(DType::F32)?.softmax()?; // [n, E]
        let (w, ids) = probs.topk(k, 1)?; // [n, k]
        let pes = self.per_expert_scale.reshape(&[1, self.n_experts])?.expand(&[n, self.n_experts])?;
        let w = w.div(&w.sum(1, true)?)?.mul(&pes.gather_axis(1, &ids)?)?;
        let x2 = self.pre_norm_2.forward(&r)?.cast(DType::F32)?;
        let quant = |t: &Tensor| -> Result<Tensor> { if self.act_quant { fake_quant_act(t) } else { Ok(t.clone()) } };
        let gu = quant(&x2)?.matmul_id(&self.gate_up, &ids)?; // [n, k, 2I]
        let inter = gu.shape()[2] / 2;
        let act = ops::gelu_tanh(&gu.slice(2, 0, inter)?)?.mul(&gu.slice(2, inter, 2 * inter)?)?;
        let act = act.reshape(&[n * k, inter])?;
        let down = quant(&act)?.matmul_id(&self.down, &ids.reshape(&[n * k, 1])?)?; // [n*k, 1, D]
        let y = down.reshape(&[n, k, d])?.mul(&w.unsqueeze(2)?)?.sum(1, false)?;
        self.post_norm_2.forward(&y.reshape(&[1, n, d])?)
    }
}
