//! Gated full attention (HF `Qwen3_5Attention`, also Qwen4-Exp): `q_proj` emits per head the
//! query and an output gate (`[q | gate]`, 2 x head_dim), q / k RMSNorm, partial half-split
//! RoPE, GQA:
//!
//! ```text
//! out = o_proj(attention(rope(q_norm(q)), rope(k_norm(k)), v) * sigmoid(gate))
//! ```

use lapid::{Result, Tensor};

use crate::attention::{KvCache, Mask, merge_heads, scaled_dot_product_attention};
use crate::layers::{Linear, RmsNorm};
use crate::ops;
use crate::rope::Rope;
use crate::weights::Loader;

pub struct GatedAttention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    pub rope: Rope,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
}

impl GatedAttention {
    /// `{p}.{q,k,v,o}_proj`, `{p}.{q,k}_norm` (no biases); `d` model width.
    pub fn load(l: &Loader<'_>, p: &str, d: usize, [n_head, n_kv, head_dim]: [usize; 3], eps: f32, rope: Rope) -> Result<GatedAttention> {
        let lin = |n: &str, o: usize, i: usize| l.linear_checked(&format!("{p}.{n}"), o, i, false);
        Ok(GatedAttention {
            q: lin("q_proj", 2 * n_head * head_dim, d)?,
            k: lin("k_proj", n_kv * head_dim, d)?,
            v: lin("v_proj", n_kv * head_dim, d)?,
            o: lin("o_proj", d, n_head * head_dim)?,
            q_norm: l.rms_norm_checked(&format!("{p}.q_norm"), head_dim, eps)?,
            k_norm: l.rms_norm_checked(&format!("{p}.k_norm"), head_dim, eps)?,
            rope,
            n_head,
            n_kv,
            head_dim,
        })
    }

    /// h [1, n, D] -> [1, n, D] at positions `offset..` (cache layer `layer`).
    pub fn forward(&self, h: &Tensor, cache: &mut KvCache, layer: usize, offset: usize, mask: &Mask) -> Result<Tensor> {
        let n = h.shape()[1];
        let (nh, nkv, dh) = (self.n_head, self.n_kv, self.head_dim);
        let qg = self.q.forward(h)?.reshape(&[1, n, nh, 2 * dh])?;
        let q = self.q_norm.forward(&qg.slice(3, 0, dh)?)?.permute(&[0, 2, 1, 3])?;
        let gate = qg.slice(3, dh, 2 * dh)?.contiguous()?.reshape(&[1, n, nh * dh])?;
        let k = self.k_norm.forward(&self.k.forward(h)?.reshape(&[1, n, nkv, dh])?)?.permute(&[0, 2, 1, 3])?;
        let v = self.v.forward(h)?.reshape(&[1, n, nkv, dh])?.permute(&[0, 2, 1, 3])?;
        let (k, v) = cache.append(layer, &self.rope.apply(&k, offset)?, &v)?;
        let a = scaled_dot_product_attention(&self.rope.apply(&q, offset)?, &k, &v, 1.0 / (dh as f32).sqrt(), mask)?;
        self.o.forward(&merge_heads(&a)?.mul(&ops::sigmoid(&gate)?)?)
    }
}
