//! Gated full attention with the QSA token indexer (HF `Qwen4ExpTextAttention` +
//! `Qwen4ExpTextQSAIndexer`, llama.cpp qwen4exp `build_layer_attn` / `build_qsa_sel`).
//!
//! Attention itself is Qwen3.5's: q (+ per-head output gate interleaved in q_proj), q/k
//! RMSNorm, partial RoPE, GQA, `o_proj(attn * sigmoid(gate))`.
//!
//! QSA restricts each query to the top `budget / ratio` blocks of `ratio` consecutive keys plus
//! its incomplete tail. A block is scored by one pooled key: the mean of the raw indexer keys
//! of its members, RMS-normed and rotated to the block's first position:
//!
//! ```text
//! score[t, b] = sum_h relu(q_h[t] . pooled[b]) / sqrt(head_dim)     (complete blocks b of query t)
//! keep key j  ⇔  block(j) in top-k(score[t]) or j >= ratio * floor((t + 1) / ratio)
//! ```
//! Raw indexer keys are cached per layer (a second KV cache). While the key range is at most
//! `budget + ratio - 1` tokens every block is selected and QSA equals dense causal attention,
//! so the indexer only scores longer contexts. Top-k is a threshold on the k-th largest score;
//! scores equal to it (exact zeros from the relu are common) are kept lowest block first, the
//! order `topk` uses (torch leaves the order of ties unspecified).

use lapid::{BinaryOp, DType, Result, Tensor};
use lapid_nn::attention::{Mask, merge_heads, scaled_dot_product_attention};
use lapid_nn::{KvCache, Linear, RmsNorm, Rope, ops};

pub(crate) struct Indexer {
    pub q: Linear,
    pub k: Linear,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    /// Rotates the first `rotary_dim` dims of a `head_dim` index head (same frequencies as attention).
    pub rope: Rope,
    pub n_head: usize,
    pub head_dim: usize,
    pub budget: usize,
    pub ratio: usize,
}

pub(crate) struct GatedAttention {
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
    pub indexer: Option<Indexer>,
}

/// Exclusive prefix sum along axis 1 of x [n, m] (Hillis-Steele scan: log2(m) shifted adds).
fn exclusive_cumsum(x: &Tensor) -> Result<Tensor> {
    let m = x.shape()[1];
    let mut acc = x.clone();
    let mut step = 1;
    while step < m {
        let shifted = acc.slice(1, 0, m - step)?.pad(1, step, 0, 0.0)?;
        acc = acc.add(&shifted)?;
        step *= 2;
    }
    acc.sub(x)
}

fn f32_leaf(v: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
    Tensor::from_f32(&v, shape)
}

impl Indexer {
    /// Additive [1, 1, n, S] mask of the keys each of the `n` queries keeps; `keys` are all raw
    /// indexer keys [1, 1, S, head_dim], queries at positions `offset..offset + n`.
    fn mask(&self, h: &Tensor, keys: &Tensor, offset: usize) -> Result<Tensor> {
        let (n, s) = (h.shape()[1], keys.shape()[2]);
        let (r, hd, nh) = (self.ratio, self.head_dim, self.n_head);
        let nb = s / r;
        let pooled = keys.slice(2, 0, nb * r)?.reshape(&[1, 1, nb, r, hd])?.cast(DType::F32)?.mean(3, false)?;
        let pooled = self.k_norm.forward(&pooled)?;
        let starts: Vec<usize> = (0..nb).map(|b| b * r).collect();
        let pooled = self.rope.apply_positions(&pooled, &starts)?.reshape(&[nb, hd])?;
        let q = self.q.forward(h)?.cast(DType::F32)?.reshape(&[1, n, nh, hd])?;
        let q = self.rope.apply(&self.q_norm.forward(&q)?.permute(&[0, 2, 1, 3])?, offset)?; // [1, nh, n, hd]
        let scores = q.contiguous()?.reshape(&[nh * n, hd])?.matmul_t(&pooled)?.reshape(&[nh, n, nb])?;
        let scores = ops::relu(&scores)?.sum(0, false)?.mul_scalar(1.0 / (hd as f64).sqrt())?; // [n, nb]
        // complete blocks of query t: b < floor((offset + t + 1) / r)
        let done: Vec<f32> = (0..n).map(|t| ((offset + t + 1) / r) as f32).collect();
        let done = f32_leaf(done, &[n, 1])?;
        let bidx = f32_leaf((0..nb).map(|b| b as f32).collect(), &[1, nb])?;
        let valid = bidx.binary(BinaryOp::CmpLt, &done)?; // [n, nb]
        let neg = Tensor::full(-1e30, &[1, 1], DType::F32);
        let scores = Tensor::where_(&valid, &scores, &neg)?;
        let k = self.budget.div_ceil(r).min(nb).max(1);
        let (top, _) = scores.topk(k, 1)?;
        let thr = top.slice(1, k - 1, k)?; // [n, 1]
        let one = Tensor::full(1.0, &[1, 1], DType::F32);
        // above the threshold, plus the lowest-index blocks among those equal to it (relu makes
        // exact ties at 0 common) until k are kept, as `topk` orders ties
        let above = thr.binary(BinaryOp::CmpLt, &scores)?; // [n, nb]
        let tied = scores.binary(BinaryOp::CmpEq, &thr)?.mul(&valid)?;
        let need = Tensor::full(k as f64, &[1, 1], DType::F32).sub(&above.sum(1, true)?)?; // [n, 1]
        let rank = exclusive_cumsum(&tied)?;
        let picked = above.add(&tied.mul(&rank.binary(BinaryOp::CmpLt, &need)?)?)?.mul(&valid)?; // [n, nb]
        let mut keep = picked.unsqueeze(2)?.expand(&[n, nb, r])?.reshape(&[n, nb * r])?;
        if nb * r < s {
            keep = Tensor::cat(&[&keep, &Tensor::full(0.0, &[n, s - nb * r], DType::F32)], 1)?;
        }
        // the incomplete tail of each query: j >= r * floor((t + 1) / r)
        let tail_start = f32_leaf((0..n).map(|t| (r * ((offset + t + 1) / r)) as f32).collect(), &[n, 1])?;
        let jidx = f32_leaf((0..s).map(|j| j as f32).collect(), &[1, s])?;
        let tail = one.sub(&jidx.binary(BinaryOp::CmpLt, &tail_start)?)?;
        let keep = keep.maximum(&tail)?;
        let hidden = Tensor::full(f64::NEG_INFINITY, &[1, 1], DType::F32);
        let zero = Tensor::full(0.0, &[1, 1], DType::F32);
        Tensor::where_(&keep, &zero, &hidden)?.reshape(&[1, 1, n, s])
    }
}

impl GatedAttention {
    /// h [1, n, D] → [1, n, D]. `idx` caches the raw indexer keys of the QSA layers.
    pub fn forward(&self, h: &Tensor, cache: &mut KvCache, idx: &mut KvCache, layer: usize, offset: usize) -> Result<Tensor> {
        let n = h.shape()[1];
        let (nh, nkv, dh) = (self.n_head, self.n_kv, self.head_dim);
        let mut mask = Mask::Causal;
        if let Some(ix) = &self.indexer {
            let raw = ix.k.forward(h)?.cast(DType::F32)?.reshape(&[1, 1, n, ix.head_dim])?;
            let (keys, _) = idx.append(layer, &raw, &raw)?;
            if offset + n > ix.budget + ix.ratio - 1 {
                mask = Mask::Custom { additive: ix.mask(h, &keys, offset)?, causal: true };
            }
        }
        let qg = self.q.forward(h)?.reshape(&[1, n, nh, 2 * dh])?;
        let q = self.q_norm.forward(&qg.slice(3, 0, dh)?)?.permute(&[0, 2, 1, 3])?;
        let gate = qg.slice(3, dh, 2 * dh)?.contiguous()?.reshape(&[1, n, nh * dh])?;
        let k = self.k_norm.forward(&self.k.forward(h)?.reshape(&[1, n, nkv, dh])?)?.permute(&[0, 2, 1, 3])?;
        let v = self.v.forward(h)?.reshape(&[1, n, nkv, dh])?.permute(&[0, 2, 1, 3])?;
        let q = self.rope.apply(&q, offset)?;
        let k = self.rope.apply(&k, offset)?;
        let (k, v) = cache.append(layer, &k, &v)?;
        let a = scaled_dot_product_attention(&q, &k, &v, 1.0 / (dh as f32).sqrt(), &mask)?;
        let a = merge_heads(&a)?.mul(&ops::sigmoid(&gate)?)?;
        self.o.forward(&a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Host reference of the HF indexer (`Qwen4ExpTextQSAIndexer.forward`) for one head row.
    fn norm_rope(x: &[f32], w: &[f32], pos: usize, rot: usize, theta: f32) -> Vec<f32> {
        let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let mut y: Vec<f32> = x.iter().zip(w).map(|(v, g)| v / (ms + 1e-6).sqrt() * g).collect();
        let half = rot / 2;
        for i in 0..half {
            let a = pos as f32 * (1.0 / theta.powf((2 * i) as f32 / rot as f32));
            let (c, s) = ((a as f64).cos() as f32, (a as f64).sin() as f32);
            let (x1, x2) = (y[i], y[i + half]);
            y[i] = x1 * c - x2 * s;
            y[i + half] = x2 * c + x1 * s;
        }
        y
    }

    /// Real QSA shapes (4 heads x 128, budget 2048, ratio 4) past the dense limit: the kept-key
    /// mask equals a direct transcription of the HF selection loop.
    #[test]
    fn qsa_mask_matches_reference_at_real_shapes() {
        let (d, nh, hd, rot, budget, r, theta) = (16usize, 4usize, 128usize, 64usize, 2048usize, 4usize, 1e7f32);
        let (s, n) = (2110usize, 3usize);
        let offset = s - n;
        let wq = rnd(nh * hd * d, 1);
        let qn = rnd(hd, 2).iter().map(|v| 1.0 + 0.2 * v).collect::<Vec<_>>();
        let kn = rnd(hd, 3).iter().map(|v| 1.0 + 0.2 * v).collect::<Vec<_>>();
        let h = rnd(n * d, 4);
        let keys = rnd(s * hd, 5);
        let ix = Indexer {
            q: Linear::new(Tensor::from_f32(&wq, &[nh * hd, d]).expect("wq"), None),
            k: Linear::new(Tensor::from_f32(&rnd(hd * d, 6), &[hd, d]).expect("wk"), None),
            q_norm: RmsNorm::new(Tensor::from_f32(&qn, &[hd]).expect("qn"), 1e-6),
            k_norm: RmsNorm::new(Tensor::from_f32(&kn, &[hd]).expect("kn"), 1e-6),
            rope: Rope::new(hd, theta).with_rotary_dim(rot),
            n_head: nh,
            head_dim: hd,
            budget,
            ratio: r,
        };
        let ht = Tensor::from_f32(&h, &[1, n, d]).expect("h");
        let kt = Tensor::from_f32(&keys, &[1, 1, s, hd]).expect("keys");
        let dev = lapid::Device::cpu();
        let m = ix.mask(&ht, &kt, offset).expect("mask");
        let got = dev.to_vec_f32(&dev.eval(&[&m]).expect("eval")[0]).expect("vec");
        for t in 0..n {
            let p = offset + t;
            let nb = (p + 1) / r;
            let blocks: Vec<Vec<f32>> = (0..nb)
                .map(|b| {
                    let mean: Vec<f32> = (0..hd).map(|e| (0..r).map(|j| keys[(b * r + j) * hd + e]).sum::<f32>() / r as f32).collect();
                    norm_rope(&mean, &kn, b * r, rot, theta)
                })
                .collect();
            let qs: Vec<Vec<f32>> = (0..nh)
                .map(|hh| {
                    let row: Vec<f32> = (0..hd).map(|o| (0..d).map(|i| h[t * d + i] * wq[(hh * hd + o) * d + i]).sum()).collect();
                    norm_rope(&row, &qn, p, rot, theta)
                })
                .collect();
            let mut score: Vec<(f32, usize)> = blocks
                .iter()
                .enumerate()
                .map(|(b, k)| (qs.iter().map(|q| q.iter().zip(k).map(|(a, c)| a * c).sum::<f32>().max(0.0)).sum::<f32>(), b))
                .collect();
            score.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let picked: Vec<usize> = score.iter().take(budget / r).map(|x| x.1).collect();
            for j in 0..s {
                let want = j <= p && (j >= nb * r || picked.contains(&(j / r)));
                let kept = got[t * s + j] == 0.0;
                // the causal rule is applied by attention itself; compare visible keys only
                if j <= p {
                    assert_eq!(kept, want, "query {t} key {j}");
                }
            }
        }
    }
}
