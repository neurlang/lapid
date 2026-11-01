//! DeepSeek V4 MoE FFN (HF `DeepseekV4SparseMoeBlock`, llama.cpp `build_moe_ffn` with
//! sqrt-softplus gating):
//!
//! ```text
//! s   = sqrt(softplus(x Wr^T))                                   [n, E]
//! ids = topk(s + bias, k)          (score layers)
//!     = tid2eid[token]             (the first `num_hash_layers` layers: hash routing)
//! w   = s[ids] / sum(s[ids]) * route_scale
//! y   = sum_r w[r] down_e(silu(min(gate_e(x), l)) * clamp(up_e(x), -l, l))   (MatMulId)
//!     + shared(x)                  (same clamped SwiGLU, always on)
//! ```
//! The experts are `lapid_nn::llm::moe::Experts` (with the clamp); routing is V4's own.

use lapid::{DType, Result, Tensor, bail};
use lapid_nn::Linear;
use lapid_nn::llm::moe::Experts;
use lapid_nn::llm::softplus;

pub(crate) struct Moe {
    /// Router `[E, D]` f32.
    pub router: Tensor,
    /// Selection bias `[E]` (score layers).
    pub bias: Option<Tensor>,
    /// Hash routing table `[vocab * k]` (hash layers).
    pub tid2eid: Option<Vec<i32>>,
    pub experts: Experts,
    /// Shared expert w1 (gate), w3 (up), w2 (down).
    pub shared: [Linear; 3],
    pub top_k: usize,
    pub n_experts: usize,
    pub route_scale: f32,
}

impl Moe {
    /// Expert ids `[n, k]` (i32) of the hash layers.
    fn hash_ids(&self, table: &[i32], tokens: &[u32]) -> Result<Tensor> {
        let k = self.top_k;
        let mut ids = Vec::with_capacity(tokens.len() * k);
        for &t in tokens {
            let row = table.get(t as usize * k..(t as usize + 1) * k);
            let Some(row) = row else { bail!("invalid_argument", "deepseek4: token {t} outside the hash routing table") };
            if let Some(&bad) = row.iter().find(|&&e| e < 0 || e as usize >= self.n_experts) {
                bail!("format", "deepseek4: hash routing table maps token {t} to expert {bad}");
            }
            ids.extend_from_slice(row);
        }
        Tensor::from_i32(&ids, &[tokens.len(), k])
    }

    /// `x [n, D]` (f32), `tokens` the n input ids -> `[n, D]`.
    pub fn forward(&self, x: &Tensor, tokens: &[u32]) -> Result<Tensor> {
        let n = x.shape()[0];
        if tokens.len() != n {
            bail!("shape_mismatch", "deepseek4 moe: {} tokens for {n} rows", tokens.len());
        }
        let k = self.top_k;
        let scores = softplus(&x.matmul_t(&self.router)?)?.sqrt()?; // [n, E]
        let ids = match (&self.tid2eid, &self.bias) {
            (Some(t), _) => self.hash_ids(t, tokens)?,
            (None, Some(b)) => scores.add(b)?.topk_indices(k, 1)?,
            (None, None) => scores.topk_indices(k, 1)?,
        };
        let w = scores.gather_axis(1, &ids)?;
        let w = w.div(&w.sum(1, true)?)?.mul_scalar(f64::from(self.route_scale))?; // [n, k]
        let routed = self.experts.forward(x, &ids, &w)?;
        let [w1, w3, w2] = &self.shared;
        let shared = w2.forward(&self.experts.glu(&w1.forward(x)?, &w3.forward(x)?)?)?;
        routed.add(&shared)?.cast(DType::F32)
    }
}
