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

use lapid::{DType, Result, Tensor, bail};
use lapid_nn::weights::fake_quant_act;
use lapid_nn::{Linear, ops};

pub(crate) struct Moe {
    /// Router `[E, D]` f32.
    pub router: Tensor,
    /// Selection bias `[E]` (score layers).
    pub bias: Option<Tensor>,
    /// Hash routing table `[vocab * k]` (hash layers).
    pub tid2eid: Option<Vec<i32>>,
    pub w1: Tensor,
    pub w3: Tensor,
    pub w2: Tensor,
    pub act_quant: bool,
    pub shared_w1: Linear,
    pub shared_w3: Linear,
    pub shared_w2: Linear,
    pub top_k: usize,
    pub n_experts: usize,
    pub route_scale: f32,
    pub limit: f32,
}

/// `log(1 + exp(x))`, overflow-safe: `max(x, 0) + log(1 + exp(-|x|))`.
fn softplus(x: &Tensor) -> Result<Tensor> {
    let pos = ops::relu(x)?;
    let neg_abs = pos.mul_scalar(-2.0)?.add(x)?; // -|x| = x - 2 max(x, 0)
    pos.add(&neg_abs.exp()?.add_scalar(1.0)?.log()?)
}

fn clamp(x: &Tensor, lo: Option<f32>, hi: f32) -> Result<Tensor> {
    let x = x.binary_scalar(lapid::BinaryOp::Min, f64::from(hi))?;
    match lo {
        Some(l) => x.binary_scalar(lapid::BinaryOp::Max, f64::from(l)),
        None => Ok(x),
    }
}

impl Moe {
    fn swiglu(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        let (g, u) = if self.limit > 0.0 {
            (clamp(gate, None, self.limit)?, clamp(up, Some(-self.limit), self.limit)?)
        } else {
            (gate.clone(), up.clone())
        };
        ops::silu(&g)?.mul(&u)
    }

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

    /// `x [n, D]` (f32), `tokens` the n input ids → `[n, D]`.
    pub fn forward(&self, x: &Tensor, tokens: &[u32]) -> Result<Tensor> {
        let (n, d) = (x.shape()[0], x.shape()[1]);
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
        let quant = |t: &Tensor| -> Result<Tensor> { if self.act_quant { fake_quant_act(t) } else { Ok(t.clone()) } };
        let xq = quant(x)?;
        let gate = xq.matmul_id(&self.w1, &ids)?; // [n, k, I]
        let up = xq.matmul_id(&self.w3, &ids)?;
        let inter = gate.shape()[2];
        let act = self.swiglu(&gate, &up)?.reshape(&[n * k, inter])?;
        let down = quant(&act)?.matmul_id(&self.w2, &ids.reshape(&[n * k, 1])?)?; // [n*k, 1, D]
        let routed = down.reshape(&[n, k, d])?.mul(&w.unsqueeze(2)?)?.sum(1, false)?;
        let shared = self.shared_w2.forward(&self.swiglu(&self.shared_w1.forward(x)?, &self.shared_w3.forward(x)?)?)?;
        routed.add(&shared)?.cast(DType::F32)
    }
}
