//! Sparse MoE FFN of `qwen3_5_moe` (HF `Qwen3_5MoeSparseMoeBlock`, llama.cpp `build_moe_ffn`
//! with softmax gating + renormalized top-k, plus the sigmoid-gated shared expert).
//!
//! ```text
//! p = softmax(x Wr^T);  (w, ids) = topk(p, k);  w /= sum(w)
//! y = sum_r w[r] * down_e(silu(gate_e(x)) * up_e(x))   (e = ids[r], MatMulId)
//!   + sigmoid(x Ws^T) * shared_mlp(x)
//! ```
//! Expert stacks are `[E, N, K]` (one b_transposed matrix per expert, quantized stacks E
//! matrices back to back).

use lapid::{DType, Result, Tensor, bail};
use lapid_nn::weights::fake_quant_act;
use lapid_nn::{GatedMlp, ops};

/// The MoE primitive (`Op::MatMulId`): x [T, K] float, w [E, N, K], ids I32 [T, R] → [T, R, N].
pub(crate) fn matmul_id(x: &Tensor, w: &Tensor, ids: &Tensor) -> Result<Tensor> {
    x.matmul_id(w, ids)
}

pub(crate) struct Experts {
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
    /// Quantize the expert matmul inputs (fake-quant activation reference).
    pub act_quant: bool,
}

pub(crate) struct SparseMoe {
    /// Router [E, D] (f32).
    pub router: Tensor,
    pub experts: Experts,
    pub top_k: usize,
    pub shared: GatedMlp,
    /// Shared expert gate [1, D].
    pub shared_gate: Tensor,
}

impl SparseMoe {
    /// h [1, n, D] → [1, n, D].
    pub fn forward(&self, h: &Tensor) -> Result<Tensor> {
        if h.rank() != 3 || h.shape()[0] != 1 {
            bail!("shape_mismatch", "moe expects [1, n, D], got {:?}", h.shape());
        }
        let (n, d) = (h.shape()[1], h.shape()[2]);
        let k = self.top_k;
        let x = h.reshape(&[n, d])?.cast(DType::F32)?;
        let probs = x.matmul_t(&self.router)?.softmax()?; // [n, E]
        let (w, ids) = probs.topk(k, 1)?; // [n, k]
        let w = w.div(&w.sum(1, true)?)?;
        let quant = |t: &Tensor| -> Result<Tensor> { if self.experts.act_quant { fake_quant_act(t) } else { Ok(t.clone()) } };
        let xq = quant(&x)?;
        let gate = matmul_id(&xq, &self.experts.gate, &ids)?; // [n, k, I]
        let up = matmul_id(&xq, &self.experts.up, &ids)?;
        let inter = gate.shape()[2];
        let act = ops::silu(&gate)?.mul(&up)?.reshape(&[n * k, inter])?;
        let down = matmul_id(&quant(&act)?, &self.experts.down, &ids.reshape(&[n * k, 1])?)?; // [n*k, 1, D]
        let routed = down.reshape(&[n, k, d])?.mul(&w.unsqueeze(2)?)?.sum(1, false)?; // [n, D]
        let xs = x.reshape(&[1, n, d])?;
        let shared = self.shared.forward(&xs)?.reshape(&[n, d])?;
        let sg = ops::sigmoid(&x.matmul_t(&self.shared_gate)?)?; // [n, 1]
        routed.add(&shared.mul(&sg)?)?.reshape(&[1, n, d])
    }
}
