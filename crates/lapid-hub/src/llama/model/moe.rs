//! Sparse MoE FFN of `qwen3_moe` (HF `Qwen3MoeSparseMoeBlock`, llama.cpp `build_moe_ffn` with
//! softmax gating; no shared expert).
//!
//! ```text
//! p = softmax(x Wr^T);  (w, ids) = topk(p, k);  w /= sum(w) (norm_topk_prob)
//! y = sum_r w[r] * down_e(silu(gate_e(x)) * up_e(x))   (e = ids[r], MatMulId)
//! ```
//! Expert stacks are `[E, N, K]` (one b_transposed matrix per expert; quantized stacks hold E
//! matrices back to back).

use lapid::{DType, Result, Tensor, bail};
use lapid_nn::ops;
use lapid_nn::weights::fake_quant_act;

pub(crate) struct SparseMoe {
    /// Router [E, D] (f32).
    pub router: Tensor,
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
    pub top_k: usize,
    pub norm_topk_prob: bool,
    /// Quantize the expert matmul inputs (fake-quant activation reference).
    pub act_quant: bool,
}

impl SparseMoe {
    /// h [1, n, D] -> [1, n, D].
    pub fn forward(&self, h: &Tensor) -> Result<Tensor> {
        if h.rank() != 3 || h.shape()[0] != 1 {
            bail!("shape_mismatch", "moe expects [1, n, D], got {:?}", h.shape());
        }
        let (n, d) = (h.shape()[1], h.shape()[2]);
        let k = self.top_k;
        let x = h.reshape(&[n, d])?.cast(DType::F32)?;
        let probs = x.matmul_t(&self.router)?.softmax()?; // [n, E]
        let (mut w, ids) = probs.topk(k, 1)?; // [n, k]
        if self.norm_topk_prob {
            w = w.div(&w.sum(1, true)?)?;
        }
        let quant = |t: &Tensor| -> Result<Tensor> { if self.act_quant { fake_quant_act(t) } else { Ok(t.clone()) } };
        let xq = quant(&x)?;
        let gate = xq.matmul_id(&self.gate, &ids)?; // [n, k, I]
        let up = xq.matmul_id(&self.up, &ids)?;
        let inter = gate.shape()[2];
        let act = ops::silu(&gate)?.mul(&up)?.reshape(&[n * k, inter])?;
        let down = quant(&act)?.matmul_id(&self.down, &ids.reshape(&[n * k, 1])?)?; // [n*k, 1, D]
        down.reshape(&[n, k, d])?.mul(&w.unsqueeze(2)?)?.sum(1, false)?.reshape(&[1, n, d])
    }
}
