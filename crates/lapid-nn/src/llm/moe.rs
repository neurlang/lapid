//! Sparse mixture-of-experts FFN (HF `Qwen3MoeSparseMoeBlock` / `Qwen3_5MoeSparseMoeBlock`,
//! llama.cpp `build_moe_ffn`). A router picks `k` experts per token, each expert is a gated
//! MLP stored as one `[E, N, K]` stack per projection and applied with `MatMulId`:
//!
//! ```text
//! (w, ids) = route(x)                                        [n, k]
//! y = sum_r w[r] * down_e(act(gate_e(x)) * up_e(x)),  e = ids[r]
//! ```
//!
//! [`Experts`] is the routed part only (any router: softmax top-k here, DeepSeek V4's
//! sqrt-softplus / hash routing, Gemma 4's scaled softmax); [`SparseMoe`] is the Qwen block:
//! softmax top-k (renormalized when `norm_topk_prob`) plus an optional shared expert, gated by
//! `sigmoid(x Ws^T)` in Qwen3.5.

use lapid::{BinaryOp, DType, Result, Tensor};

use crate::mlp::{Activation, GatedMlp};
use crate::ops;
use crate::weights::{Loader, fake_quant_act};

/// Gate and up projections of the experts: two stacks, or one fused `[E, 2I, D]` stack
/// (gate rows first).
pub enum GateUp {
    Split { gate: Tensor, up: Tensor },
    Fused(Tensor),
}

/// Routed gated-MLP experts.
pub struct Experts {
    pub gate_up: GateUp,
    /// `[E, D, I]`.
    pub down: Tensor,
    pub act: Activation,
    /// SwiGLU clamp (DeepSeek V4 `swiglu_limit`): `gate <= l`, `-l <= up <= l`.
    pub limit: Option<f32>,
    /// Fake-quant (int8 per block) inputs of the expert matmuls: the reference arithmetic of a
    /// file whose stacks are quantized.
    pub act_quant: bool,
}

impl Experts {
    /// SiLU experts from the stacks `gate` / `up` `[E, inter, d]` and `down` `[E, d, inter]`.
    pub fn load(l: &Loader<'_>, [gate, up, down]: [&str; 3], experts: usize, inter: usize, d: usize) -> Result<Experts> {
        Ok(Experts {
            act_quant: l.act_quant(gate)?,
            gate_up: GateUp::Split { gate: l.tensor_checked(gate, &[experts, inter, d])?, up: l.tensor_checked(up, &[experts, inter, d])? },
            down: l.tensor_checked(down, &[experts, d, inter])?,
            act: Activation::Silu,
            limit: None,
        })
    }

    pub fn with_limit(mut self, limit: f32) -> Experts {
        self.limit = (limit > 0.0).then_some(limit);
        self
    }

    /// `act(gate) * up` with the optional clamp (also used by DeepSeek V4's shared expert).
    pub fn glu(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        match self.limit {
            Some(l) => {
                let l = f64::from(l);
                let up = up.binary_scalar(BinaryOp::Min, l)?.binary_scalar(BinaryOp::Max, -l)?;
                self.act.apply(&gate.binary_scalar(BinaryOp::Min, l)?)?.mul(&up)
            }
            None => self.act.apply(gate)?.mul(up),
        }
    }

    /// `x [n, D]` (f32), expert `ids [n, k]` (i32) and their weights `w [n, k]` -> `[n, D]`.
    pub fn forward(&self, x: &Tensor, ids: &Tensor, w: &Tensor) -> Result<Tensor> {
        let (n, d, k) = (x.shape()[0], x.shape()[1], ids.shape()[1]);
        let quant = |t: &Tensor| if self.act_quant { fake_quant_act(t) } else { Ok(t.clone()) };
        let xq = quant(x)?;
        let (gate, up) = match &self.gate_up {
            GateUp::Split { gate, up } => (xq.matmul_id(gate, ids)?, xq.matmul_id(up, ids)?), // [n, k, I]
            GateUp::Fused(gu) => {
                let gu = xq.matmul_id(gu, ids)?;
                let i = gu.shape()[2] / 2;
                (gu.slice(2, 0, i)?, gu.slice(2, i, 2 * i)?)
            }
        };
        let act = self.glu(&gate, &up)?.reshape(&[n * k, gate.shape()[2]])?;
        let down = quant(&act)?.matmul_id(&self.down, &ids.reshape(&[n * k, 1])?)?; // [n*k, 1, D]
        down.reshape(&[n, k, d])?.mul(&w.unsqueeze(2)?)?.sum(1, false)
    }
}

/// Softmax router: `p = softmax(logits)`, `(w, ids) = topk(p, k)`, `w /= sum(w)` when `renorm`.
pub fn softmax_topk(logits: &Tensor, k: usize, renorm: bool) -> Result<(Tensor, Tensor)> {
    let (w, ids) = logits.cast(DType::F32)?.softmax()?.topk(k, 1)?;
    Ok((if renorm { w.div(&w.sum(1, true)?)? } else { w }, ids))
}

/// Shared expert of [`SparseMoe`], always on; `gate` `[1, D]` scales it by `sigmoid(x gate^T)`.
pub struct SharedExpert {
    pub mlp: GatedMlp,
    pub gate: Option<Tensor>,
}

/// Qwen-style sparse MoE block: softmax top-k routed experts plus an optional shared expert.
pub struct SparseMoe {
    /// Router `[E, D]` (f32).
    pub router: Tensor,
    pub experts: Experts,
    pub top_k: usize,
    pub norm_topk_prob: bool,
    pub shared: Option<SharedExpert>,
}

impl SparseMoe {
    /// h `[1, n, D]` -> `[1, n, D]` (f32).
    pub fn forward(&self, h: &Tensor) -> Result<Tensor> {
        let (n, d) = (h.shape()[1], h.shape()[2]);
        let x = h.reshape(&[n, d])?.cast(DType::F32)?;
        let (w, ids) = softmax_topk(&x.matmul_t(&self.router)?, self.top_k, self.norm_topk_prob)?;
        let mut y = self.experts.forward(&x, &ids, &w)?;
        if let Some(s) = &self.shared {
            let mut sh = s.mlp.forward(&x.reshape(&[1, n, d])?)?.reshape(&[n, d])?;
            if let Some(g) = &s.gate {
                sh = sh.mul(&ops::sigmoid(&x.matmul_t(g)?)?)?; // [n, 1]
            }
            y = y.add(&sh)?;
        }
        y.reshape(&[1, n, d])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{assert_close, eval, rand_vec};

    /// Experts vs a host loop over the selected experts (SwiGLU, clamp, fused stack).
    #[test]
    fn experts_match_host() {
        let (e, n, d, i, k) = (4usize, 3usize, 8usize, 6usize, 2usize);
        let (x, g, u, dn) = (rand_vec(n * d, 1, 1.0), rand_vec(e * i * d, 2, 0.5), rand_vec(e * i * d, 3, 0.5), rand_vec(e * d * i, 4, 0.5));
        let ids: Vec<i32> = vec![0, 3, 2, 1, 3, 3];
        let w: Vec<f32> = vec![0.7, 0.3, 0.5, 0.5, 0.9, 0.1];
        let t = |v: &[f32], s: &[usize]| Tensor::from_f32(v, s).expect("tensor");
        let silu = |v: f32| v / (1.0 + (-v).exp());
        for (fused, limit) in [(false, None), (true, Some(0.5f32))] {
            let gate_up = if fused {
                let mut gu = Vec::new();
                for x in 0..e {
                    gu.extend_from_slice(&g[x * i * d..(x + 1) * i * d]);
                    gu.extend_from_slice(&u[x * i * d..(x + 1) * i * d]);
                }
                GateUp::Fused(t(&gu, &[e, 2 * i, d]))
            } else {
                GateUp::Split { gate: t(&g, &[e, i, d]), up: t(&u, &[e, i, d]) }
            };
            let ex = Experts { gate_up, down: t(&dn, &[e, d, i]), act: Activation::Silu, limit, act_quant: false };
            let y = ex.forward(&t(&x, &[n, d]), &Tensor::from_i32(&ids, &[n, k]).expect("ids"), &t(&w, &[n, k])).expect("forward");
            let mut want = vec![0.0f32; n * d];
            for r in 0..n {
                for s in 0..k {
                    let ex = ids[r * k + s] as usize;
                    let dot = |m: &[f32], o: usize| (0..d).map(|c| m[(ex * i + o) * d + c] * x[r * d + c]).sum::<f32>();
                    let a: Vec<f32> = (0..i)
                        .map(|o| match limit {
                            Some(l) => silu(dot(&g, o).min(l)) * dot(&u, o).clamp(-l, l),
                            None => silu(dot(&g, o)) * dot(&u, o),
                        })
                        .collect();
                    for c in 0..d {
                        want[r * d + c] += w[r * k + s] * (0..i).map(|o| dn[(ex * d + c) * i + o] * a[o]).sum::<f32>();
                    }
                }
            }
            assert_close(&eval(&y), &want, 1e-5);
        }
    }
}
