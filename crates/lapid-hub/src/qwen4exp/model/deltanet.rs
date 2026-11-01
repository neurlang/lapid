//! Gated DeltaNet token mixer (HF `Qwen4ExpTextGatedDeltaNet`, llama.cpp qwen4exp
//! `build_layer_attn_linear`). Copy of `qwen35::model::deltanet` (private there) with the
//! output-gate activation as a parameter: Qwen4-Exp gates with sigmoid, Qwen3.5 with silu.
//!
//! ```text
//! qkv = conv1d_causal(in_proj_qkv(h)) |> silu      (depthwise, kernel 4, streamed state)
//! q, k = l2norm(split(qkv))  (repeated to the v heads, tiled order)
//! beta = sigmoid(in_proj_b(h)),  g = A * softplus(in_proj_a(h) + dt_bias)   (A = -exp(A_log))
//! o, S = gated_delta_rule(q, k, v, g, beta, S, scale = 1/sqrt(Dk))
//! out = out_proj(rmsnorm(o) * act(in_proj_z(h)))      (act = sigmoid | silu)
//! ```
//!
//! Head order: v heads are stored *tiled* (llama.cpp GGUF order): v head `h` reads k head
//! `h % Hk`. HF checkpoints (grouped: `h / (Hv/Hk)`) are reordered at conversion.

use lapid::{DType, Result, Tensor, UnaryOp, bail};
use lapid_nn::{Conv1d, Linear, RmsNorm, ops};

use crate::qwen4exp::config::GateAct;

/// The Gated DeltaNet primitive (`Op::GatedDeltaNet`). q/k [B,T,H,Dk], v [B,T,H,Dv], g/beta
/// [B,T,H], state [B,H,Dk,Dv], all f32 → [B, H, T + Dk, Dv] (outputs, then the final state).
pub(crate) fn gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    Tensor::gated_delta_net(q, k, v, g, beta, state, scale)
}

/// softplus(x) = max(x, 0) + log(1 + exp(-|x|)) (stable for large |x|).
pub(crate) fn softplus(x: &Tensor) -> Result<Tensor> {
    let pos = x.maximum(&Tensor::zeros(x.shape(), x.dtype()))?;
    pos.add(&x.unary(UnaryOp::Abs)?.neg()?.exp()?.add_scalar(1.0)?.log()?)
}

/// x * rsqrt(sum(x², last) + eps) (HF `l2norm`, FLA convention).
pub(crate) fn l2norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let ss = x.mul(x)?.sum(-1, true)?;
    x.mul(&ss.add_scalar(eps)?.unary(UnaryOp::Rsqrt)?)
}

pub(crate) struct DeltaNet {
    pub in_qkv: Linear,
    pub in_z: Linear,
    pub in_b: Linear,
    pub in_a: Linear,
    pub conv: Conv1d,
    /// -exp(A_log) [Hv] f32.
    pub a: Tensor,
    pub dt_bias: Tensor,
    pub norm: RmsNorm,
    pub out: Linear,
    pub hk: usize,
    pub hv: usize,
    pub dk: usize,
    pub dv: usize,
    pub gate: GateAct,
}

/// New lazy recurrent state of one step: (conv window, delta-rule state).
pub(crate) struct StepState {
    pub conv: Tensor,
    pub ssm: Tensor,
}

impl DeltaNet {
    /// h [1, n, D] → [1, n, D]. `conv` / `ssm` are the committed states (`None` = zeros at the
    /// start of a sequence); the new ones are returned lazily for staging.
    pub fn forward(&self, h: &Tensor, conv: Option<&Tensor>, ssm: Option<&Tensor>) -> Result<(Tensor, StepState)> {
        if h.rank() != 3 {
            bail!("shape_mismatch", "deltanet expects [B, n, D], got {:?}", h.shape());
        }
        let (b, n) = (h.shape()[0], h.shape()[1]);
        let (hk, hv, dk, dv) = (self.hk, self.hv, self.dk, self.dv);
        let kd = hk * dk;
        let qkv = self.in_qkv.forward(h)?.cast(DType::F32)?; // [B, n, C]
        let c = qkv.shape()[2];
        if c != 2 * kd + hv * dv {
            bail!("shape_mismatch", "deltanet: in_proj_qkv gives {c} channels, expected {}", 2 * kd + hv * dv);
        }
        let mut st = conv.cloned();
        let mixed = self.conv.forward_stream(&qkv.transpose(1, 2)?, &mut st)?; // [B, C, n]
        let new_conv = match st {
            Some(s) => s,
            None => bail!("internal", "deltanet: conv produced no state"),
        };
        let mixed = ops::silu(&mixed.cast(DType::F32)?)?.transpose(1, 2)?; // [B, n, C]
        let q = l2norm(&mixed.slice(2, 0, kd)?.reshape(&[b, n, hk, dk])?, 1e-6)?;
        let k = l2norm(&mixed.slice(2, kd, 2 * kd)?.reshape(&[b, n, hk, dk])?, 1e-6)?;
        let v = mixed.slice(2, 2 * kd, c)?.reshape(&[b, n, hv, dv])?;
        let (q, k) = if hv != hk {
            let r = hv / hk;
            let tile = |x: &Tensor| -> Result<Tensor> { x.unsqueeze(2)?.expand(&[b, n, r, hk, dk])?.reshape(&[b, n, hv, dk]) };
            (tile(&q)?, tile(&k)?)
        } else {
            (q, k)
        };
        let beta = ops::sigmoid(&self.in_b.forward(h)?.cast(DType::F32)?)?; // [B, n, Hv]
        let a = self.in_a.forward(h)?.cast(DType::F32)?.add(&self.dt_bias)?;
        let g = softplus(&a)?.mul(&self.a)?; // [B, n, Hv], <= 0
        let state = match ssm {
            Some(s) => s.clone(),
            None => Tensor::zeros(&[b, hv, dk, dv], DType::F32),
        };
        let out = gated_delta_rule(&q, &k, &v, &g, &beta, &state, 1.0 / (dk as f32).sqrt())?; // [B, Hv, n + Dk, Dv]
        let o = out.slice(2, 0, n)?.permute(&[0, 2, 1, 3])?; // [B, n, Hv, Dv]
        let new_ssm = out.slice(2, n, n + dk)?;
        let z = self.in_z.forward(h)?.cast(DType::F32)?.reshape(&[b, n, hv, dv])?;
        let z = match self.gate {
            GateAct::Sigmoid => ops::sigmoid(&z)?,
            GateAct::Silu => ops::silu(&z)?,
        };
        let o = self.norm.forward(&o)?.mul(&z)?.reshape(&[b, n, hv * dv])?;
        Ok((self.out.forward(&o)?, StepState { conv: new_conv, ssm: new_ssm }))
    }
}
