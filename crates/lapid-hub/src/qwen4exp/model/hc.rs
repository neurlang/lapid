//! Hyper-connections (HF `Qwen4ExpTextGatedResidual`, llama.cpp `build_hc_mix` /
//! `build_hc_combine`). The residual is `hc` parallel streams `x [1, n, hc, D]`:
//!
//! ```text
//! xn    = rmsnorm_per_stream(x) * w_norm            (w_norm [hc, D], effective scale)
//! lo    = silu(down(flat(xn)) / hc)                 (rank hc_lowrank)
//! mixed = mean_s(sigmoid(up(lo))_s * xn_s)          → block input [1, n, D]
//! w     = 2 * sigmoid(inject(flat(xn)) / hc)        → [1, n, hc, 1]
//! x'    = x + block(mixed) * w                       (combine)
//! ```
//! The final mixer (no inject) replaces the output norm.

use lapid::{DType, Result, Tensor, bail};
use lapid_nn::{Linear, RmsNorm, ops};

/// RMS norm over the last dim of every stream, scaled by a `[hc, D]` weight.
pub(crate) fn grouped_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let d = x.shape().last().copied().unwrap_or(0);
    RmsNorm::new(Tensor::full(1.0, &[d], DType::F32), eps).forward(x)?.mul(weight)
}

pub(crate) struct HyperConnection {
    /// [hc, D] effective scale.
    pub norm: Tensor,
    pub down: Linear,
    pub up: Linear,
    /// `None` for the final mixer.
    pub inject: Option<Linear>,
    pub hc: usize,
    pub eps: f32,
}

/// Output of [`HyperConnection::mix`].
pub(crate) struct Mixed {
    /// Block input [1, n, D].
    pub input: Tensor,
    /// Scatter weights [1, n, hc, 1] (`None` for the final mixer).
    pub weights: Option<Tensor>,
}

impl HyperConnection {
    pub fn mix(&self, x: &Tensor) -> Result<Mixed> {
        if x.rank() != 4 || x.shape()[2] != self.hc {
            bail!("shape_mismatch", "hyper-connection expects [1, n, {}, D], got {:?}", self.hc, x.shape());
        }
        let (n, hc, d) = (x.shape()[1], self.hc, x.shape()[3]);
        let inv = 1.0 / hc as f64;
        let xn = grouped_rms_norm(x, &self.norm, self.eps)?; // [1, n, hc, D]
        let flat = xn.reshape(&[1, n, hc * d])?;
        let lo = ops::silu(&self.down.forward(&flat)?.cast(DType::F32)?.mul_scalar(inv)?)?;
        let gate = ops::sigmoid(&self.up.forward(&lo)?.cast(DType::F32)?)?.reshape(&[1, n, hc, d])?;
        let input = gate.mul(&xn)?.sum(2, false)?.mul_scalar(inv)?; // [1, n, D]
        let weights = match &self.inject {
            Some(inj) => Some(ops::sigmoid(&inj.forward(&flat)?.cast(DType::F32)?.mul_scalar(inv)?)?.mul_scalar(2.0)?.reshape(&[1, n, hc, 1])?),
            None => None,
        };
        Ok(Mixed { input, weights })
    }
}

/// x [1, n, hc, D] + out [1, n, D] * w [1, n, hc, 1].
pub(crate) fn combine(x: &Tensor, out: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.add(&out.cast(DType::F32)?.unsqueeze(2)?.mul(w)?)
}
