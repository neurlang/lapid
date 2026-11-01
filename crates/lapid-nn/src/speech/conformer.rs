//! Conformer encoder block (Gulati et al. 2020; NeMo FastConformer, HF Parakeet), batch 1,
//! channels-last [1, T, C]:
//!
//! ```text
//! x + ½ FF1(LN x) → x + RelPosMHA(LN x) → x + Conv(LN x) → x + ½ FF2(LN x) → LN
//! ```
//!
//! The convolution module is point-wise conv (C → 2C) → GLU → depthwise conv (kernel k,
//! "same" padding, BatchNorm folded into it at conversion) → SiLU → point-wise conv.

use lapid::{Result, Tensor};

use super::rel_pos::RelPosAttention;
use super::taps::DepthwiseConv;
use crate::{LayerNorm, Linear, Mlp};

/// The conformer convolution module (1x1 convs as `Linear`).
pub struct ConvModule {
    pub pw1: Linear,
    /// Depthwise conv with the BatchNorm folded in (weight [C, K], bias [C]).
    pub dw: DepthwiseConv,
    pub pw2: Linear,
}

impl ConvModule {
    /// x [1, T, C] -> [1, T, C].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = crate::ops::glu(&self.pw1.forward(x)?, -1)?;
        let pad = (self.dw.kernel_size() - 1) / 2;
        let y = self.dw.forward(&h, 1, (pad, pad))?;
        self.pw2.forward(&crate::ops::silu(&y)?)
    }
}

/// One conformer block (macaron feed-forwards with half-step residuals).
pub struct ConformerBlock {
    pub norm_ff1: LayerNorm,
    pub ff1: Mlp,
    pub norm_att: LayerNorm,
    pub attn: RelPosAttention,
    pub norm_conv: LayerNorm,
    pub conv: ConvModule,
    pub norm_ff2: LayerNorm,
    pub ff2: Mlp,
    pub norm_out: LayerNorm,
}

impl ConformerBlock {
    /// x [1, T, C], relative positions [1, 2T-1, C] -> [1, T, C].
    pub fn forward(&self, x: &Tensor, pos: &Tensor) -> Result<Tensor> {
        let x = x.add(&self.ff1.forward(&self.norm_ff1.forward(x)?)?.mul_scalar(0.5)?)?;
        let x = x.add(&self.attn.forward(&self.norm_att.forward(&x)?, pos)?)?;
        let x = x.add(&self.conv.forward(&self.norm_conv.forward(&x)?)?)?;
        let x = x.add(&self.ff2.forward(&self.norm_ff2.forward(&x)?)?.mul_scalar(0.5)?)?;
        self.norm_out.forward(&x)
    }
}
