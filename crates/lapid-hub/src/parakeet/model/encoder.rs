//! FastConformer encoder: conv subsampling, then blocks of
//! `x + ½ FF1(LN x)` → `x + RelPosMHA(LN x)` → `x + Conv(LN x)` → `x + ½ FF2(LN x)` → LN.
//! The convolution module is point-wise conv (2d) → GLU → depthwise conv (k, BatchNorm
//! folded at conversion) → SiLU → point-wise conv, run channels-last [1, T, C].
//! The encoder projection to the joint dimension (`encoder_projector`) is applied here too.

use lapid::{Result, Tensor};
use lapid_nn::{Activation, LayerNorm, Linear, Loader, Mlp};

use super::attention::{RelPosAttention, rel_pos_embedding};
use super::subsampling::Subsampling;
use super::{LN_EPS, kernel_taps, linear_checked, weighted_sum};
use crate::parakeet::config::ParakeetConfig;

struct ConvModule {
    pw1: Linear,
    /// Depthwise per-tap weights [C] (f32), bias [C].
    dw: Vec<Tensor>,
    dw_bias: Tensor,
    pw2: Linear,
}

impl ConvModule {
    fn load(l: &Loader<'_>, p: &str, d: usize, k: usize) -> Result<ConvModule> {
        let w = l.tensor_f32(&format!("{p}.depthwise_conv.weight"))?;
        l.expect_shape(&w, &format!("{p}.depthwise_conv.weight"), &[d, k])?;
        Ok(ConvModule {
            pw1: linear_checked(l, &format!("{p}.pointwise_conv1"), 2 * d, d)?,
            dw: kernel_taps(&w)?,
            dw_bias: l.tensor_f32(&format!("{p}.depthwise_conv.bias"))?,
            pw2: linear_checked(l, &format!("{p}.pointwise_conv2"), d, d)?,
        })
    }

    /// x [1, T, C] -> [1, T, C].
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = lapid_nn::ops::glu(&self.pw1.forward(x)?, -1)?;
        let k = self.dw.len();
        let t = h.shape()[1];
        let pad = (k - 1) / 2;
        let hp = h.pad(1, pad, pad, 0.0)?;
        let taps: Vec<Tensor> = (0..k).map(|i| hp.slice(1, i, i + t)).collect::<Result<_>>()?;
        let y = weighted_sum(&taps, &self.dw, Some(&self.dw_bias))?;
        self.pw2.forward(&lapid_nn::ops::silu(&y)?)
    }
}

struct Block {
    norm_ff1: LayerNorm,
    ff1: Mlp,
    norm_att: LayerNorm,
    attn: RelPosAttention,
    norm_conv: LayerNorm,
    conv: ConvModule,
    norm_ff2: LayerNorm,
    ff2: Mlp,
    norm_out: LayerNorm,
}

impl Block {
    fn load(l: &Loader<'_>, p: &str, cfg: &ParakeetConfig) -> Result<Block> {
        let d = cfg.d_model;
        let ln = |n: &str| l.layer_norm(&format!("{p}.{n}"), LN_EPS);
        let ff = |n: &str| -> Result<Mlp> {
            Ok(Mlp::new(
                linear_checked(l, &format!("{p}.{n}.linear1"), cfg.ff_dim, d)?,
                linear_checked(l, &format!("{p}.{n}.linear2"), d, cfg.ff_dim)?,
                Activation::Silu,
            ))
        };
        Ok(Block {
            norm_ff1: ln("norm_feed_forward1")?,
            ff1: ff("feed_forward1")?,
            norm_att: ln("norm_self_att")?,
            attn: RelPosAttention::load(l, &format!("{p}.self_attn"), d, cfg.n_heads)?,
            norm_conv: ln("norm_conv")?,
            conv: ConvModule::load(l, &format!("{p}.conv"), d, cfg.conv_kernel)?,
            norm_ff2: ln("norm_feed_forward2")?,
            ff2: ff("feed_forward2")?,
            norm_out: ln("norm_out")?,
        })
    }

    fn forward(&self, x: &Tensor, pos: &Tensor) -> Result<Tensor> {
        let x = x.add(&self.ff1.forward(&self.norm_ff1.forward(x)?)?.mul_scalar(0.5)?)?;
        let x = x.add(&self.attn.forward(&self.norm_att.forward(&x)?, pos)?)?;
        let x = x.add(&self.conv.forward(&self.norm_conv.forward(&x)?)?)?;
        let x = x.add(&self.ff2.forward(&self.norm_ff2.forward(&x)?)?.mul_scalar(0.5)?)?;
        self.norm_out.forward(&x)
    }
}

pub(crate) struct Encoder {
    subsampling: Subsampling,
    blocks: Vec<Block>,
    projector: Linear,
    d_model: usize,
}

/// Encoder graph outputs: the projected frames plus named intermediates.
pub(crate) struct EncoderGraph {
    /// [1, T', joint_hidden]
    pub projected: Tensor,
    /// "encoder.subsampling", "encoder.layers.{i}", "encoder.projected".
    pub traced: Vec<Tensor>,
}

impl Encoder {
    pub fn load(l: &Loader<'_>, cfg: &ParakeetConfig) -> Result<Encoder> {
        let blocks = (0..cfg.n_layers)
            .map(|i| Block::load(l, &format!("encoder.layers.{i}"), cfg))
            .collect::<Result<_>>()?;
        Ok(Encoder {
            subsampling: Subsampling::load(l, cfg)?,
            blocks,
            projector: linear_checked(l, "encoder_projector", cfg.decoder_hidden, cfg.d_model)?,
            d_model: cfg.d_model,
        })
    }

    /// Graph for mel [T, n_mels] (frames-major). `trace` names every block output.
    pub fn graph(&self, mel: &[f32], frames: usize, trace: bool) -> Result<EncoderGraph> {
        let mut x = self.subsampling.forward(mel, frames)?.named("encoder.subsampling");
        let mut traced = Vec::new();
        if trace {
            traced.push(x.clone());
        }
        let pos = rel_pos_embedding(x.shape()[1], self.d_model)?;
        for (i, b) in self.blocks.iter().enumerate() {
            x = b.forward(&x, &pos)?;
            if trace {
                x = x.named(format!("encoder.layers.{i}"));
                traced.push(x.clone());
            }
        }
        let projected = self.projector.forward(&x)?.named("encoder.projected");
        if trace {
            traced.push(projected.clone());
        }
        Ok(EncoderGraph { projected, traced })
    }
}
