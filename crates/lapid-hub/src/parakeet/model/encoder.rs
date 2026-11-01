//! FastConformer encoder: conv subsampling, conformer blocks (`lapid_nn::speech`) over
//! sinusoidal relative positions, and the projection to the joint dimension
//! (`encoder_projector`).

use lapid::{Result, Tensor};
use lapid_nn::speech::{ConformerBlock, ConvModule, DepthwiseConv, RelPosAttention, sinusoidal_rel_positions};
use lapid_nn::{Activation, Linear, Loader, Mlp};

use super::subsampling::Subsampling;
use super::{LN_EPS, linear_checked};
use crate::parakeet::config::ParakeetConfig;

/// Block `p` (HF names) of a `cfg`-shaped encoder.
fn load_block(l: &Loader<'_>, p: &str, cfg: &ParakeetConfig) -> Result<ConformerBlock> {
    let d = cfg.d_model;
    let ln = |n: &str| l.layer_norm(&format!("{p}.{n}"), LN_EPS);
    let lin = |n: &str, out: usize, inp: usize| linear_checked(l, &format!("{p}.{n}"), out, inp);
    let ff = |n: &str| -> Result<Mlp> {
        Ok(Mlp::new(lin(&format!("{n}.linear1"), cfg.ff_dim, d)?, lin(&format!("{n}.linear2"), d, cfg.ff_dim)?, Activation::Silu))
    };
    let attn_bias = |n: &str| -> Result<Tensor> {
        let name = format!("{p}.self_attn.{n}");
        let t = l.tensor_f32(&name)?;
        l.expect_shape(&t, &name, &[cfg.n_heads, d / cfg.n_heads])?;
        Ok(t)
    };
    let attn = RelPosAttention::new(
        lin("self_attn.q_proj", d, d)?,
        lin("self_attn.k_proj", d, d)?,
        lin("self_attn.v_proj", d, d)?,
        lin("self_attn.o_proj", d, d)?,
        lin("self_attn.relative_k_proj", d, d)?,
        &attn_bias("bias_u")?,
        &attn_bias("bias_v")?,
        cfg.n_heads,
    )?;
    let dw_name = format!("{p}.conv.depthwise_conv.weight");
    let dw = l.tensor_f32(&dw_name)?;
    l.expect_shape(&dw, &dw_name, &[d, cfg.conv_kernel])?;
    let conv = ConvModule {
        pw1: lin("conv.pointwise_conv1", 2 * d, d)?,
        dw: DepthwiseConv::new(&dw, Some(l.tensor_f32(&format!("{p}.conv.depthwise_conv.bias"))?))?,
        pw2: lin("conv.pointwise_conv2", d, d)?,
    };
    Ok(ConformerBlock {
        norm_ff1: ln("norm_feed_forward1")?,
        ff1: ff("feed_forward1")?,
        norm_att: ln("norm_self_att")?,
        attn,
        norm_conv: ln("norm_conv")?,
        conv,
        norm_ff2: ln("norm_feed_forward2")?,
        ff2: ff("feed_forward2")?,
        norm_out: ln("norm_out")?,
    })
}

pub(crate) struct Encoder {
    subsampling: Subsampling,
    blocks: Vec<ConformerBlock>,
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
            .map(|i| load_block(l, &format!("encoder.layers.{i}"), cfg))
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
        let pos = sinusoidal_rel_positions(x.shape()[1], self.d_model)?;
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
