//! Whisper audio encoder: conv1 (k3, p1) + GELU, conv2 (k3, s2, p1) + GELU, positional
//! embedding, pre-LN transformer blocks, final LayerNorm.

use lapid::{Error, Result, Tensor};
use lapid_nn::{Attention, Conv1d, LayerNorm, Loader, Mask, Mlp};

use super::config::WhisperConfig;
use super::{attention, layer_norm, mlp};

struct EncoderLayer {
    attn_ln: LayerNorm,
    attn: Attention,
    mlp_ln: LayerNorm,
    mlp: Mlp,
}

pub(crate) struct Encoder {
    conv1: Conv1d,
    conv2: Conv1d,
    /// f32 [n_audio_ctx, d]
    positions: Tensor,
    layers: Vec<EncoderLayer>,
    ln: LayerNorm,
    n_mels: usize,
    n_frames: usize,
}

impl Encoder {
    pub fn load(l: &Loader<'_>, cfg: &WhisperConfig) -> Result<Encoder> {
        let p = "model.encoder";
        let d = cfg.n_audio_state;
        let conv1_w = l.tensor(&format!("{p}.conv1.weight"))?;
        l.expect_shape(&conv1_w, "conv1.weight", &[d, cfg.n_mels, 3])?;
        let conv1 = Conv1d::new(conv1_w, Some(l.tensor(&format!("{p}.conv1.bias"))?)).with_padding(1);
        let conv2 = l.conv1d(&format!("{p}.conv2"), true)?.with_stride(2).with_padding(1);
        let positions = l.tensor_f32(&format!("{p}.embed_positions.weight"))?;
        l.expect_shape(&positions, "encoder.embed_positions.weight", &[cfg.n_audio_ctx, d])?;
        let mut layers = Vec::with_capacity(cfg.n_audio_layer);
        for i in 0..cfg.n_audio_layer {
            let lp = format!("{p}.layers.{i}");
            layers.push(EncoderLayer {
                attn_ln: layer_norm(l, &format!("{lp}.self_attn_layer_norm"))?,
                attn: attention(l, &format!("{lp}.self_attn"), d, cfg.n_audio_head)?,
                mlp_ln: layer_norm(l, &format!("{lp}.final_layer_norm"))?,
                mlp: mlp(l, &lp)?,
            });
        }
        Ok(Encoder {
            conv1,
            conv2,
            positions,
            layers,
            ln: layer_norm(l, &format!("{p}.layer_norm"))?,
            n_mels: cfg.n_mels,
            n_frames: 2 * cfg.n_audio_ctx,
        })
    }

    /// Build the encoder graph for one [n_mels, 2*n_audio_ctx] log-mel window.
    /// Returns (output [1, n_audio_ctx, d], named intermediates incl. the output).
    pub fn graph(&self, mel: &[f32]) -> Result<(Tensor, Vec<Tensor>)> {
        if mel.len() != self.n_mels * self.n_frames {
            return Err(Error::new(
                "invalid_argument",
                format!(
                    "mel has {} values, expected n_mels * frames = {} * {}",
                    mel.len(),
                    self.n_mels,
                    self.n_frames
                ),
            ));
        }
        let x = Tensor::from_f32(mel, &[1, self.n_mels, self.n_frames])?;
        let x = self.conv1.forward(&x)?.gelu()?;
        let x = self.conv2.forward(&x)?.gelu()?;
        // [1, d, ctx] -> [1, ctx, d]
        let x = x.permute(&[0, 2, 1])?;
        let mut x = x.add(&self.positions)?.named("encoder.embed");
        let mut traced = vec![x.clone()];
        for (i, layer) in self.layers.iter().enumerate() {
            let h = layer.attn_ln.forward(&x)?;
            let (k, v) = layer.attn.project_kv(&h)?;
            x = x.add(&layer.attn.attend(&h, &k, &v, &Mask::None)?)?;
            let h = layer.mlp_ln.forward(&x)?;
            x = x.add(&layer.mlp.forward(&h)?)?.named(format!("encoder.layers.{i}"));
            traced.push(x.clone());
        }
        let out = self.ln.forward(&x)?.named("encoder.out");
        traced.push(out.clone());
        Ok((out, traced))
    }
}
