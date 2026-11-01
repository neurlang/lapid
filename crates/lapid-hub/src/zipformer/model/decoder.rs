//! Stateless transducer decoder and joiner (icefall `decoder.py`, `joiner.py`).
//!
//! * Decoder: embeddings of the last `context_size` tokens (id -1 = zero padding, as icefall's
//!   initial hypothesis) → grouped Conv1d(kernel = context, groups = D / 4, no bias) → ReLU →
//!   `joiner.decoder_proj`. The converter stores the grouped conv as one dense block-diagonal
//!   matrix [D, context · D] (`decoder.conv.weight`, column = t · D + input channel).
//! * Joiner: `output_linear(tanh(enc_proj + dec_proj))`; the encoder projection is applied
//!   in the encoder graph.

use lapid::{Result, Tensor, bail};
use lapid_nn::{Embedding, Linear, Loader};

use super::linear;
use crate::zipformer::config::ZipformerConfig;

pub(crate) struct Decoder {
    embedding: Embedding,
    conv: Option<Linear>,
    proj: Linear,
    context: usize,
    dim: usize,
}

impl Decoder {
    pub fn load(l: &Loader<'_>, cfg: &ZipformerConfig) -> Result<Decoder> {
        let d = cfg.decoder_dim;
        let embedding = l.embedding("decoder.embedding")?;
        l.expect_shape(&embedding.weight, "decoder.embedding.weight", &[cfg.vocab_size, d])?;
        let conv = if cfg.context_size > 1 {
            let w = l.tensor("decoder.conv.weight")?;
            l.expect_shape(&w, "decoder.conv.weight", &[d, cfg.context_size * d])?;
            Some(Linear::new(w, None).with_act_quant(l.act_quant("decoder.conv.weight")?))
        } else {
            None
        };
        Ok(Decoder { embedding, conv, proj: linear(l, "joiner.decoder_proj", cfg.joiner_dim, d)?, context: cfg.context_size, dim: d })
    }

    /// Projected decoder output [1, joiner_dim] for the last `context_size` tokens.
    pub fn forward(&self, context: &[i64]) -> Result<Tensor> {
        if context.len() != self.context {
            bail!("internal", "decoder context has {} tokens, expected {}", context.len(), self.context);
        }
        let v = self.embedding.vocab_size() as i64;
        let ids: Vec<i32> = context.iter().map(|&t| t.clamp(0, v - 1) as i32).collect();
        let keep: Vec<f32> = context.iter().map(|&t| if t >= 0 { 1.0 } else { 0.0 }).collect();
        let emb = self.embedding.forward(&Tensor::from_i32(&ids, &[self.context])?)?; // [ctx, D]
        let emb = if keep.iter().all(|&k| k == 1.0) { emb } else { emb.mul(&Tensor::from_f32(&keep, &[self.context, 1])?)? };
        let x = emb.reshape(&[1, self.context * self.dim])?;
        let x = match &self.conv {
            Some(conv) => lapid_nn::ops::relu(&conv.forward(&x)?)?,
            None => x,
        };
        self.proj.forward(&x)
    }
}

pub(crate) struct Joiner {
    output: Linear,
}

impl Joiner {
    pub fn load(l: &Loader<'_>, cfg: &ZipformerConfig) -> Result<Joiner> {
        Ok(Joiner { output: linear(l, "joiner.output_linear", cfg.vocab_size, cfg.joiner_dim)? })
    }

    /// enc [T, J] (projected), dec [1, J] (projected) -> logits [T, vocab].
    pub fn logits(&self, enc: &Tensor, dec: &Tensor) -> Result<Tensor> {
        self.output.forward(&enc.add(dec)?.tanh()?)
    }
}

/// Dense block-diagonal form of a grouped Conv1d weight `w` [D, D / G, K] (G = D / 4 groups):
/// [D, K · D] with `dense[o, t D + i] = w[o, i - group(o) · (D / G), t]`.
pub(crate) fn dense_group_conv(w: &[f32], d: usize, per_group: usize, k: usize) -> Result<Vec<f32>> {
    if per_group == 0 || d % per_group != 0 || w.len() != d * per_group * k {
        bail!("shape_mismatch", "decoder.conv.weight: {} values for D {d}, {per_group} per group, kernel {k}", w.len());
    }
    let mut out = vec![0.0f32; d * k * d];
    for o in 0..d {
        let g0 = (o / per_group) * per_group;
        for il in 0..per_group {
            for t in 0..k {
                out[o * k * d + t * d + g0 + il] = w[(o * per_group + il) * k + t];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_conv_matches_grouped() {
        let (d, per, k) = (8usize, 4usize, 2usize);
        let w: Vec<f32> = (0..d * per * k).map(|i| i as f32 * 0.1 - 1.0).collect();
        let x: Vec<f32> = (0..k * d).map(|i| (i as f32 * 0.37).sin()).collect(); // [t, c]
        let dense = dense_group_conv(&w, d, per, k).expect("dense");
        for o in 0..d {
            let g = o / per;
            let mut want = 0.0;
            for il in 0..per {
                for t in 0..k {
                    want += w[(o * per + il) * k + t] * x[t * d + g * per + il];
                }
            }
            let got: f32 = (0..k * d).map(|j| dense[o * k * d + j] * x[j]).sum();
            assert!((got - want).abs() < 1e-5, "out {o}: {got} vs {want}");
        }
    }
}
