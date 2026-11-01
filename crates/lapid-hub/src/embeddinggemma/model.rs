//! EmbeddingGemma network as lazy lapid graphs: Gemma 3 bidirectional encoder (HF
//! `Gemma3TextModel` with `use_bidirectional_attention`), mean pooling and the Dense head.
//!
//! A batch is right-padded to a common length S. Padded keys are masked out of every
//! attention; sliding layers additionally hide keys with `|i - j| >= window` from real
//! queries (padded queries keep every real key, so their rows stay finite). Mean pooling,
//! normalization and Matryoshka truncation are the shared pipeline (`runtime::embedding`).

use lapid::{Result, Tensor, bail};
use lapid_nn::{Activation, Attention, Embedding, GatedMlp, Linear, Loader, Mask, RmsNorm, Rope};

use super::config::{DenseActivation, EmbeddingGemmaConfig};
use crate::runtime::Taps;
use crate::runtime::embedding::TokenBatch;

struct Layer {
    input_norm: RmsNorm,
    attn: Attention,
    post_attn_norm: RmsNorm,
    pre_ff_norm: RmsNorm,
    mlp: GatedMlp,
    post_ff_norm: RmsNorm,
    sliding: bool,
}

/// Additive [B, 1, S, S] mask of a bidirectional sliding window: key j is visible to a
/// real query i iff `j < len` and `|i - j| < window`; padded queries see every real key.
pub(crate) fn sliding_padding_mask(lengths: &[usize], seq: usize, window: usize) -> Result<Tensor> {
    let mut data = vec![0.0f32; lengths.len() * seq * seq];
    for (b, &len) in lengths.iter().enumerate() {
        for i in 0..seq {
            let row = &mut data[(b * seq + i) * seq..(b * seq + i + 1) * seq];
            for (j, v) in row.iter_mut().enumerate() {
                let hidden = j >= len || (i < len && i.abs_diff(j) >= window);
                if hidden {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
    }
    Tensor::from_f32(&data, &[lengths.len(), 1, seq, seq])
}

pub(crate) struct Model {
    embed: Embedding,
    embed_scale: f32,
    layers: Vec<Layer>,
    norm: RmsNorm,
    rope_global: Rope,
    rope_local: Rope,
    window: usize,
    dense: Vec<(Linear, DenseActivation)>,
}

fn gemma_norm(l: &Loader<'_>, name: &str, eps: f32) -> Result<RmsNorm> {
    Ok(l.rms_norm(name, eps)?.with_weight_offset(1.0))
}

impl Model {
    pub fn load(l: &Loader<'_>, cfg: &EmbeddingGemmaConfig) -> Result<Model> {
        let eps = cfg.rms_norm_eps;
        let act = Activation::from_name(&cfg.hidden_activation).ok_or_else(|| {
            lapid::Error::new("unsupported_model", format!("hidden activation {:?}", cfg.hidden_activation))
        })?;
        let embed = l.embedding("model.embed_tokens")?;
        l.expect_shape(&embed.weight, "model.embed_tokens.weight", &[cfg.vocab_size, cfg.hidden_size])?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for (i, &sliding) in cfg.sliding_layers.iter().enumerate() {
            let p = format!("model.layers.{i}");
            let a = format!("{p}.self_attn");
            let attn = Attention::new(
                l.linear(&format!("{a}.q_proj"), false)?,
                l.linear(&format!("{a}.k_proj"), false)?,
                l.linear(&format!("{a}.v_proj"), false)?,
                l.linear(&format!("{a}.o_proj"), false)?,
                cfg.n_head,
                cfg.n_kv_head,
            )?
            .with_qk_norm(gemma_norm(l, &format!("{a}.q_norm"), eps)?, gemma_norm(l, &format!("{a}.k_norm"), eps)?)
            .with_scale(cfg.attn_scale());
            if attn.head_dim != cfg.head_dim {
                bail!("shape_mismatch", "{a}: head_dim {} != config {}", attn.head_dim, cfg.head_dim);
            }
            let m = format!("{p}.mlp");
            layers.push(Layer {
                input_norm: gemma_norm(l, &format!("{p}.input_layernorm"), eps)?,
                attn,
                post_attn_norm: gemma_norm(l, &format!("{p}.post_attention_layernorm"), eps)?,
                pre_ff_norm: gemma_norm(l, &format!("{p}.pre_feedforward_layernorm"), eps)?,
                mlp: GatedMlp::new(
                    l.linear(&format!("{m}.gate_proj"), false)?,
                    l.linear(&format!("{m}.up_proj"), false)?,
                    l.linear(&format!("{m}.down_proj"), false)?,
                    act,
                ),
                post_ff_norm: gemma_norm(l, &format!("{p}.post_feedforward_layernorm"), eps)?,
                sliding,
            });
        }
        let mut dense = Vec::with_capacity(cfg.dense.len());
        for d in &cfg.dense {
            let lin = l.linear(d.prefix.trim_end_matches('.'), d.bias)?;
            l.expect_shape(&lin.weight, &format!("{}weight", d.prefix), &[d.out_features, d.in_features])?;
            dense.push((lin, d.activation));
        }
        Ok(Model {
            embed,
            embed_scale: cfg.embed_scale(),
            layers,
            norm: gemma_norm(l, "model.norm", eps)?,
            rope_global: Rope::new(cfg.head_dim, cfg.rope_theta),
            rope_local: Rope::new(cfg.head_dim, cfg.rope_local_theta),
            window: cfg.bidirectional_window,
            dense,
        })
    }

    /// Encoder token states [B, S, D] of a right-padded batch.
    pub fn encode(&self, batch: &TokenBatch, taps: &mut Taps) -> Result<Tensor> {
        let s = batch.seq;
        let mut x = self.embed.forward(&batch.ids_tensor()?)?.mul_scalar(f64::from(self.embed_scale))?;
        taps.tap("embed", &x);
        let full_mask = if batch.padded() { Mask::Custom { additive: batch.key_padding_mask()?, causal: false } } else { Mask::None };
        // Within the window everywhere: the sliding layers see what full layers see.
        let sliding_mask = if s <= self.window {
            full_mask.clone()
        } else {
            Mask::Custom { additive: sliding_padding_mask(&batch.lengths, s, self.window)?, causal: false }
        };
        for (i, layer) in self.layers.iter().enumerate() {
            let (rope, mask) = if layer.sliding { (&self.rope_local, &sliding_mask) } else { (&self.rope_global, &full_mask) };
            let h = layer.input_norm.forward(&x)?;
            let a = layer.attn.forward(&h, Some(rope), None, mask)?;
            x = x.add(&layer.post_attn_norm.forward(&a)?)?;
            let m = layer.mlp.forward(&layer.pre_ff_norm.forward(&x)?)?;
            x = x.add(&layer.post_ff_norm.forward(&m)?)?;
            taps.tap(format_args!("layers.{i}"), &x);
        }
        let tokens = self.norm.forward(&x)?;
        taps.tap("norm", &tokens);
        Ok(tokens)
    }

    /// The sentence-transformers Dense head on pooled [B, D].
    pub fn project(&self, mut y: Tensor, taps: &mut Taps) -> Result<Tensor> {
        for (k, (lin, act)) in self.dense.iter().enumerate() {
            y = lin.forward(&y)?;
            if *act == DenseActivation::Tanh {
                y = y.tanh()?;
            }
            taps.tap(format_args!("dense.{k}"), &y);
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_mask_rule() {
        let m = sliding_padding_mask(&[4], 5, 2).expect("mask");
        let host = lapid::Device::cpu().to_vec_f32(&m).expect("host");
        let vis = |i: usize, j: usize| host[i * 5 + j] == 0.0;
        assert!(vis(0, 1) && !vis(0, 2) && vis(2, 3) && vis(3, 2) && !vis(3, 1));
        assert!(!vis(2, 4), "padded key hidden");
        // Padded query 4 keeps every real key (no all -inf rows).
        assert!((0..4).all(|j| vis(4, j)));
    }
}
