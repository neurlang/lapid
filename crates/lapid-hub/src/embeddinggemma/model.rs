//! EmbeddingGemma network as lazy lapid graphs: Gemma 3 bidirectional encoder (HF
//! `Gemma3TextModel` with `use_bidirectional_attention`), mean pooling and the Dense head.
//!
//! A batch is right-padded to a common length S. Padded keys are masked out of every
//! attention; sliding layers additionally hide keys with `|i - j| >= window` from real
//! queries (padded queries keep every real key, so their rows stay finite). Pooling is a
//! batched matmul with per-row weights `1 / len` on real tokens and 0 on padding.

use lapid::{Result, Tensor, bail};
use lapid_nn::attention::key_padding_mask;
use lapid_nn::{Activation, Attention, Embedding, GatedMlp, Linear, Loader, Mask, RmsNorm, Rope};

use super::config::{DenseActivation, EmbeddingGemmaConfig};

struct Layer {
    input_norm: RmsNorm,
    attn: Attention,
    post_attn_norm: RmsNorm,
    pre_ff_norm: RmsNorm,
    mlp: GatedMlp,
    post_ff_norm: RmsNorm,
    sliding: bool,
}

/// One padded batch of token ids.
pub(crate) struct Batch {
    /// Row-major [B, S] ids (padding = pad id).
    pub ids: Vec<i32>,
    /// Real token count per row (each >= 1).
    pub lengths: Vec<usize>,
    pub seq: usize,
}

impl Batch {
    /// Right-pad `rows` to `seq` (>= the longest row) with `pad`.
    pub fn new(rows: &[&[u32]], seq: usize, pad: u32) -> Result<Batch> {
        let mut ids = Vec::with_capacity(rows.len() * seq);
        let mut lengths = Vec::with_capacity(rows.len());
        for r in rows {
            if r.is_empty() || r.len() > seq {
                bail!("invalid_argument", "batch row of {} tokens does not fit 1..={seq}", r.len());
            }
            ids.extend(r.iter().map(|&t| t as i32));
            ids.extend(std::iter::repeat_n(pad as i32, seq - r.len()));
            lengths.push(r.len());
        }
        if rows.is_empty() {
            bail!("invalid_argument", "empty batch");
        }
        Ok(Batch { ids, lengths, seq })
    }

    pub fn size(&self) -> usize {
        self.lengths.len()
    }

    fn padded(&self) -> bool {
        self.lengths.iter().any(|&l| l != self.seq)
    }
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

/// Pooling weights [B, 1, S]: `1 / len` on real tokens, 0 on padding.
fn mean_weights(lengths: &[usize], seq: usize) -> Result<Tensor> {
    let mut data = vec![0.0f32; lengths.len() * seq];
    for (b, &len) in lengths.iter().enumerate() {
        data[b * seq..b * seq + len].fill(1.0 / len as f32);
    }
    Tensor::from_f32(&data, &[lengths.len(), 1, seq])
}

/// Graph outputs of one batch.
pub(crate) struct Outputs {
    /// [B, out_dim] head output (before L2 normalization).
    pub embedding: Tensor,
    /// Named intermediates when traced: "embed", "layers.{i}", "norm" ([B, S, D]),
    /// "pooled", "dense.{k}" ([B, D_k]).
    pub traced: Vec<Tensor>,
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

    /// Build the graph of one batch; `trace` collects named intermediates.
    pub fn graph(&self, batch: &Batch, trace: bool) -> Result<Outputs> {
        let (b, s) = (batch.size(), batch.seq);
        let mut traced = Vec::new();
        let mut keep = |t: Tensor, name: String| -> Tensor {
            let t = t.named(name);
            if trace {
                traced.push(t.clone());
            }
            t
        };
        let ids = Tensor::from_i32(&batch.ids, &[b, s])?;
        let mut x = keep(self.embed.forward(&ids)?.mul_scalar(f64::from(self.embed_scale))?, "embed".into());

        let full_mask = if batch.padded() {
            Mask::Custom { additive: key_padding_mask(&batch.lengths, s)?, causal: false }
        } else {
            Mask::None
        };
        // Within the window everywhere: the sliding layers see what full layers see.
        let sliding_mask = if s <= self.window {
            full_mask.clone()
        } else {
            Mask::Custom { additive: sliding_padding_mask(&batch.lengths, s, self.window)?, causal: false }
        };

        for (i, layer) in self.layers.iter().enumerate() {
            let (rope, mask) =
                if layer.sliding { (&self.rope_local, &sliding_mask) } else { (&self.rope_global, &full_mask) };
            let h = layer.input_norm.forward(&x)?;
            let a = layer.attn.forward(&h, Some(rope), None, mask)?;
            x = x.add(&layer.post_attn_norm.forward(&a)?)?;
            let h = layer.pre_ff_norm.forward(&x)?;
            let m = layer.mlp.forward(&h)?;
            x = keep(x.add(&layer.post_ff_norm.forward(&m)?)?, format!("layers.{i}"));
        }
        let tokens = keep(self.norm.forward(&x)?, "norm".into());

        let d = tokens.shape()[2];
        let pooled = mean_weights(&batch.lengths, s)?.matmul(&tokens)?.reshape(&[b, d])?;
        let mut y = keep(pooled, "pooled".into());
        for (k, (lin, act)) in self.dense.iter().enumerate() {
            let z = lin.forward(&y)?;
            let z = match act {
                DenseActivation::Identity => z,
                DenseActivation::Tanh => z.tanh()?,
            };
            y = keep(z, format!("dense.{k}"));
        }
        Ok(Outputs { embedding: y, traced })
    }
}

/// `x / max(||x||, 1e-12)` (torch `F.normalize`), in place.
pub(crate) fn l2_normalize(x: &mut [f32]) {
    let n = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt().max(1e-12);
    for v in x {
        *v = (f64::from(*v) / n) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_padding() {
        let b = Batch::new(&[&[2, 5, 1], &[2, 1]], 4, 0).expect("batch");
        assert_eq!(b.ids, vec![2, 5, 1, 0, 2, 1, 0, 0]);
        assert_eq!(b.lengths, vec![3, 2]);
        assert!(Batch::new(&[&[1, 2, 3]], 2, 0).is_err());
        assert!(Batch::new(&[], 2, 0).is_err());
    }

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

    #[test]
    fn normalize() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-7 && (v[1] - 0.8).abs() < 1e-7);
        let mut z = vec![0.0, 0.0];
        l2_normalize(&mut z);
        assert_eq!(z, vec![0.0, 0.0]);
    }
}
