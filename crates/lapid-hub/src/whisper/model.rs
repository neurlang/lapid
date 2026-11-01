//! Whisper network (HF `WhisperForConditionalGeneration` layout) as lazy graphs.
//!
//! * Encoder: conv1 (k3, p1) + GELU, conv2 (k3, s2, p1) + GELU, positional embedding, pre-LN
//!   transformer blocks, final LayerNorm.
//! * Decoder: token + positional embedding, blocks of self-attention (realized KV cache),
//!   cross-attention on per-window K/V and an MLP; logits use the tied token embedding.

use lapid::{Device, Error, Result, Tensor, bail};
use lapid_nn::weights::fake_quant_act;
use lapid_nn::{Activation, Attention, Conv1d, Embedding, KvCache, LayerNorm, Loader, Mask, Mlp};

use super::config::WhisperConfig;

const LN_EPS: f32 = 1e-5;

fn layer_norm(l: &Loader<'_>, prefix: &str) -> Result<LayerNorm> {
    l.layer_norm(prefix, LN_EPS)
}

/// Multi-head attention: `q_proj`, `k_proj` (no bias), `v_proj`, `out_proj`.
fn attention(l: &Loader<'_>, p: &str, d: usize, heads: usize) -> Result<Attention> {
    let q = l.linear_checked(&format!("{p}.q_proj"), d, d, true)?;
    let (k, v, o) = (l.linear(&format!("{p}.k_proj"), false)?, l.linear(&format!("{p}.v_proj"), true)?, l.linear(&format!("{p}.out_proj"), true)?);
    Attention::new(q, k, v, o, heads, heads)
}

/// fc2(gelu(fc1(x))).
fn mlp(l: &Loader<'_>, p: &str) -> Result<Mlp> {
    Ok(Mlp::new(l.linear(&format!("{p}.fc1"), true)?, l.linear(&format!("{p}.fc2"), true)?, Activation::Gelu))
}

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
        let (p, d) = ("model.encoder", cfg.n_audio_state);
        let conv1_w = l.tensor_checked(&format!("{p}.conv1.weight"), &[d, cfg.n_mels, 3])?;
        let conv1 = Conv1d::new(conv1_w, Some(l.tensor(&format!("{p}.conv1.bias"))?)).with_padding(1);
        let positions = l.tensor_f32(&format!("{p}.embed_positions.weight"))?;
        l.expect_shape(&positions, "encoder.embed_positions.weight", &[cfg.n_audio_ctx, d])?;
        let layers = (0..cfg.n_audio_layer)
            .map(|i| {
                let lp = format!("{p}.layers.{i}");
                Ok(EncoderLayer {
                    attn_ln: layer_norm(l, &format!("{lp}.self_attn_layer_norm"))?,
                    attn: attention(l, &format!("{lp}.self_attn"), d, cfg.n_audio_head)?,
                    mlp_ln: layer_norm(l, &format!("{lp}.final_layer_norm"))?,
                    mlp: mlp(l, &lp)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Encoder {
            conv1,
            conv2: l.conv1d(&format!("{p}.conv2"), true)?.with_stride(2).with_padding(1),
            positions,
            layers,
            ln: layer_norm(l, &format!("{p}.layer_norm"))?,
            n_mels: cfg.n_mels,
            n_frames: 2 * cfg.n_audio_ctx,
        })
    }

    /// Graph of one `[n_mels, 2 n_audio_ctx]` log-mel window: (output `[1, n_audio_ctx, d]`,
    /// the named intermediates "encoder.embed", "encoder.layers.{i}", "encoder.out").
    pub fn graph(&self, mel: &[f32]) -> Result<(Tensor, Vec<Tensor>)> {
        if mel.len() != self.n_mels * self.n_frames {
            bail!("invalid_argument", "mel has {} values, expected n_mels * frames = {} * {}", mel.len(), self.n_mels, self.n_frames);
        }
        let x = Tensor::from_f32(mel, &[1, self.n_mels, self.n_frames])?;
        let x = self.conv2.forward(&self.conv1.forward(&x)?.gelu()?)?.gelu()?;
        let mut x = x.permute(&[0, 2, 1])?.add(&self.positions)?.named("encoder.embed");
        let mut traced = vec![x.clone()];
        for (i, layer) in self.layers.iter().enumerate() {
            let h = layer.attn_ln.forward(&x)?;
            let (k, v) = layer.attn.project_kv(&h)?;
            x = x.add(&layer.attn.attend(&h, &k, &v, &Mask::None)?)?;
            x = x.add(&layer.mlp.forward(&layer.mlp_ln.forward(&x)?)?)?.named(format!("encoder.layers.{i}"));
            traced.push(x.clone());
        }
        let out = self.ln.forward(&x)?.named("encoder.out");
        traced.push(out.clone());
        Ok((out, traced))
    }
}

struct DecoderLayer {
    self_ln: LayerNorm,
    self_attn: Attention,
    cross_ln: LayerNorm,
    cross_attn: Attention,
    mlp_ln: LayerNorm,
    mlp: Mlp,
}

/// Cross-attention keys / values per layer, `[1, H, n_audio_ctx, Dh]`.
pub struct CrossKv {
    pub k: Vec<Tensor>,
    pub v: Vec<Tensor>,
}

pub(crate) struct Decoder {
    embed_tokens: Tensor,
    /// Fake-quantize the input of the logits projection (reference models only).
    logits_act_quant: bool,
    /// f32 [n_text_ctx, d]
    positions: Tensor,
    layers: Vec<DecoderLayer>,
    ln: LayerNorm,
    n_ctx: usize,
}

impl Decoder {
    pub fn load(l: &Loader<'_>, cfg: &WhisperConfig) -> Result<Decoder> {
        let (p, d, h) = ("model.decoder", cfg.n_text_state, cfg.n_text_head);
        let positions = l.tensor_f32(&format!("{p}.embed_positions.weight"))?;
        l.expect_shape(&positions, "decoder.embed_positions.weight", &[cfg.n_text_ctx, d])?;
        let layers = (0..cfg.n_text_layer)
            .map(|i| {
                let lp = format!("{p}.layers.{i}");
                Ok(DecoderLayer {
                    self_ln: layer_norm(l, &format!("{lp}.self_attn_layer_norm"))?,
                    self_attn: attention(l, &format!("{lp}.self_attn"), d, h)?,
                    cross_ln: layer_norm(l, &format!("{lp}.encoder_attn_layer_norm"))?,
                    cross_attn: attention(l, &format!("{lp}.encoder_attn"), d, h)?,
                    mlp_ln: layer_norm(l, &format!("{lp}.final_layer_norm"))?,
                    mlp: mlp(l, &lp)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Decoder {
            embed_tokens: l.tensor_checked(&format!("{p}.embed_tokens.weight"), &[cfg.n_vocab, d])?,
            logits_act_quant: l.act_quant(&format!("{p}.embed_tokens.weight"))?,
            positions,
            layers,
            ln: layer_norm(l, &format!("{p}.layer_norm"))?,
            n_ctx: cfg.n_text_ctx,
        })
    }

    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    /// An empty self-attention cache preallocated for the full text context.
    pub fn new_cache(&self) -> KvCache {
        KvCache::with_capacity(vec![None; self.layers.len()], self.n_ctx)
    }

    /// Lazy cross-attention K/V of the encoder output `[1, ctx, d]`.
    pub fn cross_kv(&self, encoded: &Tensor) -> Result<CrossKv> {
        let (mut k, mut v) = (Vec::new(), Vec::new());
        for layer in &self.layers {
            let (lk, lv) = layer.cross_attn.project_kv(encoded)?;
            k.push(lk.contiguous()?);
            v.push(lv.contiguous()?);
        }
        Ok(CrossKv { k, v })
    }

    /// Decoder graph for `ids` at positions `cache.offset()..`, staging the new self-attention
    /// K/V in `cache`; logits of `rows` (all rows when `None`): `[1, r, n_vocab]`.
    pub fn forward(&self, ids: &[u32], cache: &mut KvCache, cross: &CrossKv, rows: Option<&[usize]>) -> Result<Tensor> {
        let out = self.build(ids, cache, cross, rows);
        if out.is_err() {
            cache.discard();
        }
        out
    }

    fn build(&self, ids: &[u32], cache: &mut KvCache, cross: &CrossKv, rows: Option<&[usize]>) -> Result<Tensor> {
        let (n, offset) = (ids.len(), cache.offset());
        if n == 0 || offset + n > self.n_ctx {
            bail!("invalid_argument", "decoder: {n} tokens at offset {offset} (n_text_ctx {})", self.n_ctx);
        }
        if cross.k.len() != self.layers.len() || cache.n_layers() != self.layers.len() {
            bail!("invalid_argument", "cross K/V or KV cache has the wrong layer count");
        }
        let mut x = Embedding::new(self.embed_tokens.clone()).lookup(ids)?.add(&self.positions.slice(0, offset, offset + n)?)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let h = layer.self_ln.forward(&x)?;
            x = x.add(&layer.self_attn.forward(&h, None, Some((&mut *cache, i)), &Mask::Causal)?)?;
            let h = layer.cross_ln.forward(&x)?;
            x = x.add(&layer.cross_attn.attend(&h, &cross.k[i], &cross.v[i], &Mask::None)?)?;
            x = x.add(&layer.mlp.forward(&layer.mlp_ln.forward(&x)?)?)?;
        }
        let x = match rows {
            Some(rows) => select_rows(&x, rows, n)?,
            None => x,
        };
        let mut h = self.ln.forward(&x)?;
        if self.logits_act_quant {
            h = fake_quant_act(&h)?;
        }
        h.matmul_t(&self.embed_tokens)
    }

    /// One cached step: runs `ids` (prompt first, then one token at a time), commits the
    /// cache and returns host logits of `rows` (`[rows.len() * n_vocab]`).
    pub fn step(&self, device: &Device, ids: &[u32], cache: &mut KvCache, cross: &CrossKv, rows: &[usize]) -> Result<Vec<f32>> {
        let logits = self.forward(ids, cache, cross, Some(rows))?;
        let mut outs: Vec<&Tensor> = vec![&logits];
        outs.extend(cache.staged());
        let mut realized = match device.eval(&outs) {
            Ok(r) if r.len() == outs.len() => r,
            other => {
                cache.discard();
                other?;
                bail!("backend", "decoder eval returned wrong output count");
            }
        };
        let new_kv = realized.split_off(1);
        let logits = device.to_vec_f32(&realized[0])?;
        cache.commit(new_kv)?;
        Ok(logits)
    }
}

/// Rows `rows` of x `[1, n, d]` along axis 1.
fn select_rows(x: &Tensor, rows: &[usize], n: usize) -> Result<Tensor> {
    if rows.is_empty() || rows.iter().any(|&r| r >= n) {
        return Err(Error::new("invalid_argument", format!("logit rows {rows:?} for {n} tokens")));
    }
    if rows.len() == n && rows.iter().enumerate().all(|(i, &r)| i == r) {
        return Ok(x.clone());
    }
    let parts: Vec<Tensor> = rows.iter().map(|&r| x.slice(1, r, r + 1)).collect::<Result<_>>()?;
    let refs: Vec<&Tensor> = parts.iter().collect();
    if refs.len() == 1 { Ok(parts[0].clone()) } else { Tensor::cat(&refs, 1) }
}
