//! Whisper text decoder with a realized self-attention KV cache and per-window cross
//! attention K/V. Logits use the tied token embedding: `x @ embed_tokens^T`.

use lapid::{Device, Error, Result, Tensor};
use lapid_nn::weights::fake_quant_act;
use lapid_nn::{Attention, Embedding, KvCache, LayerNorm, Loader, Mask, Mlp};

use super::config::WhisperConfig;
use super::{attention, layer_norm, mlp};

struct DecoderLayer {
    self_ln: LayerNorm,
    self_attn: Attention,
    cross_ln: LayerNorm,
    cross_attn: Attention,
    mlp_ln: LayerNorm,
    mlp: Mlp,
}

/// Cross-attention keys / values per layer, [1, H, n_audio_ctx, Dh].
pub(crate) struct CrossKv {
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
        let p = "model.decoder";
        let d = cfg.n_text_state;
        let embed_tokens = l.tensor(&format!("{p}.embed_tokens.weight"))?;
        l.expect_shape(&embed_tokens, "decoder.embed_tokens.weight", &[cfg.n_vocab, d])?;
        let positions = l.tensor_f32(&format!("{p}.embed_positions.weight"))?;
        l.expect_shape(&positions, "decoder.embed_positions.weight", &[cfg.n_text_ctx, d])?;
        let mut layers = Vec::with_capacity(cfg.n_text_layer);
        for i in 0..cfg.n_text_layer {
            let lp = format!("{p}.layers.{i}");
            let h = cfg.n_text_head;
            layers.push(DecoderLayer {
                self_ln: layer_norm(l, &format!("{lp}.self_attn_layer_norm"))?,
                self_attn: attention(l, &format!("{lp}.self_attn"), d, h)?,
                cross_ln: layer_norm(l, &format!("{lp}.encoder_attn_layer_norm"))?,
                cross_attn: attention(l, &format!("{lp}.encoder_attn"), d, h)?,
                mlp_ln: layer_norm(l, &format!("{lp}.final_layer_norm"))?,
                mlp: mlp(l, &lp)?,
            });
        }
        Ok(Decoder {
            logits_act_quant: l.act_quant(&format!("{p}.embed_tokens.weight"))?,
            embed_tokens,
            positions,
            layers,
            ln: layer_norm(l, &format!("{p}.layer_norm"))?,
            n_ctx: cfg.n_text_ctx,
        })
    }

    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    /// An empty self-attention cache for this decoder, preallocated for the full text
    /// context (`n_ctx` positions, 448 for Whisper) so it never grows.
    pub fn new_cache(&self) -> KvCache {
        KvCache::with_capacity(vec![None; self.layers.len()], self.n_ctx)
    }

    /// Lazy cross-attention K/V graphs from the encoder output [1, ctx, d].
    pub fn cross_kv_graph(&self, encoded: &Tensor) -> Result<CrossKv> {
        let mut k = Vec::with_capacity(self.layers.len());
        let mut v = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (lk, lv) = layer.cross_attn.project_kv(encoded)?;
            k.push(lk.contiguous()?);
            v.push(lv.contiguous()?);
        }
        Ok(CrossKv { k, v })
    }

    /// Cross-attention K/V realized on the device (once per 30 s window).
    pub fn cross_kv(&self, device: &Device, encoded: &Tensor) -> Result<CrossKv> {
        let lazy = self.cross_kv_graph(encoded)?;
        let outs: Vec<&Tensor> = lazy.k.iter().chain(lazy.v.iter()).collect();
        let mut realized = device.eval(&outs)?;
        if realized.len() != 2 * self.layers.len() {
            return Err(Error::new("backend", "cross K/V eval returned wrong output count"));
        }
        let v = realized.split_off(self.layers.len());
        Ok(CrossKv { k: realized, v })
    }

    /// Decoder graph for `ids` at positions `cache.offset()..+n`, attending to the cache and
    /// the cross K/V; the new self-attention K/V are staged in `cache` (commit after eval,
    /// or drop the cache). Logits are computed for `rows` (indices into `ids`) or for every
    /// row when `rows` is None: [1, r, n_vocab].
    pub fn forward(&self, ids: &[u32], cache: &mut KvCache, cross: &CrossKv, rows: Option<&[usize]>) -> Result<Tensor> {
        let out = self.build(ids, cache, cross, rows);
        if out.is_err() {
            cache.discard();
        }
        out
    }

    fn build(&self, ids: &[u32], cache: &mut KvCache, cross: &CrossKv, rows: Option<&[usize]>) -> Result<Tensor> {
        let n = ids.len();
        let offset = cache.offset();
        if n == 0 {
            return Err(Error::new("invalid_argument", "decoder called with no tokens"));
        }
        if offset + n > self.n_ctx {
            return Err(Error::new(
                "invalid_argument",
                format!("decoder context overflow: {} tokens > n_text_ctx {}", offset + n, self.n_ctx),
            ));
        }
        if cross.k.len() != self.layers.len() || cross.v.len() != self.layers.len() {
            return Err(Error::new("invalid_argument", "cross K/V has the wrong layer count"));
        }
        if cache.n_layers() != self.layers.len() {
            return Err(Error::new("invalid_argument", "KV cache has the wrong layer count"));
        }

        let tok = Embedding::new(self.embed_tokens.clone()).lookup(ids)?;
        let pos = self.positions.slice(0, offset, offset + n)?;
        let mut x = tok.add(&pos)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let h = layer.self_ln.forward(&x)?;
            x = x.add(&layer.self_attn.forward(&h, None, Some((&mut *cache, i)), &Mask::Causal)?)?;

            let h = layer.cross_ln.forward(&x)?;
            x = x.add(&layer.cross_attn.attend(&h, &cross.k[i], &cross.v[i], &Mask::None)?)?;

            let h = layer.mlp_ln.forward(&x)?;
            x = x.add(&layer.mlp.forward(&h)?)?;
        }

        let x = match rows {
            None => x,
            Some(rows) => select_rows(&x, rows, n)?,
        };
        let mut h = self.ln.forward(&x)?;
        if self.logits_act_quant {
            h = fake_quant_act(&h)?;
        }
        h.matmul_t(&self.embed_tokens)
    }

    /// One cached decoding step: runs `ids` (whole prompt first, then one token at a time),
    /// extends `cache` and returns f32 logits for `rows` ([rows.len() * n_vocab]).
    pub fn step(
        &self,
        device: &Device,
        ids: &[u32],
        cache: &mut KvCache,
        cross: &CrossKv,
        rows: &[usize],
    ) -> Result<Vec<f32>> {
        let logits = self.forward(ids, cache, cross, Some(rows))?;
        let mut outs: Vec<&Tensor> = Vec::with_capacity(1 + 2 * self.layers.len());
        outs.push(&logits);
        outs.extend(cache.staged());
        let mut realized = match device.eval(&outs) {
            Ok(r) if r.len() == outs.len() => r,
            Ok(_) => {
                cache.discard();
                return Err(Error::new("backend", "decoder eval returned wrong output count"));
            }
            Err(e) => {
                cache.discard();
                return Err(e);
            }
        };
        let new_kv = realized.split_off(1);
        let logits = device.to_vec_f32(&realized[0])?;
        cache.commit(new_kv)?;
        Ok(logits)
    }
}

/// Rows `rows` of x [1, n, d] along axis 1 -> [1, rows.len(), d].
fn select_rows(x: &Tensor, rows: &[usize], n: usize) -> Result<Tensor> {
    if rows.is_empty() {
        return Err(Error::new("invalid_argument", "no logit rows requested"));
    }
    if let Some(&bad) = rows.iter().find(|&&r| r >= n) {
        return Err(Error::new("invalid_argument", format!("logit row {bad} >= {n} tokens")));
    }
    if rows.len() == n && rows.iter().enumerate().all(|(i, &r)| i == r) {
        return Ok(x.clone());
    }
    let parts: Vec<Tensor> =
        rows.iter().map(|&r| x.slice(1, r, r + 1)).collect::<Result<_>>()?;
    if parts.len() == 1 {
        return Ok(parts[0].clone());
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 1)
}
