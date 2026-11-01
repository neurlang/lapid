//! Qwen3-VL: the generic vision encoder (`vision::VisionEncoder`, merger + DeepStack) and a
//! Qwen3 text decoder (HF `Qwen3VLTextModel`).
//!
//! Layer: `x += o(attn(rope(q_norm(q x)), rope(k_norm(k x)), v x))` on `input_layernorm(x)`
//! (GQA, causal), `x += swiglu(post_attention_layernorm(x))`; the first DeepStack layers add
//! the encoder's DeepStack features to the image rows after the layer. RoPE is M-RoPE: each
//! token has (t, h, w) positions, each frequency is driven by one of them
//! (`TextConfig::mrope_stream`); text tokens have t = h = w, an image of `H x W` merged
//! tokens at cursor `c` gets (c, c + row, c + col) and moves the cursor to `c + max(H, W)`
//! (HF `get_rope_index`). State: KV cache plus that cursor.

use lapid::{DType, Error, Result, Tensor, bail};
use lapid_nn::attention::{merge_heads, scaled_dot_product_attention};
use lapid_nn::weights::fake_quant_act;
use lapid_nn::{Embedding, GatedMlp, KvCache, Linear, Loader, Mask, RmsNorm};
use lapid_verify::harness::Reference;
use serde_json::Value;

use super::config::{Qwen3VlConfig, TextConfig};
use crate::runtime::{CausalLm, LmDims, LmState, Media, MediaPrompt, Rows, Taps, VisionLm};
use crate::vision::encoder::{apply_rope_tables, rope_tables};
use crate::vision::{Image, ImageProcessor, Patches, VisionEncoder};

pub const EMBED: &str = "model.embed_tokens.weight";
pub const LM_HEAD: &str = "lm_head.weight";

/// KV cache + M-RoPE cursor (next text position; ahead of the token count after an image).
pub struct MropeCache {
    pub kv: KvCache,
    pos: usize,
    next: usize,
    staged: Option<usize>,
}

impl LmState for MropeCache {
    fn offset(&self) -> usize {
        self.pos
    }

    fn staged(&self) -> Vec<&Tensor> {
        self.kv.staged()
    }

    fn commit(&mut self, realized: Vec<Tensor>, n: usize) -> Result<()> {
        self.kv.commit(realized)?;
        self.pos += n;
        self.next = self.staged.take().unwrap_or(self.next + n);
        Ok(())
    }

    fn discard(&mut self) {
        self.kv.discard();
        self.staged = None;
    }

    fn clear(&mut self) {
        self.kv.clear();
        (self.pos, self.next, self.staged) = (0, 0, None);
    }

    /// Text-only sequences (positions = token indices) can rewind.
    fn truncate(&mut self, len: usize) -> bool {
        if len > self.pos || self.next != self.pos || self.kv.truncate(len).is_err() {
            return false;
        }
        (self.pos, self.next) = (len, len);
        true
    }
}

/// M-RoPE positions of `n` tokens with image runs `media` from cursor `c`; the next cursor.
pub fn mrope_positions(n: usize, media: &[(usize, usize, (usize, usize))], mut c: usize) -> Result<(Vec<[usize; 3]>, usize)> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        match media.iter().find(|m| m.0 == out.len()) {
            Some(&(_, len, (gh, gw))) => {
                if len != gh * gw {
                    bail!("invalid_argument", "qwen3vl: image run of {len} tokens for a {gh}x{gw} grid");
                }
                out.extend((0..gh).flat_map(|r| (0..gw).map(move |col| [c, c + r, c + col])));
                c += gh.max(gw);
            }
            None => {
                out.push([c; 3]);
                c += 1;
            }
        }
    }
    out.truncate(n);
    Ok((out, c))
}

struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    norm1: RmsNorm,
    norm2: RmsNorm,
    mlp: GatedMlp,
}

pub struct Qwen3VlModel {
    pub cfg: Qwen3VlConfig,
    vision: VisionEncoder,
    processor: ImageProcessor,
    embed: Tensor,
    head: Option<Linear>,
    logits_act_quant: bool,
    layers: Vec<Layer>,
    norm: RmsNorm,
    inv_freq: Vec<f32>,
}

impl Qwen3VlModel {
    /// Pixel budget per image (`max_pixels`), e.g. to bound memory.
    pub fn set_max_pixels(&mut self, n: usize) {
        self.processor.max_pixels = n.max(self.processor.min_pixels);
    }

    /// cos / sin [n, Dh/2] of M-RoPE positions.
    fn rope(&self, pos: &[[usize; 3]]) -> Result<(Tensor, Tensor)> {
        let t = &self.cfg.text;
        let half = self.inv_freq.len();
        let a: Vec<f32> = pos.iter().flat_map(|p| self.inv_freq.iter().enumerate().map(move |(i, f)| p[t.mrope_stream(i)] as f32 * f)).collect();
        rope_tables(&a, pos.len(), half)
    }

    fn attn(&self, ly: &Layer, h: &Tensor, rope: &(Tensor, Tensor), kv: &mut KvCache, i: usize) -> Result<Tensor> {
        let (n, t) = (h.shape()[1], &self.cfg.text);
        let heads = |x: Tensor, nh: usize, norm: Option<&RmsNorm>| -> Result<Tensor> {
            let x = x.reshape(&[1, n, nh, t.head_dim])?;
            let x = match norm {
                Some(nm) => nm.forward(&x)?,
                None => x,
            };
            x.permute(&[0, 2, 1, 3])
        };
        let q = apply_rope_tables(&heads(ly.q.forward(h)?, t.n_head, Some(&ly.q_norm))?, &rope.0, &rope.1)?;
        let k = apply_rope_tables(&heads(ly.k.forward(h)?, t.n_kv_head, Some(&ly.k_norm))?, &rope.0, &rope.1)?;
        let (k, v) = kv.append(i, &k, &heads(ly.v.forward(h)?, t.n_kv_head, None)?)?;
        let a = scaled_dot_product_attention(&q, &k, &v, 1.0 / (t.head_dim as f32).sqrt(), &Mask::Causal)?;
        ly.o.forward(&merge_heads(&a)?)
    }
}

fn load_text(l: &Loader<'_>, t: &TextConfig) -> Result<Vec<Layer>> {
    let (d, eps, b, dh, ff) = (t.hidden_size, t.norm_eps, t.attention_bias, t.head_dim, t.intermediate_size);
    (0..t.n_layers)
        .map(|i| {
            let p = format!("model.layers.{i}");
            let lin = |n: &str, o: usize, inp: usize, bias: bool| l.linear_checked(&format!("{p}.{n}"), o, inp, bias);
            let norm = |n: &str, dim: usize| l.rms_norm_checked(&format!("{p}.{n}"), dim, eps);
            Ok(Layer {
                q: lin("self_attn.q_proj", t.n_head * dh, d, b)?,
                k: lin("self_attn.k_proj", t.n_kv_head * dh, d, b)?,
                v: lin("self_attn.v_proj", t.n_kv_head * dh, d, b)?,
                o: lin("self_attn.o_proj", d, t.n_head * dh, b)?,
                q_norm: norm("self_attn.q_norm", dh)?,
                k_norm: norm("self_attn.k_norm", dh)?,
                norm1: norm("input_layernorm", d)?,
                norm2: norm("post_attention_layernorm", d)?,
                mlp: GatedMlp::swiglu(lin("mlp.gate_proj", ff, d, false)?, lin("mlp.up_proj", ff, d, false)?, lin("mlp.down_proj", d, ff, false)?),
            })
        })
        .collect()
}

impl CausalLm for Qwen3VlModel {
    const ARCH: &'static str = "qwen3vl";
    type State = MropeCache;

    fn load(l: &Loader<'_>, meta: &Value) -> Result<Qwen3VlModel> {
        let cfg = Qwen3VlConfig::from_json(meta, None)?;
        let t = &cfg.text;
        let untied = !t.tie_embeddings && l.contains(LM_HEAD);
        let dh = t.head_dim;
        Ok(Qwen3VlModel {
            vision: VisionEncoder::load(l, &cfg.vision)?,
            processor: cfg.processor.clone(),
            embed: l.tensor_checked(EMBED, &[t.vocab_size, t.hidden_size])?,
            head: if untied { Some(l.linear_checked("lm_head", t.vocab_size, t.hidden_size, false)?) } else { None },
            logits_act_quant: !untied && l.act_quant(EMBED)?,
            layers: load_text(l, t)?,
            norm: l.rms_norm_checked("model.norm", t.hidden_size, t.norm_eps)?,
            inv_freq: (0..dh / 2).map(|i| 1.0 / t.rope_theta.powf((2 * i) as f32 / dh as f32)).collect(),
            cfg,
        })
    }

    fn dims(&self) -> LmDims {
        let t = &self.cfg.text;
        LmDims { vocab: t.vocab_size, hidden: t.hidden_size, layers: t.n_layers }
    }

    fn new_state(&self) -> MropeCache {
        MropeCache { kv: KvCache::new(self.layers.len()), pos: 0, next: 0, staged: None }
    }

    fn end_of_turn(&self) -> &'static [&'static str] {
        &["<|im_end|>", "<|endoftext|>"]
    }

    fn forward(&self, ids: &[u32], st: &mut MropeCache, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>> {
        self.forward_media(ids, &[], st, rows, taps)
    }
}

impl VisionLm for Qwen3VlModel {
    type Image = Patches;

    fn has_vision(&self) -> bool {
        true
    }

    fn image_tokens(&self) -> (u32, Option<u32>, Option<u32>) {
        (self.cfg.image_token_id, None, None)
    }

    fn prepare_image(&self, img: &Image) -> Result<Patches> {
        self.processor.process(img)
    }

    fn image_size(&self, p: &Patches) -> (usize, (usize, usize)) {
        let m = self.cfg.vision.spatial_merge_size;
        (p.merged_tokens(m), (p.grid_h / m, p.grid_w / m))
    }

    /// Merged tokens, then the DeepStack features (taps `vision.*`).
    fn encode_image(&self, p: &Patches, taps: &mut Taps) -> Result<Vec<Tensor>> {
        self.vision.build(p, taps)
    }

    /// Taps: `embeds`, `layers.{i}` (before DeepStack), `norm`.
    fn forward_media(&self, ids: &[u32], media: &[Media], st: &mut MropeCache, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>> {
        let (n, d) = (ids.len(), self.cfg.text.hidden_size);
        if n == 0 {
            bail!("invalid_argument", "qwen3vl: forward of zero tokens");
        }
        let runs: Vec<_> = media.iter().map(|m| (m.start, m.len, m.grid)).collect();
        let (pos, next) = mrope_positions(n, &runs, st.next)?;
        st.staged = Some(next);
        let rows_of = |t: &Tensor, m: &Media| t.cast(DType::F32)?.reshape(&[1, m.len, d]);
        let mut x = Embedding::new(self.embed.clone()).lookup(ids)?.cast(DType::F32)?;
        if !media.is_empty() {
            let (mut parts, mut at) = (Vec::new(), 0);
            for m in media {
                if m.start > at {
                    parts.push(x.slice(1, at, m.start)?);
                }
                parts.push(rows_of(&m.embeds[0], m)?);
                at = m.start + m.len;
            }
            if at < n {
                parts.push(x.slice(1, at, n)?);
            }
            x = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?;
        }
        taps.tap("embeds", &x);
        let rope = self.rope(&pos)?;
        for (i, ly) in self.layers.iter().enumerate() {
            x = x.add(&self.attn(ly, &ly.norm1.forward(&x)?, &rope, &mut st.kv, i)?)?;
            x = x.add(&ly.mlp.forward(&ly.norm2.forward(&x)?)?)?;
            taps.tap(format_args!("layers.{i}"), &x);
            for m in media {
                if let Some(ds) = m.embeds.get(1 + i) {
                    x = x.add(&rows_of(ds, m)?.pad(1, m.start, n - m.start - m.len, 0.0)?)?;
                }
            }
        }
        let x = match rows {
            Rows::None => return Ok(None),
            Rows::Last if n > 1 => x.slice(1, n - 1, n)?,
            _ => x,
        };
        let mut h = self.norm.forward(&x)?;
        if rows == Rows::All {
            taps.tap("norm", &h);
        }
        if let Some(head) = &self.head {
            return Ok(Some(head.forward(&h)?.cast(DType::F32)?));
        }
        if self.logits_act_quant {
            h = fake_quant_act(&h)?;
        }
        Ok(Some(h.matmul_t(&self.embed)?.cast(DType::F32)?))
    }

    fn image_pixels<'a>(&self, p: &'a Patches) -> &'a [f32] {
        &p.data
    }

    /// `<case>.pixel_values` [N, 3 P P] (first temporal frame) and `grid_thw`.
    fn reference_image(&self, r: &Reference, case: &str) -> Result<Patches> {
        let c = r.case(case);
        let g = |i: usize| c.pointer(&format!("/grid_thw/{i}")).and_then(Value::as_u64).map(|x| x as usize);
        let (Some(gh), Some(gw)) = (g(1), g(2)) else { return Err(Error::new("reference", format!("qwen3vl: case {case} has no grid_thw"))) };
        let data = r.acts.get(&format!("{case}.pixel_values"))?;
        Ok(Patches { dim: data.len() / (gh * gw).max(1), data, grid_h: gh, grid_w: gw })
    }

    fn media_taps(&self) -> Vec<String> {
        let mut v = vec!["vision.merged".to_string()];
        v.extend((0..self.cfg.vision.deepstack_layers.len()).map(|j| format!("vision.deepstack.{j}")));
        v
    }

    /// The prompt's M-RoPE positions vs the reference `positions` ([3][n]).
    fn check_prompt(&self, p: &MediaPrompt<Patches>, case: &Value) -> Option<bool> {
        let want = case.get("positions")?.as_array()?;
        let row = |k: usize| want.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect::<Vec<_>>());
        let runs: Vec<_> = p.images.iter().map(|i| (i.start, i.len, i.grid)).collect();
        let Ok((got, _)) = mrope_positions(p.ids.len(), &runs, 0) else { return Some(false) };
        let (t, h, w) = (row(0)?, row(1)?, row(2)?);
        Some(t.len() == got.len() && got.iter().enumerate().all(|(i, x)| *x == [t[i], h[i], w[i]]))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn positions() {
        // 2 text tokens, a 2x3 image, 1 text token.
        let (p, next) = super::mrope_positions(9, &[(2, 6, (2, 3))], 0).expect("positions");
        assert_eq!((p[1], p[2], p[7], p[8], next), ([1; 3], [2, 2, 2], [2, 3, 4], [5; 3], 6));
    }
}
