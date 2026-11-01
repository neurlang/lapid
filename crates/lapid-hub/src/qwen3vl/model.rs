//! Qwen3-VL text decoder (HF `Qwen3VLTextModel`, a Qwen3 decoder) as lazy lapid graphs.
//!
//! Layer: `x += o(attn(rope(q_norm(q x)), rope(k_norm(k x)), v x))` on `input_layernorm(x)`
//! (GQA, causal), `x += swiglu(post_attention_layernorm(x))`. RoPE is M-RoPE: every token
//! has (t, h, w) positions and each frequency of the head is driven by one of them
//! (`TextConfig::mrope_stream`); text tokens have t = h = w. Image tokens take the merged
//! vision features instead of their embedding, and the first DeepStack layers add the
//! encoder's DeepStack features to the image rows after the layer.
//!
//! Streaming state ([`State`]): KV cache plus the M-RoPE cursor (next text position).

use lapid::{DType, Device, Error, Result, Tensor};
use lapid_nn::attention::{merge_heads, scaled_dot_product_attention};
use lapid_nn::weights::fake_quant_act;
use lapid_nn::{Embedding, GatedMlp, KvCache, Linear, Loader, Mask, RmsNorm};

use super::config::TextConfig;
use crate::vision::encoder::{VisionOutput, apply_rope_tables, rope_tables};

pub(crate) const EMBED: &str = "model.embed_tokens.weight";
pub(crate) const LM_HEAD: &str = "lm_head.weight";
pub(crate) const PREFILL_CHUNK: usize = 512;

/// M-RoPE position (t, h, w) of one token.
pub type Pos3 = [usize; 3];

/// Vision features placed at token `start` (relative to the ids of a step), one row per token.
#[derive(Clone)]
pub struct ImageSpan<'a> {
    pub start: usize,
    pub out: &'a VisionOutput,
}

impl ImageSpan<'_> {
    pub(crate) fn len(&self) -> usize {
        self.out.merged.shape()[0]
    }
}

/// The part of an image inside one chunk: rows `src..src + len` at chunk row `row`.
struct Seg<'a> {
    row: usize,
    src: usize,
    len: usize,
    out: &'a VisionOutput,
}

struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
}

impl Attn {
    fn forward(&self, h: &Tensor, rope: &(Tensor, Tensor), cache: &mut KvCache, layer: usize) -> Result<Tensor> {
        let n = h.shape()[1];
        let (nh, nkv, dh) = (self.n_head, self.n_kv, self.head_dim);
        let q = self.q_norm.forward(&self.q.forward(h)?.reshape(&[1, n, nh, dh])?)?.permute(&[0, 2, 1, 3])?;
        let k = self.k_norm.forward(&self.k.forward(h)?.reshape(&[1, n, nkv, dh])?)?.permute(&[0, 2, 1, 3])?;
        let v = self.v.forward(h)?.reshape(&[1, n, nkv, dh])?.permute(&[0, 2, 1, 3])?;
        let q = apply_rope_tables(&q, &rope.0, &rope.1)?;
        let k = apply_rope_tables(&k, &rope.0, &rope.1)?;
        let (k, v) = cache.append(layer, &k, &v)?;
        let a = scaled_dot_product_attention(&q, &k, &v, 1.0 / (dh as f32).sqrt(), &Mask::Causal)?;
        self.o.forward(&merge_heads(&a)?)
    }
}

struct Layer {
    attn: Attn,
    norm1: RmsNorm,
    norm2: RmsNorm,
    mlp: GatedMlp,
}

pub struct State {
    kv: KvCache,
    next_pos: usize,
}

impl State {
    /// Tokens consumed so far.
    pub fn offset(&self) -> usize {
        self.kv.offset()
    }

    /// M-RoPE position of the next text token.
    pub fn next_pos(&self) -> usize {
        self.next_pos
    }

    pub fn clear(&mut self) {
        self.kv.clear();
        self.next_pos = 0;
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Rows {
    Last,
    All,
    None,
}

pub(crate) struct TextModel {
    pub cfg: TextConfig,
    embed: Tensor,
    head: Option<Linear>,
    logits_act_quant: bool,
    layers: Vec<Layer>,
    norm: RmsNorm,
    inv_freq: Vec<f32>,
}

fn norm(l: &Loader<'_>, name: &str, d: usize, eps: f32) -> Result<RmsNorm> {
    let n = l.rms_norm(name, eps)?;
    l.expect_shape(&n.weight, &format!("{name}.weight"), &[d])?;
    Ok(n)
}

fn linear(l: &Loader<'_>, name: &str, out: usize, inp: usize, bias: bool) -> Result<Linear> {
    let lin = l.linear(name, bias)?;
    l.expect_shape(&lin.weight, &format!("{name}.weight"), &[out, inp])?;
    Ok(lin)
}

impl TextModel {
    pub fn load(l: &Loader<'_>, cfg: &TextConfig) -> Result<TextModel> {
        let (d, eps, b) = (cfg.hidden_size, cfg.norm_eps, cfg.attention_bias);
        let embed = l.tensor(EMBED)?;
        l.expect_shape(&embed, EMBED, &[cfg.vocab_size, d])?;
        let (nh, nkv, dh) = (cfg.n_head, cfg.n_kv_head, cfg.head_dim);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("model.layers.{i}");
            let a = format!("{p}.self_attn");
            let m = format!("{p}.mlp");
            let ff = cfg.intermediate_size;
            layers.push(Layer {
                attn: Attn {
                    q: linear(l, &format!("{a}.q_proj"), nh * dh, d, b)?,
                    k: linear(l, &format!("{a}.k_proj"), nkv * dh, d, b)?,
                    v: linear(l, &format!("{a}.v_proj"), nkv * dh, d, b)?,
                    o: linear(l, &format!("{a}.o_proj"), d, nh * dh, b)?,
                    q_norm: norm(l, &format!("{a}.q_norm"), dh, eps)?,
                    k_norm: norm(l, &format!("{a}.k_norm"), dh, eps)?,
                    n_head: nh,
                    n_kv: nkv,
                    head_dim: dh,
                },
                norm1: norm(l, &format!("{p}.input_layernorm"), d, eps)?,
                norm2: norm(l, &format!("{p}.post_attention_layernorm"), d, eps)?,
                mlp: GatedMlp::swiglu(
                    linear(l, &format!("{m}.gate_proj"), ff, d, false)?,
                    linear(l, &format!("{m}.up_proj"), ff, d, false)?,
                    linear(l, &format!("{m}.down_proj"), d, ff, false)?,
                ),
            });
        }
        let untied = !cfg.tie_embeddings && l.contains(LM_HEAD);
        let head = if untied { Some(linear(l, "lm_head", cfg.vocab_size, d, false)?) } else { None };
        let half = dh / 2;
        let inv_freq = (0..half).map(|i| 1.0 / cfg.rope_theta.powf((2 * i) as f32 / dh as f32)).collect();
        Ok(TextModel {
            cfg: cfg.clone(),
            logits_act_quant: !untied && l.act_quant(EMBED)?,
            embed,
            head,
            layers,
            norm: norm(l, "model.norm", d, eps)?,
            inv_freq,
        })
    }

    pub fn new_state(&self) -> State {
        State { kv: KvCache::with_windows(vec![None; self.layers.len()]), next_pos: 0 }
    }

    /// cos / sin [n, Dh/2] of M-RoPE positions.
    fn rope(&self, pos: &[Pos3]) -> Result<(Tensor, Tensor)> {
        let half = self.inv_freq.len();
        let streams: Vec<usize> = (0..half).map(|i| self.cfg.mrope_stream(i)).collect();
        let mut a = Vec::with_capacity(pos.len() * half);
        for p in pos {
            for (i, f) in self.inv_freq.iter().enumerate() {
                a.push(p[streams[i]] as f32 * f);
            }
        }
        rope_tables(&a, pos.len(), half)
    }

    /// Token embeddings [1, n, D] with image rows replaced by vision features.
    fn embed(&self, ids: &[u32], segs: &[Seg<'_>]) -> Result<Tensor> {
        let x = Embedding::new(self.embed.clone()).lookup(ids)?.cast(DType::F32)?;
        if segs.is_empty() {
            return Ok(x);
        }
        let d = self.cfg.hidden_size;
        let mut parts = Vec::new();
        let mut at = 0;
        for s in segs {
            if s.row > at {
                parts.push(x.slice(1, at, s.row)?);
            }
            parts.push(s.out.merged.slice(0, s.src, s.src + s.len)?.cast(DType::F32)?.reshape(&[1, s.len, d])?);
            at = s.row + s.len;
        }
        if at < ids.len() {
            parts.push(x.slice(1, at, ids.len())?);
        }
        Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)
    }

    /// Graph for `ids` (positions `pos`) appended to `state`; stages K/V. With `trace`:
    /// ("embeds" | "layers.{i}" (before DeepStack) | "norm", t).
    fn build(&self, ids: &[u32], pos: &[Pos3], segs: &[Seg<'_>], state: &mut State, rows: Rows, mut trace: Option<&mut Vec<(String, Tensor)>>) -> Result<Option<Tensor>> {
        let n = ids.len();
        if n == 0 || pos.len() != n {
            return Err(Error::new("invalid_argument", format!("qwen3vl: {n} tokens with {} positions", pos.len())));
        }
        let mut x = self.embed(ids, segs)?;
        let mut push = |name: String, t: &Tensor| {
            if let Some(tr) = trace.as_deref_mut() {
                tr.push((name, t.clone()));
            }
        };
        push("embeds".into(), &x);
        let rope = self.rope(pos)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let a = layer.attn.forward(&layer.norm1.forward(&x)?, &rope, &mut state.kv, i)?;
            x = x.add(&a)?;
            x = x.add(&layer.mlp.forward(&layer.norm2.forward(&x)?)?)?;
            push(format!("layers.{i}"), &x);
            for s in segs {
                if let Some(ds) = s.out.deepstack.get(i) {
                    let f = ds.slice(0, s.src, s.src + s.len)?.cast(DType::F32)?.reshape(&[1, s.len, self.cfg.hidden_size])?;
                    x = x.add(&f.pad(1, s.row, n - s.row - s.len, 0.0)?)?;
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
            push("norm".into(), &h);
        }
        if let Some(head) = &self.head {
            return Ok(Some(head.forward(&h)?.cast(DType::F32)?));
        }
        if self.logits_act_quant {
            h = fake_quant_act(&h)?;
        }
        Ok(Some(h.matmul_t(&self.embed)?.cast(DType::F32)?))
    }

    fn eval_commit(&self, device: &Device, outputs: &[&Tensor], state: &mut State) -> Result<Vec<Tensor>> {
        let mut outs: Vec<&Tensor> = outputs.to_vec();
        outs.extend(state.kv.staged());
        let mut realized = match device.eval(&outs) {
            Ok(r) if r.len() == outs.len() => r,
            Ok(_) => {
                state.kv.discard();
                return Err(Error::new("backend", "qwen3vl eval returned the wrong number of outputs"));
            }
            Err(e) => {
                state.kv.discard();
                return Err(e);
            }
        };
        let st = realized.split_off(outputs.len());
        state.kv.commit(st)?;
        Ok(realized)
    }

    /// Feed `ids` at M-RoPE positions `pos` (chunked) with `images` placed at their spans;
    /// returns the f32 logits of the last position.
    pub fn step(&self, device: &Device, ids: &[u32], pos: &[Pos3], images: &[ImageSpan<'_>], state: &mut State) -> Result<Vec<f32>> {
        if ids.is_empty() || pos.len() != ids.len() {
            return Err(Error::new("invalid_argument", "qwen3vl: step needs tokens and one position per token"));
        }
        let n_chunks = ids.len().div_ceil(PREFILL_CHUNK);
        for ci in 0..n_chunks {
            let (a, b) = (ci * PREFILL_CHUNK, ((ci + 1) * PREFILL_CHUNK).min(ids.len()));
            let segs = segments(images, a, b);
            let rows = if ci + 1 == n_chunks { Rows::Last } else { Rows::None };
            let out = match self.build(&ids[a..b], &pos[a..b], &segs, state, rows, None) {
                Ok(o) => o,
                Err(e) => {
                    state.kv.discard();
                    return Err(e);
                }
            };
            let outputs: Vec<&Tensor> = out.iter().collect();
            let realized = self.eval_commit(device, &outputs, state)?;
            let max = pos[a..b].iter().flat_map(|p| p.iter()).copied().max().unwrap_or(0);
            state.next_pos = state.next_pos.max(max + 1);
            if ci + 1 == n_chunks {
                let logits = realized.first().ok_or_else(|| Error::new("backend", "qwen3vl: no logits realized"))?;
                return device.to_vec_f32(logits);
            }
        }
        Err(Error::new("internal", "qwen3vl: prefill produced no logits"))
    }

    /// One uncached forward with every intermediate realized (see [`TextModel::build`]).
    pub fn forward_traced(&self, device: &Device, ids: &[u32], pos: &[Pos3], images: &[ImageSpan<'_>], rows: Rows) -> Result<Vec<(String, Vec<f32>)>> {
        let mut state = self.new_state();
        let mut trace = Vec::new();
        let segs = segments(images, 0, ids.len());
        if let Some(l) = self.build(ids, pos, &segs, &mut state, rows, Some(&mut trace))? {
            trace.push(("logits".to_string(), l));
        }
        let refs: Vec<&Tensor> = trace.iter().map(|(_, t)| t).collect();
        let realized = device.eval(&refs)?;
        trace.iter().zip(&realized).map(|((n, _), t)| Ok((n.clone(), device.to_vec_f32(t)?))).collect()
    }
}

/// Image rows inside the chunk `[a, b)` of the step's ids.
fn segments<'a>(images: &[ImageSpan<'a>], a: usize, b: usize) -> Vec<Seg<'a>> {
    let mut v: Vec<Seg<'a>> = images
        .iter()
        .filter_map(|im| {
            let (s, e) = (im.start.max(a), (im.start + im.len()).min(b));
            (s < e).then(|| Seg { row: s - a, src: s - im.start, len: e - s, out: im.out })
        })
        .collect();
    v.sort_by_key(|s| s.row);
    v
}
