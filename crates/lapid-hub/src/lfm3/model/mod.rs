//! LFM3 network (HF `Lfm3ForCausalLM` layout) as lazy lapid graphs built from `lapid_nn` blocks.
//!
//! Decoder layer (sandwich norms):
//! `x += post_op_norm(op(operator_norm(x)))`, `x += post_ffn_norm(mlp(ffn_norm(x)))`, where
//! `op` is a gated short convolution (`conv` layers) or GQA attention with q/k RMSNorm, RoPE,
//! a per-head sigmoid output gate and an optional sliding window. Final `embedding_norm`,
//! logits through the tied token embedding.
//!
//! Streaming state ([`State`]): a `KvCache` (sliding layers keep their last `window` rows)
//! plus one conv window per conv layer (`Conv1d::forward_stream`). One step is
//! build → eval (logits + staged K/V + staged conv windows) → commit.

pub(crate) mod rope;

use lapid::{DType, Device, Error, Result, Tensor};
use lapid_nn::attention::{merge_heads, scaled_dot_product_attention};
use lapid_nn::weights::fake_quant_act;
use lapid_nn::{Attention, Conv1d, Embedding, GatedMlp, KvCache, Linear, Loader, Mask, RmsNorm, ops};

use super::config::{LayerType, Lfm3Config};
use rope::ZeroTailRope;

/// Token embedding; also the tied output projection.
pub(crate) const EMBED: &str = "model.embed_tokens.weight";
/// Prompt tokens per evaluation during prefill (bounds attention scratch memory).
pub(crate) const PREFILL_CHUNK: usize = 512;

/// Gated short conv: `out_proj(C * conv(B * x))` with `B, C, x = in_proj(h)` split by channel.
struct ShortConv {
    in_proj: Linear,
    conv: Conv1d,
    out_proj: Linear,
    d: usize,
}

impl ShortConv {
    /// h [1, n, D] -> [1, n, D]; `state` is the conv window (replaced by a lazy new window).
    fn forward(&self, h: &Tensor, state: &mut Option<Tensor>) -> Result<Tensor> {
        let d = self.d;
        let bcx = self.in_proj.forward(h)?; // [1, n, 3D]
        let b = bcx.slice(2, 0, d)?;
        let c = bcx.slice(2, d, 2 * d)?;
        let x = bcx.slice(2, 2 * d, 3 * d)?;
        let bx = b.mul(&x)?.transpose(1, 2)?; // [1, D, n]
        let conv = self.conv.forward_stream(&bx, state)?.transpose(1, 2)?; // [1, n, D]
        self.out_proj.forward(&c.mul(&conv)?)
    }
}

/// Attention with q/k norm, zero-tail RoPE, sigmoid output gate per head.
struct GatedAttention {
    attn: Attention,
    gate: Linear,
    rope: ZeroTailRope,
    mask: Mask,
}

impl GatedAttention {
    fn forward(&self, h: &Tensor, cache: &mut KvCache, layer: usize) -> Result<Tensor> {
        let offset = cache.offset();
        let q = self.rope.apply(&self.attn.project_q(h)?, offset)?;
        let (k, v) = self.attn.project_kv(h)?;
        let k = self.rope.apply(&k, offset)?;
        let (k, v) = cache.append(layer, &k, &v)?;
        let a = scaled_dot_product_attention(&q, &k, &v, self.attn.scale, &self.mask)?; // [1, H, n, Dh]
        // gate [1, n, H] -> [1, H, n, 1]
        let g = ops::sigmoid(&self.gate.forward(h)?)?.permute(&[0, 2, 1])?.unsqueeze(3)?;
        self.attn.o.forward(&merge_heads(&a.mul(&g)?)?)
    }
}

enum Operator {
    Conv(ShortConv),
    Attention(GatedAttention),
}

struct Layer {
    op: Operator,
    op_norm: RmsNorm,
    post_op_norm: Option<RmsNorm>,
    ffn_norm: RmsNorm,
    post_ffn_norm: Option<RmsNorm>,
    mlp: GatedMlp,
}

/// Mutable per-sequence state: KV cache, conv windows, staged (uncommitted) conv windows.
pub struct State {
    kv: KvCache,
    conv: Vec<Option<Tensor>>,
    staged_conv: Vec<(usize, Tensor)>,
}

impl State {
    /// Tokens consumed so far (absolute position of the next token).
    pub fn offset(&self) -> usize {
        self.kv.offset()
    }

    fn discard(&mut self) {
        self.kv.discard();
        self.staged_conv.clear();
    }

    /// Forget everything (start of a new sequence).
    pub fn clear(&mut self) {
        self.kv.clear();
        self.conv.iter_mut().for_each(|c| *c = None);
        self.staged_conv.clear();
    }

    fn staged(&self) -> Vec<&Tensor> {
        let mut v = self.kv.staged();
        v.extend(self.staged_conv.iter().map(|(_, t)| t));
        v
    }

    fn commit(&mut self, realized: Vec<Tensor>) -> Result<()> {
        let n_kv = self.kv.staged().len();
        if realized.len() != n_kv + self.staged_conv.len() {
            self.discard();
            return Err(Error::new("backend", "lfm3 step: wrong number of realized state tensors"));
        }
        let mut realized = realized;
        let conv = realized.split_off(n_kv);
        for ((layer, _), t) in self.staged_conv.drain(..).zip(conv) {
            self.conv[layer] = Some(t);
        }
        self.kv.commit(realized)
    }
}

/// Which logit rows to compute.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Rows {
    Last,
    All,
    None,
}

pub(crate) struct Lfm3Model {
    pub cfg: Lfm3Config,
    embed: Tensor,
    logits_act_quant: bool,
    layers: Vec<Layer>,
    norm: RmsNorm,
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

impl Lfm3Model {
    pub fn load(l: &Loader<'_>, cfg: &Lfm3Config) -> Result<Lfm3Model> {
        let (d, eps, hd) = (cfg.hidden_size, cfg.norm_eps, cfg.head_dim);
        let embed = l.tensor(EMBED)?;
        l.expect_shape(&embed, EMBED, &[cfg.vocab_size, d])?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for (i, &kind) in cfg.layer_types.iter().enumerate() {
            let p = format!("model.layers.{i}");
            let op = match kind {
                LayerType::Conv => {
                    let w = l.tensor(&format!("{p}.conv.conv.weight"))?;
                    l.expect_shape(&w, &format!("{p}.conv.conv.weight"), &[d, 1, cfg.conv_kernel])?;
                    let bias = if cfg.conv_bias { Some(l.tensor(&format!("{p}.conv.conv.bias"))?) } else { None };
                    Operator::Conv(ShortConv {
                        in_proj: linear(l, &format!("{p}.conv.in_proj"), 3 * d, d, cfg.conv_bias)?,
                        conv: Conv1d::depthwise(w, bias),
                        out_proj: linear(l, &format!("{p}.conv.out_proj"), d, d, cfg.conv_bias)?,
                        d,
                    })
                }
                LayerType::FullAttention | LayerType::SlidingAttention => {
                    let a = format!("{p}.self_attn");
                    let attn = Attention::new(
                        linear(l, &format!("{a}.q_proj"), cfg.n_head * hd, d, false)?,
                        linear(l, &format!("{a}.k_proj"), cfg.n_kv_head * hd, d, false)?,
                        linear(l, &format!("{a}.v_proj"), cfg.n_kv_head * hd, d, false)?,
                        linear(l, &format!("{a}.o_proj"), d, cfg.n_head * hd, false)?,
                        cfg.n_head,
                        cfg.n_kv_head,
                    )?
                    .with_qk_norm(norm(l, &format!("{a}.q_norm"), hd, eps)?, norm(l, &format!("{a}.k_norm"), hd, eps)?);
                    let (rp, mask) = if kind == LayerType::SlidingAttention {
                        (cfg.rope_sliding, Mask::Sliding(cfg.sliding_window))
                    } else {
                        (cfg.rope_full, Mask::Causal)
                    };
                    Operator::Attention(GatedAttention {
                        attn,
                        gate: linear(l, &format!("{a}.gate_proj"), cfg.n_head, d, false)?,
                        rope: ZeroTailRope::new(hd, rp.theta, rp.partial_rotary_factor),
                        mask,
                    })
                }
            };
            let f = format!("{p}.feed_forward");
            let sandwich = |n: &str| -> Result<Option<RmsNorm>> {
                if cfg.sandwich_norm { Ok(Some(norm(l, &format!("{p}.{n}"), d, eps)?)) } else { Ok(None) }
            };
            layers.push(Layer {
                op,
                op_norm: norm(l, &format!("{p}.operator_norm"), d, eps)?,
                post_op_norm: sandwich("post_operator_norm")?,
                ffn_norm: norm(l, &format!("{p}.ffn_norm"), d, eps)?,
                post_ffn_norm: sandwich("post_ffn_norm")?,
                mlp: GatedMlp::swiglu(
                    linear(l, &format!("{f}.w1"), cfg.intermediate_size, d, false)?,
                    linear(l, &format!("{f}.w3"), cfg.intermediate_size, d, false)?,
                    linear(l, &format!("{f}.w2"), d, cfg.intermediate_size, false)?,
                ),
            });
        }
        Ok(Lfm3Model {
            cfg: cfg.clone(),
            logits_act_quant: l.act_quant(EMBED)?,
            embed,
            layers,
            norm: norm(l, "model.embedding_norm", d, eps)?,
        })
    }

    pub fn new_state(&self) -> State {
        State {
            kv: KvCache::with_windows(self.cfg.kv_windows()),
            conv: vec![None; self.layers.len()],
            staged_conv: Vec::new(),
        }
    }

    /// Graph for `ids` at positions `state.offset()..`; stages new state. With `trace`,
    /// intermediates are collected as ("embed" | "layers.{i}.op" | "layers.{i}" | "norm", t).
    fn build(
        &self,
        ids: &[u32],
        state: &mut State,
        rows: Rows,
        mut trace: Option<&mut Vec<(String, Tensor)>>,
    ) -> Result<Option<Tensor>> {
        if ids.is_empty() {
            return Err(Error::new("invalid_argument", "lfm3: forward of zero tokens"));
        }
        let mut x = Embedding::new(self.embed.clone()).lookup(ids)?.cast(DType::F32)?;
        let mut push = |name: String, t: &Tensor| {
            if let Some(tr) = trace.as_deref_mut() {
                tr.push((name, t.clone()));
            }
        };
        push("embed".into(), &x);
        for (i, layer) in self.layers.iter().enumerate() {
            let h = layer.op_norm.forward(&x)?;
            let mut y = match &layer.op {
                Operator::Conv(c) => {
                    let mut st = state.conv[i].clone();
                    let y = c.forward(&h, &mut st)?;
                    if let Some(s) = st {
                        state.staged_conv.push((i, s));
                    }
                    y
                }
                Operator::Attention(a) => a.forward(&h, &mut state.kv, i)?,
            };
            push(format!("layers.{i}.op"), &y);
            if let Some(n) = &layer.post_op_norm {
                y = n.forward(&y)?;
            }
            x = x.add(&y)?;
            let mut f = layer.mlp.forward(&layer.ffn_norm.forward(&x)?)?;
            if let Some(n) = &layer.post_ffn_norm {
                f = n.forward(&f)?;
            }
            x = x.add(&f)?;
            push(format!("layers.{i}"), &x);
        }
        let n = ids.len();
        let x = match rows {
            Rows::None => return Ok(None),
            Rows::Last if n > 1 => x.slice(1, n - 1, n)?,
            _ => x,
        };
        let mut h = self.norm.forward(&x)?;
        if rows == Rows::All {
            push("norm".into(), &h);
        }
        if self.logits_act_quant {
            h = fake_quant_act(&h)?;
        }
        Ok(Some(h.matmul_t(&self.embed)?))
    }

    /// Evaluate `outputs` together with the staged state and commit it. Returns the realized outputs.
    fn eval_commit(&self, device: &Device, outputs: &[&Tensor], state: &mut State) -> Result<Vec<Tensor>> {
        let mut outs: Vec<&Tensor> = outputs.to_vec();
        outs.extend(state.staged());
        let mut realized = match device.eval(&outs) {
            Ok(r) if r.len() == outs.len() => r,
            Ok(_) => {
                state.discard();
                return Err(Error::new("backend", "lfm3 eval returned the wrong number of outputs"));
            }
            Err(e) => {
                state.discard();
                return Err(e);
            }
        };
        let st = realized.split_off(outputs.len());
        state.commit(st)?;
        Ok(realized)
    }

    /// Lazy last-position logits of one uncached forward over `ids` from a fresh state
    /// (nothing evaluated; used to capture linear inputs for calibrated quantization).
    pub(crate) fn forward_graph(&self, ids: &[u32]) -> Result<Tensor> {
        let mut state = self.new_state();
        self.build(ids, &mut state, Rows::Last, None)?
            .ok_or_else(|| Error::new("internal", "lfm3: forward graph has no logits"))
    }

    /// Feed `ids` (a prompt — chunked — or one token) and return the f32 logits of the
    /// last position [n_vocab].
    pub fn step(&self, device: &Device, ids: &[u32], state: &mut State) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Err(Error::new("invalid_argument", "lfm3: step with no tokens"));
        }
        let chunks: Vec<&[u32]> = ids.chunks(PREFILL_CHUNK).collect();
        let last = chunks.len() - 1;
        for (ci, chunk) in chunks.into_iter().enumerate() {
            let rows = if ci == last { Rows::Last } else { Rows::None };
            let out = match self.build(chunk, state, rows, None) {
                Ok(o) => o,
                Err(e) => {
                    state.discard();
                    return Err(e);
                }
            };
            let outputs: Vec<&Tensor> = out.iter().collect();
            let realized = self.eval_commit(device, &outputs, state)?;
            if ci == last {
                let logits = realized
                    .first()
                    .ok_or_else(|| Error::new("backend", "lfm3: no logits realized"))?;
                return device.to_vec_f32(logits);
            }
        }
        Err(Error::new("internal", "lfm3: prefill produced no logits"))
    }

    /// One uncached forward over `ids` from a fresh state with every intermediate realized:
    /// ("embed", [n*D]), ("layers.{i}.op"), ("layers.{i}"), ("norm") and ("logits", rows).
    pub fn forward_traced(&self, device: &Device, ids: &[u32], rows: Rows) -> Result<Vec<(String, Vec<f32>)>> {
        let mut state = self.new_state();
        let mut trace = Vec::new();
        let logits = self.build(ids, &mut state, rows, Some(&mut trace))?;
        if let Some(l) = logits {
            trace.push(("logits".to_string(), l));
        }
        let refs: Vec<&Tensor> = trace.iter().map(|(_, t)| t).collect();
        let realized = device.eval(&refs)?;
        trace
            .iter()
            .zip(&realized)
            .map(|((n, _), t)| Ok((n.clone(), device.to_vec_f32(t)?)))
            .collect()
    }
}
