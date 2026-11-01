//! Qwen3.5 network (`Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` text decoder) as lazy lapid
//! graphs built from `lapid_nn` blocks.
//!
//! Decoder layer (pre-norm): `x += mixer(input_layernorm(x))`, `x += ffn(post_attention_layernorm(x))`.
//! `mixer` is Gated DeltaNet (3 of 4 layers) or gated full attention: q (+ a per-head output
//! gate interleaved in q_proj), q/k RMSNorm, partial RoPE (first 64 of 256 dims, half-split;
//! text-only MRoPE = plain RoPE), GQA, causal; output `o_proj(attn * sigmoid(gate))`.
//! `ffn` is SwiGLU (dense) or the sparse MoE block. Final norm, logits through the tied
//! embedding or `lm_head`.
//!
//! Weight conventions of the `.lapid` file (see `convert`): every RMSNorm stores the effective
//! scale (HF `1 + w` folded in), DeltaNet `A = -exp(A_log)`, v heads in tiled order.
//!
//! Streaming state ([`State`]): KV cache for attention layers, conv window + delta-rule state
//! for DeltaNet layers. One step is build → eval (logits + staged state) → commit.

pub(crate) mod deltanet;
pub(crate) mod moe;

use lapid::{DType, Device, Error, Result, Tensor};
use lapid_nn::attention::{merge_heads, scaled_dot_product_attention};
use lapid_nn::weights::fake_quant_act;
use lapid_nn::{Conv1d, Embedding, GatedMlp, KvCache, Linear, Loader, Mask, Precision, RmsNorm, Rope, ops};

use super::config::{LayerType, Qwen35Config};
use deltanet::DeltaNet;
use moe::{Experts, SparseMoe};

pub(crate) const EMBED: &str = "model.embed_tokens.weight";
pub(crate) const LM_HEAD: &str = "lm_head.weight";
/// Prompt tokens per evaluation during prefill.
pub(crate) const PREFILL_CHUNK: usize = 512;

struct GatedAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rope: Rope,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
}

impl GatedAttention {
    fn forward(&self, h: &Tensor, cache: &mut KvCache, layer: usize, offset: usize) -> Result<Tensor> {
        let n = h.shape()[1];
        let (nh, nkv, dh) = (self.n_head, self.n_kv, self.head_dim);
        let qg = self.q.forward(h)?.reshape(&[1, n, nh, 2 * dh])?;
        let q = self.q_norm.forward(&qg.slice(3, 0, dh)?)?.permute(&[0, 2, 1, 3])?;
        let gate = qg.slice(3, dh, 2 * dh)?.contiguous()?.reshape(&[1, n, nh * dh])?;
        let k = self.k_norm.forward(&self.k.forward(h)?.reshape(&[1, n, nkv, dh])?)?.permute(&[0, 2, 1, 3])?;
        let v = self.v.forward(h)?.reshape(&[1, n, nkv, dh])?.permute(&[0, 2, 1, 3])?;
        let q = self.rope.apply(&q, offset)?;
        let k = self.rope.apply(&k, offset)?;
        let (k, v) = cache.append(layer, &k, &v)?;
        let a = scaled_dot_product_attention(&q, &k, &v, 1.0 / (dh as f32).sqrt(), &Mask::Causal)?;
        let a = merge_heads(&a)?.mul(&ops::sigmoid(&gate)?)?;
        self.o.forward(&a)
    }
}

enum Mixer {
    Linear(Box<DeltaNet>),
    Attention(GatedAttention),
}

enum Ffn {
    Dense(GatedMlp),
    Moe(Box<SparseMoe>),
}

impl Ffn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Ffn::Dense(m) => m.forward(x),
            Ffn::Moe(m) => m.forward(x),
        }
    }
}

struct Layer {
    mixer: Mixer,
    norm1: RmsNorm,
    norm2: RmsNorm,
    ffn: Ffn,
}

/// Mutable per-sequence state.
pub struct State {
    kv: KvCache,
    conv: Vec<Option<Tensor>>,
    ssm: Vec<Option<Tensor>>,
    staged: Vec<(usize, Tensor, Tensor)>,
    pos: usize,
    staged_n: usize,
}

impl State {
    /// Tokens consumed so far (absolute position of the next token).
    pub fn offset(&self) -> usize {
        self.pos
    }

    fn discard(&mut self) {
        self.kv.discard();
        self.staged.clear();
        self.staged_n = 0;
    }

    pub fn clear(&mut self) {
        self.kv.clear();
        self.conv.iter_mut().for_each(|c| *c = None);
        self.ssm.iter_mut().for_each(|c| *c = None);
        self.staged.clear();
        self.pos = 0;
        self.staged_n = 0;
    }

    fn staged(&self) -> Vec<&Tensor> {
        let mut v = self.kv.staged();
        for (_, c, s) in &self.staged {
            v.push(c);
            v.push(s);
        }
        v
    }

    fn commit(&mut self, mut realized: Vec<Tensor>) -> Result<()> {
        let n_kv = self.kv.staged().len();
        if realized.len() != n_kv + 2 * self.staged.len() {
            self.discard();
            return Err(Error::new("backend", "qwen35 step: wrong number of realized state tensors"));
        }
        let rec = realized.split_off(n_kv);
        let mut it = rec.into_iter();
        for (layer, _, _) in std::mem::take(&mut self.staged) {
            self.conv[layer] = it.next();
            self.ssm[layer] = it.next();
        }
        self.kv.commit(realized)?;
        self.pos += self.staged_n;
        self.staged_n = 0;
        Ok(())
    }
}

/// Which logit rows to compute.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Rows {
    Last,
    All,
    None,
}

pub(crate) struct Qwen35Model {
    pub cfg: Qwen35Config,
    embed: Tensor,
    head: Option<Linear>,
    logits_act_quant: bool,
    layers: Vec<Layer>,
    norm: RmsNorm,
}

fn norm(l: &Loader<'_>, name: &str, d: usize, eps: f32) -> Result<RmsNorm> {
    let n = l.rms_norm(name, eps)?;
    l.expect_shape(&n.weight, &format!("{name}.weight"), &[d])?;
    Ok(n)
}

/// Weight `name` on the device. In the fake-quant precisions every quantized weight is
/// dequantized to f32 here — including K-quant formats and `[E, N, K]` expert stacks, which
/// `Loader::tensor` (2-D int8 / int4 only) does not handle yet.
fn weight(l: &Loader<'_>, name: &str) -> Result<Tensor> {
    if l.precision == Precision::Stored {
        return l.tensor(name);
    }
    let t = l.file.get(name)?;
    let Some(fmt) = t.dtype().quant_format() else { return l.tensor(name) };
    let shape = t.shape().to_vec();
    let (mats, rows, cols) = match shape.as_slice() {
        [r, c] => (1, *r, *c),
        [e, r, c] => (*e, *r, *c),
        _ => return Err(Error::new("shape_mismatch", format!("{name}: quantized weight must be 2-D or 3-D, got {shape:?}"))),
    };
    if mats == 1 && !fmt.is_aos() {
        return l.tensor(name);
    }
    let bytes = l.file.bytes(name)?;
    let per = lapid::quant::layout(fmt, rows, cols).total;
    if bytes.len() != per * mats {
        return Err(Error::new("format", format!("{name}: {} bytes, layout needs {}", bytes.len(), per * mats)));
    }
    let mut data = Vec::with_capacity(mats * rows * cols);
    for m in 0..mats {
        data.extend(lapid::quant::dequantize(fmt, &bytes[m * per..(m + 1) * per], rows, cols));
    }
    l.device.upload(&Tensor::from_f32(&data, &shape)?)
}

fn linear(l: &Loader<'_>, name: &str, out: usize, inp: usize) -> Result<Linear> {
    let w = format!("{name}.weight");
    let lin = Linear::new(weight(l, &w)?, None).with_act_quant(l.act_quant(&w)?);
    l.expect_shape(&lin.weight, &w, &[out, inp])?;
    Ok(lin)
}

fn tensor(l: &Loader<'_>, name: &str, shape: &[usize]) -> Result<Tensor> {
    let t = weight(l, name)?;
    l.expect_shape(&t, name, shape)?;
    Ok(t)
}

fn f32_param(l: &Loader<'_>, name: &str, shape: &[usize]) -> Result<Tensor> {
    tensor(l, name, shape)?.cast(DType::F32)
}

fn swiglu(l: &Loader<'_>, p: &str, d: usize, inter: usize) -> Result<GatedMlp> {
    Ok(GatedMlp::swiglu(
        linear(l, &format!("{p}.gate_proj"), inter, d)?,
        linear(l, &format!("{p}.up_proj"), inter, d)?,
        linear(l, &format!("{p}.down_proj"), d, inter)?,
    ))
}

fn load_mixer(l: &Loader<'_>, cfg: &Qwen35Config, p: &str, kind: LayerType) -> Result<Mixer> {
    let (d, eps) = (cfg.hidden_size, cfg.norm_eps);
    Ok(match kind {
        LayerType::FullAttention => {
            let (nh, nkv, dh) = (cfg.n_head, cfg.n_kv_head, cfg.head_dim);
            let a = format!("{p}.self_attn");
            Mixer::Attention(GatedAttention {
                q: linear(l, &format!("{a}.q_proj"), 2 * nh * dh, d)?,
                k: linear(l, &format!("{a}.k_proj"), nkv * dh, d)?,
                v: linear(l, &format!("{a}.v_proj"), nkv * dh, d)?,
                o: linear(l, &format!("{a}.o_proj"), d, nh * dh)?,
                q_norm: norm(l, &format!("{a}.q_norm"), dh, eps)?,
                k_norm: norm(l, &format!("{a}.k_norm"), dh, eps)?,
                rope: Rope::new(dh, cfg.rope_theta).with_rotary_dim(cfg.rotary_dim()),
                n_head: nh,
                n_kv: nkv,
                head_dim: dh,
            })
        }
        LayerType::LinearAttention => {
            let a = format!("{p}.linear_attn");
            let (hk, hv, dk, dv) = (cfg.linear_k_heads, cfg.linear_v_heads, cfg.linear_k_dim, cfg.linear_v_dim);
            let c = cfg.conv_dim();
            let w = tensor(l, &format!("{a}.conv1d.weight"), &[c, 1, cfg.conv_kernel])?;
            Mixer::Linear(Box::new(DeltaNet {
                in_qkv: linear(l, &format!("{a}.in_proj_qkv"), c, d)?,
                in_z: linear(l, &format!("{a}.in_proj_z"), hv * dv, d)?,
                in_b: linear(l, &format!("{a}.in_proj_b"), hv, d)?,
                in_a: linear(l, &format!("{a}.in_proj_a"), hv, d)?,
                conv: Conv1d::depthwise(w, None),
                a: f32_param(l, &format!("{a}.A"), &[hv])?,
                dt_bias: f32_param(l, &format!("{a}.dt_bias"), &[hv])?,
                norm: norm(l, &format!("{a}.norm"), dv, eps)?,
                out: linear(l, &format!("{a}.out_proj"), d, hv * dv)?,
                hk,
                hv,
                dk,
                dv,
            }))
        }
    })
}

fn load_moe(l: &Loader<'_>, cfg: &Qwen35Config, p: &str) -> Result<Ffn> {
    let d = cfg.hidden_size;
    let m = cfg.moe.as_ref().ok_or_else(|| Error::new("internal", "qwen35: moe layer without moe config"))?;
    let (e, inter) = (m.n_experts, m.expert_intermediate);
    let gate = format!("{p}.mlp.experts.gate_proj");
    Ok(Ffn::Moe(Box::new(SparseMoe {
        router: f32_param(l, &format!("{p}.mlp.gate.weight"), &[e, d])?,
        experts: Experts {
            act_quant: l.act_quant(&gate)?,
            gate: tensor(l, &gate, &[e, inter, d])?,
            up: tensor(l, &format!("{p}.mlp.experts.up_proj"), &[e, inter, d])?,
            down: tensor(l, &format!("{p}.mlp.experts.down_proj"), &[e, d, inter])?,
        },
        top_k: m.top_k,
        shared: swiglu(l, &format!("{p}.mlp.shared_expert"), d, m.shared_intermediate)?,
        shared_gate: f32_param(l, &format!("{p}.mlp.shared_expert_gate.weight"), &[1, d])?,
    })))
}

impl Qwen35Model {
    pub fn load(l: &Loader<'_>, cfg: &Qwen35Config) -> Result<Qwen35Model> {
        let (d, eps) = (cfg.hidden_size, cfg.norm_eps);
        let embed = tensor(l, EMBED, &[cfg.vocab_size, d])?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for (i, &kind) in cfg.layer_types.iter().enumerate() {
            let p = format!("model.layers.{i}");
            let ffn = if cfg.layer_is_moe(i) { load_moe(l, cfg, &p)? } else { Ffn::Dense(swiglu(l, &format!("{p}.mlp"), d, cfg.intermediate_size)?) };
            layers.push(Layer {
                mixer: load_mixer(l, cfg, &p, kind)?,
                norm1: norm(l, &format!("{p}.input_layernorm"), d, eps)?,
                norm2: norm(l, &format!("{p}.post_attention_layernorm"), d, eps)?,
                ffn,
            });
        }
        let untied = !cfg.tie_embeddings && l.contains(LM_HEAD);
        let head = if untied { Some(linear(l, "lm_head", cfg.vocab_size, d)?) } else { None };
        Ok(Qwen35Model {
            cfg: cfg.clone(),
            logits_act_quant: !untied && l.act_quant(EMBED)?,
            embed,
            head,
            layers,
            norm: norm(l, "model.norm", d, eps)?,
        })
    }

    pub fn new_state(&self) -> State {
        let n = self.layers.len();
        State {
            kv: KvCache::with_windows(vec![None; n]),
            conv: vec![None; n],
            ssm: vec![None; n],
            staged: Vec::new(),
            pos: 0,
            staged_n: 0,
        }
    }

    /// Graph for `ids` at positions `state.offset()..`; stages new state. With `trace`,
    /// intermediates are collected as ("embed" | "layers.{i}.op" | "layers.{i}" | "norm", t).
    fn build(&self, ids: &[u32], state: &mut State, rows: Rows, mut trace: Option<&mut Vec<(String, Tensor)>>) -> Result<Option<Tensor>> {
        if ids.is_empty() {
            return Err(Error::new("invalid_argument", "qwen35: forward of zero tokens"));
        }
        let offset = state.pos;
        let mut x = Embedding::new(self.embed.clone()).lookup(ids)?.cast(DType::F32)?;
        let mut push = |name: String, t: &Tensor| {
            if let Some(tr) = trace.as_deref_mut() {
                tr.push((name, t.clone()));
            }
        };
        push("embed".into(), &x);
        for (i, layer) in self.layers.iter().enumerate() {
            let h = layer.norm1.forward(&x)?;
            let y = match &layer.mixer {
                Mixer::Attention(a) => a.forward(&h, &mut state.kv, i, offset)?,
                Mixer::Linear(dn) => {
                    let (y, st) = dn.forward(&h, state.conv[i].as_ref(), state.ssm[i].as_ref())?;
                    state.staged.push((i, st.conv, st.ssm));
                    y
                }
            };
            push(format!("layers.{i}.op"), &y);
            x = x.add(&y)?;
            x = x.add(&layer.ffn.forward(&layer.norm2.forward(&x)?)?)?;
            push(format!("layers.{i}"), &x);
        }
        state.staged_n = ids.len();
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
        if let Some(head) = &self.head {
            return Ok(Some(head.forward(&h)?));
        }
        if self.logits_act_quant {
            h = fake_quant_act(&h)?;
        }
        Ok(Some(h.matmul_t(&self.embed)?))
    }

    fn eval_commit(&self, device: &Device, outputs: &[&Tensor], state: &mut State) -> Result<Vec<Tensor>> {
        let mut outs: Vec<&Tensor> = outputs.to_vec();
        outs.extend(state.staged());
        let mut realized = match device.eval(&outs) {
            Ok(r) if r.len() == outs.len() => r,
            Ok(_) => {
                state.discard();
                return Err(Error::new("backend", "qwen35 eval returned the wrong number of outputs"));
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

    /// Feed `ids` (a prompt — chunked — or one token); f32 logits of the last position.
    pub fn step(&self, device: &Device, ids: &[u32], state: &mut State) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Err(Error::new("invalid_argument", "qwen35: step with no tokens"));
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
                let logits = realized.first().ok_or_else(|| Error::new("backend", "qwen35: no logits realized"))?;
                return device.to_vec_f32(logits);
            }
        }
        Err(Error::new("internal", "qwen35: prefill produced no logits"))
    }

    /// One uncached forward over `ids` with every intermediate realized.
    pub fn forward_traced(&self, device: &Device, ids: &[u32], rows: Rows) -> Result<Vec<(String, Vec<f32>)>> {
        let mut state = self.new_state();
        let mut trace = Vec::new();
        let logits = self.build(ids, &mut state, rows, Some(&mut trace))?;
        if let Some(l) = logits {
            trace.push(("logits".to_string(), l));
        }
        let refs: Vec<&Tensor> = trace.iter().map(|(_, t)| t).collect();
        let realized = device.eval(&refs)?;
        trace.iter().zip(&realized).map(|((n, _), t)| Ok((n.clone(), device.to_vec_f32(t)?))).collect()
    }
}
