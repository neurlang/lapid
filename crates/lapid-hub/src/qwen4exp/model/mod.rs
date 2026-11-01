//! Qwen4-Exp network (`Qwen4ExpForCausalLM` text decoder) as lazy lapid graphs.
//!
//! The residual is `hc` streams `x [1, n, hc, D]` (start: hc copies of the embedding). Layer i:
//!
//! ```text
//! x += ple(x)                                  (the PLE layer only, see `ple`)
//! m = attn_hc.mix(x);  x = combine(x, mixer(m.input), m.weights)
//! m = mlp_hc.mix(x);   x = combine(x, moe(m.input),   m.weights)
//! ```
//! `mixer` is Gated DeltaNet (sigmoid output gate) or gated attention with the QSA indexer.
//! The final mixer (`hyper_connection_mixer`) collapses the streams and replaces the output
//! norm; logits through `lm_head`.
//!
//! Weight conventions of the `.lapid` file (see `convert`): every RMSNorm stores the effective
//! scale (HF `1 + w` folded in; the DeltaNet gated norm is plain `w`), DeltaNet `A = -exp(A_log)`,
//! v heads in tiled order (llama.cpp's GGUF values, so GGUF tensors are copied as-is).
//!
//! Streaming state ([`State`]): KV cache + raw indexer keys for attention layers, conv window +
//! delta-rule state for DeltaNet layers, the PLE conv history and the last n-gram tokens.

pub(crate) mod attn;
pub(crate) mod deltanet;
pub(crate) mod hc;
pub(crate) mod moe;
pub(crate) mod ple;

use lapid::{DType, Device, Error, Result, Tensor};
use lapid_nn::weights::fake_quant_act;
use lapid_nn::{Conv1d, Embedding, GatedMlp, KvCache, Linear, Loader, Precision, RmsNorm, Rope};

use super::config::Qwen4ExpConfig;
use crate::qwen35::config::LayerType;
use attn::{GatedAttention, Indexer};
use deltanet::DeltaNet;
use hc::{HyperConnection, combine};
use moe::{Experts, SparseMoe};
use ple::{NgramTable, Ple};

pub(crate) const EMBED: &str = "model.embed_tokens.weight";
pub(crate) const LM_HEAD: &str = "lm_head.weight";
pub(crate) const MIXER: &str = "model.hyper_connection_mixer";
/// Prompt tokens per evaluation during prefill.
pub(crate) const PREFILL_CHUNK: usize = 512;

enum Mixer {
    Linear(Box<DeltaNet>),
    Attention(Box<GatedAttention>),
}

struct Layer {
    mixer: Mixer,
    attn_hc: HyperConnection,
    mlp_hc: HyperConnection,
    moe: SparseMoe,
    ple: Option<Box<Ple>>,
}

/// Mutable per-sequence state.
pub struct State {
    kv: KvCache,
    idx: KvCache,
    conv: Vec<Option<Tensor>>,
    ssm: Vec<Option<Tensor>>,
    ple_conv: Option<Tensor>,
    /// Last `ngram - 1` consumed tokens (PLE context).
    hist: Vec<u32>,
    staged: Vec<(usize, Tensor, Tensor)>,
    staged_ple: Option<Tensor>,
    staged_hist: Option<Vec<u32>>,
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
        self.idx.discard();
        self.staged.clear();
        self.staged_ple = None;
        self.staged_hist = None;
        self.staged_n = 0;
    }

    pub fn clear(&mut self) {
        self.kv.clear();
        self.idx.clear();
        self.conv.iter_mut().for_each(|c| *c = None);
        self.ssm.iter_mut().for_each(|c| *c = None);
        self.ple_conv = None;
        self.hist.clear();
        self.discard();
        self.pos = 0;
    }

    fn staged(&self) -> Vec<&Tensor> {
        let mut v = self.kv.staged();
        v.extend(self.idx.staged());
        for (_, c, s) in &self.staged {
            v.push(c);
            v.push(s);
        }
        v.extend(self.staged_ple.as_ref());
        v
    }

    fn commit(&mut self, mut realized: Vec<Tensor>) -> Result<()> {
        let (n_kv, n_idx) = (self.kv.staged().len(), self.idx.staged().len());
        let n_ple = usize::from(self.staged_ple.is_some());
        if realized.len() != n_kv + n_idx + 2 * self.staged.len() + n_ple {
            self.discard();
            return Err(Error::new("backend", "qwen4exp step: wrong number of realized state tensors"));
        }
        let ple = if n_ple == 1 { realized.pop() } else { None };
        let rec = realized.split_off(n_kv + n_idx);
        let idx = realized.split_off(n_kv);
        let mut it = rec.into_iter();
        for (layer, _, _) in std::mem::take(&mut self.staged) {
            self.conv[layer] = it.next();
            self.ssm[layer] = it.next();
        }
        self.kv.commit(realized)?;
        self.idx.commit(idx)?;
        if ple.is_some() {
            self.ple_conv = ple;
        }
        self.staged_ple = None;
        if let Some(h) = self.staged_hist.take() {
            self.hist = h;
        }
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

pub(crate) struct Qwen4ExpModel {
    pub cfg: Qwen4ExpConfig,
    embed: Tensor,
    head: Option<Linear>,
    logits_act_quant: bool,
    layers: Vec<Layer>,
    final_hc: HyperConnection,
}

/// Weight `name` on the device. In the fake-quant precisions every quantized weight is
/// dequantized to f32 here, including K-quant formats and `[E, N, K]` expert stacks.
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

fn norm(l: &Loader<'_>, name: &str, d: usize, eps: f32) -> Result<RmsNorm> {
    Ok(RmsNorm::new(f32_param(l, &format!("{name}.weight"), &[d])?, eps))
}

fn hyper(l: &Loader<'_>, cfg: &Qwen4ExpConfig, p: &str, inject: bool) -> Result<HyperConnection> {
    let (hc, d, lr) = (cfg.hc, cfg.hidden_size(), cfg.hc_lowrank);
    Ok(HyperConnection {
        norm: f32_param(l, &format!("{p}.hc_norm.weight"), &[hc, d])?,
        down: linear(l, &format!("{p}.input_mix_weight_down"), lr, hc * d)?,
        up: linear(l, &format!("{p}.input_mix_weight_up"), hc * d, lr)?,
        inject: if inject { Some(linear(l, &format!("{p}.block_inject_weight"), hc, hc * d)?) } else { None },
        hc,
        eps: cfg.base.norm_eps,
    })
}

fn load_attention(l: &Loader<'_>, cfg: &Qwen4ExpConfig, p: &str) -> Result<GatedAttention> {
    let b = &cfg.base;
    let (d, eps, nh, nkv, dh) = (b.hidden_size, b.norm_eps, b.n_head, b.n_kv_head, b.head_dim);
    let a = format!("{p}.self_attn");
    let indexer = match &cfg.qsa {
        Some(q) => {
            let ix = format!("{a}.indexer");
            Some(Indexer {
                q: linear(l, &format!("{ix}.q_proj"), q.n_heads * q.head_dim, d)?,
                k: linear(l, &format!("{ix}.k_proj"), q.head_dim, d)?,
                q_norm: norm(l, &format!("{ix}.q_norm"), q.head_dim, eps)?,
                k_norm: norm(l, &format!("{ix}.k_norm"), q.head_dim, eps)?,
                rope: Rope::new(q.head_dim, b.rope_theta).with_rotary_dim(b.rotary_dim()),
                n_head: q.n_heads,
                head_dim: q.head_dim,
                budget: q.budget,
                ratio: q.ratio,
            })
        }
        None => None,
    };
    Ok(GatedAttention {
        q: linear(l, &format!("{a}.q_proj"), 2 * nh * dh, d)?,
        k: linear(l, &format!("{a}.k_proj"), nkv * dh, d)?,
        v: linear(l, &format!("{a}.v_proj"), nkv * dh, d)?,
        o: linear(l, &format!("{a}.o_proj"), d, nh * dh)?,
        q_norm: norm(l, &format!("{a}.q_norm"), dh, eps)?,
        k_norm: norm(l, &format!("{a}.k_norm"), dh, eps)?,
        rope: Rope::new(dh, b.rope_theta).with_rotary_dim(b.rotary_dim()),
        n_head: nh,
        n_kv: nkv,
        head_dim: dh,
        indexer,
    })
}

fn load_deltanet(l: &Loader<'_>, cfg: &Qwen4ExpConfig, p: &str) -> Result<DeltaNet> {
    let b = &cfg.base;
    let a = format!("{p}.linear_attn");
    let (d, hk, hv, dk, dv) = (b.hidden_size, b.linear_k_heads, b.linear_v_heads, b.linear_k_dim, b.linear_v_dim);
    let c = b.conv_dim();
    let w = tensor(l, &format!("{a}.conv1d.weight"), &[c, 1, b.conv_kernel])?;
    Ok(DeltaNet {
        in_qkv: linear(l, &format!("{a}.in_proj_qkv"), c, d)?,
        in_z: linear(l, &format!("{a}.in_proj_z"), hv * dv, d)?,
        in_b: linear(l, &format!("{a}.in_proj_b"), hv, d)?,
        in_a: linear(l, &format!("{a}.in_proj_a"), hv, d)?,
        conv: Conv1d::depthwise(w, None),
        a: f32_param(l, &format!("{a}.A"), &[hv])?,
        dt_bias: f32_param(l, &format!("{a}.dt_bias"), &[hv])?,
        norm: norm(l, &format!("{a}.norm"), dv, b.norm_eps)?,
        out: linear(l, &format!("{a}.out_proj"), d, hv * dv)?,
        hk,
        hv,
        dk,
        dv,
        gate: cfg.output_gate,
    })
}

fn load_moe(l: &Loader<'_>, cfg: &Qwen4ExpConfig, p: &str) -> Result<SparseMoe> {
    let d = cfg.hidden_size();
    let m = cfg.moe()?;
    let (e, inter) = (m.n_experts, m.expert_intermediate);
    let gate = format!("{p}.mlp.experts.gate_proj");
    let sh = format!("{p}.mlp.shared_expert");
    Ok(SparseMoe {
        router: f32_param(l, &format!("{p}.mlp.gate.weight"), &[e, d])?,
        experts: Experts {
            act_quant: l.act_quant(&gate)?,
            gate: tensor(l, &gate, &[e, inter, d])?,
            up: tensor(l, &format!("{p}.mlp.experts.up_proj"), &[e, inter, d])?,
            down: tensor(l, &format!("{p}.mlp.experts.down_proj"), &[e, d, inter])?,
        },
        top_k: m.top_k,
        shared: GatedMlp::swiglu(
            linear(l, &format!("{sh}.gate_proj"), m.shared_intermediate, d)?,
            linear(l, &format!("{sh}.up_proj"), m.shared_intermediate, d)?,
            linear(l, &format!("{sh}.down_proj"), d, m.shared_intermediate)?,
        ),
        shared_gate: f32_param(l, &format!("{p}.mlp.shared_expert_gate.weight"), &[1, d])?,
    })
}

fn load_ple(l: &Loader<'_>, cfg: &Qwen4ExpConfig, p: &str) -> Result<Ple> {
    let pc = cfg.ple.clone().ok_or_else(|| Error::new("internal", "qwen4exp: PLE layer without PLE config"))?;
    let (hc, d) = (cfg.hc, cfg.hidden_size());
    let e = pc.embed_dim();
    let q = format!("{p}.ple");
    // host only: rows are gathered on the CPU (see `ple`)
    let table = l.file.get(&format!("{q}.ngram_embedding.weight"))?;
    if table.shape() != [pc.rows, pc.row_dim] {
        return Err(Error::new("shape_mismatch", format!("{q}.ngram_embedding.weight: {:?}, expected [{}, {}]", table.shape(), pc.rows, pc.row_dim)));
    }
    let conv = tensor(l, &format!("{q}.conv1d.weight"), &[hc * d, 1, pc.conv_kernel])?.cast(DType::F32)?;
    Ok(Ple {
        table: NgramTable { table, rows: pc.rows, cols: pc.row_dim },
        key: linear(l, &format!("{q}.key_proj"), hc * d, e)?,
        value: linear(l, &format!("{q}.value_proj"), d, e)?,
        norm_key: f32_param(l, &format!("{q}.norm_key.weight"), &[hc, d])?,
        norm_query: f32_param(l, &format!("{q}.norm_query.weight"), &[hc, d])?,
        norm_conv: f32_param(l, &format!("{q}.norm_conv.weight"), &[hc, d])?,
        conv: Conv1d::depthwise(conv, None).with_dilation(pc.ngram),
        hc,
        eps: cfg.base.norm_eps,
        cfg: pc,
    })
}

impl Qwen4ExpModel {
    pub fn load(l: &Loader<'_>, cfg: &Qwen4ExpConfig) -> Result<Qwen4ExpModel> {
        let d = cfg.hidden_size();
        let embed = tensor(l, EMBED, &[cfg.vocab_size(), d])?;
        let mut layers = Vec::with_capacity(cfg.n_layers());
        for (i, &kind) in cfg.base.layer_types.iter().enumerate() {
            let p = format!("model.layers.{i}");
            let mixer = match kind {
                LayerType::FullAttention => Mixer::Attention(Box::new(load_attention(l, cfg, &p)?)),
                LayerType::LinearAttention => Mixer::Linear(Box::new(load_deltanet(l, cfg, &p)?)),
            };
            let ple = match &cfg.ple {
                Some(pc) if pc.layer == i => Some(Box::new(load_ple(l, cfg, &p)?)),
                _ => None,
            };
            layers.push(Layer {
                mixer,
                attn_hc: hyper(l, cfg, &format!("{p}.attn_hyper_connection"), true)?,
                mlp_hc: hyper(l, cfg, &format!("{p}.mlp_hyper_connection"), true)?,
                moe: load_moe(l, cfg, &p)?,
                ple,
            });
        }
        let untied = l.contains(LM_HEAD);
        let head = if untied { Some(linear(l, "lm_head", cfg.vocab_size(), d)?) } else { None };
        Ok(Qwen4ExpModel {
            cfg: cfg.clone(),
            logits_act_quant: !untied && l.act_quant(EMBED)?,
            embed,
            head,
            layers,
            final_hc: hyper(l, cfg, MIXER, false)?,
        })
    }

    pub fn new_state(&self) -> State {
        let n = self.layers.len();
        State {
            kv: KvCache::with_windows(vec![None; n]),
            idx: KvCache::with_windows(vec![None; n]),
            conv: vec![None; n],
            ssm: vec![None; n],
            ple_conv: None,
            hist: Vec::new(),
            staged: Vec::new(),
            staged_ple: None,
            staged_hist: None,
            pos: 0,
            staged_n: 0,
        }
    }

    /// Graph for `ids` at positions `state.offset()..`; stages new state. With `trace`,
    /// intermediates are collected as ("embed" | "layers.{i}.ple" | "layers.{i}.op" |
    /// "layers.{i}" | "norm", t); layer outputs are the hc-wide residual [n, hc * D].
    fn build(&self, ids: &[u32], state: &mut State, rows: Rows, mut trace: Option<&mut Vec<(String, Tensor)>>) -> Result<Option<Tensor>> {
        if ids.is_empty() {
            return Err(Error::new("invalid_argument", "qwen4exp: forward of zero tokens"));
        }
        let (n, d, hc) = (ids.len(), self.cfg.hidden_size(), self.cfg.hc);
        let offset = state.pos;
        let e = Embedding::new(self.embed.clone()).lookup(ids)?.cast(DType::F32)?;
        let mut push = |name: String, t: &Tensor| {
            if let Some(tr) = trace.as_deref_mut() {
                tr.push((name, t.clone()));
            }
        };
        push("embed".into(), &e);
        let mut x = e.reshape(&[1, n, 1, d])?.expand(&[1, n, hc, d])?;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ple) = &layer.ple {
                let emb = ple.embed(&state.hist, ids)?;
                let mut conv = state.ple_conv.clone();
                let inj = ple.forward(&x, &emb, &mut conv)?;
                push(format!("layers.{i}.ple"), &inj);
                x = x.add(&inj)?;
                state.staged_ple = conv;
            }
            let m = layer.attn_hc.mix(&x)?;
            let y = match &layer.mixer {
                Mixer::Attention(a) => a.forward(&m.input, &mut state.kv, &mut state.idx, i, offset)?,
                Mixer::Linear(dn) => {
                    let (y, st) = dn.forward(&m.input, state.conv[i].as_ref(), state.ssm[i].as_ref())?;
                    state.staged.push((i, st.conv, st.ssm));
                    y
                }
            };
            push(format!("layers.{i}.op"), &y);
            let w = m.weights.ok_or_else(|| Error::new("internal", "qwen4exp: layer mixer without inject"))?;
            x = combine(&x, &y, &w)?;
            let m = layer.mlp_hc.mix(&x)?;
            let f = layer.moe.forward(&m.input)?;
            let w = m.weights.ok_or_else(|| Error::new("internal", "qwen4exp: layer mixer without inject"))?;
            x = combine(&x, &f, &w)?;
            push(format!("layers.{i}"), &x);
        }
        if let Some(p) = &self.cfg.ple {
            let keep = p.ngram - 1;
            let mut h: Vec<u32> = state.hist.iter().chain(ids).copied().collect();
            h.drain(..h.len().saturating_sub(keep));
            state.staged_hist = Some(h);
        }
        state.staged_n = n;
        let x = match rows {
            Rows::None => return Ok(None),
            Rows::Last if n > 1 => x.slice(1, n - 1, n)?,
            _ => x,
        };
        let mut h = self.final_hc.mix(&x)?.input;
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
                return Err(Error::new("backend", "qwen4exp eval returned the wrong number of outputs"));
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

    /// Feed `ids` (a prompt, chunked, or one token); f32 logits of the last position.
    pub fn step(&self, device: &Device, ids: &[u32], state: &mut State) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Err(Error::new("invalid_argument", "qwen4exp: step with no tokens"));
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
                let logits = realized.first().ok_or_else(|| Error::new("backend", "qwen4exp: no logits realized"))?;
                return device.to_vec_f32(logits);
            }
        }
        Err(Error::new("internal", "qwen4exp: prefill produced no logits"))
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
