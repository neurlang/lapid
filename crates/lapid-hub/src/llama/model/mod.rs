//! Plain pre-norm transformer decoder (HF `LlamaForCausalLM`, `MistralForCausalLM`,
//! `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`) as lazy lapid graphs.
//!
//! Layer: `x += o(attn(rope(q_norm?(q(h))), rope(k_norm?(k(h))), v(h)))` with
//! `h = input_layernorm(x)`, then `x += ffn(post_attention_layernorm(x))`; `ffn` is SwiGLU or
//! the sparse MoE block. GQA, causal (optionally sliding-window) attention, scale `1/sqrt(Dh)`,
//! half-split RoPE over the whole head with the config's inverse frequencies (default, linear,
//! llama3 or per-frequency factors). Final RMSNorm, logits through `lm_head` or the tied
//! embedding.
//!
//! Streaming state ([`State`]) is the KV cache; one step is build → eval (logits + staged
//! K/V) → commit. The RoPE cos / sin tables are built once per step and shared by all layers.

pub(crate) mod moe;
pub(crate) mod weights;

use lapid::{DType, Device, Error, Result, Tensor};
use lapid_nn::attention::{merge_heads, scaled_dot_product_attention};
use lapid_nn::weights::fake_quant_act;
use lapid_nn::{Embedding, GatedMlp, KvCache, Linear, Loader, Mask, RmsNorm};

use super::config::LlamaConfig;
use moe::SparseMoe;
use weights::{f32_param, linear, norm, tensor};

pub(crate) const EMBED: &str = "model.embed_tokens.weight";
pub(crate) const LM_HEAD: &str = "lm_head.weight";
/// Prompt tokens per evaluation during prefill (bounds attention scratch memory).
pub(crate) const PREFILL_CHUNK: usize = 512;

/// Half-split rotation of x [B, H, S, Dh] with cos / sin [S, Dh / 2].
fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dh = x.shape()[3];
    let half = dh / 2;
    let x1 = x.slice(3, 0, half)?;
    let x2 = x.slice(3, half, dh)?;
    let o1 = x1.mul(cos)?.sub(&x2.mul(sin)?)?;
    let o2 = x2.mul(cos)?.add(&x1.mul(sin)?)?;
    Tensor::cat(&[&o1, &o2], 3)
}

struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    mask: Mask,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn heads(&self, lin: &Linear, norm: Option<&RmsNorm>, h: &Tensor, heads: usize) -> Result<Tensor> {
        let n = h.shape()[1];
        let t = lin.forward(h)?.reshape(&[1, n, heads, self.head_dim])?;
        let t = match norm {
            Some(nm) => nm.forward(&t)?,
            None => t,
        };
        t.permute(&[0, 2, 1, 3])
    }

    fn forward(&self, h: &Tensor, cache: &mut KvCache, layer: usize, cs: &(Tensor, Tensor)) -> Result<Tensor> {
        let q = rope(&self.heads(&self.q, self.q_norm.as_ref(), h, self.n_head)?, &cs.0, &cs.1)?;
        let k = rope(&self.heads(&self.k, self.k_norm.as_ref(), h, self.n_kv)?, &cs.0, &cs.1)?;
        let v = self.heads(&self.v, None, h, self.n_kv)?;
        let (k, v) = cache.append(layer, &k, &v)?;
        let a = scaled_dot_product_attention(&q, &k, &v, 1.0 / (self.head_dim as f32).sqrt(), &self.mask)?;
        self.o.forward(&merge_heads(&a)?)
    }
}

enum Ffn {
    Dense(GatedMlp),
    Moe(Box<SparseMoe>),
}

struct Layer {
    attn: SelfAttention,
    norm1: RmsNorm,
    norm2: RmsNorm,
    ffn: Ffn,
}

/// Mutable per-sequence state: the KV cache.
pub struct State {
    kv: KvCache,
}

impl State {
    /// Tokens consumed so far (absolute position of the next token).
    pub fn offset(&self) -> usize {
        self.kv.offset()
    }

    pub fn clear(&mut self) {
        self.kv.clear();
    }
}

/// Which logit rows to compute.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Rows {
    Last,
    All,
    None,
}

pub(crate) struct LlamaModel {
    pub cfg: LlamaConfig,
    embed: Tensor,
    head: Option<Linear>,
    logits_act_quant: bool,
    layers: Vec<Layer>,
    norm: RmsNorm,
    inv_freq: Vec<f32>,
}

fn load_layer(l: &Loader<'_>, cfg: &LlamaConfig, i: usize) -> Result<Layer> {
    let (d, eps, dh) = (cfg.hidden_size, cfg.norm_eps, cfg.head_dim);
    let (nh, nkv) = (cfg.n_head, cfg.n_kv_head);
    let p = format!("model.layers.{i}");
    let a = format!("{p}.self_attn");
    let qk_norm = |n: &str| -> Result<Option<RmsNorm>> {
        if cfg.qk_norm { Ok(Some(norm(l, &format!("{a}.{n}"), dh, eps)?)) } else { Ok(None) }
    };
    let attn = SelfAttention {
        q: linear(l, &format!("{a}.q_proj"), nh * dh, d, cfg.qkv_bias)?,
        k: linear(l, &format!("{a}.k_proj"), nkv * dh, d, cfg.qkv_bias)?,
        v: linear(l, &format!("{a}.v_proj"), nkv * dh, d, cfg.qkv_bias)?,
        o: linear(l, &format!("{a}.o_proj"), d, nh * dh, cfg.o_bias)?,
        q_norm: qk_norm("q_norm")?,
        k_norm: qk_norm("k_norm")?,
        mask: match cfg.windows[i] {
            Some(w) => Mask::Sliding(w),
            None => Mask::Causal,
        },
        n_head: nh,
        n_kv: nkv,
        head_dim: dh,
    };
    let ffn = if cfg.layer_is_moe(i) {
        let m = cfg.moe.as_ref().ok_or_else(|| Error::new("internal", "llama: moe layer without moe config"))?;
        let (e, inter) = (m.n_experts, m.expert_intermediate);
        let gate = format!("{p}.mlp.experts.gate_proj");
        Ffn::Moe(Box::new(SparseMoe {
            router: f32_param(l, &format!("{p}.mlp.gate.weight"), &[e, d])?,
            act_quant: l.act_quant(&gate)?,
            gate: tensor(l, &gate, &[e, inter, d])?,
            up: tensor(l, &format!("{p}.mlp.experts.up_proj"), &[e, inter, d])?,
            down: tensor(l, &format!("{p}.mlp.experts.down_proj"), &[e, d, inter])?,
            top_k: m.top_k,
            norm_topk_prob: m.norm_topk_prob,
        }))
    } else {
        let (f, inter, b) = (format!("{p}.mlp"), cfg.intermediate_size, cfg.mlp_bias);
        Ffn::Dense(GatedMlp::swiglu(
            linear(l, &format!("{f}.gate_proj"), inter, d, b)?,
            linear(l, &format!("{f}.up_proj"), inter, d, b)?,
            linear(l, &format!("{f}.down_proj"), d, inter, b)?,
        ))
    };
    Ok(Layer {
        attn,
        norm1: norm(l, &format!("{p}.input_layernorm"), d, eps)?,
        norm2: norm(l, &format!("{p}.post_attention_layernorm"), d, eps)?,
        ffn,
    })
}

impl LlamaModel {
    pub fn load(l: &Loader<'_>, cfg: &LlamaConfig) -> Result<LlamaModel> {
        let d = cfg.hidden_size;
        let embed = tensor(l, EMBED, &[cfg.vocab_size, d])?;
        let layers = (0..cfg.n_layers).map(|i| load_layer(l, cfg, i)).collect::<Result<Vec<_>>>()?;
        let untied = !cfg.tie_embeddings && l.contains(LM_HEAD);
        let head = if untied { Some(linear(l, "lm_head", cfg.vocab_size, d, false)?) } else { None };
        Ok(LlamaModel {
            cfg: cfg.clone(),
            logits_act_quant: !untied && l.act_quant(EMBED)?,
            embed,
            head,
            layers,
            norm: norm(l, "model.norm", d, cfg.norm_eps)?,
            inv_freq: cfg.inv_freq(),
        })
    }

    pub fn new_state(&self) -> State {
        State { kv: KvCache::with_windows(self.cfg.windows.clone()) }
    }

    /// (cos, sin) [n, Dh / 2] for positions `offset..offset + n` (f32 angles like HF).
    fn cos_sin(&self, offset: usize, n: usize) -> Result<(Tensor, Tensor)> {
        let half = self.inv_freq.len();
        let (mut cos, mut sin) = (Vec::with_capacity(n * half), Vec::with_capacity(n * half));
        for r in 0..n {
            let p = (offset + r) as f32;
            for &f in &self.inv_freq {
                let a = (p * f) as f64;
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
        Ok((Tensor::from_f32(&cos, &[n, half])?, Tensor::from_f32(&sin, &[n, half])?))
    }

    /// Graph for `ids` at positions `state.offset()..`; stages new K/V. With `trace`,
    /// intermediates are collected as ("embed" | "layers.{i}.op" | "layers.{i}" | "norm", t).
    fn build(&self, ids: &[u32], state: &mut State, rows: Rows, mut trace: Option<&mut Vec<(String, Tensor)>>) -> Result<Option<Tensor>> {
        if ids.is_empty() {
            return Err(Error::new("invalid_argument", "llama: forward of zero tokens"));
        }
        let n = ids.len();
        let cs = self.cos_sin(state.offset(), n)?;
        let mut x = Embedding::new(self.embed.clone()).lookup(ids)?.cast(DType::F32)?;
        let mut push = |name: String, t: &Tensor| {
            if let Some(tr) = trace.as_deref_mut() {
                tr.push((name, t.clone()));
            }
        };
        push("embed".into(), &x);
        for (i, layer) in self.layers.iter().enumerate() {
            let y = layer.attn.forward(&layer.norm1.forward(&x)?, &mut state.kv, i, &cs)?;
            push(format!("layers.{i}.op"), &y);
            x = x.add(&y)?;
            let h = layer.norm2.forward(&x)?;
            let f = match &layer.ffn {
                Ffn::Dense(m) => m.forward(&h)?,
                Ffn::Moe(m) => m.forward(&h)?,
            };
            x = x.add(&f)?;
            push(format!("layers.{i}"), &x);
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
            return Ok(Some(head.forward(&h)?));
        }
        if self.logits_act_quant {
            h = fake_quant_act(&h)?;
        }
        Ok(Some(h.matmul_t(&self.embed)?))
    }

    fn eval_commit(&self, device: &Device, outputs: &[&Tensor], state: &mut State) -> Result<Vec<Tensor>> {
        let mut outs: Vec<&Tensor> = outputs.to_vec();
        outs.extend(state.kv.staged());
        let mut realized = match device.eval(&outs) {
            Ok(r) if r.len() == outs.len() => r,
            Ok(_) => {
                state.kv.discard();
                return Err(Error::new("backend", "llama eval returned the wrong number of outputs"));
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

    /// Feed `ids` (a prompt, chunked, or one token); f32 logits of the last position.
    pub fn step(&self, device: &Device, ids: &[u32], state: &mut State) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Err(Error::new("invalid_argument", "llama: step with no tokens"));
        }
        let chunks: Vec<&[u32]> = ids.chunks(PREFILL_CHUNK).collect();
        let last = chunks.len() - 1;
        for (ci, chunk) in chunks.into_iter().enumerate() {
            let rows = if ci == last { Rows::Last } else { Rows::None };
            let out = match self.build(chunk, state, rows, None) {
                Ok(o) => o,
                Err(e) => {
                    state.kv.discard();
                    return Err(e);
                }
            };
            let outputs: Vec<&Tensor> = out.iter().collect();
            let realized = self.eval_commit(device, &outputs, state)?;
            if ci == last {
                let logits = realized.first().ok_or_else(|| Error::new("backend", "llama: no logits realized"))?;
                return device.to_vec_f32(logits);
            }
        }
        Err(Error::new("internal", "llama: prefill produced no logits"))
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
