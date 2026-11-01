//! DeepSeek V4 network as lazy lapid graphs (HF `DeepseekV4ForCausalLM`, llama.cpp `deepseek4`).
//!
//! ```text
//! x = embed(ids) repeated to hc streams                 [n, hc, D]
//! per layer:  x = hc_attn.post(attention(attn_norm(hc_attn.pre(x))), x)
//!             x = hc_ffn.post(moe(ffn_norm(hc_ffn.pre(x))), x)
//! logits = head(norm(hc_head(x)))
//! ```
//! Streaming state ([`State`]): per layer the raw sliding window and the compressor /
//! indexer states (see `compress.rs`). One step is build → eval (logits + staged state) →
//! commit; prompts are fed in chunks of [`PREFILL_CHUNK`].
//!
//! Tensor names follow the upstream checkpoints (`embed.weight`, `layers.{i}.attn.wq_a.weight`,
//! `layers.{i}.ffn.experts.w1` `[E, I, D]` stacks, `layers.{i}.attn.wo_a.weight` as
//! `[G, o_lora, H / G * head_dim]`, ...; see `convert`).

pub(crate) mod attention;
pub(crate) mod compress;
pub(crate) mod hc;
pub(crate) mod moe;
pub(crate) mod rope;
#[cfg(test)]
mod tests;

use lapid::{DType, Device, Error, Result, Tensor};
use lapid_nn::{Embedding, Linear, Loader, RmsNorm};

use super::config::Deepseek4Config;
use attention::{Attention, Indexer, LayerState};
use compress::Compressor;
use hc::{HyperConn, HyperHead};
use moe::Moe;
use rope::TailRope;

pub(crate) const EMBED: &str = "embed.weight";
pub(crate) const HEAD: &str = "head.weight";
/// Prompt tokens per evaluation during prefill.
pub(crate) const PREFILL_CHUNK: usize = 512;

struct Layer {
    attn_hc: HyperConn,
    ffn_hc: HyperConn,
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    attn: Attention,
    moe: Moe,
}

/// Mutable per-sequence state.
pub struct State {
    layers: Vec<LayerState>,
    pos: usize,
}

impl State {
    /// Tokens consumed so far (absolute position of the next token).
    pub fn offset(&self) -> usize {
        self.pos
    }

    pub fn clear(&mut self) {
        self.layers.iter_mut().for_each(|l| *l = LayerState::default());
        self.pos = 0;
    }
}

/// Which logit rows to compute.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Rows {
    Last,
    All,
    None,
}

pub(crate) struct Deepseek4Model {
    pub cfg: Deepseek4Config,
    embed: Embedding,
    head: Linear,
    norm: RmsNorm,
    hc_head: HyperHead,
    layers: Vec<Layer>,
}

fn norm(l: &Loader<'_>, name: &str, d: usize, eps: f32) -> Result<RmsNorm> {
    let n = RmsNorm::new(l.tensor(name)?, eps);
    l.expect_shape(&n.weight, name, &[d])?;
    Ok(n)
}

fn linear(l: &Loader<'_>, name: &str, out: usize, inp: usize) -> Result<Linear> {
    let lin = Linear::new(l.tensor(name)?, None).with_act_quant(l.act_quant(name)?);
    l.expect_shape(&lin.weight, name, &[out, inp])?;
    Ok(lin)
}

fn tensor(l: &Loader<'_>, name: &str, shape: &[usize]) -> Result<Tensor> {
    let t = l.tensor(name)?;
    l.expect_shape(&t, name, shape)?;
    Ok(t)
}

fn f32_param(l: &Loader<'_>, name: &str, shape: &[usize]) -> Result<Tensor> {
    tensor(l, name, shape)?.cast(DType::F32)
}

fn load_compressor(l: &Loader<'_>, p: &str, d: usize, ratio: usize, hd: usize, eps: f32) -> Result<Compressor> {
    let c = Deepseek4Config::compress_coff(ratio) * hd;
    Ok(Compressor {
        wkv: linear(l, &format!("{p}.wkv.weight"), c, d)?,
        wgate: linear(l, &format!("{p}.wgate.weight"), c, d)?,
        ape: f32_param(l, &format!("{p}.ape"), &[ratio, c])?,
        norm: norm(l, &format!("{p}.norm.weight"), hd, eps)?,
        ratio,
        head_dim: hd,
        overlap: ratio == 4,
    })
}

fn load_attention(l: &Loader<'_>, cfg: &Deepseek4Config, i: usize) -> Result<Attention> {
    let p = format!("layers.{i}.attn");
    let (d, h, hd, eps) = (cfg.hidden_size, cfg.n_head, cfg.head_dim, cfg.norm_eps);
    let (ql, g, ol) = (cfg.q_lora_rank, cfg.o_groups, cfg.o_lora_rank);
    let ratio = cfg.compress_ratios[i];
    let comp_rope = TailRope::new(cfg.rope_dim, cfg.compress_rope_theta, cfg.yarn.as_ref());
    let rope = if ratio == 0 { TailRope::new(cfg.rope_dim, cfg.rope_theta, None) } else { comp_rope.clone() };
    let comp = if ratio > 0 { Some(load_compressor(l, &format!("{p}.compressor"), d, ratio, hd, eps)?) } else { None };
    let indexer = if ratio == 4 {
        let (ih, id) = (cfg.index_n_heads, cfg.index_head_dim);
        Some(Indexer {
            wq_b: linear(l, &format!("{p}.indexer.wq_b.weight"), ih * id, ql)?,
            weights_proj: linear(l, &format!("{p}.indexer.weights_proj.weight"), ih, d)?,
            comp: load_compressor(l, &format!("{p}.indexer.compressor"), d, ratio, id, eps)?,
            heads: ih,
            dim: id,
            topk: cfg.index_topk,
        })
    } else {
        None
    };
    let wo_a_name = format!("{p}.wo_a.weight");
    Ok(Attention {
        wq_a: linear(l, &format!("{p}.wq_a.weight"), ql, d)?,
        q_norm: norm(l, &format!("{p}.q_norm.weight"), ql, eps)?,
        wq_b: linear(l, &format!("{p}.wq_b.weight"), h * hd, ql)?,
        wkv: linear(l, &format!("{p}.wkv.weight"), hd, d)?,
        kv_norm: norm(l, &format!("{p}.kv_norm.weight"), hd, eps)?,
        wo_a: tensor(l, &wo_a_name, &[g, ol, h / g * hd])?,
        wo_a_act_quant: l.act_quant(&wo_a_name)?,
        wo_b: linear(l, &format!("{p}.wo_b.weight"), d, g * ol)?,
        sinks: f32_param(l, &format!("{p}.attn_sink"), &[h])?.reshape(&[1, h, 1])?,
        comp,
        indexer,
        rope,
        comp_rope,
        ones: l.device.upload(&Tensor::full(1.0, &[hd], DType::F32))?,
        n_head: h,
        head_dim: hd,
        groups: g,
        window: cfg.sliding_window,
        ratio,
        eps,
    })
}

fn load_moe(l: &Loader<'_>, cfg: &Deepseek4Config, i: usize) -> Result<Moe> {
    let p = format!("layers.{i}.ffn");
    let (d, e, inter, k) = (cfg.hidden_size, cfg.n_experts, cfg.moe_intermediate, cfg.top_k);
    let hash = i < cfg.n_hash_layers;
    let tid2eid = if hash {
        let name = format!("{p}.gate.tid2eid");
        let t = l.file.get(&name)?;
        if t.dtype() != DType::I32 || t.shape() != [cfg.vocab_size, k] {
            return Err(Error::new("format", format!("{name}: expected i32 [{}, {k}], got {} {:?}", cfg.vocab_size, t.dtype().name(), t.shape())));
        }
        let b = l.file.bytes(&name)?;
        Some(b.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect())
    } else {
        None
    };
    let w1 = format!("{p}.experts.w1");
    Ok(Moe {
        router: f32_param(l, &format!("{p}.gate.weight"), &[e, d])?,
        bias: if hash { None } else { Some(f32_param(l, &format!("{p}.gate.bias"), &[e])?) },
        tid2eid,
        act_quant: l.act_quant(&w1)?,
        w1: tensor(l, &w1, &[e, inter, d])?,
        w3: tensor(l, &format!("{p}.experts.w3"), &[e, inter, d])?,
        w2: tensor(l, &format!("{p}.experts.w2"), &[e, d, inter])?,
        shared_w1: linear(l, &format!("{p}.shared_experts.w1.weight"), inter, d)?,
        shared_w3: linear(l, &format!("{p}.shared_experts.w3.weight"), inter, d)?,
        shared_w2: linear(l, &format!("{p}.shared_experts.w2.weight"), d, inter)?,
        top_k: k,
        n_experts: e,
        route_scale: cfg.route_scale,
        limit: cfg.swiglu_limit,
    })
}

type Staged = Vec<(usize, usize, Tensor)>;

impl Deepseek4Model {
    pub fn load(l: &Loader<'_>, cfg: &Deepseek4Config) -> Result<Deepseek4Model> {
        let (d, eps, hc) = (cfg.hidden_size, cfg.norm_eps, cfg.hc_mult);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("layers.{i}");
            let conn = |kind: &str| HyperConn::load(l, &format!("{p}.hc_{kind}"), hc, d, cfg.hc_sinkhorn_iters, cfg.hc_eps, eps);
            layers.push(Layer {
                attn_hc: conn("attn")?,
                ffn_hc: conn("ffn")?,
                attn_norm: norm(l, &format!("{p}.attn_norm.weight"), d, eps)?,
                ffn_norm: norm(l, &format!("{p}.ffn_norm.weight"), d, eps)?,
                attn: load_attention(l, cfg, i)?,
                moe: load_moe(l, cfg, i)?,
            });
        }
        Ok(Deepseek4Model {
            cfg: cfg.clone(),
            embed: Embedding::new(tensor(l, EMBED, &[cfg.vocab_size, d])?),
            head: linear(l, HEAD, cfg.vocab_size, d)?,
            norm: norm(l, "norm.weight", d, eps)?,
            hc_head: HyperHead::load(l, hc, d, cfg.hc_eps, eps)?,
            layers,
        })
    }

    pub fn new_state(&self) -> State {
        State { layers: vec![LayerState::default(); self.layers.len()], pos: 0 }
    }

    /// Graph for `ids` at positions `state.offset()..`. Returns the logits (per `rows`), the
    /// next per-layer states and the new (lazy) state tensors to realize. With `trace`,
    /// intermediates are collected as ("embed" | "layers.{i}.op" | "layers.{i}" | "norm", t).
    fn build(
        &self,
        ids: &[u32],
        state: &State,
        rows: Rows,
        mut trace: Option<&mut Vec<(String, Tensor)>>,
    ) -> Result<(Option<Tensor>, Vec<LayerState>, Staged)> {
        if ids.is_empty() {
            return Err(Error::new("invalid_argument", "deepseek4: forward of zero tokens"));
        }
        let (n, d, hc) = (ids.len(), self.cfg.hidden_size, self.cfg.hc_mult);
        let pos = state.pos;
        let e = self.embed.lookup(ids)?.cast(DType::F32)?.reshape(&[n, d])?;
        let mut push = |name: String, t: &Tensor| {
            if let Some(tr) = trace.as_deref_mut() {
                tr.push((name, t.clone()));
            }
        };
        push("embed".into(), &e);
        let mut x = e.unsqueeze(1)?.expand(&[n, hc, d])?.contiguous()?;
        let mut next = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            let pre = layer.attn_hc.pre(&x)?;
            let h = layer.attn_norm.forward(&pre.collapsed)?;
            let (a, st) = layer.attn.forward(&h, pos, &state.layers[i])?;
            next.push(st);
            push(format!("layers.{i}.op"), &a);
            x = layer.attn_hc.post(&a, &x, &pre.post, &pre.comb)?;
            let pre = layer.ffn_hc.pre(&x)?;
            let h = layer.ffn_norm.forward(&pre.collapsed)?;
            let m = layer.moe.forward(&h, ids)?;
            x = layer.ffn_hc.post(&m, &x, &pre.post, &pre.comb)?;
            push(format!("layers.{i}"), &x.reshape(&[n, hc * d])?);
        }
        let mut staged = Vec::new();
        for (i, (old, new)) in state.layers.iter().zip(&next).enumerate() {
            for (j, (o, nw)) in old.slots().into_iter().zip(new.slots()).enumerate() {
                if let Some(t) = nw
                    && o.as_ref().is_none_or(|o| o.id() != t.id())
                {
                    staged.push((i, j, t.clone()));
                }
            }
        }
        let x = match rows {
            Rows::None => return Ok((None, next, staged)),
            Rows::Last if n > 1 => x.slice(0, n - 1, n)?,
            _ => x,
        };
        let h = self.norm.forward(&self.hc_head.forward(&x)?)?;
        if rows == Rows::All {
            push("norm".into(), &h);
        }
        Ok((Some(self.head.forward(&h)?), next, staged))
    }

    fn eval_commit(&self, device: &Device, outputs: &[&Tensor], state: &mut State, mut next: Vec<LayerState>, staged: Staged, n: usize) -> Result<Vec<Tensor>> {
        let mut outs: Vec<&Tensor> = outputs.to_vec();
        outs.extend(staged.iter().map(|(_, _, t)| t));
        let mut realized = device.eval(&outs)?;
        if realized.len() != outs.len() {
            return Err(Error::new("backend", "deepseek4 eval returned the wrong number of outputs"));
        }
        let st = realized.split_off(outputs.len());
        for ((i, j, _), t) in staged.into_iter().zip(st) {
            if let Some(slot) = next[i].slots_mut().into_iter().nth(j) {
                *slot = Some(t);
            }
        }
        state.layers = next;
        state.pos += n;
        Ok(realized)
    }

    /// Feed `ids` (a prompt, chunked, or one token); f32 logits of the last position.
    /// On error the state is left unchanged.
    pub fn step(&self, device: &Device, ids: &[u32], state: &mut State) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Err(Error::new("invalid_argument", "deepseek4: step with no tokens"));
        }
        let chunks: Vec<&[u32]> = ids.chunks(PREFILL_CHUNK).collect();
        let last = chunks.len() - 1;
        for (ci, chunk) in chunks.into_iter().enumerate() {
            let rows = if ci == last { Rows::Last } else { Rows::None };
            let (out, next, staged) = self.build(chunk, state, rows, None)?;
            let outputs: Vec<&Tensor> = out.iter().collect();
            let realized = self.eval_commit(device, &outputs, state, next, staged, chunk.len())?;
            if ci == last {
                let logits = realized.first().ok_or_else(|| Error::new("backend", "deepseek4: no logits realized"))?;
                return device.to_vec_f32(logits);
            }
        }
        Err(Error::new("internal", "deepseek4: prefill produced no logits"))
    }

    /// One uncached forward over `ids` with every intermediate realized.
    pub fn forward_traced(&self, device: &Device, ids: &[u32], rows: Rows) -> Result<Vec<(String, Vec<f32>)>> {
        let state = self.new_state();
        let mut trace = Vec::new();
        let (logits, _, _) = self.build(ids, &state, rows, Some(&mut trace))?;
        if let Some(l) = logits {
            trace.push(("logits".to_string(), l));
        }
        let refs: Vec<&Tensor> = trace.iter().map(|(_, t)| t).collect();
        let realized = device.eval(&refs)?;
        trace.iter().zip(&realized).map(|((n, _), t)| Ok((n.clone(), device.to_vec_f32(t)?))).collect()
    }
}
