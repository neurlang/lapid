//! Gemma 4 text decoder, dense or MoE (HF `Gemma4TextModel`), as lazy lapid graphs, with
//! embedding injection for multimodal inputs (HF `Gemma4Model.forward`).
//!
//! Input: `embed(ids) * sqrt(D)`, with injected rows (image soft tokens, already projected
//! to the text width) replacing the rows of their placeholder tokens; per-layer embeddings
//! (PLE, E2B / E4B) look up the pad id at injected positions and project the merged
//! embeddings. Decoder layer (sandwich norms):
//! `x += post_attn_norm(attn(input_norm(x)))`;
//! `f = mlp(pre_ff_norm(x))`, MoE layers: `f = post_ff_norm_1(f) + moe(x)`;
//! `x += post_ff_norm(f)`; PLE block; `x *= layer_scalar`. Final norm, tied logits,
//! `tanh(logits / c) * c`.
//!
//! With `use_bidirectional_attention == "vision"` (26B-A4B, 31B) sliding layers let the
//! tokens of one injected block see each other in both directions (full layers stay causal);
//! prefill chunks never split a block.
//!
//! Streaming state ([`State`]): a `KvCache` (sliding layers keep their last `window` rows);
//! one step is build → eval (logits + staged K/V) → commit.

pub(crate) mod layers;
pub(crate) mod rope;
mod weights;

use lapid::{DType, Device, Error, Result, Tensor};
use lapid_nn::weights::fake_quant_act;
use lapid_nn::{Embedding, GatedMlp, KvCache, Loader, RmsNorm, ops};

use super::config::Gemma4MoeConfig;
use layers::{Attn, Moe, SharedKv};

pub(crate) const EMBED: &str = "model.embed_tokens.weight";
pub(crate) const PLE_EMBED: &str = "model.embed_tokens_per_layer.weight";
pub(crate) const PLE_PROJ: &str = "model.per_layer_model_projection";
pub(crate) const PLE_NORM: &str = "model.per_layer_projection_norm";
pub(crate) const FINAL_NORM: &str = "model.norm";
/// Prompt tokens per evaluation during prefill (bounds attention scratch memory).
pub(crate) const PREFILL_CHUNK: usize = 512;

/// Rows of `embeds` ([n, D], any float dtype) replace the embeddings of `ids[start..start + n]`.
#[derive(Clone)]
pub(crate) struct Injection {
    pub start: usize,
    pub embeds: Tensor,
}

impl Injection {
    pub fn len(&self) -> usize {
        self.embeds.shape().first().copied().unwrap_or(0)
    }
}

/// Per-layer-embedding block of one layer.
struct Ple {
    gate: lapid_nn::Linear,
    proj: lapid_nn::Linear,
    post_norm: RmsNorm,
}

struct Layer {
    attn: Attn,
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    pre_ff_norm: RmsNorm,
    post_ff_norm: RmsNorm,
    mlp: GatedMlp,
    /// MoE layers: (post_feedforward_layernorm_1, experts).
    moe: Option<(RmsNorm, Moe)>,
    ple: Option<Ple>,
    scalar: f32,
}

/// Mutable per-sequence state (KV cache).
pub struct State {
    kv: KvCache,
}

impl State {
    /// Tokens consumed so far (absolute position of the next token).
    pub fn offset(&self) -> usize {
        self.kv.offset()
    }

    /// Forget everything (start of a new sequence).
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

pub(crate) struct Gemma4MoeModel {
    cfg: Gemma4MoeConfig,
    embed: Tensor,
    logits_act_quant: bool,
    /// PLE table as column chunks [V, cols_i].
    ple_embed: Vec<Tensor>,
    ple_proj: Option<lapid_nn::Linear>,
    ple_norm: Option<RmsNorm>,
    layers: Vec<Layer>,
    norm: RmsNorm,
}

/// Check injections (in range, ordered, non-overlapping, [n, D]); returns per-row block ids
/// (-1 = text) when there are any.
fn block_ids(n: usize, d: usize, inject: &[Injection]) -> Result<Option<Vec<i32>>> {
    if inject.is_empty() {
        return Ok(None);
    }
    let mut blocks = vec![-1i32; n];
    let mut end = 0;
    for (b, inj) in inject.iter().enumerate() {
        let len = inj.len();
        if inj.embeds.rank() != 2 || inj.embeds.shape()[1] != d || len == 0 {
            return Err(Error::new("shape_mismatch", format!("gemma4moe: injection {b} must be [n, {d}], got {:?}", inj.embeds.shape())));
        }
        if inj.start < end || inj.start + len > n {
            return Err(Error::new("invalid_argument", format!("gemma4moe: injection {b} at {}..{} overlaps or exceeds {n} tokens", inj.start, inj.start + len)));
        }
        end = inj.start + len;
        blocks[inj.start..end].iter_mut().for_each(|x| *x = b as i32);
    }
    Ok(Some(blocks))
}

impl Gemma4MoeModel {
    pub(crate) fn load(l: &Loader<'_>, cfg: &Gemma4MoeConfig) -> Result<Gemma4MoeModel> {
        weights::load(l, cfg)
    }

    pub(crate) fn cfg(&self) -> &Gemma4MoeConfig {
        &self.cfg
    }

    pub(crate) fn new_state(&self) -> State {
        State { kv: KvCache::with_windows(self.cfg.text.kv_windows()) }
    }

    /// PLE inputs [1, n, L, P] from the (pad-substituted) ids and the merged embedding x.
    fn ple_inputs(&self, ids: &[u32], x: &Tensor) -> Result<Option<Tensor>> {
        let (Some(proj), Some(pnorm)) = (&self.ple_proj, &self.ple_norm) else {
            return Ok(None);
        };
        let t = &self.cfg.text;
        let (n, nl, p) = (ids.len(), t.n_layers, t.ple_dim);
        let parts: Vec<Tensor> = self
            .ple_embed
            .iter()
            .map(|tb| Embedding::new(tb.clone()).lookup(ids)?.cast(DType::F32))
            .collect::<Result<_>>()?;
        let tok = match parts.as_slice() {
            [one] => one.clone(),
            many => Tensor::cat(&many.iter().collect::<Vec<_>>(), 2)?,
        };
        let tok = tok.mul_scalar((p as f64).sqrt())?.reshape(&[1, n, nl, p])?;
        let pr = proj.forward(x)?.mul_scalar(1.0 / (t.hidden_size as f64).sqrt())?.reshape(&[1, n, nl, p])?;
        let pr = pnorm.forward(&pr)?;
        Ok(Some(pr.add(&tok)?.mul_scalar(std::f64::consts::FRAC_1_SQRT_2)?))
    }

    /// Input embeddings [1, n, D] (f32) with the injected rows merged, and the ids the PLE
    /// lookup uses (pad at injected positions).
    fn embeddings(&self, ids: &[u32], inject: &[Injection]) -> Result<(Tensor, Vec<u32>)> {
        let d = self.cfg.text.hidden_size;
        let mut eff = ids.to_vec();
        for inj in inject {
            eff[inj.start..inj.start + inj.len()].iter_mut().for_each(|x| *x = self.cfg.pad_id());
        }
        let x = Embedding::new(self.embed.clone()).lookup(&eff)?.cast(DType::F32)?.mul_scalar((d as f64).sqrt())?;
        if inject.is_empty() {
            return Ok((x, eff));
        }
        let n = ids.len();
        let mut parts = Vec::new();
        let mut at = 0;
        for inj in inject {
            if inj.start > at {
                parts.push(x.slice(1, at, inj.start)?);
            }
            parts.push(inj.embeds.cast(DType::F32)?.reshape(&[1, inj.len(), d])?);
            at = inj.start + inj.len();
        }
        if at < n {
            parts.push(x.slice(1, at, n)?);
        }
        let merged = match parts.as_slice() {
            [one] => one.clone(),
            many => Tensor::cat(&many.iter().collect::<Vec<_>>(), 1)?,
        };
        Ok((merged, eff))
    }

    /// Graph for `ids` at positions `state.offset()..`; stages new K/V. With `trace`,
    /// intermediates are collected as ("embed" | "layers.{i}.attn" | "layers.{i}" | "norm", t).
    fn build(
        &self,
        ids: &[u32],
        inject: &[Injection],
        state: &mut State,
        rows: Rows,
        mut trace: Option<&mut Vec<(String, Tensor)>>,
    ) -> Result<Option<Tensor>> {
        if ids.is_empty() {
            return Err(Error::new("invalid_argument", "gemma4moe: forward of zero tokens"));
        }
        let t = &self.cfg.text;
        let n = ids.len();
        let blocks = block_ids(n, t.hidden_size, inject)?;
        let blocks = blocks.filter(|_| self.cfg.bidirectional_vision);
        let (mut x, eff) = self.embeddings(ids, inject)?;
        let mut push = |name: String, tn: &Tensor| {
            if let Some(tr) = trace.as_deref_mut() {
                tr.push((name, tn.clone()));
            }
        };
        push("embed".into(), &x);
        let ple = self.ple_inputs(&eff, &x)?;
        let mut shared: SharedKv = vec![None; self.layers.len()];
        for (i, layer) in self.layers.iter().enumerate() {
            let a = layer.attn.forward(&layer.input_norm.forward(&x)?, &mut state.kv, i, &mut shared, blocks.as_deref())?;
            push(format!("layers.{i}.attn"), &a);
            x = x.add(&layer.post_attn_norm.forward(&a)?)?;
            let mut f = layer.mlp.forward(&layer.pre_ff_norm.forward(&x)?)?;
            if let Some((post1, moe)) = &layer.moe {
                f = post1.forward(&f)?.add(&moe.forward(&x)?)?;
            }
            x = x.add(&layer.post_ff_norm.forward(&f)?)?;
            if let (Some(pb), Some(pi)) = (&layer.ple, &ple) {
                let pin = pi.slice(2, i, i + 1)?.reshape(&[1, n, t.ple_dim])?;
                let g = ops::gelu_tanh(&pb.gate.forward(&x)?)?.mul(&pin)?;
                x = x.add(&pb.post_norm.forward(&pb.proj.forward(&g)?)?)?;
            }
            if layer.scalar != 1.0 {
                x = x.mul_scalar(f64::from(layer.scalar))?;
            }
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
        if self.logits_act_quant {
            h = fake_quant_act(&h)?;
        }
        let mut logits = h.matmul_t(&self.embed)?.cast(DType::F32)?;
        if let Some(c) = t.final_logit_softcapping.filter(|c| *c > 0.0) {
            logits = logits.mul_scalar(1.0 / f64::from(c))?.tanh()?.mul_scalar(f64::from(c))?;
        }
        Ok(Some(logits))
    }

    /// Evaluate `outputs` together with the staged K/V and commit it.
    fn eval_commit(&self, device: &Device, outputs: &[&Tensor], state: &mut State) -> Result<Vec<Tensor>> {
        let mut outs: Vec<&Tensor> = outputs.to_vec();
        outs.extend(state.kv.staged());
        let mut realized = match device.eval(&outs) {
            Ok(r) if r.len() == outs.len() => r,
            Ok(_) => {
                state.kv.discard();
                return Err(Error::new("backend", "gemma4moe eval returned the wrong number of outputs"));
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

    /// Feed `ids` (a prompt, chunked, or one token) and return the f32 logits of the last
    /// position [n_vocab].
    pub(crate) fn step(&self, device: &Device, ids: &[u32], state: &mut State) -> Result<Vec<f32>> {
        self.step_with(device, ids, &[], state)
    }

    /// [`step`](Self::step) with injected embedding rows (see [`Injection`]).
    pub(crate) fn step_with(&self, device: &Device, ids: &[u32], inject: &[Injection], state: &mut State) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Err(Error::new("invalid_argument", "gemma4moe: step with no tokens"));
        }
        block_ids(ids.len(), self.cfg.text.hidden_size, inject)?;
        let chunks = chunk_bounds(ids.len(), inject);
        let last = chunks.len() - 1;
        for (ci, &(a, b)) in chunks.iter().enumerate() {
            let rows = if ci == last { Rows::Last } else { Rows::None };
            let local: Vec<Injection> = inject
                .iter()
                .filter(|j| j.start >= a && j.start < b)
                .map(|j| Injection { start: j.start - a, embeds: j.embeds.clone() })
                .collect();
            let out = match self.build(&ids[a..b], &local, state, rows, None) {
                Ok(o) => o,
                Err(e) => {
                    state.kv.discard();
                    return Err(e);
                }
            };
            let outputs: Vec<&Tensor> = out.iter().collect();
            let realized = self.eval_commit(device, &outputs, state)?;
            if ci == last {
                let logits = realized.first().ok_or_else(|| Error::new("backend", "gemma4moe: no logits realized"))?;
                return device.to_vec_f32(logits);
            }
        }
        Err(Error::new("internal", "gemma4moe: prefill produced no logits"))
    }

    /// One uncached forward over `ids` (with injections) from a fresh state, every
    /// intermediate realized: ("embed"), ("layers.{i}.attn"), ("layers.{i}"), ("norm"),
    /// ("logits", rows).
    pub(crate) fn forward_traced_with(&self, device: &Device, ids: &[u32], inject: &[Injection], rows: Rows) -> Result<Vec<(String, Vec<f32>)>> {
        let mut state = self.new_state();
        let mut trace = Vec::new();
        let logits = self.build(ids, inject, &mut state, rows, Some(&mut trace))?;
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

/// Prefill chunks `[a, b)` of at most `PREFILL_CHUNK` tokens, extended so that no injected
/// block is split.
fn chunk_bounds(n: usize, inject: &[Injection]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut a = 0;
    while a < n {
        let mut b = (a + PREFILL_CHUNK).min(n);
        for j in inject {
            let (s, e) = (j.start, j.start + j.len());
            if s < b && e > b {
                b = if s > a { s } else { e };
            }
        }
        out.push((a, b));
        a = b;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inj(start: usize, n: usize) -> Injection {
        Injection { start, embeds: Tensor::zeros(&[n, 4], DType::F32) }
    }

    #[test]
    fn chunks_keep_blocks_whole() {
        assert_eq!(chunk_bounds(1000, &[]), vec![(0, 512), (512, 1000)]);
        assert_eq!(chunk_bounds(1000, &[inj(500, 20)]), vec![(0, 500), (500, 1000)]);
        assert_eq!(chunk_bounds(1300, &[inj(0, 600)]), vec![(0, 600), (600, 1112), (1112, 1300)]);
    }

    #[test]
    fn blocks() {
        assert_eq!(block_ids(5, 4, &[inj(1, 2)]).expect("ok"), Some(vec![-1, 0, 0, -1, -1]));
        assert!(block_ids(5, 4, &[inj(4, 2)]).is_err());
        assert!(block_ids(5, 4, &[inj(1, 2), inj(2, 1)]).is_err());
    }

    #[test]
    fn sliding_block_mask() {
        let m = layers::block_sliding_mask(&[-1, 0, 0], 4, Some(8)).expect("mask");
        let dev = Device::cpu();
        let r = dev.eval(&[&m]).expect("eval");
        let v = dev.to_vec_f32(&r[0]).expect("host");
        // query row 1 (block 0) sees key 3 (its block, in the future); row 0 does not.
        assert_eq!(v[4 + 3], 0.0);
        assert_eq!(v[3], f32::NEG_INFINITY);
    }
}
