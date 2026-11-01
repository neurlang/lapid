//! Gemma 4 text decoder (HF `Gemma4TextModel`) as lazy lapid graphs built from `lapid_nn`
//! blocks.
//!
//! Input: `embed(ids) * sqrt(D)`; per-layer embeddings (PLE)
//! `ple = (norm(proj(x) / sqrt(D)) + embed_per_layer(ids) * sqrt(P)) / sqrt(2)` [n, L, P].
//! Decoder layer (sandwich norms):
//! `x += post_attn_norm(attn(input_norm(x)))`, `x += post_ff_norm(geglu(pre_ff_norm(x)))`,
//! `x += post_ple_norm(proj(gelu_tanh(gate(x)) * ple[i]))`, `x *= layer_scalar`.
//! Attention: q/k RMSNorm (`x * w`), weight-free v RMSNorm, scale 1.0, GQA, zero-tail RoPE
//! (sliding layers: full rotary, theta 1e4, window 512; full layers: head dim 512,
//! "proportional" partial 0.25, theta 1e6). The last `num_kv_shared_layers` layers have no
//! K/V of their own: they attend to the K/V of the last earlier layer of their type.
//! Final norm, tied logits, `tanh(logits / c) * c` soft-capping.
//!
//! Streaming state ([`State`]): a `KvCache` (sliding layers keep their last `window` rows);
//! one step is build → eval (logits + staged K/V) → commit.

pub(crate) mod rope;

use lapid::{DType, Device, Error, HostData, Result, Tensor, UnaryOp};
use lapid_nn::attention::{merge_heads, repeat_kv, scaled_dot_product_attention};
use lapid_nn::weights::fake_quant_act;
use lapid_nn::{Embedding, GatedMlp, KvCache, Linear, Loader, Mask, RmsNorm, ops};

use super::config::{Gemma4Config, LayerType};
use rope::ZeroTailRope;

/// Token embedding; also the tied output projection.
pub(crate) const EMBED: &str = "model.embed_tokens.weight";
/// Per-layer token embedding [V, L * P] (a gather table).
pub(crate) const PLE_EMBED: &str = "model.embed_tokens_per_layer.weight";
pub(crate) const PLE_PROJ: &str = "model.per_layer_model_projection";
pub(crate) const PLE_NORM: &str = "model.per_layer_projection_norm";
pub(crate) const FINAL_NORM: &str = "model.norm";
/// Prompt tokens per evaluation during prefill (bounds attention scratch memory).
pub(crate) const PREFILL_CHUNK: usize = 512;
/// Largest head dim the fused attention kernels take on every backend (CUDA / Vulkan: 256).
const FUSED_ATTENTION_MAX_HEAD_DIM: usize = 256;
/// Float gather tables above this size are split into column chunks (Vulkan
/// `maxStorageBufferRange` is 4 GiB; the f16 PLE table of E2B is 4.7 GB).
const MAX_TABLE_BYTES: usize = 1 << 31;

/// Causal attention from primitives (matmul + softmax) for head dims the fused kernels do
/// not take (Gemma 4 full-attention layers: 512). q [1, H, n, D], k / v [1, Hkv, Sk, D].
fn attention_unfused(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let (h, n) = (q.shape()[1], q.shape()[2]);
    let hkv = k.shape()[1];
    let sk = k.shape()[2];
    let (k, v) = (repeat_kv(k, h / hkv.max(1))?, repeat_kv(v, h / hkv.max(1))?);
    let qf = q.cast(DType::F32)?;
    let mut s = qf.matmul_t(&k.cast(DType::F32)?)?;
    if scale != 1.0 {
        s = s.mul_scalar(f64::from(scale))?;
    }
    if n > 1 {
        let off = sk - n;
        let mut m = vec![0.0f32; n * sk];
        for i in 0..n {
            for j in (i + off + 1)..sk {
                m[i * sk + j] = f32::NEG_INFINITY;
            }
        }
        s = s.add(&Tensor::from_f32(&m, &[n, sk])?)?;
    }
    s.softmax()?.matmul(&v.cast(DType::F32)?)?.cast(q.dtype())
}

/// RMSNorm without weight (Gemma 4 `v_norm`), in f32.
fn rms_plain(x: &Tensor, eps: f32) -> Result<Tensor> {
    let xf = x.cast(DType::F32)?;
    let inv = xf.mul(&xf)?.mean(-1, true)?.add_scalar(f64::from(eps))?.unary(UnaryOp::Rsqrt)?;
    xf.mul(&inv)
}

struct Attn {
    q: Linear,
    /// `None` on kv-shared layers.
    k: Option<Linear>,
    /// `None` on kv-shared layers and on k_eq_v full layers (V = K before k_norm).
    v: Option<Linear>,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: Option<RmsNorm>,
    rope: ZeroTailRope,
    mask: Mask,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    /// Layer whose K/V this layer attends to (itself unless kv-shared).
    source: usize,
    eps: f32,
}

type SharedKv = Vec<Option<(Tensor, Tensor)>>;

impl Attn {
    /// h [1, n, D] -> [1, n, D].
    fn forward(&self, h: &Tensor, cache: &mut KvCache, layer: usize, shared: &mut SharedKv) -> Result<Tensor> {
        let n = h.shape()[1];
        let offset = cache.offset();
        let q = self.q.forward(h)?.reshape(&[1, n, self.n_head, self.head_dim])?;
        let q = self.q_norm.forward(&q)?.permute(&[0, 2, 1, 3])?;
        let q = self.rope.apply(&q, offset)?;
        let (k, v) = match (&self.k, &self.k_norm) {
            (Some(kp), Some(kn)) => {
                let kraw = kp.forward(h)?.reshape(&[1, n, self.n_kv, self.head_dim])?;
                let vraw = match &self.v {
                    Some(vp) => vp.forward(h)?.reshape(&[1, n, self.n_kv, self.head_dim])?,
                    None => kraw.clone(),
                };
                let k = self.rope.apply(&kn.forward(&kraw)?.permute(&[0, 2, 1, 3])?, offset)?;
                let v = rms_plain(&vraw, self.eps)?.permute(&[0, 2, 1, 3])?;
                let kv = cache.append(layer, &k, &v)?;
                shared[layer] = Some(kv.clone());
                kv
            }
            _ => shared
                .get(self.source)
                .cloned()
                .flatten()
                .ok_or_else(|| Error::new("internal", format!("gemma4: layer {layer} reads K/V of layer {} before it ran", self.source)))?,
        };
        let a = if self.head_dim > FUSED_ATTENTION_MAX_HEAD_DIM && matches!(self.mask, Mask::Causal) {
            attention_unfused(&q, &k, &v, 1.0)?
        } else {
            scaled_dot_product_attention(&q, &k, &v, 1.0, &self.mask)?
        };
        self.o.forward(&merge_heads(&a)?)
    }
}

/// Per-layer-embedding block of one layer.
struct Ple {
    gate: Linear,
    proj: Linear,
    post_norm: RmsNorm,
}

struct Layer {
    attn: Attn,
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    pre_ff_norm: RmsNorm,
    post_ff_norm: RmsNorm,
    mlp: GatedMlp,
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

pub(crate) struct Gemma4Model {
    pub cfg: Gemma4Config,
    embed: Tensor,
    logits_act_quant: bool,
    /// PLE table as column chunks [V, cols_i] (one chunk unless it is a huge float table).
    ple_embed: Vec<Tensor>,
    ple_proj: Option<Linear>,
    ple_norm: Option<RmsNorm>,
    layers: Vec<Layer>,
    norm: RmsNorm,
}

fn norm(l: &Loader<'_>, name: &str, d: usize, eps: f32) -> Result<RmsNorm> {
    let n = l.rms_norm(name, eps)?;
    l.expect_shape(&n.weight, &format!("{name}.weight"), &[d])?;
    Ok(n)
}

fn linear(l: &Loader<'_>, name: &str, out: usize, inp: usize) -> Result<Linear> {
    let lin = l.linear(name, false)?;
    l.expect_shape(&lin.weight, &format!("{name}.weight"), &[out, inp])?;
    Ok(lin)
}

/// Host value of a 1-element weight (`layer_scalar`), 1.0 when absent.
fn scalar(l: &Loader<'_>, name: &str) -> Result<f32> {
    if !l.contains(name) {
        return Ok(1.0);
    }
    let t = l.tensor_f32(name)?;
    let v = l.device.to_vec_f32(&t)?;
    match v.as_slice() {
        [x] => Ok(*x),
        _ => Err(Error::new("shape_mismatch", format!("{name}: expected one value, got {}", v.len()))),
    }
}

/// The PLE table, split into column chunks of whole layers when it is a float table larger
/// than `MAX_TABLE_BYTES` (copied on the host once at load).
fn ple_table(l: &Loader<'_>, cfg: &Gemma4Config) -> Result<Vec<Tensor>> {
    let lp = cfg.n_layers * cfg.ple_dim;
    let host = l.file.get(PLE_EMBED)?;
    l.expect_shape(&host, PLE_EMBED, &[cfg.ple_vocab, lp])?;
    let dt = host.dtype();
    let quant_or_fake = dt.is_quant() || l.precision != lapid_nn::Precision::Stored;
    let total = if dt.is_quant() { 0 } else { cfg.ple_vocab * lp * dt.size() };
    if quant_or_fake || total <= MAX_TABLE_BYTES {
        return Ok(vec![l.tensor(PLE_EMBED)?]);
    }
    let bytes = l.file.bytes(PLE_EMBED)?;
    let es = dt.size();
    let n_chunks = total.div_ceil(MAX_TABLE_BYTES);
    let layers_per = cfg.n_layers.div_ceil(n_chunks);
    let mut out = Vec::new();
    let mut c0 = 0;
    while c0 < lp {
        let c1 = (c0 + layers_per * cfg.ple_dim).min(lp);
        let w = c1 - c0;
        let mut v = Vec::with_capacity(cfg.ple_vocab * w * es);
        for r in 0..cfg.ple_vocab {
            let row = r * lp * es;
            v.extend_from_slice(&bytes[row + c0 * es..row + c1 * es]);
        }
        let t = Tensor::from_host(HostData::from_vec(v), &[cfg.ple_vocab, w], dt)?;
        out.push(l.device.upload(&t)?);
        c0 = c1;
    }
    Ok(out)
}

impl Gemma4Model {
    pub fn load(l: &Loader<'_>, cfg: &Gemma4Config) -> Result<Gemma4Model> {
        let (d, eps) = (cfg.hidden_size, cfg.norm_eps);
        let embed = l.tensor(EMBED)?;
        l.expect_shape(&embed, EMBED, &[cfg.vocab_size, d])?;
        let lp = cfg.n_layers * cfg.ple_dim;
        let (ple_embed, ple_proj, ple_norm) = if cfg.ple_dim > 0 {
            (ple_table(l, cfg)?, Some(linear(l, PLE_PROJ, lp, d)?), Some(norm(l, PLE_NORM, cfg.ple_dim, eps)?))
        } else {
            (Vec::new(), None, None)
        };
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("model.layers.{i}");
            let a = format!("{p}.self_attn");
            let (hd, kvh) = (cfg.layer_head_dim(i), cfg.layer_kv_heads(i));
            let shared = cfg.is_shared(i);
            let (k, v, k_norm) = if shared {
                (None, None, None)
            } else {
                let v = if cfg.k_eq_v(i) { None } else { Some(linear(l, &format!("{a}.v_proj"), kvh * hd, d)?) };
                (Some(linear(l, &format!("{a}.k_proj"), kvh * hd, d)?), v, Some(norm(l, &format!("{a}.k_norm"), hd, eps)?))
            };
            let (rp, mask) = match cfg.layer_types[i] {
                LayerType::SlidingAttention => (cfg.rope_sliding, Mask::Sliding(cfg.sliding_window)),
                LayerType::FullAttention => (cfg.rope_full, Mask::Causal),
            };
            let source = cfg
                .kv_source(i)
                .ok_or_else(|| Error::new("metadata", format!("gemma4: layer {i} has no kv source")))?;
            let attn = Attn {
                q: linear(l, &format!("{a}.q_proj"), cfg.n_head * hd, d)?,
                k,
                v,
                o: linear(l, &format!("{a}.o_proj"), d, cfg.n_head * hd)?,
                q_norm: norm(l, &format!("{a}.q_norm"), hd, eps)?,
                k_norm,
                rope: ZeroTailRope::new(hd, rp.theta, rp.partial_rotary_factor),
                mask,
                n_head: cfg.n_head,
                n_kv: kvh,
                head_dim: hd,
                source,
                eps,
            };
            let ff = cfg.layer_intermediate(i);
            let m = format!("{p}.mlp");
            let ple = if cfg.ple_dim > 0 {
                Some(Ple {
                    gate: linear(l, &format!("{p}.per_layer_input_gate"), cfg.ple_dim, d)?,
                    proj: linear(l, &format!("{p}.per_layer_projection"), d, cfg.ple_dim)?,
                    post_norm: norm(l, &format!("{p}.post_per_layer_input_norm"), d, eps)?,
                })
            } else {
                None
            };
            layers.push(Layer {
                attn,
                input_norm: norm(l, &format!("{p}.input_layernorm"), d, eps)?,
                post_attn_norm: norm(l, &format!("{p}.post_attention_layernorm"), d, eps)?,
                pre_ff_norm: norm(l, &format!("{p}.pre_feedforward_layernorm"), d, eps)?,
                post_ff_norm: norm(l, &format!("{p}.post_feedforward_layernorm"), d, eps)?,
                mlp: GatedMlp::geglu(
                    linear(l, &format!("{m}.gate_proj"), ff, d)?,
                    linear(l, &format!("{m}.up_proj"), ff, d)?,
                    linear(l, &format!("{m}.down_proj"), d, ff)?,
                ),
                ple,
                scalar: scalar(l, &format!("{p}.layer_scalar"))?,
            });
        }
        Ok(Gemma4Model {
            cfg: cfg.clone(),
            logits_act_quant: l.act_quant(EMBED)?,
            embed,
            ple_embed,
            ple_proj,
            ple_norm,
            layers,
            norm: norm(l, FINAL_NORM, d, eps)?,
        })
    }

    pub fn new_state(&self) -> State {
        State { kv: KvCache::with_windows(self.cfg.kv_windows()) }
    }

    /// PLE inputs [1, n, L, P] from the ids and the scaled embedding x [1, n, D].
    fn ple_inputs(&self, ids: &[u32], x: &Tensor) -> Result<Option<Tensor>> {
        let (Some(proj), Some(pnorm)) = (&self.ple_proj, &self.ple_norm) else {
            return Ok(None);
        };
        let (n, nl, p) = (ids.len(), self.cfg.n_layers, self.cfg.ple_dim);
        let parts: Vec<Tensor> = self
            .ple_embed
            .iter()
            .map(|t| Embedding::new(t.clone()).lookup(ids)?.cast(DType::F32))
            .collect::<Result<_>>()?;
        let tok = match parts.as_slice() {
            [one] => one.clone(),
            many => Tensor::cat(&many.iter().collect::<Vec<_>>(), 2)?,
        };
        let tok = tok
            .mul_scalar((p as f64).sqrt())?
            .reshape(&[1, n, nl, p])?;
        let pr = proj.forward(x)?.mul_scalar(1.0 / (self.cfg.hidden_size as f64).sqrt())?.reshape(&[1, n, nl, p])?;
        let pr = pnorm.forward(&pr)?;
        Ok(Some(pr.add(&tok)?.mul_scalar(std::f64::consts::FRAC_1_SQRT_2)?))
    }

    /// Graph for `ids` at positions `state.offset()..`; stages new K/V. With `trace`,
    /// intermediates are collected as ("embed" | "layers.{i}.attn" | "layers.{i}" | "norm", t).
    fn build(
        &self,
        ids: &[u32],
        state: &mut State,
        rows: Rows,
        mut trace: Option<&mut Vec<(String, Tensor)>>,
    ) -> Result<Option<Tensor>> {
        if ids.is_empty() {
            return Err(Error::new("invalid_argument", "gemma4: forward of zero tokens"));
        }
        let n = ids.len();
        let mut x = Embedding::new(self.embed.clone())
            .lookup(ids)?
            .cast(DType::F32)?
            .mul_scalar((self.cfg.hidden_size as f64).sqrt())?;
        let mut push = |name: String, t: &Tensor| {
            if let Some(tr) = trace.as_deref_mut() {
                tr.push((name, t.clone()));
            }
        };
        push("embed".into(), &x);
        let ple = self.ple_inputs(ids, &x)?;
        let mut shared: SharedKv = vec![None; self.layers.len()];
        for (i, layer) in self.layers.iter().enumerate() {
            let a = layer.attn.forward(&layer.input_norm.forward(&x)?, &mut state.kv, i, &mut shared)?;
            push(format!("layers.{i}.attn"), &a);
            x = x.add(&layer.post_attn_norm.forward(&a)?)?;
            let f = layer.mlp.forward(&layer.pre_ff_norm.forward(&x)?)?;
            x = x.add(&layer.post_ff_norm.forward(&f)?)?;
            if let (Some(pb), Some(pi)) = (&layer.ple, &ple) {
                let pin = pi.slice(2, i, i + 1)?.reshape(&[1, n, self.cfg.ple_dim])?;
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
        if let Some(c) = self.cfg.final_logit_softcapping.filter(|c| *c > 0.0) {
            logits = logits.mul_scalar(1.0 / f64::from(c))?.tanh()?.mul_scalar(f64::from(c))?;
        }
        Ok(Some(logits))
    }

    /// Evaluate `outputs` together with the staged K/V and commit it. Returns the realized outputs.
    fn eval_commit(&self, device: &Device, outputs: &[&Tensor], state: &mut State) -> Result<Vec<Tensor>> {
        let mut outs: Vec<&Tensor> = outputs.to_vec();
        outs.extend(state.kv.staged());
        let mut realized = match device.eval(&outs) {
            Ok(r) if r.len() == outs.len() => r,
            Ok(_) => {
                state.kv.discard();
                return Err(Error::new("backend", "gemma4 eval returned the wrong number of outputs"));
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

    /// Feed `ids` (a prompt — chunked — or one token) and return the f32 logits of the
    /// last position [n_vocab].
    pub fn step(&self, device: &Device, ids: &[u32], state: &mut State) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Err(Error::new("invalid_argument", "gemma4: step with no tokens"));
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
                let logits = realized.first().ok_or_else(|| Error::new("backend", "gemma4: no logits realized"))?;
                return device.to_vec_f32(logits);
            }
        }
        Err(Error::new("internal", "gemma4: prefill produced no logits"))
    }

    /// One uncached forward over `ids` from a fresh state with every intermediate realized:
    /// ("embed", [n*D]), ("layers.{i}.attn"), ("layers.{i}"), ("norm") and ("logits", rows).
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
