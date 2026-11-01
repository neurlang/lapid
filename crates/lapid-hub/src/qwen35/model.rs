//! Qwen3.5 network (`Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` text decoder).
//!
//! Decoder layer (pre-norm): `x += mixer(input_layernorm(x))`, `x += ffn(post_attention_layernorm(x))`.
//! `mixer` is Gated DeltaNet (3 of 4 layers) or gated full attention: q (+ a per-head output
//! gate interleaved in q_proj), q/k RMSNorm, partial RoPE (half-split; text-only MRoPE = plain
//! RoPE), GQA, causal; output `o_proj(attn * sigmoid(gate))`. `ffn` is SwiGLU (dense) or the
//! sparse MoE block with a sigmoid-gated shared expert. Final norm, logits through the tied
//! embedding or `lm_head`.
//!
//! File conventions (see `convert`): RMSNorms store the effective scale (HF `1 + w`),
//! DeltaNet `A = -exp(A_log)`, v heads tiled. State (`LmCache`): KV cache for attention
//! layers; slot `i` = conv window, slot `L + i` = delta-rule state of DeltaNet layer `i`.
//! The attention, DeltaNet and MoE blocks are reused by Qwen4-Exp.

use lapid::{DType, Result, Tensor, UnaryOp, bail};
use lapid_nn::attention::{merge_heads, scaled_dot_product_attention};
use lapid_nn::{Conv1d, GatedMlp, KvCache, Linear, Loader, Mask, RmsNorm, Rope, ops};
use serde_json::Value;

use super::config::{LayerType, Qwen35Config};
use crate::llama::model::{LmHead, SparseMoe};
use crate::runtime::{CausalLm, LmCache, LmDims, LmState, Rows, Taps};

/// Gated full attention.
pub(crate) struct GatedAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    pub rope: Rope,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
}

impl GatedAttention {
    /// `{p}.self_attn.*`.
    pub fn load(l: &Loader<'_>, cfg: &Qwen35Config, p: &str) -> Result<GatedAttention> {
        let (d, eps, nh, nkv, dh) = (cfg.hidden_size, cfg.norm_eps, cfg.n_head, cfg.n_kv_head, cfg.head_dim);
        let lin = |n: &str, o: usize, i: usize| l.linear_checked(&format!("{p}.self_attn.{n}"), o, i, false);
        Ok(GatedAttention {
            q: lin("q_proj", 2 * nh * dh, d)?,
            k: lin("k_proj", nkv * dh, d)?,
            v: lin("v_proj", nkv * dh, d)?,
            o: lin("o_proj", d, nh * dh)?,
            q_norm: l.rms_norm_checked(&format!("{p}.self_attn.q_norm"), dh, eps)?,
            k_norm: l.rms_norm_checked(&format!("{p}.self_attn.k_norm"), dh, eps)?,
            rope: Rope::new(dh, cfg.rope_theta).with_rotary_dim(cfg.rotary_dim()),
            n_head: nh,
            n_kv: nkv,
            head_dim: dh,
        })
    }

    /// h [1, n, D] -> [1, n, D] at positions `offset..`.
    pub fn forward(&self, h: &Tensor, cache: &mut KvCache, layer: usize, offset: usize, mask: &Mask) -> Result<Tensor> {
        let n = h.shape()[1];
        let (nh, nkv, dh) = (self.n_head, self.n_kv, self.head_dim);
        let qg = self.q.forward(h)?.reshape(&[1, n, nh, 2 * dh])?;
        let q = self.q_norm.forward(&qg.slice(3, 0, dh)?)?.permute(&[0, 2, 1, 3])?;
        let gate = qg.slice(3, dh, 2 * dh)?.contiguous()?.reshape(&[1, n, nh * dh])?;
        let k = self.k_norm.forward(&self.k.forward(h)?.reshape(&[1, n, nkv, dh])?)?.permute(&[0, 2, 1, 3])?;
        let v = self.v.forward(h)?.reshape(&[1, n, nkv, dh])?.permute(&[0, 2, 1, 3])?;
        let (k, v) = cache.append(layer, &self.rope.apply(&k, offset)?, &v)?;
        let a = scaled_dot_product_attention(&self.rope.apply(&q, offset)?, &k, &v, 1.0 / (dh as f32).sqrt(), mask)?;
        self.o.forward(&merge_heads(&a)?.mul(&ops::sigmoid(&gate)?)?)
    }
}

/// softplus(x) = max(x, 0) + log(1 + exp(-|x|)) (stable for large |x|).
fn softplus(x: &Tensor) -> Result<Tensor> {
    let pos = x.maximum(&Tensor::zeros(x.shape(), x.dtype()))?;
    pos.add(&x.unary(UnaryOp::Abs)?.neg()?.exp()?.add_scalar(1.0)?.log()?)
}

/// x * rsqrt(sum(x^2, last) + eps) (HF `l2norm`, FLA convention).
fn l2norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    x.mul(&x.mul(x)?.sum(-1, true)?.add_scalar(eps)?.unary(UnaryOp::Rsqrt)?)
}

/// Gated DeltaNet token mixer (HF `Qwen3_5GatedDeltaNet`, llama.cpp `build_layer_attn_linear`):
///
/// ```text
/// qkv = silu(conv1d_causal(in_proj_qkv(h)))   (depthwise, streamed window)
/// q, k = l2norm(split(qkv))  (repeated to the v heads, tiled order: v head h reads k head h % Hk)
/// beta = sigmoid(in_proj_b(h)),  g = A * softplus(in_proj_a(h) + dt_bias)
/// o, S = gated_delta_rule(q, k, v, g, beta, S, scale = 1/sqrt(Dk))
/// out = out_proj(rmsnorm(o) * act(in_proj_z(h)))   (act = silu; Qwen4-Exp: sigmoid)
/// ```
pub(crate) struct DeltaNet {
    in_qkv: Linear,
    in_z: Linear,
    in_b: Linear,
    in_a: Linear,
    conv: Conv1d,
    /// -exp(A_log) [Hv] f32.
    a: Tensor,
    dt_bias: Tensor,
    norm: RmsNorm,
    out: Linear,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
    sigmoid_gate: bool,
}

impl DeltaNet {
    /// `{p}.linear_attn.*`.
    pub fn load(l: &Loader<'_>, cfg: &Qwen35Config, p: &str, sigmoid_gate: bool) -> Result<DeltaNet> {
        let (d, hk, hv, dk, dv, c) = (cfg.hidden_size, cfg.linear_k_heads, cfg.linear_v_heads, cfg.linear_k_dim, cfg.linear_v_dim, cfg.conv_dim());
        let a = format!("{p}.linear_attn");
        let lin = |n: &str, o: usize, i: usize| l.linear_checked(&format!("{a}.{n}"), o, i, false);
        let f32_param = |n: &str| l.tensor_checked(&format!("{a}.{n}"), &[hv])?.cast(DType::F32);
        Ok(DeltaNet {
            in_qkv: lin("in_proj_qkv", c, d)?,
            in_z: lin("in_proj_z", hv * dv, d)?,
            in_b: lin("in_proj_b", hv, d)?,
            in_a: lin("in_proj_a", hv, d)?,
            conv: Conv1d::depthwise(l.tensor_checked(&format!("{a}.conv1d.weight"), &[c, 1, cfg.conv_kernel])?, None),
            a: f32_param("A")?,
            dt_bias: f32_param("dt_bias")?,
            norm: l.rms_norm_checked(&format!("{a}.norm"), dv, cfg.norm_eps)?,
            out: lin("out_proj", d, hv * dv)?,
            hk,
            hv,
            dk,
            dv,
            sigmoid_gate,
        })
    }

    /// h [1, n, D] -> [1, n, D]; `conv` / `ssm` are the committed states (`None` = zeros) and
    /// are replaced by the new (lazy) ones.
    pub fn forward(&self, h: &Tensor, conv: &mut Option<Tensor>, ssm: &mut Option<Tensor>) -> Result<Tensor> {
        let (b, n) = (h.shape()[0], h.shape()[1]);
        let (hk, hv, dk, dv) = (self.hk, self.hv, self.dk, self.dv);
        let kd = hk * dk;
        let qkv = self.in_qkv.forward(h)?.cast(DType::F32)?; // [B, n, C]
        let c = qkv.shape()[2];
        if c != 2 * kd + hv * dv {
            bail!("shape_mismatch", "deltanet: in_proj_qkv gives {c} channels, expected {}", 2 * kd + hv * dv);
        }
        let mixed = self.conv.forward_stream(&qkv.transpose(1, 2)?, conv)?; // [B, C, n]
        let mixed = ops::silu(&mixed.cast(DType::F32)?)?.transpose(1, 2)?; // [B, n, C]
        let q = l2norm(&mixed.slice(2, 0, kd)?.reshape(&[b, n, hk, dk])?, 1e-6)?;
        let k = l2norm(&mixed.slice(2, kd, 2 * kd)?.reshape(&[b, n, hk, dk])?, 1e-6)?;
        let v = mixed.slice(2, 2 * kd, c)?.reshape(&[b, n, hv, dv])?;
        let tile = |x: Tensor| -> Result<Tensor> {
            if hv == hk {
                return Ok(x);
            }
            x.unsqueeze(2)?.expand(&[b, n, hv / hk, hk, dk])?.reshape(&[b, n, hv, dk])
        };
        let beta = ops::sigmoid(&self.in_b.forward(h)?.cast(DType::F32)?)?; // [B, n, Hv]
        let g = softplus(&self.in_a.forward(h)?.cast(DType::F32)?.add(&self.dt_bias)?)?.mul(&self.a)?; // <= 0
        let state = ssm.take().unwrap_or_else(|| Tensor::zeros(&[b, hv, dk, dv], DType::F32));
        let out = Tensor::gated_delta_net(&tile(q)?, &tile(k)?, &v, &g, &beta, &state, 1.0 / (dk as f32).sqrt())?; // [B, Hv, n + Dk, Dv]
        *ssm = Some(out.slice(2, n, n + dk)?);
        let o = out.slice(2, 0, n)?.permute(&[0, 2, 1, 3])?; // [B, n, Hv, Dv]
        let z = self.in_z.forward(h)?.cast(DType::F32)?.reshape(&[b, n, hv, dv])?;
        let z = if self.sigmoid_gate { ops::sigmoid(&z)? } else { ops::silu(&z)? };
        self.out.forward(&self.norm.forward(&o)?.mul(&z)?.reshape(&[b, n, hv * dv])?)
    }
}

/// Sparse MoE (HF `Qwen3_5MoeSparseMoeBlock`): renormalized top-k routed experts plus the
/// shared expert gated by `sigmoid(x Ws^T)`.
pub(crate) struct Moe {
    routed: SparseMoe,
    shared: GatedMlp,
    /// Shared expert gate [1, D] (f32).
    shared_gate: Tensor,
}

impl Moe {
    /// `{p}.mlp.*` of an MoE layer.
    pub fn load(l: &Loader<'_>, cfg: &Qwen35Config, p: &str) -> Result<Moe> {
        let d = cfg.hidden_size;
        let Some(m) = &cfg.moe else { bail!("internal", "qwen35: moe layer without moe config") };
        let s = format!("{p}.mlp.shared_expert");
        let lin = |n: &str, o: usize, i: usize| l.linear_checked(&format!("{s}.{n}"), o, i, false);
        let si = m.shared_intermediate;
        Ok(Moe {
            routed: SparseMoe::load(l, p, d, m.n_experts, m.expert_intermediate, m.top_k, true)?,
            shared: GatedMlp::swiglu(lin("gate_proj", si, d)?, lin("up_proj", si, d)?, lin("down_proj", d, si)?),
            shared_gate: l.tensor_checked(&format!("{p}.mlp.shared_expert_gate.weight"), &[1, d])?.cast(DType::F32)?,
        })
    }

    /// h [1, n, D] -> [1, n, D].
    pub fn forward(&self, h: &Tensor) -> Result<Tensor> {
        let (n, d) = (h.shape()[1], h.shape()[2]);
        let x = h.reshape(&[n, d])?.cast(DType::F32)?;
        let shared = self.shared.forward(&x.reshape(&[1, n, d])?)?.reshape(&[n, d])?;
        let sg = ops::sigmoid(&x.matmul_t(&self.shared_gate)?)?; // [n, 1]
        self.routed.forward(&x)?.add(&shared.mul(&sg)?)?.reshape(&[1, n, d])
    }
}

enum Mixer {
    Linear(Box<DeltaNet>),
    Attention(GatedAttention),
}

enum Ffn {
    Dense(GatedMlp),
    Moe(Box<Moe>),
}

struct Layer {
    mixer: Mixer,
    norm1: RmsNorm,
    norm2: RmsNorm,
    ffn: Ffn,
}

pub struct Qwen35Model {
    pub cfg: Qwen35Config,
    head: LmHead,
    layers: Vec<Layer>,
    norm: RmsNorm,
}

fn layer(l: &Loader<'_>, cfg: &Qwen35Config, i: usize, kind: LayerType) -> Result<Layer> {
    let (d, eps, p) = (cfg.hidden_size, cfg.norm_eps, format!("model.layers.{i}"));
    let ffn = if cfg.layer_is_moe(i) {
        Ffn::Moe(Box::new(Moe::load(l, cfg, &p)?))
    } else {
        let lin = |n: &str, o: usize, i: usize| l.linear_checked(&format!("{p}.mlp.{n}"), o, i, false);
        let f = cfg.intermediate_size;
        Ffn::Dense(GatedMlp::swiglu(lin("gate_proj", f, d)?, lin("up_proj", f, d)?, lin("down_proj", d, f)?))
    };
    Ok(Layer {
        mixer: match kind {
            LayerType::FullAttention => Mixer::Attention(GatedAttention::load(l, cfg, &p)?),
            LayerType::LinearAttention => Mixer::Linear(Box::new(DeltaNet::load(l, cfg, &p, false)?)),
        },
        norm1: l.rms_norm_checked(&format!("{p}.input_layernorm"), d, eps)?,
        norm2: l.rms_norm_checked(&format!("{p}.post_attention_layernorm"), d, eps)?,
        ffn,
    })
}

impl CausalLm for Qwen35Model {
    const ARCH: &'static str = "qwen35";
    type State = LmCache;

    fn load(l: &Loader<'_>, meta: &Value) -> Result<Qwen35Model> {
        let cfg = Qwen35Config::from_json(meta)?;
        let layers = cfg.layer_types.iter().enumerate().map(|(i, &k)| layer(l, &cfg, i, k)).collect::<Result<_>>()?;
        Ok(Qwen35Model {
            head: LmHead::load(l, cfg.vocab_size, cfg.hidden_size, !cfg.tie_embeddings)?,
            norm: l.rms_norm_checked("model.norm", cfg.hidden_size, cfg.norm_eps)?,
            layers,
            cfg,
        })
    }

    fn dims(&self) -> LmDims {
        LmDims { vocab: self.cfg.vocab_size, hidden: self.cfg.hidden_size, layers: self.layers.len() }
    }

    fn new_state(&self) -> LmCache {
        let n = self.layers.len();
        LmCache::new(KvCache::new(n), 2 * n)
    }

    fn end_of_turn(&self) -> &'static [&'static str] {
        &["<|im_end|>", "<|endoftext|>"]
    }

    /// Taps: `embed`, `layers.{i}.op` (mixer output), `layers.{i}`, `norm`.
    fn forward(&self, ids: &[u32], st: &mut LmCache, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>> {
        let (offset, nl) = (st.offset(), self.layers.len());
        let mut x = self.head.embed(ids)?;
        taps.tap("embed", &x);
        for (i, layer) in self.layers.iter().enumerate() {
            let h = layer.norm1.forward(&x)?;
            let y = match &layer.mixer {
                Mixer::Attention(a) => a.forward(&h, &mut st.kv, i, offset, &Mask::Causal)?,
                Mixer::Linear(dn) => deltanet_step(dn, &h, st, i, nl + i)?,
            };
            taps.tap(format_args!("layers.{i}.op"), &y);
            x = x.add(&y)?;
            let h = layer.norm2.forward(&x)?;
            x = x.add(&match &layer.ffn {
                Ffn::Dense(m) => m.forward(&h)?,
                Ffn::Moe(m) => m.forward(&h)?,
            })?;
            taps.tap(format_args!("layers.{i}"), &x);
        }
        self.head.logits(&x, rows, taps, |x| self.norm.forward(x))
    }
}

/// DeltaNet over `h` with its conv window in slot `conv` and delta-rule state in slot `ssm`.
pub(crate) fn deltanet_step(dn: &DeltaNet, h: &Tensor, st: &mut LmCache, conv: usize, ssm: usize) -> Result<Tensor> {
    let (mut c, mut s) = (st.slot(conv), st.slot(ssm));
    let y = dn.forward(h, &mut c, &mut s)?;
    for (i, t) in [(conv, c), (ssm, s)] {
        if let Some(t) = t {
            st.stage_slot(i, t);
        }
    }
    Ok(y)
}
