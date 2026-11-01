//! Loading the decoder weights (HF names without the `language_model` level).

use lapid::{Error, HostData, Result, Tensor};
use lapid_nn::{GatedMlp, Linear, Loader, Precision, RmsNorm};

use super::layers::{Attn, Moe};
use super::rope::ZeroTailRope;
use super::{EMBED, FINAL_NORM, Gemma4MoeModel, Layer, PLE_EMBED, PLE_NORM, PLE_PROJ, Ple};
use crate::gemma4::config::{Gemma4Config, LayerType};
use crate::gemma4moe::config::Gemma4MoeConfig;

/// Float gather tables above this size are split into column chunks (Vulkan
/// `maxStorageBufferRange` is 4 GiB; the f16 PLE table of E2B is 4.7 GB).
const MAX_TABLE_BYTES: usize = 1 << 31;

/// A weight as stored, or for the fake-quant precisions dequantized to f32 (any quantized
/// format, 2-D matrices and 3-D expert stacks).
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

fn norm(l: &Loader<'_>, name: &str, d: usize, eps: f32) -> Result<RmsNorm> {
    let n = l.rms_norm(name, eps)?;
    l.expect_shape(&n.weight, &format!("{name}.weight"), &[d])?;
    Ok(n)
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
    let t = l.tensor_f32(name)?;
    l.expect_shape(&t, name, shape)?;
    Ok(t)
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
    let quant_or_fake = dt.is_quant() || l.precision != Precision::Stored;
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

fn moe_block(l: &Loader<'_>, p: &str, mc: &Gemma4MoeConfig) -> Result<Option<(RmsNorm, Moe)>> {
    let Some(m) = mc.moe else { return Ok(None) };
    let (d, eps) = (mc.text.hidden_size, mc.text.norm_eps);
    let (e, i) = (m.num_experts, m.moe_intermediate_size);
    let gu = format!("{p}.experts.gate_up_proj");
    let moe = Moe {
        router: tensor(l, &format!("{p}.router.proj.weight"), &[e, d])?,
        router_scale: f32_param(l, &format!("{p}.router.scale"), &[d])?,
        per_expert_scale: f32_param(l, &format!("{p}.router.per_expert_scale"), &[e])?,
        gate_up: tensor(l, &gu, &[e, 2 * i, d])?,
        down: tensor(l, &format!("{p}.experts.down_proj"), &[e, d, i])?,
        top_k: m.top_k,
        n_experts: e,
        act_quant: l.act_quant(&gu)?,
        pre_norm_2: norm(l, &format!("{p}.pre_feedforward_layernorm_2"), d, eps)?,
        post_norm_2: norm(l, &format!("{p}.post_feedforward_layernorm_2"), d, eps)?,
        eps,
    };
    Ok(Some((norm(l, &format!("{p}.post_feedforward_layernorm_1"), d, eps)?, moe)))
}

pub(super) fn load(l: &Loader<'_>, mc: &Gemma4MoeConfig) -> Result<Gemma4MoeModel> {
    let cfg = &mc.text;
    let (d, eps) = (cfg.hidden_size, cfg.norm_eps);
    let embed = weight(l, EMBED)?;
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
        let (k, v, k_norm) = if cfg.is_shared(i) {
            (None, None, None)
        } else {
            let v = if cfg.k_eq_v(i) { None } else { Some(linear(l, &format!("{a}.v_proj"), kvh * hd, d)?) };
            (Some(linear(l, &format!("{a}.k_proj"), kvh * hd, d)?), v, Some(norm(l, &format!("{a}.k_norm"), hd, eps)?))
        };
        let (rp, window) = match cfg.layer_types[i] {
            LayerType::SlidingAttention => (cfg.rope_sliding, Some(cfg.sliding_window)),
            LayerType::FullAttention => (cfg.rope_full, None),
        };
        let source = cfg.kv_source(i).ok_or_else(|| Error::new("metadata", format!("gemma4moe: layer {i} has no kv source")))?;
        let attn = Attn {
            q: linear(l, &format!("{a}.q_proj"), cfg.n_head * hd, d)?,
            k,
            v,
            o: linear(l, &format!("{a}.o_proj"), d, cfg.n_head * hd)?,
            q_norm: norm(l, &format!("{a}.q_norm"), hd, eps)?,
            k_norm,
            rope: ZeroTailRope::new(hd, rp.theta, rp.partial_rotary_factor),
            window,
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
            moe: moe_block(l, &p, mc)?,
            ple,
            scalar: scalar(l, &format!("{p}.layer_scalar"))?,
        });
    }
    Ok(Gemma4MoeModel {
        cfg: mc.clone(),
        logits_act_quant: l.act_quant(EMBED)?,
        embed,
        ple_embed,
        ple_proj,
        ple_norm,
        layers,
        norm: norm(l, FINAL_NORM, d, eps)?,
    })
}
