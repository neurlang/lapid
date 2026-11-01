//! Qwen3.5 checkpoints -> `.lapid`: HF safetensors (`Qwen3_5ForConditionalGeneration` /
//! `...ForCausalLM`, dense and MoE; vision tower and MTP head dropped) and GGUF (`qwen35` /
//! `qwen35moe`, llama.cpp naming; MTP / nextn blocks dropped).
//!
//! File conventions = llama.cpp's GGUF values, so GGUF tensors are renamed and copied (fused
//! `ffn_gate_up_exps` split on raw rows): RMSNorm weights are effective scales (HF `1 + w`;
//! the DeltaNet gated norm is plain `w`), `A = -exp(A_log)`, DeltaNet v heads tiled (HF
//! grouped order is permuted when `linear_num_value_heads > linear_num_key_heads`).

use lapid::{Error, Result};
use lapid_convert::engine::{ConvertHooks, ConvertSpec, Emit, SourceKind, TensorSource};
use lapid_convert::gguf::{GgufFile, GgufValue};
use lapid_convert::mapping::{RoleRule, Rule};
use lapid_convert::policy::{Role, layer_index};
use lapid_convert::transforms::{gather_rows, group_order};
use serde_json::{Value, json};

use super::config::{LayerType, Qwen35Config};
use crate::llama::convert::expert_stacks;
use crate::llama::model::{EMBED, LM_HEAD};

macro_rules! blk {
    ($g:literal, $h:literal) => {
        Rule::new(concat!("blk.{i}.", $g), concat!("model.layers.{i}.", $h))
    };
}

/// HF `model.language_model.*` (multimodal checkpoints) -> `model.*`, any depth.
macro_rules! lm {
    ($($s:literal),*) => {
        Rule::new(concat!("model.language_model", $(".{", $s, "}"),*), concat!("model", $(".{", $s, "}"),*))
    };
}

/// `a` then `b` (`C` = their total length).
pub(crate) const fn concat<const A: usize, const B: usize, const C: usize>(a: [Rule; A], b: [Rule; B]) -> [Rule; C] {
    let mut out = [Rule::drop(""); C];
    let mut i = 0;
    while i < A + B {
        out[i] = if i < A { a[i] } else { b[i - A] };
        i += 1;
    }
    out
}

/// HF `model.language_model.*` (multimodal checkpoints) -> `model.*`, any depth.
pub(crate) const HF_LM: [Rule; 6] = [
    lm!("a", "b"),
    lm!("a", "b", "c"),
    lm!("a", "b", "c", "d"),
    lm!("a", "b", "c", "d", "e"),
    lm!("a", "b", "c", "d", "e", "f"),
    lm!("a", "b", "c", "d", "e", "f", "g"),
];

const HF_NAMES: [Rule; 10] = concat(
    [
        Rule::drop("mtp.*"),
        Rule::drop("*visual.*"),
        Rule::new("model.language_model.layers.{i}.linear_attn.A_log", "model.layers.{i}.linear_attn.A"),
        Rule::new("model.layers.{i}.linear_attn.A_log", "model.layers.{i}.linear_attn.A"),
    ],
    HF_LM,
);

/// llama.cpp per-layer names of the Qwen3.5 blocks.
pub(crate) const GGUF_LAYERS: [Rule; 29] = [
    blk!("attn_norm.weight", "input_layernorm.weight"),
    blk!("post_attention_norm.weight", "post_attention_layernorm.weight"),
    blk!("attn_q.weight", "self_attn.q_proj.weight"),
    blk!("attn_k.weight", "self_attn.k_proj.weight"),
    blk!("attn_v.weight", "self_attn.v_proj.weight"),
    blk!("attn_output.weight", "self_attn.o_proj.weight"),
    blk!("attn_q_norm.weight", "self_attn.q_norm.weight"),
    blk!("attn_k_norm.weight", "self_attn.k_norm.weight"),
    blk!("attn_qkv.weight", "linear_attn.in_proj_qkv.weight"),
    blk!("attn_gate.weight", "linear_attn.in_proj_z.weight"),
    blk!("ssm_beta.weight", "linear_attn.in_proj_b.weight"),
    blk!("ssm_alpha.weight", "linear_attn.in_proj_a.weight"),
    blk!("ssm_out.weight", "linear_attn.out_proj.weight"),
    blk!("ssm_conv1d.weight", "linear_attn.conv1d.weight"),
    blk!("ssm_dt.bias", "linear_attn.dt_bias"),
    blk!("ssm_a", "linear_attn.A"),
    blk!("ssm_norm.weight", "linear_attn.norm.weight"),
    blk!("ffn_gate.weight", "mlp.gate_proj.weight"),
    blk!("ffn_up.weight", "mlp.up_proj.weight"),
    blk!("ffn_down.weight", "mlp.down_proj.weight"),
    blk!("ffn_gate_inp.weight", "mlp.gate.weight"),
    blk!("ffn_gate_exps.weight", "mlp.experts.gate_proj"),
    blk!("ffn_up_exps.weight", "mlp.experts.up_proj"),
    blk!("ffn_down_exps.weight", "mlp.experts.down_proj"),
    blk!("ffn_gate_up_exps.weight", "mlp.experts.gate_up_proj"),
    blk!("ffn_gate_shexp.weight", "mlp.shared_expert.gate_proj.weight"),
    blk!("ffn_up_shexp.weight", "mlp.shared_expert.up_proj.weight"),
    blk!("ffn_down_shexp.weight", "mlp.shared_expert.down_proj.weight"),
    blk!("ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight"),
];

const GGUF_NAMES: [Rule; 34] = concat(
    [
        Rule::new("token_embd.weight", EMBED),
        Rule::new("output_norm.weight", "model.norm.weight"),
        Rule::new("output.weight", LM_HEAD),
        Rule::drop("rope_freqs.weight"),
        Rule::drop("blk.{i}.nextn.*"),
    ],
    GGUF_LAYERS,
);

pub static CONVERT: ConvertSpec = ConvertSpec {
    architecture: "qwen35",
    license: "apache-2.0",
    hf_model_types: &["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"],
    gguf_architectures: &["qwen35", "qwen35moe"],
    hf_names: &HF_NAMES,
    gguf_names: &GGUF_NAMES,
    roles: ROLES,
    text_assets: true,
    hooks: &Hooks,
};

/// Embedding at most int8; router and shared-expert gate f32; expert stacks like linears.
pub(crate) const ROLES: &[RoleRule] = &[
    RoleRule::new(EMBED, Role::LinearInt8),
    RoleRule::new("*.mlp.gate.weight", Role::Param),
    RoleRule::new("*shared_expert_gate.weight", Role::Param),
    RoleRule::new("*.mlp.experts.*", Role::Linear),
];

struct Hooks;

/// Expected mixer and FFN tensors of every layer (Qwen4-Exp adds its own on top);
/// `norms`: the pre-norms of both blocks.
pub(crate) fn expected_layers(cfg: &Qwen35Config, norms: bool, v: &mut Vec<String>) {
    for (i, t) in cfg.layer_types.iter().enumerate() {
        let p = format!("model.layers.{i}");
        if norms {
            v.extend(["input_layernorm", "post_attention_layernorm"].map(|n| format!("{p}.{n}.weight")));
        }
        let names: &[&str] = match t {
            LayerType::FullAttention => &["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"],
            LayerType::LinearAttention => &["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj", "conv1d", "norm"],
        };
        let module = if *t == LayerType::FullAttention { "self_attn" } else { "linear_attn" };
        v.extend(names.iter().map(|n| format!("{p}.{module}.{n}.weight")));
        if *t == LayerType::LinearAttention {
            v.extend(["A", "dt_bias"].map(|n| format!("{p}.linear_attn.{n}")));
        }
        if cfg.layer_is_moe(i) {
            v.push(format!("{p}.mlp.gate.weight"));
            for n in ["gate_proj", "up_proj", "down_proj"] {
                v.push(format!("{p}.mlp.experts.{n}"));
                v.push(format!("{p}.mlp.shared_expert.{n}.weight"));
            }
            v.push(format!("{p}.mlp.shared_expert_gate.weight"));
        } else {
            v.extend(["gate_proj", "up_proj", "down_proj"].map(|n| format!("{p}.mlp.{n}.weight")));
        }
    }
}

/// HF -> file values of one tensor (see the module docs); `None` = unchanged. `r` v heads per
/// k head, `kd` q + k channels of the DeltaNet in_proj_qkv / conv1d.
pub(crate) fn hf_values(cfg: &Qwen35Config, name: &str, shape: &[usize], data: Vec<f32>) -> Result<Option<Vec<f32>>> {
    let (hk, dv) = (cfg.linear_k_heads, cfg.linear_v_dim);
    let r = cfg.linear_v_heads / hk.max(1);
    let mut data = data;
    let mut changed = false;
    if name.ends_with("norm.weight") && !name.ends_with("linear_attn.norm.weight") {
        data.iter_mut().for_each(|x| *x += 1.0);
        changed = true;
    }
    if name.ends_with(".linear_attn.A") {
        data.iter_mut().for_each(|x| *x = -x.exp());
        changed = true;
    }
    if r > 1 && name.contains(".linear_attn.") && !name.ends_with("norm.weight") {
        // tiled head j = vi * hk + g  <-  grouped head g * r + vi
        let heads: Vec<usize> = (0..hk * r).map(|j| (j % hk) * r + j / hk).collect();
        let rows = shape.first().copied().unwrap_or(1);
        let cols = data.len() / rows.max(1);
        let kd = 2 * hk * cfg.linear_k_dim;
        let prefix = |n: usize, order: Vec<usize>| (0..n).chain(order.into_iter().map(|x| x + n)).collect::<Vec<_>>();
        let order = if name.ends_with("in_proj_qkv.weight") || name.ends_with("conv1d.weight") {
            Some(prefix(kd, group_order(&heads, dv)))
        } else if name.ends_with("in_proj_z.weight") {
            Some(group_order(&heads, dv))
        } else if ["in_proj_a.weight", "in_proj_b.weight", ".A", ".dt_bias"].iter().any(|s| name.ends_with(s)) {
            Some(heads.clone())
        } else {
            None
        };
        if let Some(o) = order {
            data = gather_rows(&data, &o)?;
            changed = true;
        } else if name.ends_with("out_proj.weight") {
            let o = group_order(&heads, dv);
            data = data.chunks_exact(cols).flat_map(|row| o.iter().map(move |&c| row[c])).collect();
            changed = true;
        }
    }
    Ok(changed.then_some(data))
}

/// Common tensor handling of Qwen3.5-style checkpoints: drop a tied `lm_head` and blocks past
/// the trunk (GGUF MTP), split fused GGUF `gate_up_exps` on raw rows, stack HF experts, apply
/// the HF value conventions.
pub(crate) fn transform(cfg: &Qwen35Config, name: &str, src: &dyn TensorSource) -> Result<Emit> {
    if (name == LM_HEAD && cfg.tie_embeddings) || layer_index(name).is_some_and(|i| i >= cfg.n_layers) {
        return Ok(Emit::Drop);
    }
    if let Some(p) = name.strip_suffix(".mlp.experts.gate_up_proj")
        && src.kind() == SourceKind::Gguf
    {
        let shape = src.shape(name).unwrap_or_default();
        let [e, i2, k] = shape[..] else { return Err(Error::new("shape_mismatch", format!("{name}: expected [E, 2I, K], got {shape:?}"))) };
        let part = |off: usize| (0..e).flat_map(|x| (x * i2 + off)..(x * i2 + off + i2 / 2)).collect::<Vec<_>>();
        let s = vec![e, i2 / 2, k];
        return Ok(Emit::Rows(vec![(format!("{p}.mlp.experts.gate_proj"), s.clone(), part(0)), (format!("{p}.mlp.experts.up_proj"), s, part(i2 / 2))]));
    }
    if src.kind() == SourceKind::Gguf {
        return Ok(Emit::Keep);
    }
    if name.contains(".mlp.experts.") {
        return expert_stacks(name, src, cfg.moe.as_ref().map_or(0, |m| m.n_experts));
    }
    let shape = src.shape(name).unwrap_or_default();
    let needs = name.contains("norm.weight") || name.contains(".linear_attn.");
    match needs.then(|| src.f32(name)).transpose()?.map(|d| hf_values(cfg, name, &shape, d)).transpose()?.flatten() {
        Some(data) => Ok(Emit::Replace(vec![(name.to_string(), shape, data)])),
        None => Ok(Emit::Keep),
    }
}

/// GGUF stores the conv as [C, K] and the shared-expert gate as [D].
pub(crate) fn reshape(name: &str, shape: &[usize]) -> Option<Vec<usize>> {
    match shape {
        [c, k] if name.ends_with("conv1d.weight") => Some(vec![*c, 1, *k]),
        [d] if name.ends_with("shared_expert_gate.weight") => Some(vec![1, *d]),
        _ => None,
    }
}

impl ConvertHooks for Hooks {
    fn hf_config(&self, c: &Value) -> Result<Value> {
        Ok(Qwen35Config::from_json(c)?.to_json())
    }

    fn gguf_config(&self, f: &GgufFile) -> Result<Value> {
        Ok(Qwen35Config::from_json(&gguf_config(f, "qwen35")?)?.to_json())
    }

    fn expected(&self, c: &Value) -> Result<Vec<String>> {
        let cfg = Qwen35Config::from_json(c)?;
        let mut v = vec![EMBED.to_string(), "model.norm.weight".to_string()];
        if !cfg.tie_embeddings {
            v.push(LM_HEAD.to_string());
        }
        expected_layers(&cfg, true, &mut v);
        Ok(v)
    }

    fn reshape(&self, name: &str, shape: &[usize], _cfg: &Value, _src: SourceKind) -> Option<Vec<usize>> {
        reshape(name, shape)
    }

    fn transform(&self, name: &str, src: &dyn TensorSource, cfg: &Value) -> Result<Emit> {
        transform(&Qwen35Config::from_json(cfg)?, name, src)
    }

    fn metadata(&self, meta: &mut Value) {
        let arch = if meta["qwen35"].get("num_experts").is_some() { "qwen35moe" } else { "qwen35" };
        meta["architecture"] = json!(arch);
        meta["components"]["qwen35"]["architecture"] = json!(arch);
        meta["qwen35"]["weight_layout"] = json!({"rms_norm": "effective_scale", "deltanet_A": "neg_exp", "v_heads": "tiled"});
    }
}

/// HF-style config from `<arch>.*` GGUF keys (`qwen35`, `qwen35moe`, `qwen4exp`); MTP /
/// nextn blocks are not counted as layers.
pub(crate) fn gguf_config(f: &GgufFile, family: &str) -> Result<Value> {
    let arch = f.architecture().unwrap_or(family).to_string();
    let get = |k: &str| f.get(&format!("{arch}.{k}"));
    let opt = |k: &str| get(k).and_then(GgufValue::as_u64).map(|v| v as usize);
    let uint = |k: &str| opt(k).ok_or_else(|| Error::new("missing_metadata", format!("GGUF: missing {arch}.{k}")));
    let blocks = uint("block_count")?;
    let n_layers = blocks - opt("nextn_predict_layers").unwrap_or(0).min(blocks);
    let (hidden, n_head, hv) = (uint("embedding_length")?, uint("attention.head_count")?, uint("ssm.time_step_rank")?);
    let head_dim = opt("attention.key_length").unwrap_or(hidden / n_head.max(1));
    let recurrent: Option<Vec<bool>> = get("attention.recurrent_layers")
        .and_then(GgufValue::as_array)
        .map(|a| a.iter().map(|v| matches!(v, GgufValue::Bool(true)) || v.as_u64().is_some_and(|x| x != 0)).collect());
    let interval = opt("full_attention_interval").unwrap_or(4).max(1);
    let layer_types: Vec<&str> = (0..n_layers)
        .map(|i| match recurrent.as_ref().map_or((i + 1) % interval != 0, |r| r.get(i).copied().unwrap_or(false)) {
            true => "linear_attention",
            false => "full_attention",
        })
        .collect();
    let vocab = f.tensor("token_embd.weight").map(|t| t.shape[0]).ok_or_else(|| Error::new("missing_tensor", "GGUF: no token_embd.weight"))?;
    let tok_id = |k: &str| f.get(&format!("tokenizer.ggml.{k}")).and_then(GgufValue::as_u64);
    let moe = opt("expert_count").is_some_and(|e| e > 0);
    let mut v = json!({
        "model_type": if moe { "qwen3_5_moe" } else { "qwen3_5" },
        "vocab_size": vocab, "hidden_size": hidden, "intermediate_size": opt("feed_forward_length").unwrap_or(0),
        "num_hidden_layers": n_layers, "num_attention_heads": n_head, "num_key_value_heads": uint("attention.head_count_kv")?,
        "head_dim": head_dim, "layer_types": layer_types,
        "rms_norm_eps": get("attention.layer_norm_rms_epsilon").and_then(GgufValue::as_f64).unwrap_or(1e-6),
        "rope_parameters": {"rope_type": "default", "rope_theta": get("rope.freq_base").and_then(GgufValue::as_f64).unwrap_or(1e7),
            "partial_rotary_factor": opt("rope.dimension_count").unwrap_or(head_dim / 4) as f64 / head_dim as f64},
        "linear_conv_kernel_dim": opt("ssm.conv_kernel").unwrap_or(4), "linear_num_key_heads": uint("ssm.group_count")?,
        "linear_num_value_heads": hv, "linear_key_head_dim": uint("ssm.state_size")?, "linear_value_head_dim": uint("ssm.inner_size")? / hv.max(1),
        "tie_word_embeddings": f.tensor("output.weight").is_none(),
        "max_position_embeddings": opt("context_length").unwrap_or(262_144),
        "eos_token_id": tok_id("eos_token_id"), "pad_token_id": tok_id("padding_token_id"),
    });
    if moe {
        let inter = uint("expert_feed_forward_length")?;
        v["num_experts"] = json!(uint("expert_count")?);
        v["num_experts_per_tok"] = json!(uint("expert_used_count")?);
        v["moe_intermediate_size"] = json!(inter);
        v["shared_expert_intermediate_size"] = json!(opt("expert_shared_feed_forward_length").unwrap_or(inter));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lapid_convert::mapping::{Renamed, rename};

    #[test]
    fn names_roles_values() {
        let h = |n: &str| rename(CONVERT.hf_names, n);
        assert_eq!(h("model.language_model.layers.0.linear_attn.A_log"), Renamed::To("model.layers.0.linear_attn.A".into()));
        assert_eq!(h("model.language_model.embed_tokens.weight"), Renamed::To(EMBED.into()));
        assert_eq!(h("model.language_model.layers.2.mlp.experts.7.up_proj.weight"), Renamed::To("model.layers.2.mlp.experts.7.up_proj.weight".into()));
        assert_eq!(h("model.visual.blocks.0.attn.qkv.weight"), Renamed::Drop);
        assert_eq!(h("mtp.fc.weight"), Renamed::Drop);
        assert_eq!(h("lm_head.weight"), Renamed::NoMatch);
        let g = |n: &str| rename(CONVERT.gguf_names, n);
        assert_eq!(g("blk.0.ssm_a"), Renamed::To("model.layers.0.linear_attn.A".into()));
        assert_eq!(g("blk.1.ffn_down_exps.weight"), Renamed::To("model.layers.1.mlp.experts.down_proj".into()));
        assert_eq!(g("blk.0.mystery.weight"), Renamed::NoMatch);
        assert_eq!(CONVERT.role("model.layers.0.linear_attn.conv1d.weight", &[6144, 1, 4]), Role::Dense);
        assert_eq!(CONVERT.role("model.layers.0.mlp.shared_expert_gate.weight", &[1, 2048]), Role::Param);
        assert_eq!(CONVERT.role("model.layers.0.mlp.experts.gate_proj", &[256, 512, 2048]), Role::Linear);
        // hk = 2 groups, r = 2 v heads per group, dim 1: grouped [g0v0, g0v1, g1v0, g1v1] -> tiled [g0v0, g1v0, g0v1, g1v1]
        let cfg = Qwen35Config::from_json(&json!({"model_type": "qwen3_5", "tie_word_embeddings": true, "text_config": {
            "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 1}}))
        .expect("cfg");
        let d = vec![0.0, 1.0, 2.0, 3.0];
        assert_eq!(hf_values(&cfg, "model.layers.0.linear_attn.dt_bias", &[4], d.clone()).expect("v"), Some(vec![0.0, 2.0, 1.0, 3.0]));
        assert_eq!(hf_values(&cfg, "model.layers.0.linear_attn.out_proj.weight", &[1, 4], d).expect("v"), Some(vec![0.0, 2.0, 1.0, 3.0]));
        // 2 globals + 4 * (2 norms + 3 mlp) + 3 linear * 9 + 1 attn * 6
        assert_eq!(Hooks.expected(&cfg.to_json()).expect("names").len(), 2 + 20 + 27 + 6);
    }
}
