//! Llama / Mistral / Qwen2 / Qwen3 / Qwen3-MoE checkpoints -> `.lapid`: HF safetensors (names
//! kept; per-expert `mlp.experts.{e}.{part}.weight` tensors and fused `gate_up_proj` stacks
//! become `mlp.experts.{gate,up,down}_proj` `[E, N, K]` stacks) and GGUF (`llama`, `qwen2`,
//! `qwen3`, `qwen3moe`; llama.cpp's q/k row permutation of `llama` files undone on raw rows,
//! so K-quant blocks are copied; llama3 RoPE scaling from `rope_freqs.weight`).
//!
//! Metadata: config object `"llama"` (normalized `LlamaConfig::to_json`), `architecture` the
//! family (`llama`, `qwen2`, `qwen3`, `qwen3moe`, `mistral`).

use lapid::{Error, Result, bail};
use lapid_convert::assets::HfFiles;
use lapid_convert::engine::{ConvertHooks, ConvertSpec, Emit, HfExtras, SourceKind, TensorSource};
use lapid_convert::gguf::{GgufFile, GgufValue};
use lapid_convert::mapping::{RoleRule, Rule};
use lapid_convert::policy::Role;
use lapid_convert::transforms::llama_qk_unpermute;
use serde_json::{Value, json};

use super::config::{LlamaConfig, MODEL_TYPES};
use super::model::{EMBED, LM_HEAD};

macro_rules! blk {
    ($g:literal, $h:literal) => {
        Rule::new(concat!("blk.{i}.", $g, ".{w}"), concat!("model.layers.{i}.", $h, ".{w}"))
    };
}

pub static CONVERT: ConvertSpec = ConvertSpec {
    architecture: "llama",
    license: "other",
    hf_model_types: MODEL_TYPES,
    gguf_architectures: &["llama", "qwen2", "qwen3", "qwen3moe", "mistral"],
    hf_names: &[Rule::drop("*rotary_emb.inv_freq"), Rule::drop("mtp.*")],
    gguf_names: &[
        Rule::new("token_embd.weight", EMBED),
        Rule::new("output_norm.weight", "model.norm.weight"),
        Rule::new("output.weight", LM_HEAD),
        Rule::drop("rope_freqs.weight"),
        Rule::drop("blk.{i}.attn_rot_embd*"),
        blk!("attn_norm", "input_layernorm"),
        blk!("ffn_norm", "post_attention_layernorm"),
        blk!("attn_q", "self_attn.q_proj"),
        blk!("attn_k", "self_attn.k_proj"),
        blk!("attn_v", "self_attn.v_proj"),
        blk!("attn_output", "self_attn.o_proj"),
        blk!("attn_q_norm", "self_attn.q_norm"),
        blk!("attn_k_norm", "self_attn.k_norm"),
        blk!("ffn_gate", "mlp.gate_proj"),
        blk!("ffn_up", "mlp.up_proj"),
        blk!("ffn_down", "mlp.down_proj"),
        blk!("ffn_gate_inp", "mlp.gate"),
        Rule::new("blk.{i}.ffn_gate_exps.weight", "model.layers.{i}.mlp.experts.gate_proj"),
        Rule::new("blk.{i}.ffn_up_exps.weight", "model.layers.{i}.mlp.experts.up_proj"),
        Rule::new("blk.{i}.ffn_down_exps.weight", "model.layers.{i}.mlp.experts.down_proj"),
    ],
    roles: ROLES,
    text_assets: true,
    hooks: &Hooks,
};

/// Embedding at most int8, router f32, expert stacks quantized like linears.
pub(crate) const ROLES: &[RoleRule] =
    &[RoleRule::new(EMBED, Role::LinearInt8), RoleRule::new("*.mlp.gate.weight", Role::Param), RoleRule::new("*.mlp.experts.*", Role::Linear)];

struct Hooks;

/// `(stack name, expert, part)` of an HF per-expert tensor `{p}.mlp.experts.{e}.{part}.weight`.
fn per_expert(name: &str) -> Option<(String, usize, &str)> {
    let (prefix, rest) = name.split_once(".mlp.experts.")?;
    let (e, part) = rest.split_once('.')?;
    let part = part.strip_suffix(".weight").filter(|p| matches!(*p, "gate_proj" | "up_proj" | "down_proj"))?;
    Some((format!("{prefix}.mlp.experts.{part}"), e.parse().ok()?, part))
}

/// HF expert layouts -> `[E, N, K]` stacks: per-expert tensors are stacked when expert 0 is
/// written (the others dropped), fused `[E, 2I, D]` `gate_up_proj` stacks split.
pub(crate) fn expert_stacks(name: &str, src: &dyn TensorSource, n_experts: usize) -> Result<Emit> {
    if let Some((stack, e, part)) = per_expert(name) {
        if e != 0 {
            return Ok(Emit::Drop);
        }
        let prefix = &name[..name.find(".mlp.experts.").unwrap_or(0)];
        let (mut data, mut inner) = (Vec::new(), Vec::new());
        for x in 0..n_experts {
            let n = format!("{prefix}.mlp.experts.{x}.{part}.weight");
            let shape = src.shape(&n).ok_or_else(|| Error::new("missing_tensor", format!("{stack}: expert {x} missing")))?;
            if x > 0 && shape != inner {
                bail!("shape_mismatch", "{stack}: experts disagree in shape ({inner:?} vs {shape:?})");
            }
            inner = shape;
            data.extend(src.f32(&n)?);
        }
        return Ok(Emit::Replace(vec![(stack, [vec![n_experts], inner].concat(), data)]));
    }
    if let Some(p) = name.strip_suffix(".mlp.experts.gate_up_proj") {
        let shape = src.shape(name).unwrap_or_default();
        let [e, i2, d] = shape[..] else { bail!("shape_mismatch", "{name}: expected [E, 2I, D], got {shape:?}") };
        let (data, half) = (src.f32(name)?, i2 / 2 * d);
        let (mut gate, mut up) = (Vec::with_capacity(data.len() / 2), Vec::with_capacity(data.len() / 2));
        for m in data.chunks_exact(i2 * d) {
            gate.extend_from_slice(&m[..half]);
            up.extend_from_slice(&m[half..]);
        }
        let s = vec![e, i2 / 2, d];
        return Ok(Emit::Replace(vec![(format!("{p}.mlp.experts.gate_proj"), s.clone(), gate), (format!("{p}.mlp.experts.up_proj"), s, up)]));
    }
    Ok(Emit::Keep)
}

impl ConvertHooks for Hooks {
    fn hf_config(&self, c: &Value) -> Result<Value> {
        Ok(LlamaConfig::from_json(c)?.to_json())
    }

    fn gguf_config(&self, f: &GgufFile) -> Result<Value> {
        Ok(LlamaConfig::from_json(&gguf_config(f)?)?.to_json())
    }

    /// License from the model card's YAML front matter.
    fn hf_extras(&self, files: &HfFiles, _cfg: &mut Value) -> Result<HfExtras> {
        let card = files.optional("README.md")?.and_then(|p| std::fs::read_to_string(p).ok()).unwrap_or_default();
        let license = card.strip_prefix("---").and_then(|b| b[..b.find("\n---")?].lines().find_map(|l| l.strip_prefix("license:")));
        Ok(license_extras(license.map(|l| l.trim().trim_matches(|c| c == '"' || c == '\'').to_string())))
    }

    fn gguf_extras(&self, f: &GgufFile, _cfg: &mut Value) -> Result<HfExtras> {
        Ok(license_extras(f.get("general.license").and_then(GgufValue::as_str).map(str::to_string)))
    }

    fn expected(&self, c: &Value) -> Result<Vec<String>> {
        let cfg = LlamaConfig::from_json(c)?;
        let mut v = vec![EMBED.to_string(), "model.norm.weight".to_string()];
        if !cfg.tie_embeddings {
            v.push(LM_HEAD.to_string());
        }
        for i in 0..cfg.n_layers {
            let p = format!("model.layers.{i}");
            v.extend(["input_layernorm", "post_attention_layernorm"].map(|n| format!("{p}.{n}.weight")));
            for n in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                v.push(format!("{p}.self_attn.{n}.weight"));
                if (n == "o_proj" && cfg.o_bias) || (n != "o_proj" && cfg.qkv_bias) {
                    v.push(format!("{p}.self_attn.{n}.bias"));
                }
            }
            if cfg.qk_norm {
                v.extend(["q_norm", "k_norm"].map(|n| format!("{p}.self_attn.{n}.weight")));
            }
            if cfg.layer_is_moe(i) {
                v.push(format!("{p}.mlp.gate.weight"));
                v.extend(["gate_proj", "up_proj", "down_proj"].map(|n| format!("{p}.mlp.experts.{n}")));
            } else {
                for n in ["gate_proj", "up_proj", "down_proj"] {
                    v.push(format!("{p}.mlp.{n}.weight"));
                    if cfg.mlp_bias {
                        v.push(format!("{p}.mlp.{n}.bias"));
                    }
                }
            }
        }
        Ok(v)
    }

    /// llama.cpp permutes `llama` q/k rows for its interleaved RoPE; restore HF order.
    fn row_order(&self, name: &str, shape: &[usize], cfg: &Value, src: SourceKind) -> Option<Vec<usize>> {
        let qk = name.contains(".self_attn.q_proj.") || name.contains(".self_attn.k_proj.");
        let head_dim = cfg["head_dim"].as_u64()? as usize;
        (src == SourceKind::Gguf && qk && cfg["model_type"] == "llama").then(|| llama_qk_unpermute(shape[0], head_dim))
    }

    fn transform(&self, name: &str, src: &dyn TensorSource, cfg: &Value) -> Result<Emit> {
        if name == LM_HEAD && cfg["tie_word_embeddings"] == true {
            return Ok(Emit::Drop);
        }
        if !name.contains(".mlp.experts.") || src.kind() == SourceKind::Gguf {
            return Ok(Emit::Keep);
        }
        expert_stacks(name, src, cfg["num_experts"].as_u64().unwrap_or(0) as usize)
    }

    /// `architecture` = the family (as earlier lapid versions wrote it).
    fn metadata(&self, meta: &mut Value) {
        let arch = LlamaConfig::from_json(&meta["llama"]).map_or("llama", |c| c.arch());
        meta["architecture"] = json!(arch);
        meta["components"]["llama"]["architecture"] = json!(arch);
    }
}

fn license_extras(license: Option<String>) -> HfExtras {
    let mut x = HfExtras::default();
    if let Some(l) = license.filter(|l| !l.is_empty()) {
        x.metadata.insert("license".into(), json!(l));
    }
    x
}

/// HF-style config from `<arch>.*` GGUF keys.
fn gguf_config(f: &GgufFile) -> Result<Value> {
    let arch = f.architecture().unwrap_or("llama").to_string();
    let get = |k: &str| f.get(&format!("{arch}.{k}"));
    let opt = |k: &str| get(k).and_then(GgufValue::as_u64).map(|v| v as usize);
    let uint = |k: &str| opt(k).ok_or_else(|| Error::new("missing_metadata", format!("GGUF: missing {arch}.{k}")));
    let (hidden, n_head, n_layers) = (uint("embedding_length")?, uint("attention.head_count")?, uint("block_count")?);
    let head_dim = opt("attention.key_length").unwrap_or(hidden / n_head.max(1));
    let vocab = f.tensor("token_embd.weight").map(|t| t.shape[0]).ok_or_else(|| Error::new("missing_tensor", "GGUF: no token_embd.weight"))?;
    let tok_id = |k: &str| f.get(&format!("tokenizer.ggml.{k}")).and_then(GgufValue::as_u64);
    let experts = opt("expert_count").unwrap_or(0);
    if experts > 0 && arch != "qwen3moe" {
        bail!("unsupported_model", "GGUF: {arch} with {experts} experts (Mixtral-style MoE) is not supported");
    }
    let block0 = |n: &str| f.tensor(&format!("blk.0.{n}")).is_some();
    let mut v = json!({
        "model_type": if arch == "qwen3moe" { "qwen3_moe" } else { arch.as_str() },
        "vocab_size": vocab, "hidden_size": hidden, "intermediate_size": opt("feed_forward_length").unwrap_or(0),
        "num_hidden_layers": n_layers, "num_attention_heads": n_head,
        "num_key_value_heads": opt("attention.head_count_kv").unwrap_or(n_head), "head_dim": head_dim,
        "rms_norm_eps": get("attention.layer_norm_rms_epsilon").and_then(GgufValue::as_f64).unwrap_or(1e-6),
        "rope_theta": get("rope.freq_base").and_then(GgufValue::as_f64).unwrap_or(10_000.0),
        "qkv_bias": block0("attn_q.bias"), "o_bias": block0("attn_output.bias"), "qk_norm": block0("attn_q_norm.weight"),
        "tie_word_embeddings": f.tensor("output.weight").is_none(),
        "max_position_embeddings": opt("context_length").unwrap_or(32_768),
        "eos_token_id": tok_id("eos_token_id"), "bos_token_id": tok_id("bos_token_id"), "pad_token_id": tok_id("padding_token_id"),
    });
    if let Some(w) = opt("attention.sliding_window") {
        v["sliding_window"] = json!(w);
        v["layer_types"] = json!(vec!["sliding_attention"; n_layers]);
    }
    if let Some(t) = f.tensor("rope_freqs.weight") {
        v["rope_freq_factors"] = json!(f.to_f32(t)?);
    } else if get("rope.scaling.type").and_then(GgufValue::as_str) == Some("linear") {
        v["rope_scaling"] = json!({"rope_type": "linear", "factor": get("rope.scaling.factor").and_then(GgufValue::as_f64).unwrap_or(1.0)});
    }
    if arch == "qwen3moe" {
        v["num_experts"] = json!(experts);
        v["num_experts_per_tok"] = json!(uint("expert_used_count")?);
        v["moe_intermediate_size"] = json!(uint("expert_feed_forward_length")?);
        v["norm_topk_prob"] = json!(true);
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lapid_convert::mapping::{Renamed, rename};

    #[test]
    fn names_and_roles() {
        let g = |n: &str| rename(CONVERT.gguf_names, n);
        assert_eq!(g("blk.3.attn_q.bias"), Renamed::To("model.layers.3.self_attn.q_proj.bias".into()));
        assert_eq!(g("blk.0.ffn_norm.weight"), Renamed::To("model.layers.0.post_attention_layernorm.weight".into()));
        assert_eq!(g("blk.1.ffn_down_exps.weight"), Renamed::To("model.layers.1.mlp.experts.down_proj".into()));
        assert_eq!(g("blk.1.ffn_gate_inp.weight"), Renamed::To("model.layers.1.mlp.gate.weight".into()));
        assert_eq!(g("rope_freqs.weight"), Renamed::Drop);
        assert_eq!(g("blk.0.mystery.weight"), Renamed::NoMatch);
        assert_eq!(rename(CONVERT.hf_names, "model.layers.0.self_attn.rotary_emb.inv_freq"), Renamed::Drop);
        assert_eq!(per_expert("model.layers.2.mlp.experts.7.up_proj.weight"), Some(("model.layers.2.mlp.experts.up_proj".into(), 7, "up_proj")));
        assert_eq!(CONVERT.role(EMBED, &[151936, 1024]), Role::LinearInt8);
        assert_eq!(CONVERT.role("model.layers.0.mlp.gate.weight", &[128, 2048]), Role::Param);
        assert_eq!(CONVERT.role("model.layers.0.mlp.experts.gate_proj", &[128, 768, 2048]), Role::Linear);
        assert_eq!(CONVERT.role(LM_HEAD, &[151936, 1024]), Role::Linear);
        let cfg = LlamaConfig::from_json(&json!({"model_type": "qwen2", "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "tie_word_embeddings": true}))
        .expect("cfg");
        // 2 globals + 2 * (2 norms + 4 attn + 3 bias + 3 mlp)
        assert_eq!(Hooks.expected(&cfg.to_json()).expect("names").len(), 2 + 2 * 12);
    }
}
