//! DeepSeek V4 checkpoints -> `.lapid`: upstream safetensors (the `deepseek-ai/DeepSeek-V4-*`
//! layout, bf16 / f16 / f32; the FP8 / FP4 releases carry `F8_E8M0` block scales that are not
//! read: use a GGUF) and GGUF (llama.cpp `deepseek4`, values stored unchanged).
//!
//! Our names = upstream names, except: routed experts stacked as `layers.{i}.ffn.experts.{w}`
//! `[E, N, K]`, `attn.wo_a.weight` as `[G, o_lora, H / G * head_dim]` (one MatMulId "expert" per
//! output group), the hash tables `ffn.gate.tid2eid` as f32 `[vocab, k]`; the MTP block and the
//! unused hash-layer selection biases are dropped. V4 has no Jinja template (prompts are
//! `encoding_dsv4`), so only the tokenizer is stored (GGUF: rebuilt with all ids kept).
//!
//! Precision: Linear weights and expert stacks quantized, embedding >= int8; compressors and
//! the indexer weights projection (fp32 in HF), router, norms, sinks, hyper-connections, ape
//! f32. GGUF Q8_0 / Q4_0 / K-quant weights are kept as stored, MXFP4 experts re-encoded.

use lapid::{Error, Result, bail};
use lapid_convert::assets::{HfFiles, TOKENIZER_BLOB, special_tokens};
use lapid_convert::engine::{ConvertHooks, ConvertSpec, Emit, HfExtras, SourceKind, TensorSource};
use lapid_convert::gguf::{GgufFile, GgufValue};
use lapid_convert::mapping::{RoleRule, Rule};
use lapid_convert::policy::Role;
use lapid_convert::writer::read_json;
use serde_json::{Map, Value, json};

use super::config::Deepseek4Config;
use super::model::{EMBED, HEAD};

macro_rules! blk {
    ($g:literal, $h:literal) => {
        Rule::new(concat!("blk.{i}.", $g), concat!("layers.{i}.", $h))
    };
}

pub static CONVERT: ConvertSpec = ConvertSpec {
    architecture: "deepseek4",
    license: "mit",
    hf_model_types: &["deepseek_v4"],
    gguf_architectures: &["deepseek4"],
    hf_names: &[Rule::drop("mtp.*")],
    gguf_names: &[
        Rule::new("token_embd.weight", EMBED),
        Rule::new("output_norm.weight", "norm.weight"),
        Rule::new("output.weight", HEAD),
        Rule::new("output_hc_{k}.weight", "hc_head_{k}"),
        Rule::drop("blk.{i}.nextn.*"),
        Rule::drop("blk.{i}.exp_probs_b_vl.bias"),
        blk!("attn_norm.weight", "attn_norm.weight"),
        blk!("ffn_norm.weight", "ffn_norm.weight"),
        blk!("hc_{k}.weight", "hc_{k}"),
        blk!("attn_sinks.weight", "attn.attn_sink"),
        blk!("attn_q_a.weight", "attn.wq_a.weight"),
        blk!("attn_q_a_norm.weight", "attn.q_norm.weight"),
        blk!("attn_q_b.weight", "attn.wq_b.weight"),
        blk!("attn_kv.weight", "attn.wkv.weight"),
        blk!("attn_kv_a_norm.weight", "attn.kv_norm.weight"),
        blk!("attn_output_a.weight", "attn.wo_a.weight"),
        blk!("attn_output_b.weight", "attn.wo_b.weight"),
        blk!("attn_compressor_kv.weight", "attn.compressor.wkv.weight"),
        blk!("attn_compressor_gate.weight", "attn.compressor.wgate.weight"),
        blk!("attn_compressor_ape.weight", "attn.compressor.ape"),
        blk!("attn_compressor_norm.weight", "attn.compressor.norm.weight"),
        blk!("indexer.attn_q_b.weight", "attn.indexer.wq_b.weight"),
        blk!("indexer.proj.weight", "attn.indexer.weights_proj.weight"),
        blk!("indexer_compressor_kv.weight", "attn.indexer.compressor.wkv.weight"),
        blk!("indexer_compressor_gate.weight", "attn.indexer.compressor.wgate.weight"),
        blk!("indexer_compressor_ape.weight", "attn.indexer.compressor.ape"),
        blk!("indexer_compressor_norm.weight", "attn.indexer.compressor.norm.weight"),
        blk!("ffn_gate_inp.weight", "ffn.gate.weight"),
        blk!("exp_probs_b.bias", "ffn.gate.bias"),
        blk!("ffn_gate_tid2eid.weight", "ffn.gate.tid2eid"),
        blk!("ffn_gate_exps.weight", "ffn.experts.w1"),
        blk!("ffn_up_exps.weight", "ffn.experts.w3"),
        blk!("ffn_down_exps.weight", "ffn.experts.w2"),
        blk!("ffn_gate_shexp.weight", "ffn.shared_experts.w1.weight"),
        blk!("ffn_up_shexp.weight", "ffn.shared_experts.w3.weight"),
        blk!("ffn_down_shexp.weight", "ffn.shared_experts.w2.weight"),
    ],
    roles: &[
        // HF `_keep_in_fp32_modules`: the compressors pool many tokens and amplify input
        // rounding, the indexer weights pick the top-k entries
        RoleRule::new("*.compressor.*", Role::Param),
        RoleRule::new("*weights_proj.weight", Role::Param),
        RoleRule::new("*.ffn.gate.*", Role::Param),
        RoleRule::new("*hc_*", Role::Param),
        RoleRule::new("*.ape", Role::Param),
        RoleRule::new(EMBED, Role::LinearInt8),
        RoleRule::new("*.ffn.experts.*", Role::Linear),
        RoleRule::new("*.attn.wo_a.weight", Role::Linear),
    ],
    text_assets: false,
    hooks: &Hooks,
};

struct Hooks;

fn cfg_of(c: &Value) -> Result<Deepseek4Config> {
    Deepseek4Config::from_json(c)
}

/// Layer index of `layers.{i}.*`.
fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix("layers.")?.split('.').next()?.parse().ok()
}

/// `(layers.{i}.ffn.experts.{w}, e)` of a per-expert upstream tensor.
fn per_expert(name: &str) -> Option<(String, usize)> {
    let (prefix, rest) = name.split_once(".ffn.experts.")?;
    let (e, part) = rest.split_once('.')?;
    let w = part.strip_suffix(".weight").filter(|w| matches!(*w, "w1" | "w2" | "w3"))?;
    Some((format!("{prefix}.ffn.experts.{w}"), e.parse().ok()?))
}

impl ConvertHooks for Hooks {
    fn hf_config(&self, c: &Value) -> Result<Value> {
        cfg_of(c).map(|_| c.clone())
    }

    fn hf_extras(&self, files: &HfFiles, cfg: &mut Value) -> Result<HfExtras> {
        let tok_cfg = files.optional("tokenizer_config.json")?.as_deref().map(read_json).transpose()?;
        cfg["special_tokens"] = special_tokens(tok_cfg.as_ref(), None);
        cfg["generation"] = match files.optional("generation_config.json")? {
            Some(p) => read_json(&p)?,
            None => json!({}),
        };
        let p = files.required("tokenizer.json")?;
        let tok = std::fs::read(&p).map_err(|e| Error::new("io", format!("{}: {e}", p.display())))?;
        Ok(tokenizer_extras(cfg, tok))
    }

    fn gguf_config(&self, f: &GgufFile) -> Result<Value> {
        let c = gguf_config(f)?;
        cfg_of(&c).map(|_| c)
    }

    fn gguf_extras(&self, f: &GgufFile, cfg: &mut Value) -> Result<HfExtras> {
        cfg["generation"] = json!({});
        Ok(tokenizer_extras(cfg, serde_json::to_vec(&gguf_tokenizer(f)?)?))
    }

    fn expected(&self, c: &Value) -> Result<Vec<String>> {
        let cfg = cfg_of(c)?;
        let mut v: Vec<String> = [EMBED, HEAD, "norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"].map(String::from).to_vec();
        for i in 0..cfg.n_layers {
            let p = format!("layers.{i}");
            let mut names: Vec<String> = ["attn_norm.weight", "ffn_norm.weight", "ffn.gate.weight"].map(String::from).to_vec();
            names.extend(["attn", "ffn"].iter().flat_map(|k| ["fn", "base", "scale"].map(|s| format!("hc_{k}_{s}"))));
            names.extend(["attn_sink", "wq_a.weight", "q_norm.weight", "wq_b.weight", "wkv.weight", "kv_norm.weight", "wo_a.weight", "wo_b.weight"].map(|n| format!("attn.{n}")));
            let comp = |c: &str| ["wkv.weight", "wgate.weight", "ape", "norm.weight"].map(|n| format!("attn.{c}compressor.{n}"));
            match cfg.compress_ratios[i] {
                0 => {}
                4 => {
                    names.extend(comp(""));
                    names.extend(comp("indexer."));
                    names.extend(["attn.indexer.wq_b.weight".into(), "attn.indexer.weights_proj.weight".into()]);
                }
                _ => names.extend(comp("")),
            }
            names.push(format!("ffn.gate.{}", if i < cfg.n_hash_layers { "tid2eid" } else { "bias" }));
            names.extend(["w1", "w2", "w3"].iter().flat_map(|w| [format!("ffn.experts.{w}"), format!("ffn.shared_experts.{w}.weight")]));
            v.extend(names.iter().map(|n| format!("{p}.{n}")));
        }
        Ok(v)
    }

    /// `attn.wo_a` `[G * o_lora, in]` -> `[G, o_lora, in]`.
    fn reshape(&self, name: &str, shape: &[usize], c: &Value, _src: SourceKind) -> Option<Vec<usize>> {
        if !(name.ends_with(".attn.wo_a.weight") && shape.len() == 2) {
            return None;
        }
        let cfg = cfg_of(c).ok()?;
        Some(vec![cfg.o_groups, cfg.o_lora_rank, shape[1]])
    }

    /// Drops the MTP layers and the hash layers' selection bias; stacks HF per-expert tensors.
    fn transform(&self, name: &str, src: &dyn TensorSource, c: &Value) -> Result<Emit> {
        let cfg = cfg_of(c)?;
        let layer = layer_of(name);
        if layer.is_some_and(|l| l >= cfg.n_layers || (l < cfg.n_hash_layers && name.ends_with(".ffn.gate.bias"))) {
            return Ok(Emit::Drop);
        }
        if name.ends_with(".scale") {
            bail!("unsupported_dtype", "{name}: FP8 / FP4 quantized upstream checkpoints are not supported; convert a GGUF instead");
        }
        let Some((stack, e)) = per_expert(name) else { return Ok(Emit::Keep) };
        if e != 0 {
            return Ok(Emit::Drop);
        }
        let inner = src.shape(name).unwrap_or_default();
        let mut data = Vec::new();
        for x in 0..cfg.n_experts {
            let n = name.replacen(".experts.0.", &format!(".experts.{x}."), 1);
            if src.shape(&n).as_ref() != Some(&inner) {
                bail!("missing_tensor", "{stack}: expert {x} missing or of another shape");
            }
            data.extend(src.f32(&n)?);
        }
        let mut shape = vec![cfg.n_experts];
        shape.extend(inner);
        Ok(Emit::Replace(vec![(stack, shape, data)]))
    }
}

/// Tokenizer blob + its metadata keys.
fn tokenizer_extras(cfg: &mut Value, tokenizer_json: Vec<u8>) -> HfExtras {
    cfg["tokenizer_blob"] = json!(TOKENIZER_BLOB);
    HfExtras { blobs: vec![(TOKENIZER_BLOB.into(), "application/json".into(), tokenizer_json)], ..Default::default() }
}

/// Upstream-style config from the `deepseek4.*` GGUF keys (MTP blocks not counted).
fn gguf_config(f: &GgufFile) -> Result<Value> {
    let get = |k: &str| f.get(&format!("deepseek4.{k}"));
    let opt = |k: &str| get(k).and_then(GgufValue::as_u64).map(|v| v as usize);
    let uint = |k: &str| opt(k).ok_or_else(|| Error::new("missing_metadata", format!("GGUF: missing deepseek4.{k}")));
    let float = |k: &str, d: f64| get(k).and_then(GgufValue::as_f64).unwrap_or(d);
    let blocks = uint("block_count")?;
    let ratios: Vec<u64> = get("attention.compress_ratios")
        .and_then(GgufValue::as_array)
        .ok_or_else(|| Error::new("metadata", "GGUF: no deepseek4.attention.compress_ratios array"))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0))
        .collect();
    let vocab = f.tensor("token_embd.weight").map(|t| t.shape[0]).ok_or_else(|| Error::new("missing_tensor", "GGUF: no token_embd.weight"))?;
    if opt("expert_gating_func").unwrap_or(4) != 4 {
        bail!("unsupported_model", "GGUF: deepseek4 expert gating other than sqrt-softplus (4)");
    }
    let swiglu = match get("swiglu_clamp_exp") {
        Some(GgufValue::Array(a)) => a.first().and_then(GgufValue::as_f64),
        v => v.and_then(GgufValue::as_f64),
    };
    let tok = |k: &str| f.get(&format!("tokenizer.ggml.{k}")).and_then(GgufValue::as_u64);
    let mut v = json!({
        "model_type": "deepseek_v4", "vocab_size": vocab, "hidden_size": uint("embedding_length")?,
        "num_hidden_layers": blocks - opt("nextn_predict_layers").unwrap_or(0).min(blocks),
        "num_attention_heads": uint("attention.head_count")?, "head_dim": uint("attention.key_length")?,
        "qk_rope_head_dim": uint("rope.dimension_count")?, "q_lora_rank": uint("attention.q_lora_rank")?,
        "o_groups": uint("attention.output_group_count")?, "o_lora_rank": uint("attention.output_lora_rank")?,
        "n_routed_experts": uint("expert_count")?, "n_shared_experts": opt("expert_shared_count").unwrap_or(1),
        "num_experts_per_tok": uint("expert_used_count")?, "moe_intermediate_size": uint("expert_feed_forward_length")?,
        "routed_scaling_factor": float("expert_weights_scale", 1.0), "swiglu_limit": swiglu.unwrap_or(0.0),
        "sliding_window": uint("attention.sliding_window")?, "compress_ratios": ratios,
        "num_hash_layers": opt("hash_layer_count").unwrap_or(0), "rope_theta": float("rope.freq_base", 10000.0),
        "compress_rope_theta": float("attention.compress_rope_freq_base", 160000.0),
        "hc_mult": opt("hyper_connection.count").unwrap_or(4), "hc_sinkhorn_iters": opt("hyper_connection.sinkhorn_iterations").unwrap_or(20),
        "hc_eps": float("hyper_connection.epsilon", 1e-6), "index_n_heads": uint("attention.indexer.head_count")?,
        "index_head_dim": uint("attention.indexer.key_length")?, "index_topk": uint("attention.indexer.top_k")?,
        "rms_norm_eps": float("attention.layer_norm_rms_epsilon", 1e-6), "max_position_embeddings": opt("context_length").unwrap_or(1 << 20),
        "bos_token_id": tok("bos_token_id"), "eos_token_id": tok("eos_token_id").unwrap_or(1),
    });
    if get("rope.scaling.type").and_then(GgufValue::as_str) == Some("yarn") {
        v["rope_scaling"] = json!({
            "type": "yarn", "factor": float("rope.scaling.factor", 1.0),
            "original_max_position_embeddings": opt("rope.scaling.original_context_length").unwrap_or(65536),
            "beta_fast": float("rope.scaling.yarn_beta_fast", 32.0), "beta_slow": float("rope.scaling.yarn_beta_slow", 1.0),
        });
    }
    Ok(v)
}

/// DeepSeek V3 / V4 pre-tokenizer splits (llama.cpp `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM`).
const SPLITS: [&str; 3] = [
    "\\p{N}{1,3}",
    "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+",
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
];

/// HF `tokenizer.json` of the GGUF byte-level BPE (DeepSeek V3 splits). Every token keeps its
/// id in the BPE vocab (added tokens below the normal range, e.g. BOS = 0, would otherwise be
/// renumbered); control (3) / user-defined (4) tokens are also added tokens.
fn gguf_tokenizer(f: &GgufFile) -> Result<Value> {
    let s = |k: &str| f.get(k).and_then(GgufValue::as_str).unwrap_or("");
    let (model, pre) = (s("tokenizer.ggml.model"), s("tokenizer.ggml.pre"));
    if model != "gpt2" || !matches!(pre, "deepseek-v3" | "joyai-llm" | "deepseek4" | "deepseek-v4") {
        bail!("unsupported_model", "GGUF tokenizer {model:?} / pre-tokenizer {pre:?} is not supported for deepseek4");
    }
    let arr = |k: &str| f.get(k).and_then(GgufValue::as_array).ok_or_else(|| Error::new("missing_metadata", format!("GGUF: missing array {k:?}")));
    let tokens: Vec<&str> = arr("tokenizer.ggml.tokens")?.iter().map(|v| v.as_str().unwrap_or("")).collect();
    let merges: Vec<&str> = arr("tokenizer.ggml.merges")?.iter().map(|v| v.as_str().unwrap_or("")).collect();
    let types: Vec<i64> = arr("tokenizer.ggml.token_type")?.iter().map(|v| v.as_f64().unwrap_or(1.0) as i64).collect();
    if types.len() != tokens.len() {
        bail!("metadata", "GGUF: {} token types for {} tokens", types.len(), tokens.len());
    }
    let mut vocab = Map::new();
    let mut added = Vec::new();
    for (id, (tok, ty)) in tokens.iter().zip(&types).enumerate() {
        vocab.insert(tok.to_string(), json!(id));
        if matches!(ty, 3 | 4) {
            added.push(json!({"id": id, "content": tok, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": *ty == 3}));
        }
    }
    let splits: Vec<Value> = SPLITS
        .iter()
        .map(|r| json!({"type": "Split", "pattern": {"Regex": r}, "behavior": "Isolated", "invert": false}))
        .chain([json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false})])
        .collect();
    let byte_level = json!({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true});
    Ok(json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": {"type": "Sequence", "normalizers": []}, "pre_tokenizer": {"type": "Sequence", "pretokenizers": splits},
        "post_processor": byte_level.clone(), "decoder": byte_level,
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null, "end_of_word_suffix": null,
                  "fuse_unk": false, "byte_fallback": false, "vocab": vocab, "merges": merges},
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lapid_convert::mapping::{Renamed, rename};

    #[test]
    fn names_and_roles() {
        let g = |n: &str| rename(CONVERT.gguf_names, n);
        assert_eq!(g("blk.3.attn_output_a.weight"), Renamed::To("layers.3.attn.wo_a.weight".into()));
        assert_eq!(g("blk.2.indexer_compressor_ape.weight"), Renamed::To("layers.2.attn.indexer.compressor.ape".into()));
        assert_eq!(g("blk.0.hc_attn_fn.weight"), Renamed::To("layers.0.hc_attn_fn".into()));
        assert_eq!(g("output_hc_fn.weight"), Renamed::To("hc_head_fn".into()));
        assert_eq!(g("blk.0.exp_probs_b_vl.bias"), Renamed::Drop);
        assert_eq!(g("blk.0.mystery.weight"), Renamed::NoMatch);
        assert_eq!(per_expert("layers.2.ffn.experts.7.w3.weight"), Some(("layers.2.ffn.experts.w3".into(), 7)));
        assert_eq!(per_expert("layers.2.ffn.shared_experts.w3.weight"), None);
        assert_eq!(CONVERT.role(EMBED, &[129280, 4096]), Role::LinearInt8);
        assert_eq!(CONVERT.role("layers.0.hc_attn_fn", &[24, 16384]), Role::Param);
        assert_eq!(CONVERT.role("layers.1.attn.compressor.wkv.weight", &[1024, 4096]), Role::Param);
        assert_eq!(CONVERT.role("layers.1.attn.indexer.wq_b.weight", &[8192, 1024]), Role::Linear);
        assert_eq!(CONVERT.role("layers.1.attn.wo_a.weight", &[8, 1024, 4096]), Role::Linear);
        assert_eq!(CONVERT.role("layers.1.ffn.experts.w1", &[8, 1024, 4096]), Role::Linear);
        assert_eq!(CONVERT.role("layers.1.ffn.gate.weight", &[256, 4096]), Role::Param);
        assert_eq!(CONVERT.role("layers.0.ffn.gate.tid2eid", &[129280, 6]), Role::Param);
    }
}
