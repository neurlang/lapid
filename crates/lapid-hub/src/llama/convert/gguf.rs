//! GGUF (llama.cpp `llama`, `qwen2`, `qwen3`, `qwen3moe`) → `.lapid`.
//!
//! | GGUF | output |
//! |---|---|
//! | `token_embd` / `output_norm` / `output` | `model.embed_tokens` / `model.norm` / `lm_head` |
//! | `blk.N.attn_norm` / `ffn_norm` | `input_layernorm` / `post_attention_layernorm` |
//! | `attn_q` `attn_k` `attn_v` `attn_output` (`.weight` / `.bias`) | `self_attn.{q,k,v,o}_proj` |
//! | `attn_q_norm` / `attn_k_norm` | `self_attn.q_norm` / `k_norm` |
//! | `ffn_gate` / `ffn_up` / `ffn_down` | `mlp.{gate,up,down}_proj` |
//! | `ffn_gate_inp` / `ffn_{gate,up,down}_exps` | `mlp.gate` / `mlp.experts.{gate,up,down}_proj` |
//!
//! `llama` GGUFs store q/k rows permuted for llama.cpp's interleaved RoPE; they are restored to
//! HF order on raw rows. llama3 RoPE scaling is read from `rope_freqs.weight` (per-frequency
//! divisors). The tokenizer (`gpt2` BPE; pre-tokenizer `llama-bpe` / `qwen2`) is rebuilt as a
//! Hugging Face `tokenizer.json`.

use std::collections::BTreeSet;
use std::path::Path;

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::gguf::{GgufFile, GgufTensor, GgufValue};
use lapid_convert::writer::{ModelWriter, check_complete, file_hash};
use serde_json::{Map, Value, json};

use super::{Assets, add_blobs, expected_tensors, metadata, role, unpermute_rows, write_ggml};
use crate::llama::config::LlamaConfig;

const ARCHS: [&str; 4] = ["llama", "qwen2", "qwen3", "qwen3moe"];
const LLAMA3_SPLIT: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const QWEN2_SPLIT: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

pub(crate) fn is_llama_gguf(f: &GgufFile) -> bool {
    f.architecture().is_some_and(|a| ARCHS.contains(&a))
}

fn key<'a>(f: &'a GgufFile, k: &str) -> Result<&'a GgufValue> {
    f.get(k).ok_or_else(|| Error::new("missing_metadata", format!("GGUF: missing {k:?}")))
}

fn uint(f: &GgufFile, k: &str) -> Result<usize> {
    key(f, k)?.as_u64().map(|v| v as usize).ok_or_else(|| Error::new("metadata", format!("GGUF: {k:?} is not an integer")))
}

fn opt_uint(f: &GgufFile, k: &str) -> Option<usize> {
    f.get(k).and_then(GgufValue::as_u64).map(|v| v as usize)
}

/// HF-style config JSON from the GGUF metadata (then parsed by `LlamaConfig::from_json`).
fn config_json(f: &GgufFile) -> Result<Value> {
    let arch = f.architecture().unwrap_or("llama").to_string();
    let p = |k: &str| format!("{arch}.{k}");
    let hidden = uint(f, &p("embedding_length"))?;
    let n_head = uint(f, &p("attention.head_count"))?;
    let head_dim = opt_uint(f, &p("attention.key_length")).unwrap_or(hidden / n_head.max(1));
    let n_layers = uint(f, &p("block_count"))?;
    let vocab = f.tensor("token_embd.weight").map(|t| t.shape[0]).ok_or_else(|| Error::new("missing_tensor", "GGUF: no token_embd.weight"))?;
    let tok_id = |k: &str| f.get(&format!("tokenizer.ggml.{k}")).and_then(GgufValue::as_u64);
    let experts = opt_uint(f, &p("expert_count")).unwrap_or(0);
    if experts > 0 && arch != "qwen3moe" {
        bail!("unsupported_model", "GGUF: {arch} with {experts} experts (Mixtral-style MoE) is not supported");
    }
    let model_type = match arch.as_str() {
        "qwen3moe" => "qwen3_moe",
        other => other,
    };
    let block0 = |n: &str| f.tensor(&format!("blk.0.{n}")).is_some();
    let mut v = json!({
        "model_type": model_type,
        "vocab_size": vocab,
        "hidden_size": hidden,
        "intermediate_size": opt_uint(f, &p("feed_forward_length")).unwrap_or(0),
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_head,
        "num_key_value_heads": opt_uint(f, &p("attention.head_count_kv")).unwrap_or(n_head),
        "head_dim": head_dim,
        "rms_norm_eps": f.get(&p("attention.layer_norm_rms_epsilon")).and_then(GgufValue::as_f64).unwrap_or(1e-6),
        "rope_theta": f.get(&p("rope.freq_base")).and_then(GgufValue::as_f64).unwrap_or(10_000.0),
        "qkv_bias": block0("attn_q.bias"),
        "o_bias": block0("attn_output.bias"),
        "qk_norm": block0("attn_q_norm.weight"),
        "tie_word_embeddings": f.tensor("output.weight").is_none(),
        "max_position_embeddings": opt_uint(f, &p("context_length")).unwrap_or(32_768),
        "eos_token_id": tok_id("eos_token_id"),
        "bos_token_id": tok_id("bos_token_id"),
        "pad_token_id": tok_id("padding_token_id"),
    });
    if let Some(w) = opt_uint(f, &p("attention.sliding_window")) {
        v["sliding_window"] = json!(w);
        v["layer_types"] = json!(vec!["sliding_attention"; n_layers]);
    }
    if let Some(t) = f.tensor("rope_freqs.weight") {
        v["rope_freq_factors"] = json!(f.to_f32(t)?);
    } else if f.get(&p("rope.scaling.type")).and_then(GgufValue::as_str) == Some("linear") {
        let factor = f.get(&p("rope.scaling.factor")).and_then(GgufValue::as_f64).unwrap_or(1.0);
        v["rope_scaling"] = json!({"rope_type": "linear", "factor": factor});
    }
    if arch == "qwen3moe" {
        v["num_experts"] = json!(experts);
        v["num_experts_per_tok"] = json!(uint(f, &p("expert_used_count"))?);
        v["moe_intermediate_size"] = json!(uint(f, &p("expert_feed_forward_length"))?);
        v["norm_topk_prob"] = json!(true);
    }
    Ok(v)
}

fn strings(f: &GgufFile, k: &str) -> Result<Vec<String>> {
    key(f, k)?
        .as_array()
        .ok_or_else(|| Error::new("metadata", format!("GGUF: {k:?} is not an array")))?
        .iter()
        .map(|v| v.as_str().map(str::to_string).ok_or_else(|| Error::new("metadata", format!("GGUF: {k:?} holds a non-string"))))
        .collect()
}

/// HF `tokenizer.json` equivalent of the GGUF byte-level BPE tokenizer: Llama 3 (`llama-bpe`:
/// digit groups of 3, `ignore_merges`) or Qwen2 (`qwen2`: single digits, NFC). Normal tokens
/// (type 1) form the vocab, control (3) / user-defined (4) tokens become added tokens
/// (special / not special), unused (5) padding is dropped.
fn tokenizer_json(f: &GgufFile) -> Result<Value> {
    let model = f.get("tokenizer.ggml.model").and_then(GgufValue::as_str).unwrap_or("");
    let pre = f.get("tokenizer.ggml.pre").and_then(GgufValue::as_str).unwrap_or("");
    let (split, normalizer, ignore_merges) = match (model, pre) {
        ("gpt2", "llama-bpe" | "llama3") => (LLAMA3_SPLIT, Value::Null, true),
        ("gpt2", "qwen2" | "qwen35") => (QWEN2_SPLIT, json!({"type": "NFC"}), false),
        _ => bail!("unsupported_model", "GGUF tokenizer {model:?} / pre-tokenizer {pre:?} is not supported for llama"),
    };
    let tokens = strings(f, "tokenizer.ggml.tokens")?;
    let merges = strings(f, "tokenizer.ggml.merges")?;
    let types: Vec<i64> = key(f, "tokenizer.ggml.token_type")?
        .as_array()
        .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(1.0) as i64).collect())
        .unwrap_or_default();
    if types.len() != tokens.len() {
        bail!("metadata", "GGUF: {} token types for {} tokens", types.len(), tokens.len());
    }
    let mut vocab = Map::new();
    let mut added = Vec::new();
    for (id, (tok, ty)) in tokens.iter().zip(&types).enumerate() {
        match ty {
            1 => {
                vocab.insert(tok.clone(), json!(id));
            }
            3 | 4 => added.push(json!({
                "id": id, "content": tok, "single_word": false, "lstrip": false, "rstrip": false,
                "normalized": false, "special": *ty == 3,
            })),
            _ => {}
        }
    }
    let byte_level = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
    Ok(json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": normalizer,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": split}, "behavior": "Isolated", "invert": false},
            byte_level.clone(),
        ]},
        "post_processor": byte_level.clone(),
        "decoder": byte_level,
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": ignore_merges,
                  "vocab": vocab, "merges": merges},
    }))
}

/// Output name of a GGUF tensor (`None` = dropped).
fn out_name(g: &str) -> Result<Option<String>> {
    match g {
        "token_embd.weight" => return Ok(Some("model.embed_tokens.weight".into())),
        "output_norm.weight" => return Ok(Some("model.norm.weight".into())),
        "output.weight" => return Ok(Some("lm_head.weight".into())),
        "rope_freqs.weight" => return Ok(None),
        _ => {}
    }
    let unknown = || Error::new("unsupported_model", format!("GGUF: unknown llama tensor {g:?}"));
    let rest = g.strip_prefix("blk.").ok_or_else(unknown)?;
    let (layer, rest) = rest.split_once('.').ok_or_else(unknown)?;
    let li: usize = layer.parse().map_err(|_| unknown())?;
    let (base, suffix) = rest.rsplit_once('.').ok_or_else(unknown)?;
    let n = match base {
        "attn_norm" => "input_layernorm",
        "ffn_norm" => "post_attention_layernorm",
        "attn_q" => "self_attn.q_proj",
        "attn_k" => "self_attn.k_proj",
        "attn_v" => "self_attn.v_proj",
        "attn_output" => "self_attn.o_proj",
        "attn_q_norm" => "self_attn.q_norm",
        "attn_k_norm" => "self_attn.k_norm",
        "ffn_gate" => "mlp.gate_proj",
        "ffn_up" => "mlp.up_proj",
        "ffn_down" => "mlp.down_proj",
        "ffn_gate_inp" => "mlp.gate",
        "ffn_gate_exps" => return Ok(Some(format!("model.layers.{li}.mlp.experts.gate_proj"))),
        "ffn_up_exps" => return Ok(Some(format!("model.layers.{li}.mlp.experts.up_proj"))),
        "ffn_down_exps" => return Ok(Some(format!("model.layers.{li}.mlp.experts.down_proj"))),
        "attn_rot_embd" => return Ok(None),
        _ => return Err(unknown()),
    };
    if !matches!(suffix, "weight" | "bias") {
        return Err(unknown());
    }
    Ok(Some(format!("model.layers.{li}.{n}.{suffix}")))
}

/// Write one tensor; `llama` q/k projections get their rows restored to HF order.
fn write_tensor(f: &GgufFile, w: &mut ModelWriter, dtype: OutDType, cfg: &LlamaConfig, permuted: bool, name: &str, t: &GgufTensor) -> Result<()> {
    let data = f.tensor_data(t);
    let qk = name.contains(".self_attn.q_proj.") || name.contains(".self_attn.k_proj.");
    if permuted && qk {
        let rows = t.shape[0];
        let row_bytes = if t.shape.len() == 2 { t.ggml_type.nbytes(t.shape[1] as u64)? as usize } else { data.len() / rows.max(1) };
        let fixed = unpermute_rows(data, row_bytes, cfg.head_dim);
        return write_ggml(w, dtype, name, t.ggml_type, &t.shape, &fixed);
    }
    write_ggml(w, dtype, name, t.ggml_type, &t.shape, data)
}

/// SPDX-like license id from `general.license` (else "other").
fn license(f: &GgufFile) -> String {
    f.get("general.license").and_then(GgufValue::as_str).filter(|s| !s.is_empty()).unwrap_or("other").to_string()
}

pub(crate) fn convert(path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    let f = GgufFile::open(path)?;
    if !is_llama_gguf(&f) {
        bail!("unsupported_model", "{}: GGUF architecture {:?} is not one of {}", path.display(), f.architecture(), ARCHS.join(", "));
    }
    let cfg = LlamaConfig::from_json(&config_json(&f)?)?;
    let token_name = |id: Option<u32>| id.and_then(|i| f.get("tokenizer.ggml.tokens")?.as_array()?.get(i as usize)?.as_str().map(str::to_string));
    let mut special = Map::new();
    for (k, id) in [("bos_token", cfg.bos), ("eos_token", cfg.eos.first().copied()), ("pad_token", cfg.pad)] {
        if let Some(s) = token_name(id) {
            special.insert(k.to_string(), json!(s));
        }
    }
    let assets = Assets {
        tokenizer_json: serde_json::to_string(&tokenizer_json(&f)?)?,
        chat_template: f
            .get("tokenizer.chat_template")
            .and_then(GgufValue::as_str)
            .map(str::to_string)
            .ok_or_else(|| Error::new("missing_metadata", "GGUF: no tokenizer.chat_template"))?,
        special_tokens: Value::Object(special),
        generation: json!({}),
        license: license(&f),
    };

    let mut mapped = Vec::new();
    let mut have = BTreeSet::new();
    for t in &f.tensors {
        if let Some(n) = out_name(&t.name)? {
            have.insert(n.clone());
            mapped.push((n, t));
        }
    }
    let expected = expected_tensors(&cfg);
    check_complete(&have, &expected)?;

    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("llama").to_string();
    let source = json!({
        "repo": f.get("general.name").and_then(GgufValue::as_str).map_or_else(|| name.clone(), |n| format!("gguf:{n}")),
        "file": path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
        "format": "gguf",
        "file_hash": file_hash(path)?,
    });
    let meta = metadata(&name, dtype, source, &cfg, &assets);
    let mut w = ModelWriter::create(out, meta, dtype, role, expected.len())?;
    add_blobs(&mut w, &assets)?;
    let permuted = f.architecture() == Some("llama");
    for (n, t) in &mapped {
        write_tensor(&f, &mut w, dtype, &cfg, permuted, n, t)?;
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let n = |g: &str| out_name(g).ok().flatten();
        assert_eq!(n("blk.3.attn_q.weight").as_deref(), Some("model.layers.3.self_attn.q_proj.weight"));
        assert_eq!(n("blk.3.attn_q.bias").as_deref(), Some("model.layers.3.self_attn.q_proj.bias"));
        assert_eq!(n("blk.0.ffn_norm.weight").as_deref(), Some("model.layers.0.post_attention_layernorm.weight"));
        assert_eq!(n("blk.1.ffn_down_exps.weight").as_deref(), Some("model.layers.1.mlp.experts.down_proj"));
        assert_eq!(n("blk.1.ffn_gate_inp.weight").as_deref(), Some("model.layers.1.mlp.gate.weight"));
        assert_eq!(n("rope_freqs.weight"), None);
        assert_eq!(out_name("blk.0.mystery.weight").err().map(|e| e.code), Some("unsupported_model"));
    }
}
