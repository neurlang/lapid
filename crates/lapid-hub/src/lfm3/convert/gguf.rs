//! GGUF (llama.cpp `general.architecture = "lfm3"`) → `.lapid`.
//!
//! Tensor names (ggml `ne` reversed = row-major; no q/k permutation — LFM3 uses NeoX-style
//! rotate-half RoPE):
//!
//! | GGUF | HF |
//! |---|---|
//! | `token_embd.weight` | `model.embed_tokens.weight` |
//! | `output_norm.weight` | `model.embedding_norm.weight` |
//! | `blk.N.attn_norm` / `ffn_norm` / `post_attention_norm` / `post_ffw_norm` | `operator_norm` / `ffn_norm` / `post_operator_norm` / `post_ffn_norm` |
//! | `blk.N.ffn_gate` / `ffn_up` / `ffn_down` | `feed_forward.w1` / `w3` / `w2` |
//! | `blk.N.shortconv.conv` [D, K] / `in_proj` / `out_proj` | `conv.conv` [D, 1, K] / `conv.in_proj` / `conv.out_proj` |
//! | `blk.N.attn_q` / `attn_k` / `attn_v` / `attn_output` / `attn_q_norm` / `attn_k_norm` / `attn_gate` | `self_attn.q_proj` / … / `o_proj` / `q_norm` / `k_norm` / `gate_proj` |
//! | `rope_freqs.weight` | dropped: 1.0 on rotated pairs, 1e30 on the zero tail → `partial_rotary_factor` |
//!
//! Config from `lfm3.*` keys: per-layer `attention.head_count_kv` (0 = conv layer),
//! `attention.sliding_window_pattern` (true = sliding), `rope.freq_base` / `rope.freq_base_swa`.
//! The tokenizer (`tokenizer.ggml.*`, pre-tokenizer `lfm2` = Llama-3 split) is rebuilt as a
//! Hugging Face `tokenizer.json`; the chat template comes from `tokenizer.chat_template`.

use std::collections::BTreeSet;
use std::path::Path;

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::gguf::{GgufFile, GgufValue};
use lapid_convert::writer::{ModelWriter, file_hash};
use serde_json::{Map, Value, json};

use super::{Assets, LICENSE, add_blobs, expected_tensors, metadata, role};
use crate::lfm3::config::Lfm3Config;

const ARCH: &str = "lfm3";
const LLAMA3_SPLIT: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// Whether an opened GGUF holds an LFM3 model.
pub(crate) fn is_lfm3_gguf(f: &GgufFile) -> bool {
    f.architecture() == Some(ARCH)
}

fn key<'a>(f: &'a GgufFile, k: &str) -> Result<&'a GgufValue> {
    f.get(k).ok_or_else(|| Error::new("missing_metadata", format!("GGUF: missing {k:?}")))
}

fn uint(f: &GgufFile, k: &str) -> Result<usize> {
    key(f, k)?.as_u64().map(|v| v as usize).ok_or_else(|| Error::new("metadata", format!("GGUF: {k:?} is not an integer")))
}

fn float(f: &GgufFile, k: &str) -> Option<f64> {
    f.get(k).and_then(GgufValue::as_f64)
}

fn bool_of(v: &GgufValue) -> bool {
    match v {
        GgufValue::Bool(b) => *b,
        other => other.as_u64().is_some_and(|x| x != 0),
    }
}

/// HF-style config JSON from the GGUF metadata (then parsed by `Lfm3Config::from_json`).
fn config_json(f: &GgufFile) -> Result<Value> {
    let p = |k: &str| format!("{ARCH}.{k}");
    let n_layers = uint(f, &p("block_count"))?;
    let hidden = uint(f, &p("embedding_length"))?;
    let n_head = uint(f, &p("attention.head_count"))?;
    let kv = key(f, &p("attention.head_count_kv"))?;
    let kv_per_layer: Vec<usize> = match kv.as_array() {
        Some(a) => a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect(),
        None => vec![kv.as_u64().unwrap_or(0) as usize; n_layers],
    };
    let swa: Vec<bool> = f
        .get(&p("attention.sliding_window_pattern"))
        .and_then(GgufValue::as_array)
        .map(|a| a.iter().map(bool_of).collect())
        .unwrap_or_default();
    if kv_per_layer.len() != n_layers {
        bail!("metadata", "GGUF: head_count_kv has {} entries for {n_layers} layers", kv_per_layer.len());
    }
    let n_kv = kv_per_layer.iter().copied().find(|&k| k > 0).ok_or_else(|| Error::new("metadata", "GGUF: no attention layers"))?;
    let layer_types: Vec<&str> = (0..n_layers)
        .map(|i| match (kv_per_layer[i], swa.get(i).copied().unwrap_or(false)) {
            (0, _) => "conv",
            (_, true) => "sliding_attention",
            _ => "full_attention",
        })
        .collect();
    let head_dim = f.get(&p("attention.key_length")).and_then(GgufValue::as_u64).map_or(hidden / n_head.max(1), |v| v as usize);
    // rope_freqs: 1.0 on rotated pairs, a huge factor (frequency ~ 0) on the zero tail.
    let partial = match f.tensor("rope_freqs.weight") {
        Some(t) => {
            let v = f.to_f32(t)?;
            let rotated = v.iter().filter(|&&x| x < 1e3).count();
            (2 * rotated) as f64 / head_dim as f64
        }
        None => 1.0,
    };
    let vocab = f
        .tensor("token_embd.weight")
        .map(|t| t.shape[0])
        .ok_or_else(|| Error::new("missing_tensor", "GGUF: no token_embd.weight"))?;
    let tok_id = |k: &str| f.get(&format!("tokenizer.ggml.{k}")).and_then(GgufValue::as_u64);
    Ok(json!({
        "model_type": ARCH,
        "vocab_size": vocab,
        "hidden_size": hidden,
        "intermediate_size": uint(f, &p("feed_forward_length"))?,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "norm_eps": float(f, &p("attention.layer_norm_rms_epsilon")).unwrap_or(1e-5),
        "conv_L_cache": f.get(&p("shortconv.l_cache")).and_then(GgufValue::as_u64).unwrap_or(7),
        "conv_bias": f.tensor("blk.0.shortconv.in_proj.bias").is_some(),
        "layer_types": layer_types,
        "sliding_window": f.get(&p("attention.sliding_window")).and_then(GgufValue::as_u64).unwrap_or(256),
        "rope_parameters": {
            "full_attention": {"rope_type": "default", "rope_theta": float(f, &p("rope.freq_base")).unwrap_or(1e6), "partial_rotary_factor": partial},
            "sliding_attention": {"rope_type": "default", "rope_theta": float(f, &p("rope.freq_base_swa")).unwrap_or(1e4)},
        },
        "use_sandwich_norm": f.tensor("blk.0.post_attention_norm.weight").is_some(),
        "max_position_embeddings": f.get(&p("context_length")).and_then(GgufValue::as_u64).unwrap_or(128_000),
        "tie_word_embeddings": true,
        "bos_token_id": tok_id("bos_token_id"),
        "eos_token_id": tok_id("eos_token_id"),
        "pad_token_id": tok_id("padding_token_id"),
    }))
}

fn strings(f: &GgufFile, k: &str) -> Result<Vec<String>> {
    key(f, k)?
        .as_array()
        .ok_or_else(|| Error::new("metadata", format!("GGUF: {k:?} is not an array")))?
        .iter()
        .map(|v| v.as_str().map(str::to_string).ok_or_else(|| Error::new("metadata", format!("GGUF: {k:?} holds a non-string"))))
        .collect()
}

/// Hugging Face `tokenizer.json` equivalent of the GGUF byte-level BPE tokenizer: normal
/// tokens (type 1) form the vocab, control (3) / user-defined (4) tokens are added tokens
/// (special / not special), unused (5) placeholders are dropped.
fn tokenizer_json(f: &GgufFile) -> Result<Value> {
    let model = f.get("tokenizer.ggml.model").and_then(GgufValue::as_str).unwrap_or("");
    let pre = f.get("tokenizer.ggml.pre").and_then(GgufValue::as_str).unwrap_or("");
    if model != "gpt2" || !matches!(pre, "lfm2" | "llama-bpe" | "lfm3") {
        bail!("unsupported_model", "GGUF tokenizer {model:?} / pre-tokenizer {pre:?} is not supported for lfm3");
    }
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
    let byte_level = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false});
    Ok(json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": LLAMA3_SPLIT}, "behavior": "Isolated", "invert": false},
            byte_level,
        ]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true},
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                  "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": true,
                  "vocab": vocab, "merges": merges},
    }))
}

/// HF name of a GGUF tensor, or `None` for tensors that are dropped.
fn hf_name(g: &str) -> Result<Option<String>> {
    match g {
        "token_embd.weight" => return Ok(Some("model.embed_tokens.weight".into())),
        "output_norm.weight" => return Ok(Some("model.embedding_norm.weight".into())),
        "rope_freqs.weight" | "output.weight" => return Ok(None),
        _ => {}
    }
    let unknown = || Error::new("unsupported_model", format!("GGUF: unknown lfm3 tensor {g:?}"));
    let rest = g.strip_prefix("blk.").ok_or_else(unknown)?;
    let (layer, rest) = rest.split_once('.').ok_or_else(unknown)?;
    let (base, suffix) = rest.rsplit_once('.').ok_or_else(unknown)?;
    let hf = match base {
        "attn_norm" => "operator_norm",
        "ffn_norm" => "ffn_norm",
        "post_attention_norm" => "post_operator_norm",
        "post_ffw_norm" => "post_ffn_norm",
        "ffn_gate" => "feed_forward.w1",
        "ffn_up" => "feed_forward.w3",
        "ffn_down" => "feed_forward.w2",
        "shortconv.conv" => "conv.conv",
        "shortconv.in_proj" => "conv.in_proj",
        "shortconv.out_proj" => "conv.out_proj",
        "attn_q" => "self_attn.q_proj",
        "attn_k" => "self_attn.k_proj",
        "attn_v" => "self_attn.v_proj",
        "attn_output" => "self_attn.o_proj",
        "attn_q_norm" => "self_attn.q_norm",
        "attn_k_norm" => "self_attn.k_norm",
        "attn_gate" => "self_attn.gate_proj",
        _ => return Err(unknown()),
    };
    Ok(Some(format!("model.layers.{layer}.{hf}.{suffix}")))
}

pub(crate) fn convert(path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    let f = GgufFile::open(path)?;
    if !is_lfm3_gguf(&f) {
        bail!("unsupported_model", "{}: GGUF architecture {:?} is not lfm3", path.display(), f.architecture());
    }
    let cfg = Lfm3Config::from_json(&config_json(&f)?)?;
    let token_name = |id: Option<u32>| id.and_then(|i| f.get("tokenizer.ggml.tokens")?.as_array()?.get(i as usize)?.as_str().map(str::to_string));
    let mut special = Map::new();
    for (k, id) in [("bos_token", cfg.special.bos), ("eos_token", cfg.special.eos.first().copied()), ("pad_token", cfg.special.pad)] {
        if let Some(s) = token_name(id) {
            special.insert(k.to_string(), json!(s));
        }
    }
    let mut generation = json!({});
    if let Some(t) = float(&f, "general.sampling.temp") {
        generation = json!({"do_sample": true, "temperature": (t * 1e4).round() / 1e4});
        if let Some(k) = f.get("general.sampling.top_k").and_then(GgufValue::as_u64) {
            generation["top_k"] = json!(k);
        }
        if let Some(p) = float(&f, "general.sampling.top_p") {
            generation["top_p"] = json!((p * 1e4).round() / 1e4);
        }
        if let Some(r) = float(&f, "general.sampling.penalty_repeat") {
            generation["repetition_penalty"] = json!((r * 1e4).round() / 1e4);
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
        generation,
    };

    let mut mapped = Vec::new();
    for t in &f.tensors {
        if let Some(n) = hf_name(&t.name)? {
            mapped.push((n, t));
        }
    }
    let expected = expected_tensors(&cfg);
    let have: BTreeSet<String> = mapped.iter().map(|(n, _)| n.clone()).collect();
    lapid_convert::writer::check_complete(&have, &expected)?;

    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("lfm3").to_string();
    let source = json!({
        "repo": f.get("general.name").and_then(GgufValue::as_str).map_or_else(|| name.clone(), |n| format!("gguf:{n}")),
        "file": path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
        "format": "gguf",
        "file_hash": file_hash(path)?,
    });
    let meta = metadata(&name, dtype, LICENSE, source, &cfg, &assets);
    let mut w = ModelWriter::create(out, meta, dtype, role, mapped.len())?;
    add_blobs(&mut w, &assets)?;
    for (n, t) in &mapped {
        let shape: Option<Vec<usize>> = if n.ends_with("conv.conv.weight") && t.shape.len() == 2 {
            Some(vec![t.shape[0], 1, t.shape[1]])
        } else {
            None
        };
        f.write_tensor(&mut w, n, t, shape.as_deref())?;
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(hf_name("blk.3.shortconv.in_proj.weight").ok().flatten().as_deref(), Some("model.layers.3.conv.in_proj.weight"));
        assert_eq!(hf_name("blk.2.attn_gate.weight").ok().flatten().as_deref(), Some("model.layers.2.self_attn.gate_proj.weight"));
        assert_eq!(hf_name("blk.0.post_ffw_norm.weight").ok().flatten().as_deref(), Some("model.layers.0.post_ffn_norm.weight"));
        assert_eq!(hf_name("rope_freqs.weight").ok().flatten(), None);
        assert_eq!(hf_name("blk.0.mystery.weight").err().map(|e| e.code), Some("unsupported_model"));
    }
}
