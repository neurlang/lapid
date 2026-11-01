//! GGUF (llama.cpp `general.architecture = "gemma4"`) → `.lapid`.
//!
//! Tensor names (ggml `ne` reversed = row-major; no q/k permutation — NeoX-style RoPE):
//!
//! | GGUF | lapid (HF without `language_model`) |
//! |---|---|
//! | `token_embd` / `per_layer_token_embd` / `per_layer_model_proj` / `per_layer_proj_norm` / `output_norm` | `model.embed_tokens` / `embed_tokens_per_layer` / `per_layer_model_projection` / `per_layer_projection_norm` / `norm` |
//! | `blk.N.attn_norm` / `post_attention_norm` / `ffn_norm` / `post_ffw_norm` / `post_norm` | `input_layernorm` / `post_attention_layernorm` / `pre_feedforward_layernorm` / `post_feedforward_layernorm` / `post_per_layer_input_norm` |
//! | `blk.N.attn_q` / `attn_k` / `attn_v` / `attn_output` / `attn_q_norm` / `attn_k_norm` | `self_attn.q_proj` / … / `o_proj` / `q_norm` / `k_norm` |
//! | `blk.N.ffn_gate` / `ffn_up` / `ffn_down` | `mlp.gate_proj` / `up_proj` / `down_proj` |
//! | `blk.N.inp_gate` / `proj` / `layer_output_scale` | `per_layer_input_gate` / `per_layer_projection` / `layer_scalar` |
//! | `rope_freqs` | dropped: 1.0 on rotated pairs, 1e30 on the zero tail → `partial_rotary_factor` |
//!
//! K/V weights of kv-shared layers (present in llama.cpp files, unused) are dropped. The
//! tokenizer (`tokenizer.ggml.model = "gemma4"`: byte-fallback BPE over `▁`-normalized
//! text) is rebuilt as a Hugging Face `tokenizer.json`; the template comes from
//! `tokenizer.chat_template`.

use std::collections::BTreeSet;
use std::path::Path;

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::gguf::{GgufFile, GgufValue};
use lapid_convert::writer::{ModelWriter, file_hash};
use serde_json::{Map, Value, json};

use super::{Assets, add_blobs, expected_tensors, metadata, role, write_gguf_native};
use crate::gemma4::config::Gemma4Config;

const ARCH: &str = "gemma4";

/// Whether an opened GGUF holds a Gemma 4 model.
pub(crate) fn is_gemma4_gguf(f: &GgufFile) -> bool {
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

/// Integer key that may be a per-layer array: (values per layer).
fn per_layer(f: &GgufFile, k: &str, n: usize) -> Result<Vec<usize>> {
    let v = key(f, k)?;
    match v.as_array() {
        Some(a) => Ok(a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect()),
        None => Ok(vec![v.as_u64().unwrap_or(0) as usize; n]),
    }
}

/// HF-style text config JSON from the GGUF metadata (then parsed by `Gemma4Config::from_json`).
fn config_json(f: &GgufFile) -> Result<Value> {
    let p = |k: &str| format!("{ARCH}.{k}");
    let n_layers = uint(f, &p("block_count"))?;
    let swa: Vec<bool> = key(f, &p("attention.sliding_window_pattern"))?
        .as_array()
        .map(|a| a.iter().map(bool_of).collect())
        .unwrap_or_default();
    if swa.len() != n_layers {
        bail!("metadata", "GGUF: sliding_window_pattern has {} entries for {n_layers} layers", swa.len());
    }
    let ff = per_layer(f, &p("feed_forward_length"), n_layers)?;
    let ff_base = ff.iter().copied().min().unwrap_or(0);
    let kv = per_layer(f, &p("attention.head_count_kv"), n_layers)?;
    let kv_swa = (0..n_layers).find(|&i| swa[i]).map_or(kv[0], |i| kv[i]);
    let kv_full = (0..n_layers).find(|&i| !swa[i]).map_or(kv_swa, |i| kv[i]);
    let global_hd = uint(f, &p("attention.key_length"))?;
    let partial = match f.tensor("rope_freqs.weight") {
        Some(t) => {
            let v = f.to_f32(t)?;
            (2 * v.iter().filter(|&&x| x < 1e3).count()) as f64 / global_hd as f64
        }
        None => 1.0,
    };
    let shared = f.get(&p("attention.shared_kv_layers")).and_then(GgufValue::as_u64).unwrap_or(0) as usize;
    let first_full = (0..n_layers.saturating_sub(shared)).find(|&i| !swa[i]);
    let k_eq_v = first_full.is_some_and(|i| f.tensor(&format!("blk.{i}.attn_v.weight")).is_none());
    let rows = |name: &str| f.tensor(name).map(|t| t.shape[0]);
    let vocab = rows("token_embd.weight").ok_or_else(|| Error::new("missing_tensor", "GGUF: no token_embd.weight"))?;
    let tok_id = |k: &str| f.get(&format!("tokenizer.ggml.{k}")).and_then(GgufValue::as_u64);
    let types: Vec<&str> = swa.iter().map(|&s| if s { "sliding_attention" } else { "full_attention" }).collect();
    Ok(json!({
        "model_type": "gemma4_text",
        "vocab_size": vocab,
        "hidden_size": uint(f, &p("embedding_length"))?,
        "intermediate_size": ff_base,
        "use_double_wide_mlp": ff.iter().any(|&x| x > ff_base),
        "num_hidden_layers": n_layers,
        "num_attention_heads": uint(f, &p("attention.head_count"))?,
        "num_key_value_heads": kv_swa,
        "num_global_key_value_heads": kv_full,
        "attention_k_eq_v": k_eq_v,
        "head_dim": f.get(&p("attention.key_length_swa")).and_then(GgufValue::as_u64).map_or(global_hd, |v| v as usize),
        "global_head_dim": global_hd,
        "rms_norm_eps": float(f, &p("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6),
        "layer_types": types,
        "sliding_window": f.get(&p("attention.sliding_window")).and_then(GgufValue::as_u64).unwrap_or(512),
        "rope_parameters": {
            "full_attention": {"rope_type": "proportional", "rope_theta": float(f, &p("rope.freq_base")).unwrap_or(1e6), "partial_rotary_factor": partial},
            "sliding_attention": {"rope_type": "default", "rope_theta": float(f, &p("rope.freq_base_swa")).unwrap_or(1e4)},
        },
        "num_kv_shared_layers": shared,
        "hidden_size_per_layer_input": f.get(&p("embedding_length_per_layer_input")).and_then(GgufValue::as_u64).unwrap_or(0),
        "vocab_size_per_layer_input": rows("per_layer_token_embd.weight").unwrap_or(0),
        "final_logit_softcapping": float(f, &p("final_logit_softcapping")),
        "max_position_embeddings": f.get(&p("context_length")).and_then(GgufValue::as_u64).unwrap_or(131_072),
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

/// Hugging Face `tokenizer.json` equivalent of the GGUF Gemma 4 tokenizer (same layout as
/// google/gemma-4-*'s own file): BPE with byte fallback over text whose spaces are
/// replaced by `▁`; control (3) / user-defined (4) tokens are special added tokens.
fn tokenizer_json(f: &GgufFile) -> Result<Value> {
    let model = f.get("tokenizer.ggml.model").and_then(GgufValue::as_str).unwrap_or("");
    if model != "gemma4" {
        bail!("unsupported_model", "GGUF tokenizer {model:?} is not supported for gemma4");
    }
    let tokens = strings(f, "tokenizer.ggml.tokens")?;
    let merges: Vec<Value> = strings(f, "tokenizer.ggml.merges")?
        .into_iter()
        .map(|m| match m.split_once(' ') {
            Some((a, b)) => Ok(json!([a, b])),
            None => Err(Error::new("metadata", format!("GGUF: malformed merge {m:?}"))),
        })
        .collect::<Result<_>>()?;
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
        vocab.insert(tok.clone(), json!(id));
        if matches!(ty, 3 | 4) {
            added.push(json!({
                "id": id, "content": tok, "single_word": false, "lstrip": false, "rstrip": false,
                "normalized": false, "special": true,
            }));
        }
    }
    let unk = f
        .get("tokenizer.ggml.unknown_token_id")
        .and_then(GgufValue::as_u64)
        .and_then(|i| tokens.get(i as usize).cloned());
    Ok(json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"},
        "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false},
        "post_processor": {"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A", "type_id": 0}}],
                           "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
                           "special_tokens": {}},
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ]},
        "model": {"type": "BPE", "dropout": null, "unk_token": unk, "continuing_subword_prefix": null,
                  "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true, "ignore_merges": false,
                  "vocab": vocab, "merges": merges},
    }))
}

/// lapid name of a GGUF tensor, or `None` for tensors that are dropped.
fn lapid_name(g: &str) -> Result<Option<String>> {
    let global = match g {
        "token_embd.weight" => Some("model.embed_tokens.weight"),
        "per_layer_token_embd.weight" => Some("model.embed_tokens_per_layer.weight"),
        "per_layer_model_proj.weight" => Some("model.per_layer_model_projection.weight"),
        "per_layer_proj_norm.weight" => Some("model.per_layer_projection_norm.weight"),
        "output_norm.weight" => Some("model.norm.weight"),
        "rope_freqs.weight" | "output.weight" => return Ok(None),
        _ => None,
    };
    if let Some(n) = global {
        return Ok(Some(n.to_string()));
    }
    let unknown = || Error::new("unsupported_model", format!("GGUF: unknown gemma4 tensor {g:?}"));
    let rest = g.strip_prefix("blk.").ok_or_else(unknown)?;
    let (layer, rest) = rest.split_once('.').ok_or_else(unknown)?;
    let base = rest.strip_suffix(".weight").ok_or_else(unknown)?;
    let hf = match base {
        "attn_norm" => "input_layernorm.weight",
        "post_attention_norm" => "post_attention_layernorm.weight",
        "ffn_norm" => "pre_feedforward_layernorm.weight",
        "post_ffw_norm" => "post_feedforward_layernorm.weight",
        "post_norm" => "post_per_layer_input_norm.weight",
        "attn_q" => "self_attn.q_proj.weight",
        "attn_k" => "self_attn.k_proj.weight",
        "attn_v" => "self_attn.v_proj.weight",
        "attn_output" => "self_attn.o_proj.weight",
        "attn_q_norm" => "self_attn.q_norm.weight",
        "attn_k_norm" => "self_attn.k_norm.weight",
        "ffn_gate" => "mlp.gate_proj.weight",
        "ffn_up" => "mlp.up_proj.weight",
        "ffn_down" => "mlp.down_proj.weight",
        "inp_gate" => "per_layer_input_gate.weight",
        "proj" => "per_layer_projection.weight",
        "layer_output_scale" => "layer_scalar",
        _ => return Err(unknown()),
    };
    Ok(Some(format!("model.layers.{layer}.{hf}")))
}

pub(crate) fn convert(path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    let f = GgufFile::open(path)?;
    if !is_gemma4_gguf(&f) {
        bail!("unsupported_model", "{}: GGUF architecture {:?} is not gemma4", path.display(), f.architecture());
    }
    let cfg = Gemma4Config::from_json(&config_json(&f)?)?;
    let token_name = |id: Option<u64>| id.and_then(|i| f.get("tokenizer.ggml.tokens")?.as_array()?.get(i as usize)?.as_str().map(str::to_string));
    let mut special = Map::new();
    for (k, id) in [("bos_token", "bos_token_id"), ("eos_token", "eos_token_id"), ("pad_token", "padding_token_id"), ("unk_token", "unknown_token_id")] {
        if let Some(s) = token_name(f.get(&format!("tokenizer.ggml.{id}")).and_then(GgufValue::as_u64)) {
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

    let expected = expected_tensors(&cfg);
    let wanted: BTreeSet<&String> = expected.iter().collect();
    let mut mapped = Vec::new();
    for t in &f.tensors {
        if let Some(n) = lapid_name(&t.name)?.filter(|n| wanted.contains(n)) {
            mapped.push((n, t));
        }
    }
    let have: BTreeSet<String> = mapped.iter().map(|(n, _)| n.clone()).collect();
    lapid_convert::writer::check_complete(&have, &expected)?;

    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("gemma4").to_string();
    let source = json!({
        "repo": f.get("general.name").and_then(GgufValue::as_str).map_or_else(|| name.clone(), |n| format!("gguf:{n}")),
        "file": path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
        "format": "gguf",
        "file_hash": file_hash(path)?,
    });
    let meta = metadata(&name, dtype, source, &cfg, &assets);
    let mut w = ModelWriter::create(out, meta, dtype, role, mapped.len())?;
    add_blobs(&mut w, &assets)?;
    for (n, t) in &mapped {
        write_gguf_native(&f, &mut w, n, t, None, dtype)?;
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let g = |s: &str| lapid_name(s).ok().flatten();
        assert_eq!(g("blk.3.inp_gate.weight").as_deref(), Some("model.layers.3.per_layer_input_gate.weight"));
        assert_eq!(g("blk.0.layer_output_scale.weight").as_deref(), Some("model.layers.0.layer_scalar"));
        assert_eq!(g("blk.9.ffn_norm.weight").as_deref(), Some("model.layers.9.pre_feedforward_layernorm.weight"));
        assert_eq!(g("per_layer_token_embd.weight").as_deref(), Some("model.embed_tokens_per_layer.weight"));
        assert_eq!(g("rope_freqs.weight"), None);
        assert_eq!(lapid_name("blk.0.mystery.weight").err().map(|e| e.code), Some("unsupported_model"));
    }
}
