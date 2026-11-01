//! GGUF pair → `.lapid`: a llama.cpp `qwen3vl` text model + its `clip` mmproj.
//!
//! | GGUF | lapid |
//! |---|---|
//! | `token_embd` / `output_norm` / `output` | `model.embed_tokens` / `model.norm` / `lm_head` |
//! | `blk.N.attn_norm` / `ffn_norm` | `input_layernorm` / `post_attention_layernorm` |
//! | `attn_q` `attn_k` `attn_v` `attn_output` `attn_q_norm` `attn_k_norm` | `self_attn.{q,k,v,o}_proj`, `q_norm`, `k_norm` |
//! | `ffn_gate` / `ffn_up` / `ffn_down` | `mlp.{gate,up,down}_proj` |
//! | mmproj `v.*`, `mm.*` | same names (see [`crate::vision::mmproj`]) |
//!
//! The tokenizer (`gpt2` BPE, pre-tokenizer `qwen2`) is rebuilt as an HF `tokenizer.json`.

use std::collections::BTreeSet;
use std::path::Path;

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::gguf::{GgufFile, GgufValue};
use lapid_convert::writer::{ModelWriter, file_hash};
use serde_json::{Map, Value, json};

use super::{Assets, add_blobs, expected_tensors, is_qwen3vl_gguf, metadata, role, text_tensors};
use crate::qwen3vl::config::{Qwen3VlConfig, TextConfig};
use crate::vision::mmproj;

const QWEN2_SPLIT: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

fn uint(f: &GgufFile, k: &str) -> Result<usize> {
    f.get(k)
        .and_then(GgufValue::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| Error::new("missing_metadata", format!("GGUF: missing integer {k:?}")))
}

fn strings(f: &GgufFile, k: &str) -> Result<Vec<String>> {
    f.get(k)
        .and_then(GgufValue::as_array)
        .ok_or_else(|| Error::new("missing_metadata", format!("GGUF: {k:?} is not an array")))?
        .iter()
        .map(|v| v.as_str().map(str::to_string).ok_or_else(|| Error::new("metadata", format!("GGUF: {k:?} holds a non-string"))))
        .collect()
}

/// HF-style `text_config` from the GGUF metadata.
fn text_config(f: &GgufFile) -> Result<TextConfig> {
    let p = |k: &str| format!("qwen3vl.{k}");
    let hidden = uint(f, &p("embedding_length"))?;
    let n_head = uint(f, &p("attention.head_count"))?;
    let head_dim = f.get(&p("attention.key_length")).and_then(GgufValue::as_u64).map_or(hidden / n_head.max(1), |x| x as usize);
    let sec: Vec<u64> = f
        .get(&p("rope.dimension_sections"))
        .and_then(GgufValue::as_array)
        .map(|a| a.iter().filter_map(GgufValue::as_u64).collect())
        .unwrap_or_else(|| vec![24, 20, 20]);
    let vocab = f.tensor("token_embd.weight").map(|t| t.shape[0]).ok_or_else(|| Error::new("missing_tensor", "GGUF: no token_embd.weight"))?;
    let v = json!({
        "vocab_size": vocab,
        "hidden_size": hidden,
        "intermediate_size": uint(f, &p("feed_forward_length"))?,
        "num_hidden_layers": uint(f, &p("block_count"))?,
        "num_attention_heads": n_head,
        "num_key_value_heads": uint(f, &p("attention.head_count_kv"))?,
        "head_dim": head_dim,
        "rms_norm_eps": f.get(&p("attention.layer_norm_rms_epsilon")).and_then(GgufValue::as_f64).unwrap_or(1e-6),
        "rope_parameters": {"rope_type": "default", "rope_theta": f.get(&p("rope.freq_base")).and_then(GgufValue::as_f64).unwrap_or(5e6),
            "mrope_section": &sec[..3.min(sec.len())], "mrope_interleaved": true},
        "tie_word_embeddings": f.tensor("output.weight").is_none(),
        "max_position_embeddings": f.get(&p("context_length")).and_then(GgufValue::as_u64).unwrap_or(262_144),
    });
    TextConfig::from_json(&v, true)
}

/// HF `tokenizer.json` equivalent of the GGUF byte-level BPE tokenizer (Qwen2 split regex,
/// NFC normalizer); normal tokens (type 1) form the vocab, control (3) / user-defined (4)
/// tokens become added tokens. Same construction as `qwen35`'s (candidate for sharing).
fn tokenizer_json(f: &GgufFile) -> Result<Value> {
    let model = f.get("tokenizer.ggml.model").and_then(GgufValue::as_str).unwrap_or("");
    let pre = f.get("tokenizer.ggml.pre").and_then(GgufValue::as_str).unwrap_or("");
    if model != "gpt2" || !matches!(pre, "qwen2") {
        bail!("unsupported_model", "GGUF tokenizer {model:?} / pre-tokenizer {pre:?} is not supported for qwen3vl");
    }
    let tokens = strings(f, "tokenizer.ggml.tokens")?;
    let merges = strings(f, "tokenizer.ggml.merges")?;
    let types: Vec<i64> = f
        .get("tokenizer.ggml.token_type")
        .and_then(GgufValue::as_array)
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
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": QWEN2_SPLIT}, "behavior": "Isolated", "invert": false},
            byte_level.clone(),
        ]},
        "post_processor": byte_level.clone(),
        "decoder": byte_level,
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                  "vocab": vocab, "merges": merges},
    }))
}

/// lapid name of a text-model GGUF tensor (`None` = dropped).
fn out_name(g: &str) -> Result<Option<String>> {
    match g {
        "token_embd.weight" => return Ok(Some("model.embed_tokens.weight".into())),
        "output_norm.weight" => return Ok(Some("model.norm.weight".into())),
        "output.weight" => return Ok(Some("lm_head.weight".into())),
        "rope_freqs.weight" => return Ok(None),
        _ => {}
    }
    let unknown = || Error::new("unsupported_model", format!("GGUF: unknown qwen3vl tensor {g:?}"));
    let rest = g.strip_prefix("blk.").ok_or_else(unknown)?;
    let (layer, rest) = rest.split_once('.').ok_or_else(unknown)?;
    let li: usize = layer.parse().map_err(|_| unknown())?;
    let (stem, suffix) = rest.rsplit_once('.').ok_or_else(unknown)?;
    let n = match stem {
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
        _ => return Err(unknown()),
    };
    Ok(Some(format!("model.layers.{li}.{n}.{suffix}")))
}

pub(crate) fn convert(path: &Path, mmproj_path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    let f = GgufFile::open(path)?;
    if !is_qwen3vl_gguf(&f) {
        bail!("unsupported_model", "{}: GGUF architecture {:?} is not qwen3vl", path.display(), f.architecture());
    }
    let mp = mmproj::open(mmproj_path)?;
    let (vision, processor) = mmproj::vision_config(&mp)?;
    let tokens = strings(&f, "tokenizer.ggml.tokens")?;
    let tok_id = |s: &str| tokens.iter().position(|t| t == s).map(|i| i as u32);
    let text = text_config(&f)?;
    let id_of = |s: &str| tok_id(s).ok_or_else(|| Error::new("missing_metadata", format!("GGUF vocabulary has no {s}")));
    let eos: Vec<u32> = f.get("tokenizer.ggml.eos_token_id").and_then(GgufValue::as_u64).map(|x| x as u32).into_iter().collect();
    let cfg = Qwen3VlConfig {
        text,
        vision,
        processor,
        image_token_id: id_of("<|image_pad|>")?,
        vision_start_token_id: id_of("<|vision_start|>")?,
        vision_end_token_id: id_of("<|vision_end|>")?,
        eos: eos.clone(),
    };
    let mut special = Map::new();
    if let Some(s) = eos.first().and_then(|&i| tokens.get(i as usize)) {
        special.insert("eos_token".into(), json!(s));
    }
    if let Some(s) = f.get("tokenizer.ggml.padding_token_id").and_then(GgufValue::as_u64).and_then(|i| tokens.get(i as usize)) {
        special.insert("pad_token".into(), json!(s));
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
    };

    let mut mapped = Vec::new();
    let mut have = BTreeSet::new();
    for t in &f.tensors {
        if let Some(n) = out_name(&t.name)? {
            have.insert(n.clone());
            mapped.push((n, t));
        }
    }
    lapid_convert::writer::check_complete(&have, &text_tensors(&cfg))?;
    let vision_names: BTreeSet<String> = mp.tensors.iter().map(|t| t.name.clone()).filter(|n| n != "v.patch_embd.weight.1").collect();
    lapid_convert::writer::check_complete(&vision_names, &cfg.vision.expected_tensors(false, false))?;

    let expected = expected_tensors(&cfg);
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("qwen3vl").to_string();
    let source = json!({
        "repo": f.get("general.name").and_then(GgufValue::as_str).map_or_else(|| name.clone(), |n| format!("gguf:{n}")),
        "file": path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
        "mmproj": mmproj_path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
        "format": "gguf",
        "file_hash": file_hash(path)?,
    });
    let meta = metadata(&name, dtype, source, &cfg, &assets);
    let mut w = ModelWriter::create(out, meta, dtype, role, expected.len())?;
    w.set_layer_count(Some(cfg.text.n_layers));
    add_blobs(&mut w, &assets)?;
    for (n, t) in &mapped {
        let matmul = t.shape.len() == 2 && !n.contains("norm");
        mmproj::write_gguf_tensor(&f, &mut w, dtype, n, t, &t.shape.clone(), matmul)?;
    }
    mmproj::write_vision(&mp, &mut w, dtype, &cfg.vision)?;
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let n = |g: &str| out_name(g).ok().flatten();
        assert_eq!(n("blk.3.attn_q.weight").as_deref(), Some("model.layers.3.self_attn.q_proj.weight"));
        assert_eq!(n("blk.0.ffn_norm.weight").as_deref(), Some("model.layers.0.post_attention_layernorm.weight"));
        assert_eq!(n("output.weight").as_deref(), Some("lm_head.weight"));
        assert!(out_name("blk.0.mystery.weight").is_err());
    }
}
