//! Upstream checkpoint files and the text assets every language model carries.
//!
//! * [`HfFiles`]: a Hugging Face repo id (downloaded into the cache) or local directory:
//!   `config.json`, the safetensors weights (single file or sharded index) and the optional
//!   tokenizer / template / generation files.
//! * [`TextAssets`]: `tokenizer.json`, the chat template, special token strings and
//!   generation defaults, read from HF files or rebuilt from GGUF metadata, stored as two
//!   blobs plus keys of the model's metadata object.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use lapid_tensor::{Error, Result, bail};
use serde_json::{Map, Value, json};

use crate::gguf::{GgufFile, GgufValue};
use crate::writer::{ModelWriter, read_json};
use crate::{download_hf, download_hf_optional};

/// Blob names of the text assets (also recorded in the metadata object).
pub const TOKENIZER_BLOB: &str = "tokenizer/tokenizer.json";
pub const TEMPLATE_BLOB: &str = "tokenizer/chat_template.jinja";

/// Llama 3 / LFM2 pre-tokenizer split.
pub const LLAMA3_SPLIT: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
/// Qwen3.5 pre-tokenizer split (Qwen2's with combining marks kept inside words).
pub const QWEN35_SPLIT: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
/// Qwen 2 / 3 pre-tokenizer split.
pub const QWEN2_SPLIT: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// Files of a Hugging Face checkpoint.
#[derive(Clone, Debug)]
pub struct HfFiles {
    /// Repo id or directory (as given, trimmed).
    pub repo: String,
    /// Downloaded from the hub (else a local directory).
    pub remote: bool,
    pub config: PathBuf,
    pub weights: Vec<PathBuf>,
    dir: Option<PathBuf>,
}

fn shard_files(index: &Value) -> Result<Vec<String>> {
    let map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| Error::new("format", "model.safetensors.index.json has no weight_map"))?;
    let files: BTreeSet<String> = map.values().filter_map(|v| v.as_str().map(str::to_string)).collect();
    if files.is_empty() {
        bail!("format", "model.safetensors.index.json lists no shards");
    }
    Ok(files.into_iter().collect())
}

impl HfFiles {
    /// Locate (download when `src` is a repo id) config and weights.
    pub fn locate(src: &str) -> Result<HfFiles> {
        let dir = Path::new(src);
        if dir.is_dir() {
            let opt = |f: &str| Some(dir.join(f)).filter(|p| p.is_file());
            let weights = if let Some(p) = opt("model.safetensors") {
                vec![p]
            } else if let Some(idx) = opt("model.safetensors.index.json") {
                shard_files(&read_json(&idx)?)?.iter().map(|f| dir.join(f)).collect()
            } else {
                bail!("not_found", "{}: no model.safetensors", dir.display());
            };
            let config = opt("config.json").ok_or_else(|| Error::new("not_found", format!("{}: no config.json", dir.display())))?;
            return Ok(HfFiles { repo: dir.display().to_string(), remote: false, config, weights, dir: Some(dir.to_path_buf()) });
        }
        let repo = src.trim_matches('/');
        let weights = match download_hf_optional(repo, "model.safetensors")? {
            Some(p) => vec![p],
            None => {
                let idx = download_hf_optional(repo, "model.safetensors.index.json")?
                    .ok_or_else(|| Error::new("not_found", format!("{repo}: no model.safetensors in repo")))?;
                shard_files(&read_json(&idx)?)?.iter().map(|f| download_hf(repo, f)).collect::<Result<_>>()?
            }
        };
        Ok(HfFiles { repo: repo.to_string(), remote: true, config: download_hf(repo, "config.json")?, weights, dir: None })
    }

    /// An optional file of the checkpoint (downloaded on demand).
    pub fn optional(&self, name: &str) -> Result<Option<PathBuf>> {
        match &self.dir {
            Some(d) => Ok(Some(d.join(name)).filter(|p| p.is_file())),
            None => download_hf_optional(&self.repo, name),
        }
    }

    /// A required file of the checkpoint.
    pub fn required(&self, name: &str) -> Result<PathBuf> {
        self.optional(name)?.ok_or_else(|| Error::new("not_found", format!("{}: no {name}", self.repo)))
    }

    /// Model name: the last path component of the repo.
    pub fn name(&self, fallback: &str) -> String {
        self.repo.trim_end_matches('/').rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or(fallback).to_string()
    }
}

fn read_text(p: &Path) -> Result<String> {
    std::fs::read_to_string(p).map_err(|e| Error::new("io", format!("{}: {e}", p.display())))
}

/// A special token entry: a string or `{"content": ...}`.
fn token_str(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("content").and_then(Value::as_str).map(str::to_string),
        _ => None,
    }
}

/// `bos_token`, `eos_token`, `pad_token`, `unk_token` strings (tokenizer_config wins).
pub fn special_tokens(tok_cfg: Option<&Value>, special_map: Option<&Value>) -> Value {
    let mut m = Map::new();
    for key in ["bos_token", "eos_token", "pad_token", "unk_token"] {
        let v = tok_cfg.and_then(|c| c.get(key)).and_then(token_str).or_else(|| special_map.and_then(|c| c.get(key)).and_then(token_str));
        if let Some(s) = v {
            m.insert(key.to_string(), json!(s));
        }
    }
    Value::Object(m)
}

/// Tokenizer, chat template, template special tokens and generation defaults.
#[derive(Clone, Debug, Default)]
pub struct TextAssets {
    pub tokenizer_json: String,
    pub chat_template: String,
    /// Template globals (`bos_token`, `eos_token`, `pad_token`, ...).
    pub special_tokens: Value,
    /// `generation_config.json`-style sampling defaults.
    pub generation: Value,
}

fn gguf_strings(f: &GgufFile, k: &str) -> Result<Vec<String>> {
    f.get(k)
        .and_then(GgufValue::as_array)
        .ok_or_else(|| Error::new("missing_metadata", format!("GGUF: missing array {k:?}")))?
        .iter()
        .map(|v| v.as_str().map(str::to_string).ok_or_else(|| Error::new("metadata", format!("GGUF: {k:?} holds a non-string"))))
        .collect()
}

/// Hugging Face `tokenizer.json` equivalent of a GGUF byte-level BPE (`gpt2`) tokenizer:
/// normal tokens (type 1) form the vocab, control (3) / user-defined (4) tokens are added
/// tokens (special / not special), unused (5) placeholders are dropped. Pre-tokenizers:
/// `lfm2`, `lfm3`, `llama-bpe`, `llama3` (Llama 3 split), `qwen2` and `qwen35` (Qwen splits, NFC).
pub fn gguf_tokenizer_json(f: &GgufFile) -> Result<Value> {
    let model = f.get("tokenizer.ggml.model").and_then(GgufValue::as_str).unwrap_or("");
    if model == "gemma4" {
        return gemma4_tokenizer_json(f);
    }
    let pre = f.get("tokenizer.ggml.pre").and_then(GgufValue::as_str).unwrap_or("");
    let (split, normalizer, ignore_merges) = match (model, pre) {
        ("gpt2", "lfm2" | "lfm3" | "llama-bpe" | "llama3") => (LLAMA3_SPLIT, Value::Null, true),
        ("gpt2", "qwen2") => (QWEN2_SPLIT, json!({"type": "NFC"}), false),
        ("gpt2", "qwen35") => (QWEN35_SPLIT, json!({"type": "NFC"}), false),
        _ => bail!("unsupported_model", "GGUF tokenizer {model:?} / pre-tokenizer {pre:?} is not supported"),
    };
    let tokens = gguf_strings(f, "tokenizer.ggml.tokens")?;
    let merges = gguf_strings(f, "tokenizer.ggml.merges")?;
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
    let byte_level = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false});
    Ok(json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": normalizer,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": split}, "behavior": "Isolated", "invert": false},
            byte_level,
        ]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true},
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                  "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": ignore_merges,
                  "vocab": vocab, "merges": merges},
    }))
}

/// Gemma 4 GGUF tokenizer (`tokenizer.ggml.model = "gemma4"`) as google/gemma-4-*'s own
/// `tokenizer.json`: BPE with byte fallback over text whose spaces are `▁`; control (3) /
/// user-defined (4) tokens are special added tokens.
fn gemma4_tokenizer_json(f: &GgufFile) -> Result<Value> {
    let tokens = gguf_strings(f, "tokenizer.ggml.tokens")?;
    let merges: Vec<Value> = gguf_strings(f, "tokenizer.ggml.merges")?
        .into_iter()
        .map(|m| match m.split_once(' ') {
            Some((a, b)) => Ok(json!([a, b])),
            None => Err(Error::new("metadata", format!("GGUF: malformed merge {m:?}"))),
        })
        .collect::<Result<_>>()?;
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
        vocab.insert(tok.clone(), json!(id));
        if matches!(ty, 3 | 4) {
            added.push(json!({"id": id, "content": tok, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}));
        }
    }
    let unk = f.get("tokenizer.ggml.unknown_token_id").and_then(GgufValue::as_u64).and_then(|i| tokens.get(i as usize).cloned());
    let seq = |a: &str, b: &str| json!([{"Sequence": {"id": a, "type_id": 0}}, {"Sequence": {"id": b, "type_id": 1}}]);
    Ok(json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"},
        "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false},
        "post_processor": {"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A", "type_id": 0}}], "pair": seq("A", "B"), "special_tokens": {}},
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"},
        ]},
        "model": {"type": "BPE", "dropout": null, "unk_token": unk, "continuing_subword_prefix": null, "end_of_word_suffix": null,
                  "fuse_unk": true, "byte_fallback": true, "ignore_merges": false, "vocab": vocab, "merges": merges},
    }))
}

impl TextAssets {
    /// From `tokenizer.json`, `tokenizer_config.json` / `special_tokens_map.json`,
    /// `chat_template.jinja` (or `tokenizer_config.json` `chat_template`) and
    /// `generation_config.json` (optional).
    pub fn from_hf(files: &HfFiles) -> Result<TextAssets> {
        let tok_cfg = files.optional("tokenizer_config.json")?.as_deref().map(read_json).transpose()?;
        let special_map = files.optional("special_tokens_map.json")?.as_deref().map(read_json).transpose()?;
        let chat_template = match files.optional("chat_template.jinja")? {
            Some(p) => read_text(&p)?,
            // a string, or the named-template list (its "default" entry)
            None => tok_cfg
                .as_ref()
                .and_then(|c| c.get("chat_template"))
                .and_then(|t| match t {
                    Value::Array(a) => a.iter().find(|e| e.get("name").and_then(Value::as_str) == Some("default")).and_then(|e| e.get("template")),
                    t => Some(t),
                })
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| Error::new("not_found", format!("{}: no chat template", files.repo)))?,
        };
        let generation = match files.optional("generation_config.json")? {
            Some(p) => read_json(&p)?,
            None => json!({}),
        };
        Ok(TextAssets {
            tokenizer_json: read_text(&files.required("tokenizer.json")?)?,
            chat_template,
            special_tokens: special_tokens(tok_cfg.as_ref(), special_map.as_ref()),
            generation,
        })
    }

    /// From GGUF metadata: rebuilt tokenizer, `tokenizer.chat_template`, special token
    /// strings of the bos / eos / padding ids, `general.sampling.*` defaults.
    pub fn from_gguf(f: &GgufFile) -> Result<TextAssets> {
        let tokens = f.get("tokenizer.ggml.tokens").and_then(GgufValue::as_array);
        let name = |k: &str| -> Option<String> {
            let id = f.get(&format!("tokenizer.ggml.{k}"))?.as_u64()? as usize;
            tokens?.get(id)?.as_str().map(str::to_string)
        };
        let mut special = Map::new();
        for (k, id) in [("bos_token", "bos_token_id"), ("eos_token", "eos_token_id"), ("pad_token", "padding_token_id"), ("unk_token", "unknown_token_id")] {
            if let Some(s) = name(id) {
                special.insert(k.to_string(), json!(s));
            }
        }
        let round = |x: f64| (x * 1e4).round() / 1e4;
        let float = |k: &str| f.get(k).and_then(GgufValue::as_f64);
        let mut generation = json!({});
        if let Some(t) = float("general.sampling.temp") {
            generation = json!({"do_sample": true, "temperature": round(t)});
            if let Some(k) = f.get("general.sampling.top_k").and_then(GgufValue::as_u64) {
                generation["top_k"] = json!(k);
            }
            if let Some(p) = float("general.sampling.top_p") {
                generation["top_p"] = json!(round(p));
            }
            if let Some(r) = float("general.sampling.penalty_repeat") {
                generation["repetition_penalty"] = json!(round(r));
            }
        }
        Ok(TextAssets {
            tokenizer_json: serde_json::to_string(&gguf_tokenizer_json(f)?)?,
            chat_template: f
                .get("tokenizer.chat_template")
                .and_then(GgufValue::as_str)
                .map(str::to_string)
                .ok_or_else(|| Error::new("missing_metadata", "GGUF: no tokenizer.chat_template"))?,
            special_tokens: Value::Object(special),
            generation,
        })
    }

    /// Record the assets in the model's metadata object (`special_tokens`, `generation`,
    /// `tokenizer_blob`, `chat_template_blob`).
    pub fn annotate(&self, cfg: &mut Value) {
        cfg["special_tokens"] = self.special_tokens.clone();
        cfg["generation"] = self.generation.clone();
        cfg["tokenizer_blob"] = json!(TOKENIZER_BLOB);
        cfg["chat_template_blob"] = json!(TEMPLATE_BLOB);
    }

    pub fn write_blobs(&self, w: &mut ModelWriter) -> Result<()> {
        w.add_blob(TOKENIZER_BLOB, "application/json", self.tokenizer_json.as_bytes())?;
        w.add_blob(TEMPLATE_BLOB, "text/x-jinja2", self.chat_template.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_token_strings() {
        let tc = json!({"bos_token": "<|startoftext|>", "eos_token": {"content": "<|im_end|>"}});
        let sm = json!({"pad_token": "<|pad|>", "eos_token": "ignored"});
        let s = special_tokens(Some(&tc), Some(&sm));
        assert_eq!(s, json!({"bos_token": "<|startoftext|>", "eos_token": "<|im_end|>", "pad_token": "<|pad|>"}));
    }
}
