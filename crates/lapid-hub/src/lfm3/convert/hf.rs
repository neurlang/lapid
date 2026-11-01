//! Hugging Face LFM3 (`Lfm3ForCausalLM` safetensors) → `.lapid`.
//!
//! Reads `config.json`, `generation_config.json` (optional), `tokenizer.json`,
//! `tokenizer_config.json` / `special_tokens_map.json` (special token strings),
//! `chat_template.jinja` (or `tokenizer_config.json` `chat_template`) and the safetensors
//! weights (single file or sharded index), streaming tensors through the dtype policy.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::safetensors::SafeTensors;
use lapid_convert::writer::{ModelWriter, check_complete, file_hash, read_json};
use lapid_convert::{download_hf, download_hf_optional as download_optional};
use serde_json::{Map, Value, json};

use super::{Assets, LICENSE, add_blobs, expected_tensors, metadata, role};
use crate::lfm3::config::Lfm3Config;

struct Files {
    config: PathBuf,
    generation: Option<PathBuf>,
    tokenizer: PathBuf,
    tokenizer_config: Option<PathBuf>,
    special_map: Option<PathBuf>,
    template: Option<PathBuf>,
    weights: Vec<PathBuf>,
    repo: String,
    remote: bool,
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

fn locate(src: &str) -> Result<Files> {
    let dir = Path::new(src);
    if dir.is_dir() {
        let opt = |f: &str| Some(dir.join(f)).filter(|p| p.is_file());
        let need = |f: &str| opt(f).ok_or_else(|| Error::new("not_found", format!("{}: no {f}", dir.display())));
        let weights = if let Some(p) = opt("model.safetensors") {
            vec![p]
        } else if let Some(idx) = opt("model.safetensors.index.json") {
            shard_files(&read_json(&idx)?)?.iter().map(|f| dir.join(f)).collect()
        } else {
            bail!("not_found", "{}: no model.safetensors", dir.display());
        };
        return Ok(Files {
            config: need("config.json")?,
            generation: opt("generation_config.json"),
            tokenizer: need("tokenizer.json")?,
            tokenizer_config: opt("tokenizer_config.json"),
            special_map: opt("special_tokens_map.json"),
            template: opt("chat_template.jinja"),
            weights,
            repo: dir.display().to_string(),
            remote: false,
        });
    }
    let repo = src.trim_matches('/');
    let weights = match download_optional(repo, "model.safetensors")? {
        Some(p) => vec![p],
        None => {
            let idx = download_optional(repo, "model.safetensors.index.json")?
                .ok_or_else(|| Error::new("not_found", format!("{repo}: no model.safetensors in repo")))?;
            let mut v = Vec::new();
            for f in shard_files(&read_json(&idx)?)? {
                v.push(download_hf(repo, &f)?);
            }
            v
        }
    };
    Ok(Files {
        config: download_hf(repo, "config.json")?,
        generation: download_optional(repo, "generation_config.json")?,
        tokenizer: download_hf(repo, "tokenizer.json")?,
        tokenizer_config: download_optional(repo, "tokenizer_config.json")?,
        special_map: download_optional(repo, "special_tokens_map.json")?,
        template: download_optional(repo, "chat_template.jinja")?,
        weights,
        repo: repo.to_string(),
        remote: true,
    })
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
pub(crate) fn special_tokens(tok_cfg: Option<&Value>, special_map: Option<&Value>) -> Value {
    let mut m = Map::new();
    for key in ["bos_token", "eos_token", "pad_token", "unk_token"] {
        let v = tok_cfg.and_then(|c| c.get(key)).and_then(token_str).or_else(|| special_map.and_then(|c| c.get(key)).and_then(token_str));
        if let Some(s) = v {
            m.insert(key.to_string(), json!(s));
        }
    }
    Value::Object(m)
}

fn read_text(p: &Path) -> Result<String> {
    std::fs::read_to_string(p).map_err(|e| Error::new("io", format!("{}: {e}", p.display())))
}

pub(crate) fn convert(src: &str, dtype: OutDType, out: &Path) -> Result<()> {
    let files = locate(src)?;
    let cfg_json = read_json(&files.config)?;
    if cfg_json.get("model_type").and_then(Value::as_str) != Some("lfm3") {
        bail!("unsupported_model", "{}: config.json model_type is not \"lfm3\"", files.repo);
    }
    let cfg = Lfm3Config::from_json(&cfg_json)?;
    let tok_cfg = files.tokenizer_config.as_deref().map(read_json).transpose()?;
    let special_map = files.special_map.as_deref().map(read_json).transpose()?;
    let chat_template = match &files.template {
        Some(p) => read_text(p)?,
        None => tok_cfg
            .as_ref()
            .and_then(|c| c.get("chat_template"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| Error::new("not_found", format!("{}: no chat template", files.repo)))?,
    };
    let generation = match &files.generation {
        Some(p) => read_json(p)?,
        None => json!({}),
    };
    let assets = Assets {
        tokenizer_json: read_text(&files.tokenizer)?,
        chat_template,
        special_tokens: special_tokens(tok_cfg.as_ref(), special_map.as_ref()),
        generation,
    };

    let shards: Vec<SafeTensors> = files.weights.iter().map(|p| SafeTensors::open(p)).collect::<Result<_>>()?;
    let mut by_name: BTreeMap<String, usize> = BTreeMap::new();
    for (i, st) in shards.iter().enumerate() {
        for n in st.names() {
            if n == "lm_head.weight" {
                continue; // tied to model.embed_tokens.weight
            }
            by_name.insert(n, i);
        }
    }
    let expected = expected_tensors(&cfg);
    let have: BTreeSet<String> = by_name.keys().cloned().collect();
    check_complete(&have, &expected)?;

    let name = files.repo.trim_end_matches('/').rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or("lfm3").to_string();
    let mut source = json!({ "repo": files.repo });
    if files.remote {
        source["revision"] = json!("main");
    }
    if files.weights.len() == 1 {
        source["file_hash"] = json!(file_hash(&files.weights[0])?);
    }
    let meta = metadata(&name, dtype, LICENSE, source, &cfg, &assets);
    let mut order = expected.clone();
    let expected_set: BTreeSet<&String> = expected.iter().collect();
    order.extend(by_name.keys().filter(|k| !expected_set.contains(k)).cloned());
    let mut w = ModelWriter::create(out, meta, dtype, role, order.len())?;
    add_blobs(&mut w, &assets)?;
    for n in &order {
        let st = &shards[by_name[n]];
        let shape = st.shape(n)?;
        w.add_f32(n, &shape, &st.to_f32(n)?)?;
    }
    w.finish(&expected)
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
