//! Hugging Face Gemma 4 (`Gemma4ForConditionalGeneration` / `Gemma4ForCausalLM` safetensors)
//! → `.lapid` (text decoder only).
//!
//! Reads `config.json` (`model_type` "gemma4" with `text_config`, or "gemma4_text"),
//! `generation_config.json`, `tokenizer.json`, `tokenizer_config.json` (special token
//! strings), `chat_template.jinja` (or `tokenizer_config.json` `chat_template`) and the
//! safetensors weights (single file or sharded), streaming the language-model tensors
//! through the dtype policy. `model.language_model.X` is stored as `model.X`; vision /
//! audio towers and the unused k/v weights of kv-shared layers are dropped.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::safetensors::SafeTensors;
use lapid_convert::writer::{ModelWriter, check_complete, file_hash, read_json};
use lapid_convert::{download_hf, download_hf_optional as download_optional};
use serde_json::{Map, Value, json};

use super::{Assets, add_blobs, expected_tensors, metadata, role};
use crate::gemma4::config::Gemma4Config;

struct Files {
    config: PathBuf,
    generation: Option<PathBuf>,
    tokenizer: PathBuf,
    tokenizer_config: Option<PathBuf>,
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
        template: download_optional(repo, "chat_template.jinja")?,
        weights,
        repo: repo.to_string(),
        remote: true,
    })
}

fn token_str(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("content").and_then(Value::as_str).map(str::to_string),
        _ => None,
    }
}

/// `bos_token`, `eos_token`, `pad_token`, `unk_token` strings from tokenizer_config.json.
fn special_tokens(tok_cfg: Option<&Value>) -> Value {
    let mut m = Map::new();
    for key in ["bos_token", "eos_token", "pad_token", "unk_token"] {
        if let Some(s) = tok_cfg.and_then(|c| c.get(key)).and_then(token_str) {
            m.insert(key.to_string(), json!(s));
        }
    }
    Value::Object(m)
}

fn read_text(p: &Path) -> Result<String> {
    std::fs::read_to_string(p).map_err(|e| Error::new("io", format!("{}: {e}", p.display())))
}

/// lapid name of an HF tensor name (`None` = not part of the text decoder).
fn lapid_name(hf: &str) -> Option<String> {
    if let Some(rest) = hf.strip_prefix("model.language_model.") {
        return Some(format!("model.{rest}"));
    }
    if hf.starts_with("model.layers.") || hf.starts_with("model.embed_tokens") || hf.starts_with("model.norm.") || hf.starts_with("model.per_layer") {
        return Some(hf.to_string());
    }
    None
}

pub(crate) fn convert(src: &str, dtype: OutDType, out: &Path) -> Result<()> {
    let files = locate(src)?;
    let cfg_json = read_json(&files.config)?;
    let mt = cfg_json.get("model_type").and_then(Value::as_str);
    if !matches!(mt, Some("gemma4" | "gemma4_text")) {
        bail!("unsupported_model", "{}: config.json model_type {mt:?} is not \"gemma4\"", files.repo);
    }
    let cfg = Gemma4Config::from_json(&cfg_json)?;
    let tok_cfg = files.tokenizer_config.as_deref().map(read_json).transpose()?;
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
        special_tokens: special_tokens(tok_cfg.as_ref()),
        generation,
    };

    let shards: Vec<SafeTensors> = files.weights.iter().map(|p| SafeTensors::open(p)).collect::<Result<_>>()?;
    let expected = expected_tensors(&cfg);
    let wanted: BTreeSet<&String> = expected.iter().collect();
    let mut by_name: BTreeMap<String, (usize, String)> = BTreeMap::new();
    for (i, st) in shards.iter().enumerate() {
        for n in st.names() {
            if let Some(g) = lapid_name(&n).filter(|g| wanted.contains(g)) {
                by_name.insert(g, (i, n));
            }
        }
    }
    let have: BTreeSet<String> = by_name.keys().cloned().collect();
    check_complete(&have, &expected)?;

    let name = files.repo.trim_end_matches('/').rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or("gemma4").to_string();
    let mut source = json!({ "repo": files.repo });
    if files.remote {
        source["revision"] = json!("main");
    }
    if files.weights.len() == 1 {
        source["file_hash"] = json!(file_hash(&files.weights[0])?);
    }
    let meta = metadata(&name, dtype, source, &cfg, &assets);
    let mut w = ModelWriter::create(out, meta, dtype, role, expected.len())?;
    add_blobs(&mut w, &assets)?;
    for n in &expected {
        let (si, hf) = &by_name[n];
        let st = &shards[*si];
        let shape = st.shape(hf)?;
        w.add_f32(n, &shape, &st.to_f32(hf)?)?;
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(lapid_name("model.language_model.layers.3.mlp.up_proj.weight").as_deref(), Some("model.layers.3.mlp.up_proj.weight"));
        assert_eq!(lapid_name("model.vision_tower.encoder.x"), None);
        assert_eq!(lapid_name("model.embed_vision.embedding_projection.weight"), None);
        let s = special_tokens(Some(&json!({"bos_token": "<bos>", "eos_token": {"content": "<eos>"}})));
        assert_eq!(s, json!({"bos_token": "<bos>", "eos_token": "<eos>"}));
    }
}
