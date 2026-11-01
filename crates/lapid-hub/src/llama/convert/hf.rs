//! Hugging Face Llama / Mistral / Qwen2 / Qwen3 / Qwen3-MoE (safetensors, single file or
//! sharded) → `.lapid`. Names are kept; per-expert tensors (`mlp.experts.{e}.{part}.weight`,
//! the `save_pretrained` layout) and fused `mlp.experts.gate_up_proj [E, 2I, D]` stacks become
//! `mlp.experts.{gate,up,down}_proj` stacks. No value transforms (Llama-family RMSNorm weights
//! are plain scales).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::safetensors::SafeTensors;
use lapid_convert::writer::{ModelWriter, check_complete, file_hash, read_json};
use lapid_convert::{download_hf, download_hf_optional as download_optional};
use serde_json::{Map, Value, json};

use super::{Assets, add_blobs, add_f32, card_license, expected_tensors, metadata, role};
use crate::llama::config::LlamaConfig;

struct Files {
    config: PathBuf,
    generation: Option<PathBuf>,
    tokenizer: PathBuf,
    tokenizer_config: Option<PathBuf>,
    template: Option<PathBuf>,
    readme: Option<PathBuf>,
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
            readme: opt("README.md"),
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
        readme: download_optional(repo, "README.md")?,
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

/// `bos_token`, `eos_token`, `pad_token`, `unk_token` strings from `tokenizer_config.json`.
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

/// Chat template: `chat_template.jinja`, else `tokenizer_config.json` `chat_template` (a string
/// or the named-template list, "default" entry).
fn chat_template(files: &Files, tok_cfg: Option<&Value>) -> Result<String> {
    if let Some(p) = &files.template {
        return read_text(p);
    }
    let t = tok_cfg.and_then(|c| c.get("chat_template"));
    let s = match t {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(a)) => a
            .iter()
            .find(|e| e.get("name").and_then(Value::as_str) == Some("default"))
            .and_then(|e| e.get("template").and_then(Value::as_str))
            .map(str::to_string),
        _ => None,
    };
    s.ok_or_else(|| Error::new("not_found", format!("{}: no chat template", files.repo)))
}

/// `(prefix.mlp.experts.{part}, expert)` of a per-expert tensor
/// `{prefix}.mlp.experts.{e}.{part}.weight`.
fn per_expert(name: &str) -> Option<(String, usize)> {
    let (prefix, rest) = name.split_once(".mlp.experts.")?;
    let (e, part) = rest.split_once('.')?;
    let part = part.strip_suffix(".weight")?;
    if !matches!(part, "gate_proj" | "up_proj" | "down_proj") {
        return None;
    }
    Some((format!("{prefix}.mlp.experts.{part}"), e.parse().ok()?))
}

/// Output name of an HF tensor (`None` = dropped: rotary buffers, MTP heads).
fn out_name(hf: &str) -> Option<String> {
    if hf.ends_with("rotary_emb.inv_freq") || hf.starts_with("mtp.") {
        return None;
    }
    if hf == "lm_head.weight" || hf.starts_with("model.") {
        return Some(hf.to_string());
    }
    None
}

type Named = (String, Vec<usize>, Vec<f32>);

/// Split a fused `[E, 2I, D]` gate_up stack into gate / up stacks.
fn split_gate_up(prefix: &str, name: &str, shape: &[usize], data: &[f32]) -> Result<Vec<Named>> {
    if shape.len() != 3 || !shape[1].is_multiple_of(2) {
        bail!("shape_mismatch", "{name}: expected [E, 2I, D], got {shape:?}");
    }
    let (e, i2, d) = (shape[0], shape[1], shape[2]);
    let i = i2 / 2;
    let (mut gate, mut up) = (Vec::with_capacity(e * i * d), Vec::with_capacity(e * i * d));
    for x in 0..e {
        let base = x * i2 * d;
        gate.extend_from_slice(&data[base..base + i * d]);
        up.extend_from_slice(&data[base + i * d..base + i2 * d]);
    }
    Ok(vec![
        (format!("{prefix}.mlp.experts.gate_proj"), vec![e, i, d], gate),
        (format!("{prefix}.mlp.experts.up_proj"), vec![e, i, d], up),
    ])
}

pub(crate) fn convert(src: &str, dtype: OutDType, out: &Path) -> Result<()> {
    let files = locate(src)?;
    let cfg = LlamaConfig::from_json(&read_json(&files.config)?)?;
    let tok_cfg = files.tokenizer_config.as_deref().map(read_json).transpose()?;
    let readme = files.readme.as_deref().map(read_text).transpose()?;
    let assets = Assets {
        tokenizer_json: read_text(&files.tokenizer)?,
        chat_template: chat_template(&files, tok_cfg.as_ref())?,
        special_tokens: special_tokens(tok_cfg.as_ref()),
        generation: match &files.generation {
            Some(p) => read_json(p)?,
            None => json!({}),
        },
        license: readme.as_deref().and_then(card_license).unwrap_or_else(|| "other".into()),
    };

    let shards: Vec<SafeTensors> = files.weights.iter().map(|p| SafeTensors::open(p)).collect::<Result<_>>()?;
    let mut by_name: BTreeMap<String, (String, usize)> = BTreeMap::new();
    let mut experts: BTreeMap<String, BTreeMap<usize, (String, usize)>> = BTreeMap::new();
    for (i, st) in shards.iter().enumerate() {
        for n in st.names() {
            if n == "lm_head.weight" && cfg.tie_embeddings {
                continue;
            }
            let Some(o) = out_name(&n) else { continue };
            match per_expert(&o) {
                Some((stack, e)) => {
                    experts.entry(stack).or_default().insert(e, (n, i));
                }
                None => {
                    by_name.insert(o, (n, i));
                }
            }
        }
    }
    let expected = expected_tensors(&cfg);
    let mut have: BTreeSet<String> = by_name.keys().cloned().collect();
    have.extend(experts.keys().cloned());
    for k in by_name.keys() {
        if let Some(p) = k.strip_suffix(".mlp.experts.gate_up_proj") {
            have.insert(format!("{p}.mlp.experts.gate_proj"));
            have.insert(format!("{p}.mlp.experts.up_proj"));
        }
    }
    check_complete(&have, &expected)?;

    let name = files.repo.trim_end_matches('/').rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or("llama").to_string();
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
    let wanted: BTreeSet<&String> = expected.iter().collect();
    for (n, (hf, shard)) in &by_name {
        let st = &shards[*shard];
        if let Some(p) = n.strip_suffix(".mlp.experts.gate_up_proj") {
            for (on, shape, data) in split_gate_up(p, n, &st.shape(hf)?, &st.to_f32(hf)?)? {
                add_f32(&mut w, dtype, &on, &shape, &data)?;
            }
        } else if wanted.contains(n) {
            add_f32(&mut w, dtype, n, &st.shape(hf)?, &st.to_f32(hf)?)?;
        }
    }
    let n_exp = cfg.moe.as_ref().map_or(0, |m| m.n_experts);
    for (stack, per) in &experts {
        if per.len() != n_exp || per.keys().copied().ne(0..n_exp) {
            bail!("missing_tensor", "{stack}: {} of {n_exp} experts in the checkpoint", per.len());
        }
        let mut data = Vec::new();
        let mut inner: Vec<usize> = Vec::new();
        for (hf, shard) in per.values() {
            let st = &shards[*shard];
            let shape = st.shape(hf)?;
            if !inner.is_empty() && inner != shape {
                bail!("shape_mismatch", "{stack}: experts disagree in shape ({inner:?} vs {shape:?})");
            }
            inner = shape;
            data.extend(st.to_f32(hf)?);
        }
        let mut shape = vec![n_exp];
        shape.extend(&inner);
        add_f32(&mut w, dtype, stack, &shape, &data)?;
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(out_name("model.layers.3.self_attn.q_proj.bias").as_deref(), Some("model.layers.3.self_attn.q_proj.bias"));
        assert_eq!(out_name("model.layers.0.self_attn.rotary_emb.inv_freq"), None);
        assert_eq!(out_name("lm_head.weight").as_deref(), Some("lm_head.weight"));
        assert_eq!(per_expert("model.layers.2.mlp.experts.7.up_proj.weight"), Some(("model.layers.2.mlp.experts.up_proj".into(), 7)));
        assert_eq!(per_expert("model.layers.2.mlp.experts.down_proj"), None);
    }
}
