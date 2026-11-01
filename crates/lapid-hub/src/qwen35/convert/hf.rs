//! Hugging Face Qwen3.5 (safetensors, single file or sharded) → `.lapid`.
//!
//! Keeps the text decoder (`model.language_model.*` or `model.*`, `lm_head`), drops the
//! vision tower (`model.visual.*`) and the MTP head (`mtp.*`). Value transforms: RMSNorm
//! `w → 1 + w` (not the DeltaNet gated norm), `A_log → A = -exp(A_log)`, v heads grouped →
//! tiled, fused `experts.gate_up_proj [E, 2I, D]` split into gate / up stacks.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::safetensors::SafeTensors;
use lapid_convert::writer::{ModelWriter, check_complete, file_hash, read_json};
use lapid_convert::{download_hf, download_hf_optional as download_optional};
use serde_json::{Map, Value, json};

use super::{Assets, add_blobs, add_f32, expected_tensors, metadata, reorder_v_heads, role};
use crate::qwen35::config::Qwen35Config;

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

/// Output name of an HF tensor, or `None` when it is dropped.
/// (name, shape, f32 data) of one output tensor.
type Named = (String, Vec<usize>, Vec<f32>);

fn out_name(hf: &str) -> Option<String> {
    if hf.starts_with("mtp.") || hf.contains(".visual.") || hf.starts_with("visual.") {
        return None;
    }
    if hf == "lm_head.weight" {
        return Some(hf.to_string());
    }
    let rest = hf.strip_prefix("model.language_model.").or_else(|| hf.strip_prefix("model."))?;
    let n = format!("model.{rest}");
    Some(match n.strip_suffix(".linear_attn.A_log") {
        Some(p) => format!("{p}.linear_attn.A"),
        None => n,
    })
}

/// Apply the value transforms of the module docs; returns (name, shape, data) outputs.
fn transform(cfg: &Qwen35Config, name: &str, shape: Vec<usize>, mut data: Vec<f32>) -> Result<Vec<Named>> {
    let (hk, hv, dk, dv) = (cfg.linear_k_heads, cfg.linear_v_heads, cfg.linear_k_dim, cfg.linear_v_dim);
    let r = hv / hk.max(1);
    let rows = shape.first().copied().unwrap_or(1);
    let cols = if shape.len() >= 2 { shape[1..].iter().product() } else { 1 };
    if name.ends_with("norm.weight") && !name.ends_with("linear_attn.norm.weight") {
        data.iter_mut().for_each(|x| *x += 1.0);
    }
    if name.ends_with(".linear_attn.A") {
        data.iter_mut().for_each(|x| *x = -x.exp());
    }
    if r > 1 && name.contains(".linear_attn.") {
        let kd = 2 * hk * dk;
        if name.ends_with("in_proj_qkv.weight") || name.ends_with("conv1d.weight") {
            let v = reorder_v_heads(&data[kd * cols..], rows - kd, cols, 0, hk, r, dv);
            data.truncate(kd * cols);
            data.extend(v);
        } else if name.ends_with("in_proj_z.weight") {
            data = reorder_v_heads(&data, rows, cols, 0, hk, r, dv);
        } else if name.ends_with("in_proj_a.weight") || name.ends_with("in_proj_b.weight") || name.ends_with(".A") || name.ends_with(".dt_bias") {
            data = reorder_v_heads(&data, rows, cols, 0, hk, r, 1);
        } else if name.ends_with("out_proj.weight") {
            data = reorder_v_heads(&data, rows, cols, 1, hk, r, dv);
        }
    }
    if let Some(p) = name.strip_suffix(".mlp.experts.gate_up_proj") {
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
        return Ok(vec![
            (format!("{p}.mlp.experts.gate_proj"), vec![e, i, d], gate),
            (format!("{p}.mlp.experts.up_proj"), vec![e, i, d], up),
        ]);
    }
    Ok(vec![(name.to_string(), shape, data)])
}

/// `(prefix.mlp.experts.{part}, expert)` of a per-expert tensor
/// `{prefix}.mlp.experts.{e}.{part}.weight` (the layout `save_pretrained` writes).
fn per_expert(name: &str) -> Option<(String, usize)> {
    let (prefix, rest) = name.split_once(".mlp.experts.")?;
    let (e, part) = rest.split_once('.')?;
    let part = part.strip_suffix(".weight")?;
    if !matches!(part, "gate_proj" | "up_proj" | "down_proj") {
        return None;
    }
    Some((format!("{prefix}.mlp.experts.{part}"), e.parse().ok()?))
}

pub(crate) fn convert(src: &str, dtype: OutDType, out: &Path) -> Result<()> {
    let files = locate(src)?;
    let cfg_json = read_json(&files.config)?;
    let cfg = Qwen35Config::from_json(&cfg_json)?;
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
    let assets = Assets {
        tokenizer_json: read_text(&files.tokenizer)?,
        chat_template,
        special_tokens: special_tokens(tok_cfg.as_ref()),
        generation: match &files.generation {
            Some(p) => read_json(p)?,
            None => json!({}),
        },
    };

    let shards: Vec<SafeTensors> = files.weights.iter().map(|p| SafeTensors::open(p)).collect::<Result<_>>()?;
    // output-ish name (before expert splitting) → (hf name, shard)
    let mut by_name: BTreeMap<String, (String, usize)> = BTreeMap::new();
    // stacked name → expert → (hf name, shard)
    let mut experts: BTreeMap<String, BTreeMap<usize, (String, usize)>> = BTreeMap::new();
    for (i, st) in shards.iter().enumerate() {
        for n in st.names() {
            if n == "lm_head.weight" && cfg.tie_embeddings {
                continue;
            }
            if let Some(o) = out_name(&n) {
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

    let name = files.repo.trim_end_matches('/').rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or("qwen35").to_string();
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
    for (n, (hf, shard)) in &by_name {
        let st = &shards[*shard];
        for (on, shape, data) in transform(&cfg, n, st.shape(hf)?, st.to_f32(hf)?)? {
            add_f32(&mut w, dtype, &on, &shape, &data)?;
        }
    }
    for (stack, per) in &experts {
        let n_exp = cfg.moe.as_ref().map_or(0, |m| m.n_experts);
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
        assert_eq!(out_name("model.language_model.layers.0.linear_attn.A_log").as_deref(), Some("model.layers.0.linear_attn.A"));
        assert_eq!(out_name("model.language_model.embed_tokens.weight").as_deref(), Some("model.embed_tokens.weight"));
        assert_eq!(out_name("model.layers.3.self_attn.q_proj.weight").as_deref(), Some("model.layers.3.self_attn.q_proj.weight"));
        assert_eq!(out_name("model.visual.blocks.0.attn.qkv.weight"), None);
        assert_eq!(out_name("mtp.fc.weight"), None);
        assert_eq!(out_name("lm_head.weight").as_deref(), Some("lm_head.weight"));
        assert_eq!(per_expert("model.layers.2.mlp.experts.7.up_proj.weight"), Some(("model.layers.2.mlp.experts.up_proj".into(), 7)));
        assert_eq!(per_expert("model.layers.2.mlp.experts.down_proj"), None);
    }
}
