//! Upstream DeepSeek V4 safetensors (single file or sharded) → `.lapid`.
//!
//! Tensor names are the upstream ones (see `convert/mod.rs`); routed experts
//! `layers.{i}.ffn.experts.{e}.w{1,2,3}.weight` are stacked, `attn.wo_a.weight` is reshaped
//! to `[G, o_lora, in]`, `ffn.gate.tid2eid` (i32 / i64) becomes i32, `mtp.*` is dropped.
//! Float tensors (bf16 / f16 / f32) only: the released FP8 / FP4 checkpoints carry
//! `F8_E8M0` block scales that `lapid_convert::safetensors` does not read yet (use a GGUF).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::safetensors::SafeTensors;
use lapid_convert::writer::{ModelWriter, check_complete, file_hash, read_json};
use lapid_convert::{download_hf, download_hf_optional as download_optional};
use serde_json::{Value, json};

use super::{Assets, add_blobs, add_f32, add_i32, expected_tensors, metadata, role};
use crate::deepseek4::config::Deepseek4Config;

struct Files {
    config: PathBuf,
    generation: Option<PathBuf>,
    tokenizer: PathBuf,
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
        weights,
        repo: repo.to_string(),
        remote: true,
    })
}

fn read_text(p: &Path) -> Result<String> {
    std::fs::read_to_string(p).map_err(|e| Error::new("io", format!("{}: {e}", p.display())))
}

/// `(layers.{i}.ffn.experts.{w}, e)` of a per-expert upstream tensor.
fn per_expert(name: &str) -> Option<(String, usize)> {
    let (prefix, rest) = name.split_once(".ffn.experts.")?;
    let (e, part) = rest.split_once('.')?;
    let w = part.strip_suffix(".weight")?;
    if !matches!(w, "w1" | "w2" | "w3") {
        return None;
    }
    Some((format!("{prefix}.ffn.experts.{w}"), e.parse().ok()?))
}

fn i32_table(st: &SafeTensors, name: &str) -> Result<Vec<i32>> {
    let b = st.bytes(name)?;
    Ok(match st.dtype(name)?.as_str() {
        "I32" => b.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect(),
        "I64" => b.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c) as i32).collect(),
        other => bail!("unsupported_dtype", "{name}: hash routing table dtype {other} (expected I32 / I64)"),
    })
}

pub(crate) fn convert(src: &str, dtype: OutDType, out: &Path) -> Result<()> {
    let files = locate(src)?;
    let cfg = Deepseek4Config::from_json(&read_json(&files.config)?)?;
    let assets = Assets {
        tokenizer_json: read_text(&files.tokenizer)?,
        generation: match &files.generation {
            Some(p) => read_json(p)?,
            None => json!({}),
        },
    };
    let shards: Vec<SafeTensors> = files.weights.iter().map(|p| SafeTensors::open(p)).collect::<Result<_>>()?;
    let mut by_name: BTreeMap<String, usize> = BTreeMap::new();
    let mut experts: BTreeMap<String, BTreeMap<usize, (String, usize)>> = BTreeMap::new();
    for (i, st) in shards.iter().enumerate() {
        for n in st.names() {
            if n.starts_with("mtp.") {
                continue;
            }
            if n.ends_with(".scale") {
                bail!("unsupported_dtype", "{n}: FP8 / FP4 quantized upstream checkpoints are not supported; convert a GGUF instead");
            }
            if let Some(l) = n.strip_prefix("layers.").and_then(|r| r.split('.').next()).and_then(|x| x.parse::<usize>().ok())
                && l >= cfg.n_layers
            {
                continue;
            }
            match per_expert(&n) {
                Some((stack, e)) => {
                    experts.entry(stack).or_default().insert(e, (n, i));
                }
                None => {
                    by_name.insert(n, i);
                }
            }
        }
    }
    // hash layers ship an unused selection bias in some exports
    by_name.retain(|n, _| {
        !(n.ends_with(".ffn.gate.bias")
            && n.strip_prefix("layers.").and_then(|r| r.split('.').next()).and_then(|x| x.parse::<usize>().ok()).is_some_and(|l| l < cfg.n_hash_layers))
    });
    let expected = expected_tensors(&cfg);
    let mut have: BTreeSet<String> = by_name.keys().cloned().collect();
    have.extend(experts.keys().cloned());
    check_complete(&have, &expected)?;
    let wanted: BTreeSet<&String> = expected.iter().collect();
    by_name.retain(|n, _| wanted.contains(n));

    let name = files.repo.trim_end_matches('/').rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or("deepseek4").to_string();
    let mut source = json!({ "repo": files.repo });
    if files.remote {
        source["revision"] = json!("main");
    }
    if files.weights.len() == 1 {
        source["file_hash"] = json!(file_hash(&files.weights[0])?);
    }
    let meta = metadata(&name, dtype, source, &cfg, &assets);
    let mut w = ModelWriter::create(out, meta, dtype, role, expected.len())?;
    w.set_layer_count(Some(cfg.n_layers));
    add_blobs(&mut w, &assets)?;
    let (h, hd, g, ol) = (cfg.n_head, cfg.head_dim, cfg.o_groups, cfg.o_lora_rank);
    for (n, shard) in &by_name {
        let st = &shards[*shard];
        let shape = st.shape(n)?;
        if n.ends_with(".ffn.gate.tid2eid") {
            add_i32(&mut w, n, &shape, &i32_table(st, n)?)?;
        } else if n.ends_with(".attn.wo_a.weight") {
            if shape != [g * ol, h / g * hd] {
                bail!("shape_mismatch", "{n}: expected [{}, {}], got {shape:?}", g * ol, h / g * hd);
            }
            add_f32(&mut w, dtype, n, &[g, ol, h / g * hd], &st.to_f32(n)?)?;
        } else {
            add_f32(&mut w, dtype, n, &shape, &st.to_f32(n)?)?;
        }
    }
    for (stack, per) in &experts {
        let e = cfg.n_experts;
        if per.len() != e || per.keys().copied().ne(0..e) {
            bail!("missing_tensor", "{stack}: {} of {e} experts in the checkpoint", per.len());
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
        let mut shape = vec![e];
        shape.extend(&inner);
        add_f32(&mut w, dtype, stack, &shape, &data)?;
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_names() {
        assert_eq!(per_expert("layers.2.ffn.experts.7.w3.weight"), Some(("layers.2.ffn.experts.w3".into(), 7)));
        assert_eq!(per_expert("layers.2.ffn.shared_experts.w3.weight"), None);
        assert_eq!(per_expert("layers.2.ffn.experts.7.w3.scale"), None);
    }
}
