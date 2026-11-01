//! Hugging Face Qwen4-Exp (safetensors, single file or sharded) → `.lapid`.
//!
//! Keeps the text decoder (`model.language_model.*` or `model.*`, `lm_head`), drops the vision
//! tower and the MTP head (`mtp.*`). Value transforms: RMSNorm `w → 1 + w` (not the DeltaNet
//! gated norm), grouped hyper-connection / PLE norms reshaped `[hc, D]`, `A_log → A = -exp(A_log)`,
//! DeltaNet v heads grouped → tiled, `indexer.index_qk_proj` split into q / k, fused
//! `experts.gate_up_proj [E, 2I, D]` split, the n-gram table shards concatenated in index order;
//! the hash-constant buffers are dropped (recomputed from the config, see `config`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::safetensors::SafeTensors;
use lapid_convert::writer::{ModelWriter, check_complete, file_hash, read_json};
use lapid_convert::{download_hf, download_hf_optional as download_optional};
use serde_json::{Map, Value, json};

use super::{Assets, add_blobs, add_f32, expected_tensors, is_grouped_norm, metadata, role};
use crate::qwen35::convert::reorder_v_heads;
use crate::qwen4exp::config::Qwen4ExpConfig;

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

/// (name, shape, f32 data) of one output tensor.
type Named = (String, Vec<usize>, Vec<f32>);

/// What an HF tensor becomes.
#[derive(Debug, PartialEq)]
enum Out {
    Drop,
    Tensor(String),
    /// Shard `k` of the n-gram table of the layer prefix.
    NgramShard(String, usize),
}

fn classify(hf: &str) -> Out {
    if hf.starts_with("mtp.") || hf.contains(".visual.") || hf.starts_with("visual.") {
        return Out::Drop;
    }
    if hf == "lm_head.weight" {
        return Out::Tensor(hf.to_string());
    }
    let Some(rest) = hf.strip_prefix("model.language_model.").or_else(|| hf.strip_prefix("model.")) else { return Out::Drop };
    let n = format!("model.{rest}");
    if let Some((prefix, tail)) = n.split_once(".ple.ple_embedding.") {
        if let Some(k) = tail.strip_prefix("ngram_embedding.shard_").and_then(|s| s.strip_suffix(".weight")).and_then(|s| s.parse().ok()) {
            return Out::NgramShard(prefix.to_string(), k);
        }
        if tail == "ngram_embedding.weight" {
            return Out::NgramShard(prefix.to_string(), 0);
        }
        return Out::Drop; // hash-constant buffers
    }
    let n = n
        .replace(".linear_attn.A_log", ".linear_attn.A")
        .replace(".indexer.q_layernorm.", ".indexer.q_norm.")
        .replace(".indexer.k_layernorm.", ".indexer.k_norm.");
    Out::Tensor(n)
}

/// Apply the value transforms of the module docs; returns (name, shape, data) outputs.
fn transform(cfg: &Qwen4ExpConfig, name: &str, shape: Vec<usize>, mut data: Vec<f32>) -> Result<Vec<Named>> {
    let b = &cfg.base;
    let (hk, hv, dk, dv) = (b.linear_k_heads, b.linear_v_heads, b.linear_k_dim, b.linear_v_dim);
    let r = hv / hk.max(1);
    let rows = shape.first().copied().unwrap_or(1);
    let cols = if shape.len() >= 2 { shape[1..].iter().product() } else { 1 };
    let is_norm = (name.ends_with("norm.weight") && !name.ends_with("linear_attn.norm.weight")) || name.contains(".ple.norm_");
    if is_norm {
        data.iter_mut().for_each(|x| *x += 1.0);
    }
    if is_grouped_norm(name) && shape.len() == 1 {
        return Ok(vec![(name.to_string(), vec![cfg.hc, rows / cfg.hc], data)]);
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
    if let Some(p) = name.strip_suffix(".indexer.index_qk_proj.weight") {
        let q = cfg.qsa.as_ref().ok_or_else(|| Error::new("metadata", "qwen4exp: indexer weights without indexer config"))?;
        let nq = q.n_heads * q.head_dim;
        if rows != nq + q.head_dim {
            bail!("shape_mismatch", "{name}: {rows} rows, expected {}", nq + q.head_dim);
        }
        let k = data.split_off(nq * cols);
        return Ok(vec![
            (format!("{p}.indexer.q_proj.weight"), vec![nq, cols], data),
            (format!("{p}.indexer.k_proj.weight"), vec![q.head_dim, cols], k),
        ]);
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
/// `{prefix}.mlp.experts.{e}.{part}.weight` (the layout `save_pretrained` may write).
fn per_expert(name: &str) -> Option<(String, usize)> {
    let (prefix, rest) = name.split_once(".mlp.experts.")?;
    let (e, part) = rest.split_once('.')?;
    let part = part.strip_suffix(".weight")?;
    if !matches!(part, "gate_proj" | "up_proj" | "down_proj") {
        return None;
    }
    Some((format!("{prefix}.mlp.experts.{part}"), e.parse().ok()?))
}

/// Output names a source name provides (for the completeness check).
fn provides(cfg: &Qwen4ExpConfig, n: &str) -> Vec<String> {
    if let Some(p) = n.strip_suffix(".mlp.experts.gate_up_proj") {
        return vec![format!("{p}.mlp.experts.gate_proj"), format!("{p}.mlp.experts.up_proj")];
    }
    if let (Some(p), true) = (n.strip_suffix(".indexer.index_qk_proj.weight"), cfg.qsa.is_some()) {
        return vec![format!("{p}.indexer.q_proj.weight"), format!("{p}.indexer.k_proj.weight")];
    }
    vec![n.to_string()]
}

pub(crate) fn convert(src: &str, dtype: OutDType, out: &Path) -> Result<()> {
    let files = locate(src)?;
    let cfg_json = read_json(&files.config)?;
    let cfg = Qwen4ExpConfig::from_json(&cfg_json)?;
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
    let mut by_name: BTreeMap<String, (String, usize)> = BTreeMap::new();
    let mut experts: BTreeMap<String, BTreeMap<usize, (String, usize)>> = BTreeMap::new();
    let mut ngram: BTreeMap<String, BTreeMap<usize, (String, usize)>> = BTreeMap::new();
    for (i, st) in shards.iter().enumerate() {
        for n in st.names() {
            if n == "lm_head.weight" && cfg.base.tie_embeddings {
                continue;
            }
            match classify(&n) {
                Out::Drop => {}
                Out::NgramShard(prefix, k) => {
                    ngram.entry(format!("{prefix}.ple.ngram_embedding.weight")).or_default().insert(k, (n, i));
                }
                Out::Tensor(o) => match per_expert(&o) {
                    Some((stack, e)) => {
                        experts.entry(stack).or_default().insert(e, (n, i));
                    }
                    None => {
                        by_name.insert(o, (n, i));
                    }
                },
            }
        }
    }
    let expected = expected_tensors(&cfg);
    let mut have: BTreeSet<String> = by_name.keys().flat_map(|k| provides(&cfg, k)).collect();
    have.extend(experts.keys().cloned());
    have.extend(ngram.keys().cloned());
    check_complete(&have, &expected)?;

    let name = files.repo.trim_end_matches('/').rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or("qwen4exp").to_string();
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
    for (table, parts) in &ngram {
        if parts.keys().copied().ne(0..parts.len()) {
            bail!("missing_tensor", "{table}: n-gram shards are not 0..{}", parts.len());
        }
        let (mut data, mut rows, mut cols) = (Vec::new(), 0, 0);
        for (hf, shard) in parts.values() {
            let st = &shards[*shard];
            let shape = st.shape(hf)?;
            if shape.len() != 2 || (cols != 0 && shape[1] != cols) {
                bail!("shape_mismatch", "{hf}: n-gram shard shape {shape:?}");
            }
            (rows, cols) = (rows + shape[0], shape[1]);
            data.extend(st.to_f32(hf)?);
        }
        add_f32(&mut w, dtype, table, &[rows, cols], &data)?;
    }
    for (stack, per) in &experts {
        let n_exp = cfg.moe()?.n_experts;
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
        let t = |s: &str| Out::Tensor(s.to_string());
        assert_eq!(classify("model.language_model.layers.0.linear_attn.A_log"), t("model.layers.0.linear_attn.A"));
        assert_eq!(classify("model.language_model.layers.3.self_attn.indexer.k_layernorm.weight"), t("model.layers.3.self_attn.indexer.k_norm.weight"));
        assert_eq!(classify("model.layers.1.ple.ple_embedding.ngram_embedding.shard_7.weight"), Out::NgramShard("model.layers.1".into(), 7));
        assert_eq!(classify("model.layers.1.ple.ple_embedding.layer_multipliers"), Out::Drop);
        assert_eq!(classify("model.visual.blocks.0.attn.qkv.weight"), Out::Drop);
        assert_eq!(classify("mtp.layers.0.mlp.gate.weight"), Out::Drop);
        assert_eq!(per_expert("model.layers.2.mlp.experts.7.up_proj.weight"), Some(("model.layers.2.mlp.experts.up_proj".into(), 7)));
    }
}
