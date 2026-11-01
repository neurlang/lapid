//! Hugging Face Qwen3-VL (`Qwen3VLForConditionalGeneration` safetensors) → `.lapid`.
//!
//! Reads `config.json`, `preprocessor_config.json`, `generation_config.json`,
//! `tokenizer.json`, `tokenizer_config.json`, the chat template (`chat_template.jinja`,
//! `chat_template.json` or `tokenizer_config.json`) and the weights (single or sharded).
//!
//! | HF | lapid |
//! |---|---|
//! | `model.language_model.X` / `lm_head.weight` | `model.X` / `lm_head.weight` (untied only) |
//! | `model.visual.patch_embed.proj.{weight [D,C,T,P,P], bias}` | `v.patch_embd.{weight [D, C*P*P] (sum over T), bias}` |
//! | `model.visual.pos_embed.weight` | `v.position_embd.weight` |
//! | `model.visual.blocks.{i}.{norm1,norm2,attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}` | `v.blk.{i}.{ln1,ln2,attn_qkv,attn_out,ffn_up,ffn_down}` |
//! | `model.visual.merger.{norm,linear_fc1,linear_fc2}` | `v.post_ln`, `mm.0`, `mm.2` |
//! | `model.visual.deepstack_merger_list.{j}.{norm,linear_fc1,linear_fc2}` | `v.deepstack.{layer_j}.{norm,fc1,fc2}` |

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::safetensors::SafeTensors;
use lapid_convert::writer::{ModelWriter, check_complete, file_hash, read_json};
use lapid_convert::{download_hf, download_hf_optional as download_optional};
use serde_json::{Map, Value, json};

use super::{Assets, add_blobs, expected_tensors, metadata, role};
use crate::qwen3vl::config::Qwen3VlConfig;

struct Files {
    config: PathBuf,
    preprocessor: Option<PathBuf>,
    generation: Option<PathBuf>,
    tokenizer: PathBuf,
    tokenizer_config: Option<PathBuf>,
    template: Option<PathBuf>,
    template_json: Option<PathBuf>,
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
            preprocessor: opt("preprocessor_config.json"),
            generation: opt("generation_config.json"),
            tokenizer: need("tokenizer.json")?,
            tokenizer_config: opt("tokenizer_config.json"),
            template: opt("chat_template.jinja"),
            template_json: opt("chat_template.json"),
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
        preprocessor: download_optional(repo, "preprocessor_config.json")?,
        generation: download_optional(repo, "generation_config.json")?,
        tokenizer: download_hf(repo, "tokenizer.json")?,
        tokenizer_config: download_optional(repo, "tokenizer_config.json")?,
        template: download_optional(repo, "chat_template.jinja")?,
        template_json: download_optional(repo, "chat_template.json")?,
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

/// lapid name of an HF tensor (`None` = dropped). `deepstack` maps merger j → encoder layer.
pub(crate) fn lapid_name(hf: &str, deepstack: &[usize]) -> Option<String> {
    if hf == "lm_head.weight" {
        return Some(hf.to_string());
    }
    if let Some(rest) = hf.strip_prefix("model.language_model.") {
        return Some(format!("model.{rest}"));
    }
    let v = hf.strip_prefix("model.visual.").or_else(|| hf.strip_prefix("visual."))?;
    let (stem, suffix) = v.rsplit_once('.')?;
    let base = match stem {
        "patch_embed.proj" => "v.patch_embd".to_string(),
        "pos_embed" => "v.position_embd".to_string(),
        "merger.norm" => "v.post_ln".to_string(),
        "merger.linear_fc1" => "mm.0".to_string(),
        "merger.linear_fc2" => "mm.2".to_string(),
        _ => {
            if let Some(r) = stem.strip_prefix("blocks.") {
                let (i, part) = r.split_once('.')?;
                let p = match part {
                    "norm1" => "ln1",
                    "norm2" => "ln2",
                    "attn.qkv" => "attn_qkv",
                    "attn.proj" => "attn_out",
                    "mlp.linear_fc1" => "ffn_up",
                    "mlp.linear_fc2" => "ffn_down",
                    _ => return None,
                };
                format!("v.blk.{i}.{p}")
            } else {
                let r = stem.strip_prefix("deepstack_merger_list.")?;
                let (j, part) = r.split_once('.')?;
                let layer = deepstack.get(j.parse::<usize>().ok()?)?;
                let p = match part {
                    "norm" => "norm",
                    "linear_fc1" => "fc1",
                    "linear_fc2" => "fc2",
                    _ => return None,
                };
                format!("v.deepstack.{layer}.{p}")
            }
        }
    };
    Some(format!("{base}.{suffix}"))
}

pub(crate) fn convert(src: &str, dtype: OutDType, out: &Path) -> Result<()> {
    let files = locate(src)?;
    let cfg_json = read_json(&files.config)?;
    let mt = cfg_json.get("model_type").and_then(Value::as_str);
    if mt != Some("qwen3_vl") {
        bail!("unsupported_model", "{}: config.json model_type {mt:?} is not \"qwen3_vl\"", files.repo);
    }
    let pre = files.preprocessor.as_deref().map(read_json).transpose()?;
    let cfg = Qwen3VlConfig::from_json(&cfg_json, pre.as_ref())?;
    let tok_cfg = files.tokenizer_config.as_deref().map(read_json).transpose()?;
    let chat_template = if let Some(p) = &files.template {
        read_text(p)?
    } else if let Some(t) = files.template_json.as_deref().map(read_json).transpose()?.and_then(|j| j.get("chat_template").and_then(Value::as_str).map(str::to_string)) {
        t
    } else {
        tok_cfg
            .as_ref()
            .and_then(|c| c.get("chat_template"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| Error::new("not_found", format!("{}: no chat template", files.repo)))?
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
    let expected = expected_tensors(&cfg);
    let wanted: BTreeSet<&String> = expected.iter().collect();
    let mut by_name: BTreeMap<String, (usize, String)> = BTreeMap::new();
    for (i, st) in shards.iter().enumerate() {
        for n in st.names() {
            if let Some(g) = lapid_name(&n, &cfg.vision.deepstack_layers).filter(|g| wanted.contains(g)) {
                by_name.insert(g, (i, n));
            }
        }
    }
    let have: BTreeSet<String> = by_name.keys().cloned().collect();
    check_complete(&have, &expected)?;

    let name = files.repo.trim_end_matches('/').rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or("qwen3vl").to_string();
    let mut source = json!({ "repo": files.repo });
    if files.remote {
        source["revision"] = json!("main");
    }
    if files.weights.len() == 1 {
        source["file_hash"] = json!(file_hash(&files.weights[0])?);
    }
    let meta = metadata(&name, dtype, source, &cfg, &assets);
    let mut w = ModelWriter::create(out, meta, dtype, role, expected.len())?;
    w.set_layer_count(Some(cfg.text.n_layers));
    add_blobs(&mut w, &assets)?;
    for n in &expected {
        let (si, hf) = &by_name[n];
        let st = &shards[*si];
        let shape = st.shape(hf)?;
        let data = st.to_f32(hf)?;
        if n == "v.patch_embd.weight" {
            let (folded, shape) = fold_temporal(&data, &shape)?;
            w.add_f32(n, &shape, &folded)?;
        } else {
            w.add_f32(n, &shape, &data)?;
        }
    }
    w.finish(&expected)
}

/// Conv3d kernel [D, C, T, P, P] → [D, C*P*P] summed over T (a still image repeats its frame).
fn fold_temporal(data: &[f32], shape: &[usize]) -> Result<(Vec<f32>, Vec<usize>)> {
    let [d, c, t, p, q] = shape else {
        bail!("shape_mismatch", "patch_embed.proj.weight: expected [D, C, T, P, P], got {shape:?}");
    };
    let (d, c, t, pq) = (*d, *c, *t, p * q);
    let mut out = vec![0.0f32; d * c * pq];
    for o in 0..d {
        for ch in 0..c {
            for ti in 0..t {
                let src = ((o * c + ch) * t + ti) * pq;
                let dst = (o * c + ch) * pq;
                for k in 0..pq {
                    out[dst + k] += data[src + k];
                }
            }
        }
    }
    Ok((out, vec![d, c * pq]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let ds = [5, 11, 17];
        let n = |s: &str| lapid_name(s, &ds);
        assert_eq!(n("model.language_model.layers.3.mlp.up_proj.weight").as_deref(), Some("model.layers.3.mlp.up_proj.weight"));
        assert_eq!(n("model.visual.blocks.2.attn.qkv.bias").as_deref(), Some("v.blk.2.attn_qkv.bias"));
        assert_eq!(n("model.visual.merger.linear_fc2.weight").as_deref(), Some("mm.2.weight"));
        assert_eq!(n("model.visual.deepstack_merger_list.1.norm.weight").as_deref(), Some("v.deepstack.11.norm.weight"));
        assert_eq!(n("model.visual.patch_embed.proj.weight").as_deref(), Some("v.patch_embd.weight"));
        let (f, s) = fold_temporal(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 1, 2]).expect("fold");
        assert_eq!((f, s), (vec![4.0, 6.0], vec![1, 2]));
    }
}
