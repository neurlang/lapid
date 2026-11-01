//! sentence-transformers EmbeddingGemma (HF repo or local directory) → one `.lapid` file.
//!
//! Reads the Gemma 3 text encoder (`config.json`, `model.safetensors`), the module list
//! (`modules.json`: Transformer → Pooling → Dense… → Normalize), each Dense module's
//! weights, the prompts (`config_sentence_transformers.json`), the max sequence length
//! (`sentence_bert_config.json`) and `tokenizer.json` (stored verbatim as a blob).
//!
//! Tensor names: encoder `model.<hf name>`, Dense module k `dense.<k>.weight` / `.bias`.
//! Components: `encoder` (gemma3_text, prefix `model.`), `pooling`, `dense_<k>`
//! (prefix `dense.<k>.`), `normalize`. Precision: projection and Dense weights follow the
//! dtype; the token embedding (a gather table) is at most int8; norms f32.

pub(crate) mod calibrate;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::policy::Role;
use lapid_convert::safetensors::SafeTensors;
use lapid_convert::writer::{ModelWriter, check_complete, file_hash, read_json};
use lapid_convert::{download_hf, download_hf_optional};
use serde_json::{Value, json};

use super::config::{DenseActivation, DenseConfig, EmbeddingGemmaConfig, META_KEY, PoolingConfig};

pub(crate) const TOKENIZER_BLOB: &str = "tokenizer/tokenizer.json";
const EMBED: &str = "model.embed_tokens.weight";
/// Matryoshka sizes EmbeddingGemma was trained with (model card).
const MATRYOSHKA: [usize; 4] = [768, 512, 256, 128];
const LICENSE: &str = "gemma";
const LICENSE_URL: &str = "https://ai.google.dev/gemma/terms";

/// Source files: repo id (downloaded on demand) or local directory.
struct Source {
    dir: Option<PathBuf>,
    repo: String,
}

impl Source {
    fn new(src: &str) -> Source {
        let p = Path::new(src);
        if p.is_dir() {
            Source { dir: Some(p.to_path_buf()), repo: p.display().to_string() }
        } else {
            Source { dir: None, repo: src.trim_matches('/').to_string() }
        }
    }

    fn required(&self, file: &str) -> Result<PathBuf> {
        match &self.dir {
            Some(d) => {
                let p = d.join(file);
                if p.is_file() { Ok(p) } else { bail!("not_found", "{}: missing {file}", d.display()) }
            }
            None => download_hf(&self.repo, file),
        }
    }

    fn optional(&self, file: &str) -> Result<Option<PathBuf>> {
        match &self.dir {
            Some(d) => Ok(Some(d.join(file)).filter(|p| p.is_file())),
            None => download_hf_optional(&self.repo, file),
        }
    }
}

/// Dtype-policy role of a tensor.
pub(crate) fn role(name: &str, shape: &[usize]) -> Role {
    if shape.len() <= 1 || name.ends_with("norm.weight") || name.ends_with(".bias") {
        Role::Param
    } else if name == EMBED {
        Role::LinearInt8
    } else {
        Role::Linear
    }
}

fn get_usize(cfg: &Value, key: &str) -> Result<usize> {
    cfg.get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| Error::new("missing_metadata", format!("config.json: missing integer {key:?}")))
}

fn get_f32(cfg: &Value, key: &str, default: Option<f32>) -> Result<f32> {
    match cfg.get(key).and_then(Value::as_f64) {
        Some(v) => Ok(v as f32),
        None => default.ok_or_else(|| Error::new("missing_metadata", format!("config.json: missing number {key:?}"))),
    }
}

/// Encoder part of the config from Gemma 3 `config.json`.
fn encoder_config(cfg: &Value) -> Result<EmbeddingGemmaConfig> {
    let n_layer = get_usize(cfg, "num_hidden_layers")?;
    let hidden_size = get_usize(cfg, "hidden_size")?;
    let n_head = get_usize(cfg, "num_attention_heads")?;
    let sliding_layers: Vec<bool> = match cfg.get("layer_types").and_then(Value::as_array) {
        Some(types) => types.iter().map(|t| t.as_str() == Some("sliding_attention")).collect(),
        None => {
            let pattern = cfg.get("sliding_window_pattern").or(cfg.get("_sliding_window_pattern")).and_then(Value::as_u64).unwrap_or(6) as usize;
            (0..n_layer).map(|i| (i + 1) % pattern.max(1) != 0).collect()
        }
    };
    if cfg.get("use_bidirectional_attention").and_then(Value::as_bool) != Some(true) {
        bail!("unsupported_model", "config.json: only bidirectional Gemma 3 encoders (EmbeddingGemma) are supported");
    }
    for k in ["attn_logit_softcapping", "final_logit_softcapping", "rope_scaling"] {
        if cfg.get(k).is_some_and(|v| !v.is_null()) {
            bail!("unsupported_model", "config.json: {k} is not supported");
        }
    }
    if cfg.get("attention_bias").and_then(Value::as_bool) == Some(true) {
        bail!("unsupported_model", "config.json: attention_bias is not supported");
    }
    let sliding_window = get_usize(cfg, "sliding_window")?;
    let head_dim = cfg.get("head_dim").and_then(Value::as_u64).map_or(hidden_size / n_head.max(1), |v| v as usize);
    Ok(EmbeddingGemmaConfig {
        hidden_size,
        n_layer,
        n_head,
        n_kv_head: get_usize(cfg, "num_key_value_heads")?,
        head_dim,
        intermediate_size: get_usize(cfg, "intermediate_size")?,
        vocab_size: get_usize(cfg, "vocab_size")?,
        rms_norm_eps: get_f32(cfg, "rms_norm_eps", Some(1e-6))?,
        rope_theta: get_f32(cfg, "rope_theta", Some(1e6))?,
        rope_local_theta: get_f32(cfg, "rope_local_base_freq", Some(1e4))?,
        sliding_window,
        // transformers Gemma3TextConfig: bidirectional => window // 2 + 1 (exclusive).
        bidirectional_window: sliding_window / 2 + 1,
        sliding_layers,
        query_pre_attn_scalar: get_f32(cfg, "query_pre_attn_scalar", Some(head_dim as f32))?,
        hidden_activation: cfg
            .get("hidden_activation")
            .or(cfg.get("hidden_act"))
            .and_then(Value::as_str)
            .unwrap_or("gelu_pytorch_tanh")
            .to_string(),
        max_seq_length: get_usize(cfg, "max_position_embeddings").unwrap_or(2048),
        pad_token_id: cfg.get("pad_token_id").and_then(Value::as_u64).unwrap_or(0) as u32,
        pooling: PoolingConfig { mode: "mean".into(), include_prompt: true },
        dense: Vec::new(),
        normalize: false,
        prompts: BTreeMap::new(),
        default_prompt_name: None,
        matryoshka_dims: Vec::new(),
        tokenizer_blob: TOKENIZER_BLOB.into(),
    })
}

/// A Dense module: its config and weight file.
struct DenseModule {
    cfg: DenseConfig,
    weights: PathBuf,
}

/// Walk `modules.json`: pooling, Dense modules (in order), normalize.
fn read_modules(src: &Source, cfg: &mut EmbeddingGemmaConfig) -> Result<Vec<DenseModule>> {
    let modules = read_json(&src.required("modules.json")?)?;
    let list = modules.as_array().ok_or_else(|| Error::new("format", "modules.json is not an array"))?;
    let mut dense = Vec::new();
    for m in list {
        let ty = m.get("type").and_then(Value::as_str).unwrap_or("");
        let path = m.get("path").and_then(Value::as_str).unwrap_or("");
        let kind = ty.rsplit('.').next().unwrap_or(ty);
        match kind {
            "Transformer" => {}
            "Pooling" => {
                let p = read_json(&src.required(&format!("{path}/config.json"))?)?;
                let flag = |k: &str| p.get(k).and_then(Value::as_bool).unwrap_or(false);
                if !flag("pooling_mode_mean_tokens") || ["cls_token", "max_tokens", "mean_sqrt_len_tokens", "weightedmean_tokens", "lasttoken"].iter().any(|k| flag(&format!("pooling_mode_{k}"))) {
                    bail!("unsupported_model", "{path}: only mean pooling is supported");
                }
                cfg.pooling = PoolingConfig { mode: "mean".into(), include_prompt: p.get("include_prompt").and_then(Value::as_bool).unwrap_or(true) };
            }
            "Dense" => {
                let d = read_json(&src.required(&format!("{path}/config.json"))?)?;
                let act_name = d.get("activation_function").and_then(Value::as_str).unwrap_or("torch.nn.modules.activation.Tanh");
                let activation = DenseActivation::from_torch(act_name)
                    .ok_or_else(|| Error::new("unsupported_model", format!("{path}: activation {act_name:?} is not supported")))?;
                let k = dense.len();
                dense.push(DenseModule {
                    cfg: DenseConfig {
                        prefix: format!("dense.{k}."),
                        in_features: get_usize(&d, "in_features")?,
                        out_features: get_usize(&d, "out_features")?,
                        bias: d.get("bias").and_then(Value::as_bool).unwrap_or(true),
                        activation,
                    },
                    weights: src.required(&format!("{path}/model.safetensors"))?,
                });
            }
            "Normalize" => cfg.normalize = true,
            other => bail!("unsupported_model", "modules.json: module type {other:?} is not supported"),
        }
    }
    cfg.dense = dense.iter().map(|d| d.cfg.clone()).collect();
    Ok(dense)
}

/// Prompts and max_seq_length from the sentence-transformers configs.
fn read_st_configs(src: &Source, cfg: &mut EmbeddingGemmaConfig) -> Result<()> {
    if let Some(p) = src.optional("config_sentence_transformers.json")? {
        let st = read_json(&p)?;
        if let Some(obj) = st.get("prompts").and_then(Value::as_object) {
            cfg.prompts = obj.iter().filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string()))).collect();
        }
        cfg.default_prompt_name = st.get("default_prompt_name").and_then(Value::as_str).map(str::to_string);
    }
    if let Some(p) = src.optional("sentence_bert_config.json")?
        && let Some(n) = read_json(&p)?.get("max_seq_length").and_then(Value::as_u64)
    {
        cfg.max_seq_length = n as usize;
    }
    let out = cfg.output_dim();
    cfg.matryoshka_dims = MATRYOSHKA.iter().copied().filter(|&d| d <= out).collect();
    if !cfg.matryoshka_dims.contains(&out) {
        cfg.matryoshka_dims.insert(0, out);
    }
    Ok(())
}

/// Encoder tensor names (HF Gemma3TextModel naming, prefixed `model.`).
pub(crate) fn expected_tensors(cfg: &EmbeddingGemmaConfig) -> Vec<String> {
    let mut v = vec![EMBED.to_string(), "model.norm.weight".to_string()];
    for i in 0..cfg.n_layer {
        let p = format!("model.layers.{i}");
        for s in [
            "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm",
            "self_attn.q_norm", "self_attn.k_norm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        ] {
            v.push(format!("{p}.{s}.weight"));
        }
    }
    for d in &cfg.dense {
        v.push(format!("{}weight", d.prefix));
        if d.bias {
            v.push(format!("{}bias", d.prefix));
        }
    }
    v
}

/// `.lapid` metadata: required keys, the config object and the component declarations.
fn metadata(name: &str, dtype: OutDType, source: Value, cfg: &EmbeddingGemmaConfig) -> Result<Value> {
    let mut m = lapid_convert::writer::metadata(name, META_KEY, dtype, LICENSE, source);
    m["license_url"] = json!(LICENSE_URL);
    m[META_KEY] = serde_json::to_value(cfg)?;
    let mut comps = serde_json::Map::new();
    comps.insert("encoder".into(), json!({"architecture": "gemma3_text", "config": META_KEY, "license": LICENSE, "tensor_prefix": "model."}));
    comps.insert("pooling".into(), json!({"architecture": "sentence_transformers.pooling", "config": serde_json::to_value(&cfg.pooling)?, "tensor_prefix": "pooling."}));
    for (k, d) in cfg.dense.iter().enumerate() {
        comps.insert(format!("dense_{k}"), json!({"architecture": "sentence_transformers.dense", "config": serde_json::to_value(d)?, "tensor_prefix": d.prefix}));
    }
    if cfg.normalize {
        comps.insert("normalize".into(), json!({"architecture": "sentence_transformers.normalize", "config": {}, "tensor_prefix": "normalize."}));
    }
    m["components"] = Value::Object(comps);
    Ok(m)
}

/// Convert `src` (repo id or directory) to `out` at `dtype`.
pub fn convert_hf(src: &str, dtype: OutDType, out: &Path) -> Result<()> {
    let src = Source::new(src);
    let hf_cfg = read_json(&src.required("config.json")?)?;
    if hf_cfg.get("model_type").and_then(Value::as_str) != Some("gemma3_text") {
        bail!("unsupported_model", "{}: config.json model_type is not \"gemma3_text\"", src.repo);
    }
    let mut cfg = encoder_config(&hf_cfg)?;
    let dense = read_modules(&src, &mut cfg)?;
    read_st_configs(&src, &mut cfg)?;
    cfg.validate()?;
    let tokenizer = std::fs::read(src.required("tokenizer.json")?).map_err(|e| Error::new("io", format!("tokenizer.json: {e}")))?;
    // Fail at conversion time, not load time, on a tokenizer we cannot parse.
    let text = std::str::from_utf8(&tokenizer).map_err(|_| Error::new("tokenizer", "tokenizer.json is not UTF-8"))?;
    lapid_nn::Tokenizer::from_tokenizer_json(text)?;

    let weights = src.required("model.safetensors")?;
    let st = SafeTensors::open(&weights)?;
    let mut by_name: BTreeMap<String, (usize, String)> = BTreeMap::new();
    for n in st.names() {
        let norm = if n.starts_with("model.") { n.clone() } else { format!("model.{n}") };
        by_name.insert(norm, (0, n));
    }
    let mut files = vec![st];
    for d in &dense {
        let dst = SafeTensors::open(&d.weights)?;
        for n in dst.names() {
            let leaf = n.rsplit('.').next().unwrap_or(&n).to_string();
            by_name.insert(format!("{}{leaf}", d.cfg.prefix), (files.len(), n));
        }
        files.push(dst);
    }
    let expected = expected_tensors(&cfg);
    let have: BTreeSet<String> = by_name.keys().cloned().collect();
    check_complete(&have, &expected)?;
    let (fi, en) = &by_name[EMBED];
    let shape = files[*fi].shape(en)?;
    if shape != [cfg.vocab_size, cfg.hidden_size] {
        bail!("shape_mismatch", "{EMBED}: shape {shape:?}, config says [{}, {}]", cfg.vocab_size, cfg.hidden_size);
    }

    let name = src.repo.trim_end_matches('/').rsplit('/').next().unwrap_or("embeddinggemma").to_string();
    let mut source = json!({ "repo": src.repo, "file_hash": file_hash(&weights)? });
    if src.dir.is_none() {
        source["revision"] = json!("main");
    }
    let meta = metadata(&name, dtype, source, &cfg)?;
    let mut w = ModelWriter::create(out, meta, dtype, role, expected.len())?;
    w.add_blob(TOKENIZER_BLOB, "application/json", &tokenizer)?;
    for n in &expected {
        let (fi, orig) = &by_name[n];
        let f = &files[*fi];
        w.add_f32(n, &f.shape(orig)?, &f.to_f32(orig)?)?;
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lapid::quant::Format;
    use lapid_convert::policy::{Target, target};

    #[test]
    fn policy() {
        let t = |n: &str, s: &[usize], d| target(role(n, s), s, d);
        assert_eq!(t(EMBED, &[262144, 768], OutDType::Int4), Target::Quant(Format::Int8));
        assert_eq!(t(EMBED, &[262144, 768], OutDType::F16), Target::F16);
        assert_eq!(t("model.layers.0.self_attn.q_proj.weight", &[768, 768], OutDType::Int4), Target::Quant(Format::Int4));
        assert_eq!(t("model.layers.0.self_attn.q_norm.weight", &[256], OutDType::Int8), Target::F32);
        assert_eq!(t("dense.0.weight", &[3072, 768], OutDType::Int8), Target::Quant(Format::Int8));
    }

    #[test]
    fn gemma_config() {
        let cfg = json!({
            "model_type": "gemma3_text", "num_hidden_layers": 6, "hidden_size": 768, "num_attention_heads": 3,
            "num_key_value_heads": 1, "head_dim": 256, "intermediate_size": 1152, "vocab_size": 262144,
            "sliding_window": 512, "use_bidirectional_attention": true, "query_pre_attn_scalar": 256,
            "rope_theta": 1000000.0, "rope_local_base_freq": 10000.0, "_sliding_window_pattern": 6
        });
        let c = encoder_config(&cfg).expect("config");
        assert_eq!(c.bidirectional_window, 257);
        assert_eq!(c.sliding_layers, vec![true, true, true, true, true, false]);
        assert_eq!(expected_tensors(&c).len(), 2 + 6 * 13);
        let mut causal = cfg.clone();
        causal["use_bidirectional_attention"] = json!(false);
        assert_eq!(encoder_config(&causal).unwrap_err().code, "unsupported_model");
    }
}
