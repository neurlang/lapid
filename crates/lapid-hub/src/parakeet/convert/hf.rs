//! Hugging Face `ParakeetForTDT` (transformers ≥ 5.9 layout: `config.json` with an
//! `encoder_config`, `processor_config.json`, `tokenizer.json`, `model.safetensors`) → `.lapid`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::safetensors::SafeTensors;
use lapid_convert::writer::{ModelWriter, file_hash, read_json};
use lapid_convert::{download_hf, download_hf_optional};
use serde_json::{Value, json};

use super::{add_tokenizer_blob, expected_tensors, fold_batch_norm, metadata, role, target_shape};
use crate::parakeet::config::ParakeetConfig;

struct Files {
    config: PathBuf,
    processor: Option<PathBuf>,
    tokenizer: PathBuf,
    weights: PathBuf,
    repo: String,
    remote: bool,
}

fn locate(src: &str) -> Result<Files> {
    let dir = Path::new(src);
    if dir.is_dir() {
        let need = |f: &str| {
            Some(dir.join(f)).filter(|p| p.is_file()).ok_or_else(|| Error::new("not_found", format!("{}: no {f}", dir.display())))
        };
        return Ok(Files {
            config: need("config.json")?,
            processor: Some(dir.join("processor_config.json")).filter(|p| p.is_file()),
            tokenizer: need("tokenizer.json")?,
            weights: need("model.safetensors")?,
            repo: dir.display().to_string(),
            remote: false,
        });
    }
    let repo = src.trim_matches('/');
    Ok(Files {
        config: download_hf(repo, "config.json")?,
        processor: download_hf_optional(repo, "processor_config.json")?,
        tokenizer: download_hf(repo, "tokenizer.json")?,
        weights: download_hf(repo, "model.safetensors")?,
        repo: repo.to_string(),
        remote: true,
    })
}

/// `ParakeetConfig` from `config.json` (+ `processor_config.json` feature extractor).
pub(crate) fn config_from_hf(cfg: &Value, processor: Option<&Value>) -> Result<ParakeetConfig> {
    if cfg.get("model_type").and_then(Value::as_str) != Some("parakeet_tdt") {
        bail!("unsupported_model", "config.json model_type is not \"parakeet_tdt\"");
    }
    let enc = cfg.get("encoder_config").ok_or_else(|| Error::new("missing_metadata", "config.json: no encoder_config"))?;
    let mut c = ParakeetConfig::v3();
    let u = |v: &Value, k: &str, d: usize| v.get(k).and_then(Value::as_u64).map_or(d, |x| x as usize);
    c.d_model = u(enc, "hidden_size", c.d_model);
    c.n_layers = u(enc, "num_hidden_layers", c.n_layers);
    c.n_heads = u(enc, "num_attention_heads", c.n_heads);
    c.ff_dim = u(enc, "intermediate_size", c.ff_dim);
    c.conv_kernel = u(enc, "conv_kernel_size", c.conv_kernel);
    c.subsampling_factor = u(enc, "subsampling_factor", c.subsampling_factor);
    c.subsampling_channels = u(enc, "subsampling_conv_channels", c.subsampling_channels);
    c.n_mels = u(enc, "num_mel_bins", c.n_mels);
    c.decoder_hidden = u(cfg, "decoder_hidden_size", c.decoder_hidden);
    c.decoder_layers = u(cfg, "num_decoder_layers", c.decoder_layers);
    c.vocab_size = u(cfg, "vocab_size", c.vocab_size);
    c.blank_id = u(cfg, "blank_token_id", c.blank_id as usize) as u32;
    c.max_symbols_per_step = u(cfg, "max_symbols_per_step", c.max_symbols_per_step);
    if let Some(d) = cfg.get("durations").and_then(Value::as_array) {
        c.durations = d.iter().filter_map(Value::as_u64).map(|x| x as usize).collect();
    }
    for (key, want) in [("hidden_act", "silu")] {
        if let Some(a) = enc.get(key).and_then(Value::as_str).filter(|a| *a != want) {
            bail!("unsupported_model", "encoder {key} {a:?} (only {want:?} is implemented)");
        }
    }
    if let Some(a) = cfg.get("hidden_act").and_then(Value::as_str).filter(|a| *a != "relu") {
        bail!("unsupported_model", "joint activation {a:?} (only \"relu\" is implemented)");
    }
    if enc.get("scale_input").and_then(Value::as_bool) == Some(true) {
        bail!("unsupported_model", "encoder scale_input = true is not implemented");
    }
    if let Some(fe) = processor.and_then(|p| p.get("feature_extractor")) {
        c.sample_rate = u(fe, "sampling_rate", c.sample_rate as usize) as u32;
        c.n_fft = u(fe, "n_fft", c.n_fft);
        c.hop_length = u(fe, "hop_length", c.hop_length);
        c.win_length = u(fe, "win_length", c.win_length);
        c.n_mels = u(fe, "feature_size", c.n_mels);
        if let Some(p) = fe.get("preemphasis").and_then(Value::as_f64) {
            c.preemphasis = p;
        }
    }
    c.validate()?;
    Ok(c)
}

pub(crate) fn convert(src: &str, dtype: OutDType, out: &Path) -> Result<()> {
    let files = locate(src)?;
    let raw_cfg = read_json(&files.config)?;
    let processor = files.processor.as_deref().map(read_json).transpose()?;
    let cfg = config_from_hf(&raw_cfg, processor.as_ref())?;
    let tokenizer = std::fs::read(&files.tokenizer)
        .map_err(|e| Error::new("io", format!("{}: {e}", files.tokenizer.display())))?;

    let st = SafeTensors::open(&files.weights)?;
    let have: BTreeSet<String> = st.names().into_iter().collect();
    let expected = expected_tensors(&cfg);

    let name = files.repo.trim_end_matches('/').rsplit('/').next().unwrap_or("parakeet").to_string();
    let license = if files.repo.starts_with("nvidia/parakeet") { "cc-by-4.0" } else { "unknown" };
    let mut source = json!({ "repo": files.repo, "format": "hf-safetensors" });
    if files.remote {
        source["revision"] = json!("main");
    }
    source["file_hash"] = json!(file_hash(&files.weights)?);
    let meta = metadata(&name, dtype, license, source, &cfg);

    let mut w = ModelWriter::create(out, meta, dtype, role, expected.len())?;
    add_tokenizer_blob(&mut w, &tokenizer)?;
    let get = |n: &str| -> Result<Vec<f32>> {
        if !have.contains(n) {
            bail!("missing_tensor", "{}: checkpoint has no {n:?}", files.repo);
        }
        st.to_f32(n)
    };
    for n in &expected {
        if n.ends_with("conv.depthwise_conv.bias") {
            continue; // written with its weight
        }
        if let Some(p) = n.strip_suffix(".depthwise_conv.weight") {
            let shape = st.shape(n)?;
            let bias = if have.contains(&format!("{p}.depthwise_conv.bias")) { Some(get(&format!("{p}.depthwise_conv.bias"))?) } else { None };
            let (fw, fb) = fold_batch_norm(
                &get(n)?,
                bias.as_deref(),
                &get(&format!("{p}.norm.weight"))?,
                &get(&format!("{p}.norm.bias"))?,
                &get(&format!("{p}.norm.running_mean"))?,
                &get(&format!("{p}.norm.running_var"))?,
            )?;
            w.add_f32(n, &target_shape(n, &shape), &fw)?;
            w.add_f32(&format!("{p}.depthwise_conv.bias"), &[fb.len()], &fb)?;
            continue;
        }
        let shape = st.shape(n).map_err(|e| Error::new("missing_tensor", format!("{}: {}", files.repo, e.message)))?;
        w.add_f32(n, &target_shape(n, &shape), &get(n)?)?;
    }
    // Optional biases (attention_bias / convolution_bias = true checkpoints).
    let expected_set: BTreeSet<&String> = expected.iter().collect();
    for n in &have {
        let extra = n.ends_with(".bias") && !n.contains(".norm.") && !expected_set.contains(n);
        if extra && !n.contains("depthwise_conv") {
            w.add_f32(n, &st.shape(n)?, &get(n)?)?;
        }
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_from_hf_json() {
        let cfg = json!({
            "model_type": "parakeet_tdt", "blank_token_id": 8192, "decoder_hidden_size": 640,
            "durations": [0, 1, 2, 3, 4], "hidden_act": "relu", "max_symbols_per_step": 10,
            "num_decoder_layers": 2, "vocab_size": 8193,
            "encoder_config": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 8,
                "intermediate_size": 4096, "conv_kernel_size": 9, "subsampling_factor": 8,
                "subsampling_conv_channels": 256, "num_mel_bins": 128, "hidden_act": "silu", "scale_input": false}
        });
        let proc = json!({"feature_extractor": {"feature_size": 128, "hop_length": 160, "n_fft": 512,
            "preemphasis": 0.97, "sampling_rate": 16000, "win_length": 400}});
        assert_eq!(config_from_hf(&cfg, Some(&proc)).expect("cfg"), ParakeetConfig::v3());
        let mut bad = cfg.clone();
        bad["model_type"] = json!("parakeet_ctc");
        assert_eq!(config_from_hf(&bad, None).unwrap_err().code, "unsupported_model");
    }
}
