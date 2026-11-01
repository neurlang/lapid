//! Exported Silero VAD (`config.json` with `model_type = "silero_vad"` + `model.safetensors`,
//! written by `tools/reference/silero_vad_ref.py` from the official v5 TorchScript model;
//! a local directory or an HF repo with the same files) → `.lapid`.
//!
//! The model has ~310k parameters (1.2 MB in f32), so every tensor is stored as f32 whatever
//! the requested dtype: quantizing would only cost accuracy. The file's `dtype` is "fp32".
//! Layout changes: the STFT basis [258, 1, 256] is stored as [258, 256] and the 1×1 head conv
//! [1, 128, 1] as [1, 128]; everything else keeps its PyTorch name and shape.

use std::path::{Path, PathBuf};

use lapid::{Error, Result, bail};
use lapid_convert::policy::Role;
use lapid_convert::safetensors::SafeTensors;
use lapid_convert::writer::{ModelWriter, file_hash, read_json};
use lapid_convert::{OutDType, download_hf};
use serde_json::{Value, json};

use super::config::VadConfig;

/// Every tensor a converted file contains.
pub(crate) fn expected_tensors(cfg: &VadConfig) -> Vec<String> {
    let mut v = vec!["stft.forward_basis_buffer".to_string()];
    for i in 0..cfg.encoder.len() {
        v.push(format!("encoder.{i}.reparam_conv.weight"));
        v.push(format!("encoder.{i}.reparam_conv.bias"));
    }
    for n in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"] {
        v.push(format!("decoder.rnn.{n}"));
    }
    v.push("decoder.decoder.2.weight".into());
    v.push("decoder.decoder.2.bias".into());
    v
}

/// Dtype-policy role: everything stays f32.
pub(crate) fn role(_name: &str, _shape: &[usize]) -> Role {
    Role::Param
}

/// `.lapid` shape of a source tensor.
fn target_shape(name: &str, shape: &[usize]) -> Vec<usize> {
    match name {
        "stft.forward_basis_buffer" | "decoder.decoder.2.weight" if shape.len() == 3 => {
            vec![shape[0], shape[1] * shape[2]]
        }
        _ => shape.to_vec(),
    }
}

/// Convert an exported Silero VAD directory or HF repo (`dtype` is accepted for the hub
/// convention; weights are always f32).
pub fn convert_hf(src_dir: &str, dtype: OutDType, out: &Path) -> Result<()> {
    let _ = dtype;
    let dir = Path::new(src_dir);
    let (config, weights, repo, remote): (PathBuf, PathBuf, String, bool) = if dir.is_dir() {
        let need = |f: &str| Some(dir.join(f)).filter(|p| p.is_file()).ok_or_else(|| Error::new("not_found", format!("{}: no {f}", dir.display())));
        (need("config.json")?, need("model.safetensors")?, dir.display().to_string(), false)
    } else {
        let repo = src_dir.trim_matches('/');
        (download_hf(repo, "config.json")?, download_hf(repo, "model.safetensors")?, repo.to_string(), true)
    };
    let raw = read_json(&config)?;
    if raw.get("model_type").and_then(Value::as_str) != Some("silero_vad") {
        bail!("unsupported_model", "{}: model_type is not \"silero_vad\"", config.display());
    }
    let cfg = VadConfig::from_json(&raw)?;
    let st = SafeTensors::open(&weights)?;
    let expected = expected_tensors(&cfg);

    let name = format!("silero-vad-{}", cfg.version);
    let mut source = json!({ "repo": repo, "format": "silero-vad-safetensors" });
    if remote {
        source["revision"] = json!("main");
    }
    source["file_hash"] = json!(file_hash(&weights)?);
    if let Some(upstream) = raw.get("source") {
        source["upstream"] = upstream.clone();
    }
    let mut meta = lapid_convert::writer::metadata(&name, "silero_vad", OutDType::F32, "MIT", source);
    meta["silero_vad"] = cfg.to_json();
    meta["components"] = json!({
        "silero_vad": {"architecture": "silero_vad", "config": "silero_vad", "license": "MIT", "tensor_prefix": ""}
    });

    let mut w = ModelWriter::create(out, meta, OutDType::F32, role, expected.len())?;
    for n in &expected {
        if !st.contains(n) {
            bail!("missing_tensor", "{repo}: checkpoint has no {n:?}");
        }
        let shape = st.shape(n)?;
        w.add_f32(n, &target_shape(n, &shape), &st.to_f32(n)?)?;
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_shapes() {
        let n = expected_tensors(&VadConfig::v5_16k());
        assert_eq!(n.len(), 1 + 8 + 4 + 2);
        assert_eq!(target_shape("stft.forward_basis_buffer", &[258, 1, 256]), vec![258, 256]);
        assert_eq!(target_shape("decoder.decoder.2.weight", &[1, 128, 1]), vec![1, 128]);
        assert_eq!(target_shape("encoder.0.reparam_conv.weight", &[128, 129, 3]), vec![128, 129, 3]);
    }
}
