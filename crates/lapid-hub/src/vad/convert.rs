//! Exported Silero VAD (`config.json` with `model_type = "silero_vad"` + `model.safetensors`,
//! written by `tools/reference/silero_vad_ref.py` from the official v5 TorchScript model;
//! a local directory or an HF repo with the same files) -> `.lapid` through the generic engine.
//!
//! The model has ~310k parameters (1.2 MB in f32), so every tensor is stored as f32 whatever
//! the requested dtype (the file's `dtype` is "fp32"). Layout changes: the STFT basis
//! `[258, 1, 256]` is stored as `[258, 256]` and the 1x1 head conv `[1, 128, 1]` as `[1, 128]`.

use lapid::Result;
use lapid_convert::assets::HfFiles;
use lapid_convert::engine::{ConvertHooks, ConvertSpec, Emit, HfExtras, SourceKind, TensorSource};
use lapid_convert::mapping::RoleRule;
use lapid_convert::policy::Role;
use serde_json::{Map, Value, json};

use super::config::VadConfig;

pub static CONVERT: ConvertSpec = ConvertSpec {
    architecture: "silero_vad",
    license: "MIT",
    hf_model_types: &["silero_vad"],
    gguf_architectures: &[],
    hf_names: &[],
    gguf_names: &[],
    roles: &[RoleRule::new("*", Role::Param)],
    text_assets: false,
    hooks: &Hooks,
};

struct Hooks;

impl ConvertHooks for Hooks {
    fn hf_config(&self, raw: &Value) -> Result<Value> {
        Ok(VadConfig::from_json(raw)?.to_json())
    }

    fn hf_extras(&self, _files: &HfFiles, cfg: &mut Value) -> Result<HfExtras> {
        let mut metadata = Map::new();
        metadata.insert("name".into(), json!(format!("silero-vad-{}", cfg["version"].as_str().unwrap_or("v5"))));
        metadata.insert("dtype".into(), json!("fp32"));
        Ok(HfExtras { metadata, ..HfExtras::default() })
    }

    fn expected(&self, cfg: &Value) -> Result<Vec<String>> {
        Ok(expected_tensors(&VadConfig::from_json(cfg)?))
    }

    fn reshape(&self, name: &str, shape: &[usize], _cfg: &Value, _src: SourceKind) -> Option<Vec<usize>> {
        let flat = matches!(name, "stft.forward_basis_buffer" | "decoder.decoder.2.weight") && shape.len() == 3;
        flat.then(|| vec![shape[0], shape[1] * shape[2]])
    }

    fn transform(&self, name: &str, _src: &dyn TensorSource, cfg: &Value) -> Result<Emit> {
        Ok(if expected_tensors(&VadConfig::from_json(cfg)?).iter().any(|n| n == name) { Emit::Keep } else { Emit::Drop })
    }
}

/// Every tensor a converted file contains.
pub(crate) fn expected_tensors(cfg: &VadConfig) -> Vec<String> {
    let mut v = vec!["stft.forward_basis_buffer".to_string()];
    for i in 0..cfg.encoder.len() {
        v.extend([format!("encoder.{i}.reparam_conv.weight"), format!("encoder.{i}.reparam_conv.bias")]);
    }
    v.extend(["weight_ih", "weight_hh", "bias_ih", "bias_hh"].map(|n| format!("decoder.rnn.{n}")));
    v.extend(["decoder.decoder.2.weight".into(), "decoder.decoder.2.bias".into()]);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_shapes_roles() {
        assert_eq!(expected_tensors(&VadConfig::v5_16k()).len(), 1 + 8 + 4 + 2);
        let r = |n: &str, s: &[usize]| Hooks.reshape(n, s, &Value::Null, SourceKind::Hf);
        assert_eq!(r("stft.forward_basis_buffer", &[258, 1, 256]), Some(vec![258, 256]));
        assert_eq!(r("encoder.0.reparam_conv.weight", &[128, 129, 3]), None);
        assert_eq!(CONVERT.role("encoder.0.reparam_conv.weight", &[128, 129, 3]), Role::Param);
    }
}
