//! Whisper checkpoints → `.lapid`: Hugging Face (`WhisperForConditionalGeneration`
//! safetensors, repo id or local directory) and whisper.cpp `ggml-*.bin`.
//!
//! Both converters emit HF tensor names (`model.encoder.conv1.weight`, …) and the same
//! `"whisper"` metadata object. Precision follows the generic `lapid_convert::policy` with
//! Whisper's tensor roles: Linear weights quantized; `model.decoder.embed_tokens.weight`
//! (also the output projection) at most int8; conv weights and positional embeddings
//! f16; LayerNorm parameters and biases f32. The token list is stored as a blob
//! (see [`assets`]); metadata declares one component, `whisper`, with tensor prefix `""`.

mod assets;
mod cpp;
mod ggml;
mod hf;
mod tokens;

use std::path::Path;

use lapid::Result;
use lapid_convert::OutDType;
use lapid_convert::policy::Role;
use serde_json::{Value, json};

pub use assets::load_token_strings;
pub(crate) use ggml::is_ggml_file;

/// Token embedding; also the (tied) output projection.
pub(crate) const EMBED_TOKENS: &str = "model.decoder.embed_tokens.weight";

/// Convert a Whisper model in Hugging Face format (repo id or local directory).
pub fn convert_hf(repo_or_dir: &str, dtype: OutDType, out: &Path) -> Result<()> {
    hf::convert(repo_or_dir, dtype, out)
}

/// Convert a whisper.cpp `ggml-*.bin` model.
pub fn convert_ggml(ggml_bin: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    cpp::convert(ggml_bin, dtype, out)
}

/// Dtype-policy role of a Whisper tensor (HF naming).
pub(crate) fn role(name: &str, shape: &[usize]) -> Role {
    if name.ends_with(".bias") || name.contains("layer_norm") || name.contains("layernorm") || shape.len() <= 1 {
        Role::Param
    } else if name == EMBED_TOKENS {
        Role::LinearInt8
    } else if shape.len() == 2 && !name.contains("embed_positions") && !name.contains("conv") {
        Role::Linear
    } else {
        Role::Dense
    }
}

/// Required `.lapid` metadata, the `"whisper"` object (plus `tokens_blob`) and the
/// single-component declaration.
fn metadata(name: &str, dtype: OutDType, license: &str, source: Value, mut whisper: Value) -> Value {
    let mut m = lapid_convert::writer::metadata(name, "whisper", dtype, license, source);
    whisper["tokens_blob"] = json!(assets::TOKENS_BLOB);
    m["whisper"] = whisper;
    m["components"] = json!({
        "whisper": {"architecture": "whisper", "config": "whisper", "license": license, "tensor_prefix": ""}
    });
    m
}

/// Write the token list blob.
fn add_token_blob(w: &mut lapid_convert::writer::ModelWriter, tokens: &[String]) -> Result<()> {
    w.add_blob(assets::TOKENS_BLOB, assets::TOKENS_MEDIA_TYPE, &assets::tokens_blob(tokens)?)
}

/// All tensor names a Whisper checkpoint must provide (HF naming).
pub(crate) fn expected_tensors(n_audio_layer: usize, n_text_layer: usize) -> Vec<String> {
    let mut v: Vec<String> = [
        "model.encoder.conv1.weight",
        "model.encoder.conv1.bias",
        "model.encoder.conv2.weight",
        "model.encoder.conv2.bias",
        "model.encoder.embed_positions.weight",
        "model.encoder.layer_norm.weight",
        "model.encoder.layer_norm.bias",
        EMBED_TOKENS,
        "model.decoder.embed_positions.weight",
        "model.decoder.layer_norm.weight",
        "model.decoder.layer_norm.bias",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let attn = |p: &str, v: &mut Vec<String>| {
        for (proj, bias) in [("q_proj", true), ("k_proj", false), ("v_proj", true), ("out_proj", true)] {
            v.push(format!("{p}.{proj}.weight"));
            if bias {
                v.push(format!("{p}.{proj}.bias"));
            }
        }
        v.push(format!("{p}_layer_norm.weight"));
        v.push(format!("{p}_layer_norm.bias"));
    };
    let mlp = |p: &str, v: &mut Vec<String>| {
        for s in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "final_layer_norm.weight", "final_layer_norm.bias"] {
            v.push(format!("{p}.{s}"));
        }
    };
    for i in 0..n_audio_layer {
        let p = format!("model.encoder.layers.{i}");
        attn(&format!("{p}.self_attn"), &mut v);
        mlp(&p, &mut v);
    }
    for i in 0..n_text_layer {
        let p = format!("model.decoder.layers.{i}");
        attn(&format!("{p}.self_attn"), &mut v);
        attn(&format!("{p}.encoder_attn"), &mut v);
        mlp(&p, &mut v);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use lapid::quant::Format;
    use lapid_convert::policy::{Target, target};

    fn t(name: &str, shape: &[usize], dtype: OutDType) -> Target {
        target(role(name, shape), shape, dtype)
    }

    #[test]
    fn policy_table() {
        let q = "model.encoder.layers.0.self_attn.q_proj.weight";
        assert_eq!(t(q, &[768, 768], OutDType::F32), Target::F32);
        assert_eq!(t(q, &[768, 768], OutDType::F16), Target::F16);
        assert_eq!(t(q, &[768, 768], OutDType::Int8), Target::Quant(Format::Int8));
        assert_eq!(t(q, &[768, 768], OutDType::Int4), Target::Quant(Format::Int4));
        assert_eq!(t(EMBED_TOKENS, &[51865, 768], OutDType::Int4), Target::Quant(Format::Int8));
        assert_eq!(t("model.encoder.conv1.weight", &[768, 80, 3], OutDType::Int8), Target::F16);
        assert_eq!(t("model.encoder.embed_positions.weight", &[1500, 768], OutDType::Int4), Target::F16);
        assert_eq!(t("model.encoder.layer_norm.weight", &[768], OutDType::F16), Target::F32);
        assert_eq!(t("model.decoder.layers.0.fc1.bias", &[3072], OutDType::Int8), Target::F32);
        assert_eq!(t("x.weight", &[10, 33], OutDType::Int8), Target::F16);
    }

    #[test]
    fn whisper_small_names() {
        let n = expected_tensors(12, 12);
        // 11 globals + 12 * (9 attn + 6 mlp) + 12 * (2 * 9 + 6)
        assert_eq!(n.len(), 11 + 12 * 15 + 12 * 24);
        assert!(n.contains(&"model.decoder.layers.11.encoder_attn_layer_norm.bias".to_string()));
        assert!(!n.contains(&"model.encoder.layers.0.self_attn.k_proj.bias".to_string()));
    }
}
