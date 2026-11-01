//! LFM3 checkpoints → one `.lapid` file: Hugging Face safetensors (repo id or directory) and
//! GGUF (`LiquidAI/LFM3-1.7B-GGUF`, llama.cpp `lfm3` naming).
//!
//! Both emit HF tensor names (`model.layers.{i}.conv.in_proj.weight`, …), the `"lfm3"`
//! metadata object (HF config keys + `special_tokens`, `generation`, blob names) and two
//! blobs: `tokenizer/tokenizer.json` and `tokenizer/chat_template.jinja`. Precision follows
//! `lapid_convert::policy`: Linear weights quantized; the tied embedding (also the output
//! projection) at most int8; the depthwise conv weights f16; norms f32.

pub(crate) mod calibrate;
mod gguf;
mod hf;

use std::path::Path;

use lapid::Result;
use lapid_convert::OutDType;
use lapid_convert::policy::Role;
use lapid_convert::writer::ModelWriter;
use serde_json::{Value, json};

use super::config::{LayerType, Lfm3Config};
use super::model::EMBED;
use super::{TEMPLATE_BLOB, TOKENIZER_BLOB};

pub(crate) use gguf::is_lfm3_gguf;

/// SPDX-like id of the LFM Open License v1.0 (as on the model cards: `license_name: lfm1.0`).
pub const LICENSE: &str = "lfm1.0";

/// Convert an LFM3 model in Hugging Face format (repo id or local directory).
pub fn convert_hf(repo_or_dir: &str, dtype: OutDType, out: &Path) -> Result<()> {
    hf::convert(repo_or_dir, dtype, out)
}

/// Convert an LFM3 GGUF file (any ggml type lapid can dequantize; Q8_0 → int8 and Q4_0 →
/// int4 are repacked losslessly).
pub fn convert_gguf(path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    gguf::convert(path, dtype, out)
}

/// Dtype-policy role of an LFM3 tensor (HF naming).
pub(crate) fn role(name: &str, shape: &[usize]) -> Role {
    if name.ends_with(".bias") || name.contains("norm") || shape.len() <= 1 {
        Role::Param
    } else if name == EMBED {
        Role::LinearInt8
    } else if shape.len() == 2 {
        Role::Linear
    } else {
        Role::Dense
    }
}

/// Every tensor an LFM3 checkpoint must provide (HF naming).
pub(crate) fn expected_tensors(cfg: &Lfm3Config) -> Vec<String> {
    let mut v = vec![EMBED.to_string(), "model.embedding_norm.weight".to_string()];
    for (i, t) in cfg.layer_types.iter().enumerate() {
        let p = format!("model.layers.{i}");
        let mut norms = vec!["operator_norm", "ffn_norm"];
        if cfg.sandwich_norm {
            norms.extend(["post_operator_norm", "post_ffn_norm"]);
        }
        v.extend(norms.iter().map(|n| format!("{p}.{n}.weight")));
        v.extend(["w1", "w2", "w3"].iter().map(|w| format!("{p}.feed_forward.{w}.weight")));
        match t {
            LayerType::Conv => {
                for n in ["conv.conv", "conv.in_proj", "conv.out_proj"] {
                    v.push(format!("{p}.{n}.weight"));
                    if cfg.conv_bias {
                        v.push(format!("{p}.{n}.bias"));
                    }
                }
            }
            _ => {
                for n in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm", "gate_proj"] {
                    v.push(format!("{p}.self_attn.{n}.weight"));
                }
            }
        }
    }
    v
}

/// Non-tensor parts of the output file.
pub(crate) struct Assets {
    pub tokenizer_json: String,
    pub chat_template: String,
    /// Template globals (`bos_token`, `eos_token`, `pad_token`).
    pub special_tokens: Value,
    /// `generation_config.json`-style defaults.
    pub generation: Value,
}

/// Required metadata + the `"lfm3"` object + the single-component declaration.
pub(crate) fn metadata(name: &str, dtype: OutDType, license: &str, source: Value, cfg: &Lfm3Config, a: &Assets) -> Value {
    let mut m = lapid_convert::writer::metadata(name, "lfm3", dtype, license, source);
    let mut l = cfg.to_json();
    l["special_tokens"] = a.special_tokens.clone();
    l["generation"] = a.generation.clone();
    l["tokenizer_blob"] = json!(TOKENIZER_BLOB);
    l["chat_template_blob"] = json!(TEMPLATE_BLOB);
    m["lfm3"] = l;
    m["components"] = json!({
        "lfm3": {"architecture": "lfm3", "config": "lfm3", "license": license, "tensor_prefix": ""}
    });
    m
}

pub(crate) fn add_blobs(w: &mut ModelWriter, a: &Assets) -> Result<()> {
    w.add_blob(TOKENIZER_BLOB, "application/json", a.tokenizer_json.as_bytes())?;
    w.add_blob(TEMPLATE_BLOB, "text/x-jinja2", a.chat_template.as_bytes())
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
        let q = "model.layers.2.self_attn.q_proj.weight";
        assert_eq!(t(q, &[2048, 2048], OutDType::Int4), Target::Quant(Format::Int4));
        assert_eq!(t(EMBED, &[128000, 2048], OutDType::Int4), Target::Quant(Format::Int8));
        assert_eq!(t("model.layers.0.conv.conv.weight", &[2048, 1, 7], OutDType::Int8), Target::F16);
        assert_eq!(t("model.layers.2.self_attn.q_norm.weight", &[128], OutDType::F16), Target::F32);
        assert_eq!(t("model.layers.2.self_attn.gate_proj.weight", &[16, 2048], OutDType::Int8), Target::Quant(Format::Int8));
        assert_eq!(t("model.layers.0.post_ffn_norm.weight", &[2048], OutDType::Int8), Target::F32);
    }

    #[test]
    fn lfm3_names() {
        let cfg = Lfm3Config::from_json(&json!({
            "vocab_size": 128000, "hidden_size": 2048, "intermediate_size": 7680, "num_hidden_layers": 24,
            "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 128
        }))
        .expect("cfg");
        let n = expected_tensors(&cfg);
        // 2 globals + 24 * (4 norms + 3 ffn) + 13 conv * 3 + 11 attn * 7 = 286 (the HF checkpoint)
        assert_eq!(n.len(), 286);
        assert!(n.contains(&"model.layers.23.conv.out_proj.weight".to_string()));
        assert!(n.contains(&"model.layers.4.self_attn.gate_proj.weight".to_string()));
    }
}
