//! Gemma 4 checkpoints → one `.lapid` file: Hugging Face safetensors (repo id or directory,
//! `Gemma4ForConditionalGeneration` — only `model.language_model.*` is kept) and GGUF
//! (llama.cpp `gemma4` naming).
//!
//! Both emit the text-decoder tensors under `model.*` (HF names without the
//! `language_model` level), the `"gemma4"` metadata object (HF text_config keys +
//! `special_tokens`, `generation`, blob names) and two blobs: `tokenizer/tokenizer.json`
//! and `tokenizer/chat_template.jinja`. Precision follows `lapid_convert::policy`: Linear
//! weights quantized; the tied embedding and the per-layer embedding table at most int8;
//! norms / `layer_scalar` f32. KV-shared layers carry no k/v projections.
//!
//! GGUF imports at a quantized dtype keep every quantized weight in its stored format
//! (Q4_K / Q5_K / Q6_K as-is, Q8_0 → int8 and Q4_0 → int4 losslessly); float weights stay
//! f16 instead of being re-quantized.

mod gguf;
mod hf;

use std::path::Path;

use lapid::quant::Format;
use lapid::{DType, Result};
use lapid_convert::OutDType;
use lapid_convert::ggml::GgmlType;
use lapid_convert::gguf::{GgufFile, GgufTensor};
use lapid_convert::policy::{Role, Target};
use lapid_convert::writer::ModelWriter;
use serde_json::{Value, json};

use super::config::Gemma4Config;
use super::model::{EMBED, FINAL_NORM, PLE_EMBED, PLE_NORM, PLE_PROJ};
use super::{TEMPLATE_BLOB, TOKENIZER_BLOB};

pub(crate) use gguf::is_gemma4_gguf;

/// License id of the Gemma 4 models (Apache 2.0 per the model card).
pub const LICENSE: &str = "apache-2.0";

/// Convert a Gemma 4 model in Hugging Face format (repo id or local directory).
pub fn convert_hf(repo_or_dir: &str, dtype: OutDType, out: &Path) -> Result<()> {
    hf::convert(repo_or_dir, dtype, out)
}

/// Convert a Gemma 4 GGUF file.
pub fn convert_gguf(path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    gguf::convert(path, dtype, out)
}

/// Dtype-policy role of a Gemma 4 tensor (lapid naming).
pub(crate) fn role(name: &str, shape: &[usize]) -> Role {
    if name.ends_with(".bias") || name.contains("norm") || name.ends_with("layer_scalar") || shape.len() <= 1 {
        Role::Param
    } else if name == EMBED || name == PLE_EMBED {
        Role::LinearInt8
    } else if shape.len() == 2 {
        Role::Linear
    } else {
        Role::Dense
    }
}

/// Every tensor a Gemma 4 text decoder must provide (lapid naming).
pub(crate) fn expected_tensors(cfg: &Gemma4Config) -> Vec<String> {
    let mut v = vec![EMBED.to_string(), format!("{FINAL_NORM}.weight")];
    if cfg.ple_dim > 0 {
        v.extend([PLE_EMBED.to_string(), format!("{PLE_PROJ}.weight"), format!("{PLE_NORM}.weight")]);
    }
    for i in 0..cfg.n_layers {
        let p = format!("model.layers.{i}");
        for n in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
            v.push(format!("{p}.{n}.weight"));
        }
        v.push(format!("{p}.layer_scalar"));
        v.extend(["gate_proj", "up_proj", "down_proj"].iter().map(|w| format!("{p}.mlp.{w}.weight")));
        v.extend(["q_proj", "o_proj", "q_norm"].iter().map(|w| format!("{p}.self_attn.{w}.weight")));
        if !cfg.is_shared(i) {
            v.push(format!("{p}.self_attn.k_proj.weight"));
            v.push(format!("{p}.self_attn.k_norm.weight"));
            if !cfg.k_eq_v(i) {
                v.push(format!("{p}.self_attn.v_proj.weight"));
            }
        }
        if cfg.ple_dim > 0 {
            for n in ["per_layer_input_gate", "per_layer_projection", "post_per_layer_input_norm"] {
                v.push(format!("{p}.{n}.weight"));
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

/// Required metadata + the `"gemma4"` object + the single-component declaration.
pub(crate) fn metadata(name: &str, dtype: OutDType, source: Value, cfg: &Gemma4Config, a: &Assets) -> Value {
    let mut m = lapid_convert::writer::metadata(name, "gemma4", dtype, LICENSE, source);
    let mut g = cfg.to_json();
    g["special_tokens"] = a.special_tokens.clone();
    g["generation"] = a.generation.clone();
    g["tokenizer_blob"] = json!(TOKENIZER_BLOB);
    g["chat_template_blob"] = json!(TEMPLATE_BLOB);
    m["gemma4"] = g;
    m["components"] = json!({
        "gemma4": {"architecture": "gemma4", "config": "gemma4", "license": LICENSE, "tensor_prefix": ""}
    });
    m
}

pub(crate) fn add_blobs(w: &mut ModelWriter, a: &Assets) -> Result<()> {
    w.add_blob(TOKENIZER_BLOB, "application/json", a.tokenizer_json.as_bytes())?;
    w.add_blob(TEMPLATE_BLOB, "text/x-jinja2", a.chat_template.as_bytes())
}

/// Write GGUF tensor `t` as `name`, keeping quantized weights in their stored format when
/// `dtype` is quantized (K-quants as-is, Q8_0 → int8, Q4_0 → int4) and float weights f16;
/// otherwise through the writer's dtype policy (`GgufFile::write_tensor`).
/// Candidate for lapid-convert (`write_tensor_native`) — Qwen3.5 needs the same.
pub(crate) fn write_gguf_native(
    f: &GgufFile,
    w: &mut ModelWriter,
    name: &str,
    t: &GgufTensor,
    shape: Option<&[usize]>,
    dtype: OutDType,
) -> Result<()> {
    let shape = shape.map_or_else(|| t.shape.clone(), <[usize]>::to_vec);
    let quantized_out = !matches!(dtype, OutDType::F32 | OutDType::F16);
    let weight = matches!(role(name, &shape), Role::Linear | Role::LinearInt8) && shape.len() == 2;
    if !(quantized_out && weight) {
        return f.write_tensor(w, name, t, Some(&shape));
    }
    let data = f.tensor_data(t);
    let (rows, cols) = (shape[0], shape[1]);
    let native_cols = cols == t.shape.last().copied().unwrap_or(0);
    let native = match t.ggml_type {
        GgmlType::Q4K => Some(Format::Q4K),
        GgmlType::Q5K => Some(Format::Q5K),
        GgmlType::Q6K => Some(Format::Q6K),
        GgmlType::Q8_0 => Some(Format::Int8),
        GgmlType::Q4_0 => Some(Format::Int4),
        _ => None,
    };
    if let Some(fmt) = native.filter(|f| native_cols && cols % f.block_elems() == 0) {
        let dt = match fmt {
            Format::Q4K => DType::Q4K,
            Format::Q5K => DType::Q5K,
            Format::Q6K => DType::Q6K,
            Format::Int8 => DType::Int8,
            Format::Int4 => DType::Int4,
        };
        return w.add_raw(name, dt, &shape, &lapid::quant::from_ggml_blocks(fmt, data, rows, cols));
    }
    match t.ggml_type {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
            let (dt, bytes) = lapid_convert::policy::encode(Target::F16, &f.to_f32(t)?, &shape)?;
            w.add_raw(name, dt, &shape, &bytes)
        }
        _ => f.write_tensor(w, name, t, Some(&shape)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lapid_convert::policy::target;

    fn t(name: &str, shape: &[usize], dtype: OutDType) -> Target {
        target(role(name, shape), shape, dtype)
    }

    #[test]
    fn policy_table() {
        let q = "model.layers.2.self_attn.q_proj.weight";
        assert_eq!(t(q, &[2048, 1536], OutDType::Int4), Target::Quant(Format::Int4));
        assert_eq!(t(EMBED, &[262144, 1536], OutDType::Int4), Target::Quant(Format::Int8));
        assert_eq!(t(PLE_EMBED, &[262144, 8960], OutDType::Int4), Target::Quant(Format::Int8));
        assert_eq!(t("model.layers.0.layer_scalar", &[1], OutDType::Int8), Target::F32);
        assert_eq!(t("model.layers.2.self_attn.q_norm.weight", &[256], OutDType::F16), Target::F32);
    }

    #[test]
    fn e2b_names() {
        let cfg = Gemma4Config::from_json(&crate::gemma4::config::test_support::e2b()).expect("cfg");
        let n = expected_tensors(&cfg);
        // 5 globals + 35 * (4 norms + scalar + 3 mlp + 3 q/o/q_norm + 3 ple) + 15 * 3 k/v/k_norm
        assert_eq!(n.len(), 5 + 35 * 14 + 15 * 3);
        assert!(!n.contains(&"model.layers.20.self_attn.k_proj.weight".to_string()));
        assert!(n.contains(&"model.layers.14.self_attn.v_proj.weight".to_string()));
    }
}
