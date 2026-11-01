//! Qwen3-VL checkpoints → one `.lapid` file (vision encoder + text decoder + tokenizer +
//! chat template): Hugging Face safetensors (`Qwen3VLForConditionalGeneration`, repo id or
//! directory) and a GGUF pair (llama.cpp `qwen3vl` model + `clip` mmproj with projector
//! `qwen3vl`).
//!
//! Text tensors use HF names without the `language_model` level (`model.embed_tokens.weight`,
//! `model.layers.{i}.self_attn.q_proj.weight`, ..., `lm_head.weight` when untied); vision
//! tensors use the mmproj names of [`crate::vision::encoder`] (`v.blk.{i}.attn_qkv.weight`,
//! `mm.0.weight`, `v.deepstack.{layer}.fc1.weight`, ...), with the Conv3d patch kernel
//! folded over its temporal axis to [D, 3*P*P].
//!
//! Precision (`lapid_convert::policy`): text Linear weights quantized; the tied embedding at
//! most int8; vision weights f16 (never quantized, see `role`); norms / biases f32.
//! GGUF imports at a quantized dtype keep stored ggml formats (K-quants as-is, Q8_0 → int8,
//! Q4_0 → int4).

mod gguf;
mod hf;

use std::path::{Path, PathBuf};

use lapid::{Error, Result};
use lapid_convert::OutDType;
use lapid_convert::gguf::GgufFile;
use lapid_convert::policy::Role;
use lapid_convert::writer::ModelWriter;
use serde_json::{Value, json};

use super::config::Qwen3VlConfig;
use super::model::{EMBED, LM_HEAD};
use super::{META, TEMPLATE_BLOB, TOKENIZER_BLOB};

pub const LICENSE: &str = "apache-2.0";

/// Convert a Qwen3-VL model in Hugging Face format (repo id or local directory).
pub fn convert_hf(repo_or_dir: &str, dtype: OutDType, out: &Path) -> Result<()> {
    hf::convert(repo_or_dir, dtype, out)
}

/// Convert a Qwen3-VL GGUF; the vision encoder comes from `$LAPID_MMPROJ` or a sibling
/// `mmproj*.gguf` in the same directory.
pub fn convert_gguf(path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    let mmproj = find_mmproj(path)?;
    gguf::convert(path, &mmproj, dtype, out)
}

/// Convert a model GGUF + mmproj GGUF pair.
pub fn convert_gguf_pair(model: &Path, mmproj: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    gguf::convert(model, mmproj, dtype, out)
}

pub(crate) fn is_qwen3vl_gguf(f: &GgufFile) -> bool {
    f.architecture() == Some("qwen3vl")
}

/// `$LAPID_MMPROJ`, else the only / first `mmproj*.gguf` next to `model`.
pub fn find_mmproj(model: &Path) -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("LAPID_MMPROJ").filter(|v| !v.is_empty()) {
        return Ok(PathBuf::from(p));
    }
    let dir = model.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| Error::new("io", format!("{}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let n = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
            n.starts_with("mmproj") && n.ends_with(".gguf")
        })
        .collect();
    found.sort();
    found.into_iter().next().ok_or_else(|| {
        Error::new("not_found", format!("{}: no mmproj*.gguf next to the model (set LAPID_MMPROJ)", model.display()))
    })
}

/// Dtype-policy role of a tensor (lapid naming).
pub(crate) fn role(name: &str, shape: &[usize]) -> Role {
    let vision = name.starts_with("v.") || name.starts_with("mm.");
    let is_norm = name.split('.').any(|s| s.starts_with("ln") || s.ends_with("_ln") || s.contains("norm"));
    if name.ends_with(".bias") || is_norm || shape.len() <= 1 {
        Role::Param
    } else if vision {
        // Never quantized: ViT activations have outliers that int8 / int4 weights (with
        // on-the-fly int8 activations) amplify through the 24-block tower.
        Role::Dense
    } else if name == EMBED {
        Role::LinearInt8
    } else if shape.len() == 2 {
        Role::Linear
    } else {
        Role::Dense
    }
}

/// Every text-decoder tensor (lapid naming).
pub(crate) fn text_tensors(cfg: &Qwen3VlConfig) -> Vec<String> {
    let t = &cfg.text;
    let mut v = vec![EMBED.to_string(), "model.norm.weight".to_string()];
    if !t.tie_embeddings {
        v.push(LM_HEAD.to_string());
    }
    for i in 0..t.n_layers {
        let p = format!("model.layers.{i}");
        v.push(format!("{p}.input_layernorm.weight"));
        v.push(format!("{p}.post_attention_layernorm.weight"));
        for w in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            v.push(format!("{p}.self_attn.{w}.weight"));
            if t.attention_bias {
                v.push(format!("{p}.self_attn.{w}.bias"));
            }
        }
        v.push(format!("{p}.self_attn.q_norm.weight"));
        v.push(format!("{p}.self_attn.k_norm.weight"));
        for w in ["gate_proj", "up_proj", "down_proj"] {
            v.push(format!("{p}.mlp.{w}.weight"));
        }
    }
    v
}

pub(crate) fn expected_tensors(cfg: &Qwen3VlConfig) -> Vec<String> {
    let mut v = text_tensors(cfg);
    v.extend(cfg.vision.expected_tensors(false, false));
    v
}

pub(crate) struct Assets {
    pub tokenizer_json: String,
    pub chat_template: String,
    pub special_tokens: Value,
    pub generation: Value,
}

pub(crate) fn metadata(name: &str, dtype: OutDType, source: Value, cfg: &Qwen3VlConfig, a: &Assets) -> Value {
    let mut m = lapid_convert::writer::metadata(name, "qwen3vl", dtype, LICENSE, source);
    let mut q = cfg.to_json();
    q["special_tokens"] = a.special_tokens.clone();
    q["generation"] = a.generation.clone();
    q["tokenizer_blob"] = json!(TOKENIZER_BLOB);
    q["chat_template_blob"] = json!(TEMPLATE_BLOB);
    m[META] = q;
    m["components"] = json!({ META: {"architecture": "qwen3vl", "config": META, "license": LICENSE, "tensor_prefix": ""} });
    m
}

pub(crate) fn add_blobs(w: &mut ModelWriter, a: &Assets) -> Result<()> {
    w.add_blob(TOKENIZER_BLOB, "application/json", a.tokenizer_json.as_bytes())?;
    w.add_blob(TEMPLATE_BLOB, "text/x-jinja2", a.chat_template.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_and_names() {
        assert_eq!(role("v.blk.0.attn_qkv.weight", &[3072, 1024]), Role::Dense);
        assert_eq!(role("v.blk.0.ln1.weight", &[1024]), Role::Param);
        assert_eq!(role("v.position_embd.weight", &[2304, 1024]), Role::Dense);
        assert_eq!(role("model.layers.0.self_attn.q_proj.weight", &[2048, 2048]), Role::Linear);
        assert_eq!(role(EMBED, &[151936, 2048]), Role::LinearInt8);
        let cfg = Qwen3VlConfig::from_json(&crate::qwen3vl::config::test_support::q2b(), None).expect("cfg");
        let n = expected_tensors(&cfg);
        // text: 2 + 28 * 11; vision: 2 + 1 + 24 * 12 + 2 + 4 + 3 * 6
        assert_eq!(n.len(), 2 + 28 * 11 + 2 + 1 + 24 * 12 + 2 + 4 + 3 * 6);
    }
}
