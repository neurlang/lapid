//! Gemma 4 checkpoints (dense or MoE) → one `.lapid` file: Hugging Face safetensors (repo id
//! or directory; the vision tower is kept when present) and GGUF (llama.cpp `gemma4` naming,
//! optionally paired with an `mmproj` GGUF holding the vision tower).
//!
//! Tensors use HF names without the `language_model` level (`model.layers.N.…`); MoE layers
//! add `router.{proj.weight,scale,per_expert_scale}`, `experts.gate_up_proj` [E, 2I, D] (gate
//! rows first), `experts.down_proj` [E, D, I] and the `pre_feedforward_layernorm_2` /
//! `post_feedforward_layernorm_{1,2}` norms. Vision tensors keep their HF names
//! (`model.vision_tower.*`, `model.embed_vision.*`). Metadata object `"gemma4"`: HF
//! text_config keys (MoE keys included) + `vision_config`, `image_processor`, image token
//! ids, `special_tokens`, `generation`, blob names.
//!
//! Precision follows `lapid_convert::policy`: Linear weights and expert stacks quantized
//! (stacks per expert); the tied embedding, the PLE table and vision linears at most int8;
//! norms, scalars, router and position tables float. GGUF imports at a quantized dtype keep
//! quantized weights in their stored format (K-quants as-is incl. [E, N, K] stacks, Q8_0 →
//! int8, Q4_0 → int4).

mod gguf;
mod hf;

use std::path::Path;

use lapid::{DType, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::ggml::GgmlType;
use lapid_convert::gguf::{GgufFile, GgufTensor};
use lapid_convert::policy::{Role, Target, encode, target};
use lapid_convert::writer::ModelWriter;
use serde_json::{Value, json};

use super::config::Gemma4MoeConfig;
use super::model::{EMBED, PLE_EMBED};
use super::{TEMPLATE_BLOB, TOKENIZER_BLOB};
pub(crate) use crate::gemma4::convert::{Assets, add_blobs};
use crate::qwen35::convert::{format_dtype, native_format};

pub(crate) use gguf::is_gemma4moe_gguf;

/// License id of the Gemma 4 models (Apache 2.0 per the model card).
pub const LICENSE: &str = "apache-2.0";

/// Convert a Gemma 4 model in Hugging Face format (repo id or local directory), keeping the
/// vision tower when the checkpoint has one.
pub fn convert_hf(repo_or_dir: &str, dtype: OutDType, out: &Path) -> Result<()> {
    hf::convert(repo_or_dir, dtype, out)
}

/// The HF converter keeping the vision tower (dense E2B / E4B too): same as [`convert_hf`].
pub fn convert_hf_full(repo_or_dir: &str, dtype: OutDType, out: &Path) -> Result<()> {
    hf::convert(repo_or_dir, dtype, out)
}

/// Convert a Gemma 4 GGUF file; a sibling `mmproj*.gguf` in the same directory adds the
/// vision tower (when that projector cannot be read, the text model is converted alone and
/// a warning is printed). `LAPID_MMPROJ=none` skips the lookup.
pub fn convert_gguf(path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    let skip = std::env::var("LAPID_MMPROJ").is_ok_and(|v| v == "none");
    let mmproj = if skip { None } else { gguf::sibling_mmproj(path) };
    if let Some(m) = &mmproj {
        match gguf::convert(path, Some(m), dtype, out) {
            Ok(()) => return Ok(()),
            Err(e) => eprintln!("[lapid-hub] gemma4moe: {} not used ({}: {}); converting the text model only", m.display(), e.code, e.message),
        }
    }
    gguf::convert(path, None, dtype, out)
}

/// Convert a Gemma 4 GGUF model file plus an optional `mmproj` GGUF (vision tower).
pub fn convert_gguf_pair(model: &Path, mmproj: Option<&Path>, dtype: OutDType, out: &Path) -> Result<()> {
    gguf::convert(model, mmproj, dtype, out)
}

fn is_vision(name: &str) -> bool {
    name.starts_with("model.vision_tower.") || name.starts_with("model.embed_vision.")
}

fn is_expert_stack(name: &str) -> bool {
    name.ends_with(".experts.gate_up_proj") || name.ends_with(".experts.down_proj")
}

/// Dtype-policy role of a tensor (lapid naming).
pub(crate) fn role(name: &str, shape: &[usize]) -> Role {
    if name.ends_with(".bias") || name.contains("norm") || name.ends_with("layer_scalar") || name.contains(".router.") || shape.len() <= 1 {
        Role::Param
    } else if is_expert_stack(name) {
        Role::Linear
    } else if name == EMBED || name == PLE_EMBED || (is_vision(name) && shape.len() == 2) {
        Role::LinearInt8
    } else if shape.len() == 2 {
        Role::Linear
    } else {
        Role::Dense
    }
}

/// Every text-decoder tensor (lapid naming).
pub(crate) fn expected_tensors(cfg: &Gemma4MoeConfig) -> Vec<String> {
    let mut v = crate::gemma4::convert::expected_tensors(&cfg.text);
    if cfg.moe.is_some() {
        for i in 0..cfg.text.n_layers {
            let p = format!("model.layers.{i}");
            for n in [
                "router.proj.weight",
                "router.scale",
                "router.per_expert_scale",
                "experts.gate_up_proj",
                "experts.down_proj",
                "pre_feedforward_layernorm_2.weight",
                "post_feedforward_layernorm_1.weight",
                "post_feedforward_layernorm_2.weight",
            ] {
                v.push(format!("{p}.{n}"));
            }
        }
    }
    v
}

/// Required metadata + the `"gemma4"` object + the single-component declaration.
pub(crate) fn metadata(name: &str, dtype: OutDType, source: Value, cfg: &Gemma4MoeConfig, a: &Assets) -> Value {
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

/// Write an f32 tensor through the policy; `[E, N, K]` expert stacks are encoded per expert
/// (the policy only quantizes 2-D matrices).
pub(crate) fn add_f32(w: &mut ModelWriter, dtype: OutDType, name: &str, shape: &[usize], data: &[f32]) -> Result<()> {
    if shape.len() == 3 && is_expert_stack(name) {
        let (e, n, k) = (shape[0], shape[1], shape[2]);
        let t = target(Role::Linear, &[n, k], dtype);
        if let Target::Quant(_) = t {
            let mut bytes = Vec::new();
            let mut dt = DType::F32;
            for x in 0..e {
                let (d, b) = encode(t, &data[x * n * k..(x + 1) * n * k], &[n, k])?;
                dt = d;
                bytes.extend_from_slice(&b);
            }
            return w.add_raw(name, dt, shape, &bytes);
        }
        let (dt, bytes) = encode(t, data, shape)?;
        return w.add_raw(name, dt, shape, &bytes);
    }
    w.add_f32(name, shape, data)
}

/// Write GGUF data as `name` (row-major `shape`). `raw` holds `mats` matrices of
/// `[rows, cols]` in ggml type `ty` back to back. With a quantized output dtype, matmul
/// weights in a format lapid runs natively are kept bit-exact (Q8_0 / Q4_0 repacked, K-quants
/// copied); float weights become f16; everything else goes through the policy.
pub(crate) fn write_ggml(w: &mut ModelWriter, dtype: OutDType, name: &str, t: &GgufTensor, raw: &[u8], f32_data: impl FnOnce() -> Result<Vec<f32>>, shape: &[usize]) -> Result<()> {
    let quant_out = !matches!(dtype, OutDType::F32 | OutDType::F16);
    let matmul_weight = matches!(role(name, shape), Role::Linear | Role::LinearInt8);
    let cols = shape.last().copied().unwrap_or(0);
    if let (true, true, Some(fmt)) = (quant_out, matmul_weight, native_format(t.ggml_type))
        && (shape.len() == 2 || shape.len() == 3)
        && cols % fmt.block_elems() == 0
    {
        let (mats, rows) = if shape.len() == 3 { (shape[0], shape[1]) } else { (1, shape[0]) };
        let per = t.ggml_type.nbytes((rows * cols) as u64)? as usize;
        if per * mats != raw.len() {
            bail!("format", "tensor {name:?}: {} bytes for {mats} x [{rows}, {cols}]", raw.len());
        }
        let mut bytes = Vec::new();
        for m in 0..mats {
            bytes.extend_from_slice(&lapid::quant::from_ggml_blocks(fmt, &raw[m * per..(m + 1) * per], rows, cols));
        }
        return w.add_raw(name, format_dtype(fmt), shape, &bytes);
    }
    let data = f32_data()?;
    if quant_out && matmul_weight && shape.len() == 2 && matches!(t.ggml_type, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
        let (dt, bytes) = encode(Target::F16, &data, shape)?;
        return w.add_raw(name, dt, shape, &bytes);
    }
    add_f32(w, dtype, name, shape, &data)
}

/// Write one GGUF tensor (see [`write_ggml`]).
pub(crate) fn write_gguf_tensor(f: &GgufFile, w: &mut ModelWriter, dtype: OutDType, name: &str, t: &GgufTensor, shape: &[usize]) -> Result<()> {
    if shape.iter().product::<usize>() != t.numel() {
        bail!("shape_mismatch", "tensor {:?}: shape {shape:?} does not match {:?}", t.name, t.shape);
    }
    write_ggml(w, dtype, name, t, f.tensor_data(t), || f.to_f32(t), shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles() {
        assert_eq!(role("model.layers.0.experts.gate_up_proj", &[128, 1408, 2816]), Role::Linear);
        assert_eq!(role("model.layers.0.router.proj.weight", &[128, 2816]), Role::Param);
        assert_eq!(role("model.layers.0.router.per_expert_scale", &[128]), Role::Param);
        assert_eq!(role(EMBED, &[262144, 2816]), Role::LinearInt8);
        assert_eq!(role("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", &[768, 768]), Role::LinearInt8);
        assert_eq!(role("model.vision_tower.patch_embedder.position_embedding_table", &[2, 10240, 768]), Role::Dense);
        assert_eq!(role("model.vision_tower.encoder.layers.0.mlp.up_proj.input_max", &[]), Role::Param);
        assert_eq!(role("model.layers.3.self_attn.q_proj.weight", &[4096, 2816]), Role::Linear);
    }

    #[test]
    fn a4b_names() {
        let cfg = Gemma4MoeConfig::from_json(&crate::gemma4moe::config::test_support::a4b()).expect("cfg");
        let n = expected_tensors(&cfg);
        // 2 globals + 30 * (4 norms + scalar + 3 mlp + 3 q/o/q_norm + k/k_norm + 8 moe) + 25 sliding v_proj
        assert_eq!(n.len(), 2 + 30 * 21 + 25);
    }
}
