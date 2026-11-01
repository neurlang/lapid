//! Qwen3.5 checkpoints → one `.lapid` file: Hugging Face safetensors (repo id or directory;
//! `Qwen3_5ForConditionalGeneration` / `…ForCausalLM`, dense and MoE) and GGUF
//! (`general.architecture` `qwen35` / `qwen35moe`, llama.cpp naming).
//!
//! Output names (the text decoder only; vision tower and MTP head dropped):
//! `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight` (untied models),
//! `model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight`,
//! `self_attn.{q,k,v,o}_proj / q_norm / k_norm`,
//! `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj,conv1d,norm}.weight`,
//! `linear_attn.{A,dt_bias}`, `mlp.{gate,up,down}_proj.weight` (dense),
//! `mlp.gate.weight`, `mlp.experts.{gate,up,down}_proj` ([E, N, K] stacks),
//! `mlp.shared_expert.{gate,up,down}_proj.weight`, `mlp.shared_expert_gate.weight` (MoE).
//!
//! Value conventions (= llama.cpp's GGUF, so GGUF tensors are copied bit-exactly):
//! RMSNorm weights are effective scales (HF `1 + w`; the DeltaNet gated norm is plain `w`),
//! `A = -exp(A_log)`, DeltaNet v heads in tiled order (HF grouped order is permuted at
//! conversion when `linear_num_value_heads > linear_num_key_heads`).
//!
//! Precision: `lapid_convert::policy` (Linear weights quantized; tied embedding ≥ int8; the
//! conv weight f16; norms, router, gates, A, dt_bias f32). Expert stacks are quantized per
//! expert. GGUF imports keep ggml quantization when a quantized dtype is requested: Q8_0 →
//! int8 and Q4_0 → int4 (lossless repack), Q4_K / Q5_K / Q6_K stored as-is.

mod gguf;
mod hf;

use std::path::Path;

use lapid::{DType, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::ggml::GgmlType;
use lapid_convert::gguf::{GgufFile, GgufTensor};
use lapid_convert::policy::{Role, Target, encode, target};
use lapid_convert::writer::ModelWriter;
use lapid::quant::Format;
use serde_json::{Value, json};

use super::config::{LayerType, Qwen35Config};
use super::model::{EMBED, LM_HEAD};
use super::{META, TEMPLATE_BLOB, TOKENIZER_BLOB};

pub(crate) use gguf::is_qwen35_gguf;

/// SPDX id of the Qwen3.5 license (model cards: `license: apache-2.0`).
pub const LICENSE: &str = "apache-2.0";

/// Convert a Qwen3.5 model in Hugging Face format (repo id or local directory).
pub fn convert_hf(repo_or_dir: &str, dtype: OutDType, out: &Path) -> Result<()> {
    hf::convert(repo_or_dir, dtype, out)
}

/// Convert a Qwen3.5 GGUF file.
pub fn convert_gguf(path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    gguf::convert(path, dtype, out)
}

fn is_expert_stack(name: &str) -> bool {
    name.contains(".mlp.experts.")
}

/// Dtype-policy role of a Qwen3.5 tensor (output naming).
pub(crate) fn role(name: &str, shape: &[usize]) -> Role {
    if name.ends_with(".bias")
        || name.contains("norm")
        || shape.len() <= 1
        || name.ends_with(".mlp.gate.weight")
        || name.ends_with("shared_expert_gate.weight")
    {
        Role::Param
    } else if name == EMBED {
        Role::LinearInt8
    } else if shape.len() == 2 || is_expert_stack(name) {
        Role::Linear
    } else {
        Role::Dense
    }
}

/// Every tensor a checkpoint must provide (output naming).
pub(crate) fn expected_tensors(cfg: &Qwen35Config) -> Vec<String> {
    let mut v = vec![EMBED.to_string(), "model.norm.weight".to_string()];
    if !cfg.tie_embeddings {
        v.push(LM_HEAD.to_string());
    }
    for (i, t) in cfg.layer_types.iter().enumerate() {
        let p = format!("model.layers.{i}");
        v.push(format!("{p}.input_layernorm.weight"));
        v.push(format!("{p}.post_attention_layernorm.weight"));
        match t {
            LayerType::FullAttention => {
                for n in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"] {
                    v.push(format!("{p}.self_attn.{n}.weight"));
                }
            }
            LayerType::LinearAttention => {
                for n in ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj", "conv1d", "norm"] {
                    v.push(format!("{p}.linear_attn.{n}.weight"));
                }
                v.push(format!("{p}.linear_attn.A"));
                v.push(format!("{p}.linear_attn.dt_bias"));
            }
        }
        if cfg.layer_is_moe(i) {
            v.push(format!("{p}.mlp.gate.weight"));
            for n in ["gate_proj", "up_proj", "down_proj"] {
                v.push(format!("{p}.mlp.experts.{n}"));
                v.push(format!("{p}.mlp.shared_expert.{n}.weight"));
            }
            v.push(format!("{p}.mlp.shared_expert_gate.weight"));
        } else {
            v.extend(["gate_proj", "up_proj", "down_proj"].iter().map(|n| format!("{p}.mlp.{n}.weight")));
        }
    }
    v
}

/// Non-tensor parts of the output file.
pub(crate) struct Assets {
    pub tokenizer_json: String,
    pub chat_template: String,
    pub special_tokens: Value,
    pub generation: Value,
}

/// Required metadata + the `"qwen35"` object + the single-component declaration.
pub(crate) fn metadata(name: &str, dtype: OutDType, source: Value, cfg: &Qwen35Config, a: &Assets) -> Value {
    let arch = if cfg.is_moe() { "qwen35moe" } else { "qwen35" };
    let mut m = lapid_convert::writer::metadata(name, arch, dtype, LICENSE, source);
    let mut q = cfg.to_json();
    q["special_tokens"] = a.special_tokens.clone();
    q["generation"] = a.generation.clone();
    q["tokenizer_blob"] = json!(TOKENIZER_BLOB);
    q["chat_template_blob"] = json!(TEMPLATE_BLOB);
    q["weight_layout"] = json!({"rms_norm": "effective_scale", "deltanet_A": "neg_exp", "v_heads": "tiled"});
    m[META] = q;
    m["components"] = json!({ META: {"architecture": arch, "config": META, "license": LICENSE, "tensor_prefix": ""} });
    m
}

pub(crate) fn add_blobs(w: &mut ModelWriter, a: &Assets) -> Result<()> {
    w.add_blob(TOKENIZER_BLOB, "application/json", a.tokenizer_json.as_bytes())?;
    w.add_blob(TEMPLATE_BLOB, "text/x-jinja2", a.chat_template.as_bytes())
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

/// Native lapid format of a ggml quantization type.
pub(crate) fn native_format(t: GgmlType) -> Option<Format> {
    match t {
        GgmlType::Q8_0 => Some(Format::Int8),
        GgmlType::Q4_0 => Some(Format::Int4),
        GgmlType::Q4K => Some(Format::Q4K),
        GgmlType::Q5K => Some(Format::Q5K),
        GgmlType::Q6K => Some(Format::Q6K),
        _ => None,
    }
}

pub(crate) fn format_dtype(f: Format) -> DType {
    match f {
        Format::Int8 => DType::Int8,
        Format::Int4 => DType::Int4,
        Format::Q4K => DType::Q4K,
        Format::Q5K => DType::Q5K,
        Format::Q6K => DType::Q6K,
    }
}

/// Write GGUF tensor `t` as `name` (row-major `shape`). With a quantized output dtype, matmul /
/// gather weights stored in a ggml format lapid runs natively are kept bit-exact (Q8_0 / Q4_0
/// repacked, K-quants copied); everything else goes through the policy (dequantized).
pub(crate) fn write_gguf_tensor(f: &GgufFile, w: &mut ModelWriter, dtype: OutDType, name: &str, t: &GgufTensor, shape: &[usize]) -> Result<()> {
    if shape.iter().product::<usize>() != t.numel() {
        bail!("shape_mismatch", "tensor {:?}: shape {shape:?} does not match {:?}", t.name, t.shape);
    }
    let quant_out = !matches!(dtype, OutDType::F32 | OutDType::F16);
    let matmul_weight = matches!(role(name, shape), Role::Linear | Role::LinearInt8);
    let cols = shape.last().copied().unwrap_or(0);
    if let (true, true, Some(fmt)) = (quant_out, matmul_weight, native_format(t.ggml_type))
        && (shape.len() == 2 || shape.len() == 3)
        && cols == t.shape.last().copied().unwrap_or(0)
        && cols % fmt.block_elems() == 0
    {
        let data = f.tensor_data(t);
        let (mats, rows) = if shape.len() == 3 { (shape[0], shape[1]) } else { (1, shape[0]) };
        let per = t.ggml_type.nbytes((rows * cols) as u64)? as usize;
        if per * mats != data.len() {
            bail!("format", "tensor {:?}: {} bytes for {mats} x [{rows}, {cols}]", t.name, data.len());
        }
        let mut bytes = Vec::new();
        for m in 0..mats {
            bytes.extend_from_slice(&lapid::quant::from_ggml_blocks(fmt, &data[m * per..(m + 1) * per], rows, cols));
        }
        return w.add_raw(name, format_dtype(fmt), shape, &bytes);
    }
    add_f32(w, dtype, name, shape, &f.to_f32(t)?)
}

/// Reorder DeltaNet v heads from HF grouped order (`[k-group][v-in-group][dim]`) to tiled
/// (`[v-in-group][k-group][dim]`) along `axis` (0 = rows of a row-major [rows, cols] matrix,
/// 1 = columns). `hk` k heads, `r` v heads per k head, `dim` elements per head.
pub(crate) fn reorder_v_heads(data: &[f32], rows: usize, cols: usize, axis: usize, hk: usize, r: usize, dim: usize) -> Vec<f32> {
    let idx = |j: usize| -> usize {
        // tiled position j = (vi * hk + g) * dim + e  ←  grouped (g * r + vi) * dim + e
        let (head, e) = (j / dim, j % dim);
        let (vi, g) = (head / hk, head % hk);
        (g * r + vi) * dim + e
    };
    let mut out = vec![0.0f32; data.len()];
    if axis == 0 {
        for i in 0..rows {
            let src = idx(i);
            out[i * cols..(i + 1) * cols].copy_from_slice(&data[src * cols..(src + 1) * cols]);
        }
    } else {
        for i in 0..rows {
            for j in 0..cols {
                out[i * cols + j] = data[i * cols + idx(j)];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles() {
        assert_eq!(role(EMBED, &[248320, 1024]), Role::LinearInt8);
        assert_eq!(role("model.layers.0.linear_attn.conv1d.weight", &[6144, 1, 4]), Role::Dense);
        assert_eq!(role("model.layers.0.linear_attn.norm.weight", &[128]), Role::Param);
        assert_eq!(role("model.layers.0.mlp.gate.weight", &[256, 2048]), Role::Param);
        assert_eq!(role("model.layers.0.mlp.experts.gate_proj", &[256, 512, 2048]), Role::Linear);
        assert_eq!(role("model.layers.0.mlp.shared_expert_gate.weight", &[1, 2048]), Role::Param);
        assert_eq!(role("model.layers.3.self_attn.q_proj.weight", &[4096, 1024]), Role::Linear);
    }

    #[test]
    fn reorder_heads() {
        // hk = 2 groups, r = 2 v per group, dim 1: grouped [g0v0, g0v1, g1v0, g1v1] → tiled [g0v0, g1v0, g0v1, g1v1]
        let d = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(reorder_v_heads(&d, 4, 1, 0, 2, 2, 1), vec![0.0, 2.0, 1.0, 3.0]);
        assert_eq!(reorder_v_heads(&d, 1, 4, 1, 2, 2, 1), vec![0.0, 2.0, 1.0, 3.0]);
    }

    #[test]
    fn names_dense() {
        let cfg = Qwen35Config::from_json(&json!({"model_type": "qwen3_5", "tie_word_embeddings": true, "text_config": {
            "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16}}))
        .expect("cfg");
        let n = expected_tensors(&cfg);
        // 2 globals + 4 * (2 norms + 3 mlp) + 3 linear * 9 + 1 attn * 6
        assert_eq!(n.len(), 2 + 20 + 27 + 6);
    }
}
