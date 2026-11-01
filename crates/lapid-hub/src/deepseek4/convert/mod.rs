//! DeepSeek V4 checkpoints → one `.lapid` file: upstream safetensors (the layout of the
//! `deepseek-ai/DeepSeek-V4-*` repos: `embed.weight`, `layers.{i}.attn.wq_a.weight`,
//! `layers.{i}.ffn.experts.{e}.w1.weight`, ...; bf16 / f16 / f32 tensors) and GGUF
//! (`general.architecture` `deepseek4`, llama.cpp naming).
//!
//! Output names = upstream names, except: routed experts stacked as
//! `layers.{i}.ffn.experts.{w1,w2,w3}` `[E, N, K]`, `attn.wo_a.weight` reshaped to
//! `[G, o_lora, H / G * head_dim]` (one MatMulId "expert" per output group), hash routing
//! tables `ffn.gate.tid2eid` stored as i32 `[vocab, k]`. The MTP block is dropped.
//!
//! Precision (`lapid_convert::policy`): Linear weights quantized for int8 / int4 / k-quant
//! outputs (expert stacks and wo_a per group), embedding ≥ int8; compressor and indexer-weight
//! projections (fp32 in HF), norms, sinks, hyper-connection parameters, ape, router and
//! biases f32. GGUF imports keep Q8_0 (→ int8), Q4_0 (→ int4)
//! and Q4_K / Q5_K / Q6_K weights as stored; MXFP4 expert stacks and other ggml types are
//! dequantized and re-encoded by the policy.

mod gguf;
mod hf;

use std::path::Path;

use lapid::quant::Format;
use lapid::{DType, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::ggml::GgmlType;
use lapid_convert::gguf::GgufTensor;
use lapid_convert::policy::{Role, Target, encode, target};
use lapid_convert::writer::ModelWriter;
use serde_json::{Value, json};

use super::config::Deepseek4Config;
use super::model::{EMBED, HEAD};
use super::{META, TOKENIZER_BLOB};

pub(crate) use gguf::is_deepseek4_gguf;

/// SPDX id of the DeepSeek V4 license (model cards: `license: mit`).
pub const LICENSE: &str = "mit";

/// Convert an upstream DeepSeek V4 checkpoint (repo id or local directory).
pub fn convert_hf(repo_or_dir: &str, dtype: OutDType, out: &Path) -> Result<()> {
    hf::convert(repo_or_dir, dtype, out)
}

/// Convert a DeepSeek V4 GGUF file.
pub fn convert_gguf(path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    gguf::convert(path, dtype, out)
}

/// 3-D tensors stored per matrix (`[E, N, K]`).
fn is_stack(name: &str) -> bool {
    name.contains(".ffn.experts.") || name.ends_with(".attn.wo_a.weight")
}

/// Dtype-policy role of a DeepSeek V4 tensor (output naming).
pub(crate) fn role(name: &str, shape: &[usize]) -> Role {
    if shape.len() <= 1
        || name.contains("norm")
        || name.ends_with(".bias")
        || name.ends_with(".ape")
        || name.ends_with("attn_sink")
        || name.contains("hc_")
        || name.ends_with(".ffn.gate.weight")
        // HF `_keep_in_fp32_modules`: the compressors pool many tokens and amplify input
        // rounding, and the indexer weights pick the top-k entries
        || name.contains(".compressor.")
        || name.ends_with("weights_proj.weight")
    {
        Role::Param
    } else if name == EMBED {
        Role::LinearInt8
    } else {
        Role::Linear
    }
}

/// Every tensor a checkpoint must provide (output naming).
pub(crate) fn expected_tensors(cfg: &Deepseek4Config) -> Vec<String> {
    let mut v: Vec<String> = [EMBED, HEAD, "norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"].iter().map(|s| s.to_string()).collect();
    for i in 0..cfg.n_layers {
        let p = format!("layers.{i}");
        for n in ["attn_norm.weight", "ffn_norm.weight"] {
            v.push(format!("{p}.{n}"));
        }
        for k in ["attn", "ffn"] {
            for s in ["fn", "base", "scale"] {
                v.push(format!("{p}.hc_{k}_{s}"));
            }
        }
        for n in ["attn_sink", "wq_a.weight", "q_norm.weight", "wq_b.weight", "wkv.weight", "kv_norm.weight", "wo_a.weight", "wo_b.weight"] {
            v.push(format!("{p}.attn.{n}"));
        }
        let ratio = cfg.compress_ratios[i];
        let comp = |v: &mut Vec<String>, c: &str| {
            for n in ["wkv.weight", "wgate.weight", "ape", "norm.weight"] {
                v.push(format!("{c}.{n}"));
            }
        };
        if ratio > 0 {
            comp(&mut v, &format!("{p}.attn.compressor"));
        }
        if ratio == 4 {
            v.push(format!("{p}.attn.indexer.wq_b.weight"));
            v.push(format!("{p}.attn.indexer.weights_proj.weight"));
            comp(&mut v, &format!("{p}.attn.indexer.compressor"));
        }
        v.push(format!("{p}.ffn.gate.weight"));
        v.push(format!("{p}.ffn.gate.{}", if i < cfg.n_hash_layers { "tid2eid" } else { "bias" }));
        for w in ["w1", "w2", "w3"] {
            v.push(format!("{p}.ffn.experts.{w}"));
            v.push(format!("{p}.ffn.shared_experts.{w}.weight"));
        }
    }
    v
}

/// Non-tensor parts of the output file.
pub(crate) struct Assets {
    pub tokenizer_json: String,
    pub generation: Value,
}

/// Required metadata + the `"deepseek4"` object + the single-component declaration.
pub(crate) fn metadata(name: &str, dtype: OutDType, source: Value, cfg: &Deepseek4Config, a: &Assets) -> Value {
    let mut m = lapid_convert::writer::metadata(name, "deepseek4", dtype, LICENSE, source);
    let mut q = cfg.to_json();
    q["generation"] = a.generation.clone();
    q["tokenizer_blob"] = json!(TOKENIZER_BLOB);
    q["prompt_format"] = json!("encoding_dsv4");
    q["weight_layout"] = json!({"experts": "stacked [E, N, K]", "wo_a": "[G, o_lora, H/G*head_dim]", "tid2eid": "i32 [vocab, k]"});
    m[META] = q;
    m["components"] = json!({ META: {"architecture": "deepseek4", "config": META, "license": LICENSE, "tensor_prefix": ""} });
    m
}

pub(crate) fn add_blobs(w: &mut ModelWriter, a: &Assets) -> Result<()> {
    w.add_blob(TOKENIZER_BLOB, "application/json", a.tokenizer_json.as_bytes())
}

/// Write an f32 tensor through the policy; 3-D stacks are encoded per matrix.
pub(crate) fn add_f32(w: &mut ModelWriter, dtype: OutDType, name: &str, shape: &[usize], data: &[f32]) -> Result<()> {
    if shape.len() == 3 && is_stack(name) {
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

/// i32 hash routing table `[vocab, k]`.
pub(crate) fn add_i32(w: &mut ModelWriter, name: &str, shape: &[usize], data: &[i32]) -> Result<()> {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    w.add_raw(name, DType::I32, shape, &bytes)
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

/// Write GGUF tensor `t` (raw ggml bytes `data`) as `name` with row-major `shape`. With a
/// quantized output dtype, Linear weights in a ggml format lapid runs natively are kept
/// bit-exact (Q8_0 / Q4_0 repacked, K-quants copied; 3-D stacks per matrix); everything else
/// goes through the policy (dequantized f32).
pub(crate) fn write_gguf_tensor(w: &mut ModelWriter, dtype: OutDType, name: &str, t: &GgufTensor, data: &[u8], f32_data: impl FnOnce() -> Result<Vec<f32>>, shape: &[usize]) -> Result<()> {
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
    add_f32(w, dtype, name, shape, &f32_data()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles() {
        assert_eq!(role(EMBED, &[129280, 4096]), Role::LinearInt8);
        assert_eq!(role("layers.0.hc_attn_fn", &[24, 16384]), Role::Param);
        assert_eq!(role("layers.1.attn.compressor.wkv.weight", &[1024, 4096]), Role::Param);
        assert_eq!(role("layers.1.attn.indexer.wq_b.weight", &[8192, 1024]), Role::Linear);
        assert_eq!(role("layers.1.attn.wo_a.weight", &[8, 1024, 4096]), Role::Linear);
        assert_eq!(role("layers.1.ffn.gate.weight", &[256, 4096]), Role::Param);
        assert_eq!(role("layers.1.attn.attn_sink", &[64]), Role::Param);
    }

    #[test]
    fn names() {
        let cfg = Deepseek4Config::from_json(&json!({"vocab_size": 100, "hidden_size": 64, "num_hidden_layers": 3,
            "num_attention_heads": 4, "head_dim": 512, "qk_rope_head_dim": 64, "q_lora_rank": 32, "o_groups": 2,
            "o_lora_rank": 16, "n_routed_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
            "sliding_window": 16, "compress_ratios": [0, 4, 128], "num_hash_layers": 1, "index_n_heads": 2,
            "index_head_dim": 128, "index_topk": 4}))
        .expect("cfg");
        let n = expected_tensors(&cfg);
        // 6 globals + 3 * (2 norms + 6 hc + 8 attn + 2 gate + 6 ffn) + 4 (HCA comp) + 10 (CSA comp + indexer)
        assert_eq!(n.len(), 6 + 3 * 24 + 4 + 10);
    }
}
