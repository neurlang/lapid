//! Llama / Mistral / Qwen2 / Qwen3 / Qwen3-MoE checkpoints → one `.lapid` file: Hugging Face
//! safetensors (repo id or directory) and GGUF (`general.architecture` `llama`, `qwen2`,
//! `qwen3`, `qwen3moe`; llama.cpp naming).
//!
//! Output names = HF names: `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`
//! (untied), `model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight`,
//! `self_attn.{q,k,v,o}_proj.{weight,bias}`, `self_attn.{q,k}_norm.weight`,
//! `mlp.{gate,up,down}_proj.weight` (dense), `mlp.gate.weight` (router) and
//! `mlp.experts.{gate,up,down}_proj` (`[E, N, K]` stacks) for MoE layers. Values are HF's:
//! llama.cpp's row permutation of Llama q/k projections (`llama` GGUFs, for its interleaved
//! RoPE) is undone on raw rows, which keeps K-quant blocks intact.
//!
//! Precision: `lapid_convert::policy` (Linear weights quantized; the embedding at most int8;
//! norms, biases and the router f32). Expert stacks are quantized per expert. GGUF imports keep
//! ggml quantization when a quantized dtype is requested: Q8_0 → int8 and Q4_0 → int4
//! (lossless repack), Q4_K / Q5_K / Q6_K stored as-is.

mod gguf;
mod hf;

use std::path::Path;

use lapid::quant::Format;
use lapid::{DType, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::ggml::GgmlType;
use lapid_convert::policy::{Role, Target, encode, target};
use lapid_convert::writer::ModelWriter;
use serde_json::{Value, json};

use super::config::LlamaConfig;
use super::model::{EMBED, LM_HEAD};
use super::{META, TEMPLATE_BLOB, TOKENIZER_BLOB};

pub(crate) use gguf::is_llama_gguf;

/// Convert a model in Hugging Face format (repo id or local directory).
pub fn convert_hf(repo_or_dir: &str, dtype: OutDType, out: &Path) -> Result<()> {
    hf::convert(repo_or_dir, dtype, out)
}

/// Convert a GGUF file.
pub fn convert_gguf(path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    gguf::convert(path, dtype, out)
}

fn is_expert_stack(name: &str) -> bool {
    name.contains(".mlp.experts.")
}

/// Dtype-policy role of a tensor (output naming).
pub(crate) fn role(name: &str, shape: &[usize]) -> Role {
    if name.ends_with(".bias") || name.contains("norm") || shape.len() <= 1 || name.ends_with(".mlp.gate.weight") {
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
pub(crate) fn expected_tensors(cfg: &LlamaConfig) -> Vec<String> {
    let mut v = vec![EMBED.to_string(), "model.norm.weight".to_string()];
    if !cfg.tie_embeddings {
        v.push(LM_HEAD.to_string());
    }
    for i in 0..cfg.n_layers {
        let p = format!("model.layers.{i}");
        v.push(format!("{p}.input_layernorm.weight"));
        v.push(format!("{p}.post_attention_layernorm.weight"));
        for n in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            v.push(format!("{p}.self_attn.{n}.weight"));
            let bias = if n == "o_proj" { cfg.o_bias } else { cfg.qkv_bias };
            if bias {
                v.push(format!("{p}.self_attn.{n}.bias"));
            }
        }
        if cfg.qk_norm {
            v.push(format!("{p}.self_attn.q_norm.weight"));
            v.push(format!("{p}.self_attn.k_norm.weight"));
        }
        if cfg.layer_is_moe(i) {
            v.push(format!("{p}.mlp.gate.weight"));
            v.extend(["gate_proj", "up_proj", "down_proj"].iter().map(|n| format!("{p}.mlp.experts.{n}")));
        } else {
            for n in ["gate_proj", "up_proj", "down_proj"] {
                v.push(format!("{p}.mlp.{n}.weight"));
                if cfg.mlp_bias {
                    v.push(format!("{p}.mlp.{n}.bias"));
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
    pub special_tokens: Value,
    pub generation: Value,
    pub license: String,
}

/// Required metadata + the `"llama"` object + the single-component declaration.
pub(crate) fn metadata(name: &str, dtype: OutDType, source: Value, cfg: &LlamaConfig, a: &Assets) -> Value {
    let arch = cfg.arch();
    let mut m = lapid_convert::writer::metadata(name, arch, dtype, &a.license, source);
    let mut q = cfg.to_json();
    q["special_tokens"] = a.special_tokens.clone();
    q["generation"] = a.generation.clone();
    q["tokenizer_blob"] = json!(TOKENIZER_BLOB);
    q["chat_template_blob"] = json!(TEMPLATE_BLOB);
    m[META] = q;
    m["components"] = json!({ META: {"architecture": arch, "config": META, "license": a.license, "tensor_prefix": ""} });
    m
}

pub(crate) fn add_blobs(w: &mut ModelWriter, a: &Assets) -> Result<()> {
    w.add_blob(TOKENIZER_BLOB, "application/json", a.tokenizer_json.as_bytes())?;
    w.add_blob(TEMPLATE_BLOB, "text/x-jinja2", a.chat_template.as_bytes())
}

/// `license:` of a model card's YAML front matter.
pub(crate) fn card_license(readme: &str) -> Option<String> {
    let body = readme.strip_prefix("---")?;
    let end = body.find("\n---")?;
    body[..end].lines().find_map(|l| {
        let v = l.strip_prefix("license:")?.trim().trim_matches(|c| c == '"' || c == '\'');
        (!v.is_empty()).then(|| v.to_string())
    })
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
fn native_format(t: GgmlType) -> Option<(Format, DType)> {
    match t {
        GgmlType::Q8_0 => Some((Format::Int8, DType::Int8)),
        GgmlType::Q4_0 => Some((Format::Int4, DType::Int4)),
        GgmlType::Q4K => Some((Format::Q4K, DType::Q4K)),
        GgmlType::Q5K => Some((Format::Q5K, DType::Q5K)),
        GgmlType::Q6K => Some((Format::Q6K, DType::Q6K)),
        _ => None,
    }
}

/// Write raw ggml data of `name` (row-major `shape`, 1-D to 3-D). With a quantized output
/// dtype, matmul / gather weights in a ggml format lapid runs natively are kept bit-exact
/// (Q8_0 / Q4_0 repacked, K-quants copied, per matrix of a stack); everything else is
/// dequantized and goes through the policy.
pub(crate) fn write_ggml(w: &mut ModelWriter, dtype: OutDType, name: &str, ty: GgmlType, shape: &[usize], bytes: &[u8]) -> Result<()> {
    let numel: usize = shape.iter().product();
    let quant_out = !matches!(dtype, OutDType::F32 | OutDType::F16);
    let matmul_weight = matches!(role(name, shape), Role::Linear | Role::LinearInt8);
    let cols = shape.last().copied().unwrap_or(0);
    if let (true, true, Some((fmt, dt))) = (quant_out, matmul_weight, native_format(ty))
        && (shape.len() == 2 || shape.len() == 3)
        && cols % fmt.block_elems() == 0
    {
        let (mats, rows) = if shape.len() == 3 { (shape[0], shape[1]) } else { (1, shape[0]) };
        let per = ty.nbytes((rows * cols) as u64)? as usize;
        if per * mats != bytes.len() {
            bail!("format", "tensor {name:?}: {} bytes for {mats} x [{rows}, {cols}]", bytes.len());
        }
        let mut out = Vec::with_capacity(bytes.len());
        for m in 0..mats {
            out.extend_from_slice(&lapid::quant::from_ggml_blocks(fmt, &bytes[m * per..(m + 1) * per], rows, cols));
        }
        return w.add_raw(name, dt, shape, &out);
    }
    let data = lapid_convert::ggml::dequantize(ty, bytes, numel)?;
    add_f32(w, dtype, name, shape, &data)
}

/// Undo llama.cpp's q/k row permutation (`LlamaModel.permute`): GGUF row `h*Dh + 2i + s`
/// holds HF row `h*Dh + s*Dh/2 + i`. `row_bytes` per row; `head_dim` rows per head.
pub(crate) fn unpermute_rows(data: &[u8], row_bytes: usize, head_dim: usize) -> Vec<u8> {
    let rows = data.len() / row_bytes.max(1);
    let half = head_dim / 2;
    let mut out = vec![0u8; data.len()];
    for r in 0..rows {
        let (h, j) = (r / head_dim, r % head_dim);
        let (s, i) = (j / half, j % half);
        let src = h * head_dim + 2 * i + s;
        out[r * row_bytes..(r + 1) * row_bytes].copy_from_slice(&data[src * row_bytes..(src + 1) * row_bytes]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles() {
        assert_eq!(role(EMBED, &[151936, 1024]), Role::LinearInt8);
        assert_eq!(role("model.layers.0.self_attn.q_proj.bias", &[1024]), Role::Param);
        assert_eq!(role("model.layers.0.self_attn.q_norm.weight", &[128]), Role::Param);
        assert_eq!(role("model.layers.0.mlp.gate.weight", &[128, 2048]), Role::Param);
        assert_eq!(role("model.layers.0.mlp.experts.gate_proj", &[128, 768, 2048]), Role::Linear);
        assert_eq!(role("model.layers.0.mlp.gate_proj.weight", &[3072, 1024]), Role::Linear);
        assert_eq!(role(LM_HEAD, &[151936, 1024]), Role::Linear);
    }

    #[test]
    fn unpermute() {
        // one head of dim 4: HF rows [a0 a1 b0 b1] (halves a, b) -> GGUF [a0 b0 a1 b1]
        let gguf = [0u8, 2, 1, 3];
        assert_eq!(unpermute_rows(&gguf, 1, 4), vec![0, 1, 2, 3]);
        let two = [0u8, 2, 1, 3, 4, 6, 5, 7];
        assert_eq!(unpermute_rows(&two, 1, 4), vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn license_of_card() {
        assert_eq!(card_license("---\nlanguage: en\nlicense: apache-2.0\n---\n# x").as_deref(), Some("apache-2.0"));
        assert_eq!(card_license("# no front matter"), None);
    }

    #[test]
    fn names_qwen2() {
        let cfg = LlamaConfig::from_json(&json!({"model_type": "qwen2", "vocab_size": 100, "hidden_size": 64,
            "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "tie_word_embeddings": true}))
        .expect("cfg");
        // 2 globals + 2 * (2 norms + 4 attn + 3 bias + 3 mlp)
        assert_eq!(expected_tensors(&cfg).len(), 2 + 2 * 12);
    }
}
