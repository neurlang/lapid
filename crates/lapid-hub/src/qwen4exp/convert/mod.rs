//! Qwen4-Exp checkpoints → one `.lapid` file: Hugging Face safetensors (repo id or directory;
//! `Qwen4ExpForConditionalGeneration` / `…ForCausalLM`) and GGUF (`general.architecture`
//! `qwen4exp`, llama.cpp naming; merge split GGUFs first with `llama-gguf-split --merge`).
//!
//! Output names (text decoder only; vision tower and MTP head dropped) follow HF:
//! `model.embed_tokens.weight`, `lm_head.weight`, `model.hyper_connection_mixer.{hc_norm,
//! input_mix_weight_down,input_mix_weight_up}.weight`, per layer `{attn,mlp}_hyper_connection.
//! {hc_norm,input_mix_weight_down,input_mix_weight_up,block_inject_weight}.weight`,
//! `self_attn.{q,k,v,o}_proj / q_norm / k_norm`, `self_attn.indexer.{q_proj,k_proj,q_norm,k_norm}`
//! (HF `index_qk_proj` split), `linear_attn.*` and `mlp.*` as Qwen3.5 MoE, and on the PLE layer
//! `ple.{key_proj,value_proj,norm_key,norm_query,norm_conv,conv1d}.weight` plus the n-gram
//! table `ple.ngram_embedding.weight` [rows, row_dim] (HF shards concatenated).
//!
//! Value conventions (= llama.cpp's GGUF): RMSNorm weights are effective scales (HF `1 + w`,
//! including the grouped hyper-connection / PLE norms stored `[hc, D]`; the DeltaNet gated
//! norm is plain `w`), `A = -exp(A_log)`, DeltaNet v heads in tiled order.
//!
//! Precision: `lapid_convert::policy` through Qwen3.5's roles (the n-gram table is a matmul-
//! style weight: quantized). GGUF imports keep ggml quantization when a quantized dtype is
//! requested (Q8_0 → int8, Q4_0 → int4, K-quants as-is); float GGUF tensors are copied
//! exactly (f32 / bf16 → f32, f16 → f16); other ggml types are dequantized and re-encoded.

mod gguf;
mod hf;

use std::path::Path;

use lapid::{DType, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::ggml::GgmlType;
use lapid_convert::policy::Role;
use lapid_convert::writer::ModelWriter;
use serde_json::{Value, json};

use super::config::Qwen4ExpConfig;
use super::model::{EMBED, LM_HEAD, MIXER};
use super::{META, TEMPLATE_BLOB, TOKENIZER_BLOB};
use crate::qwen35::config::LayerType;
pub(crate) use crate::qwen35::convert::{Assets, add_blobs, add_f32, format_dtype, native_format, role};

pub(crate) use gguf::is_qwen4exp_gguf;

/// SPDX id of the Qwen4-Exp license (Qwen3.8-Flash-Next: `apache-2.0`).
pub const LICENSE: &str = "apache-2.0";

/// Convert a Qwen4-Exp model in Hugging Face format (repo id or local directory).
pub fn convert_hf(repo_or_dir: &str, dtype: OutDType, out: &Path) -> Result<()> {
    hf::convert(repo_or_dir, dtype, out)
}

/// Convert a (merged, single-file) Qwen4-Exp GGUF file.
pub fn convert_gguf(path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    gguf::convert(path, dtype, out)
}

const HC_PARTS: [&str; 4] = ["hc_norm", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"];

/// Every tensor a checkpoint must provide (output naming).
pub(crate) fn expected_tensors(cfg: &Qwen4ExpConfig) -> Vec<String> {
    let mut v = vec![EMBED.to_string()];
    if !cfg.base.tie_embeddings {
        v.push(LM_HEAD.to_string());
    }
    v.extend(HC_PARTS[..3].iter().map(|n| format!("{MIXER}.{n}.weight")));
    for (i, t) in cfg.base.layer_types.iter().enumerate() {
        let p = format!("model.layers.{i}");
        for hc in ["attn_hyper_connection", "mlp_hyper_connection"] {
            v.extend(HC_PARTS.iter().map(|n| format!("{p}.{hc}.{n}.weight")));
        }
        match t {
            LayerType::FullAttention => {
                for n in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"] {
                    v.push(format!("{p}.self_attn.{n}.weight"));
                }
                if cfg.qsa.is_some() {
                    v.extend(["q_proj", "k_proj", "q_norm", "k_norm"].iter().map(|n| format!("{p}.self_attn.indexer.{n}.weight")));
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
        if cfg.ple.as_ref().is_some_and(|q| q.layer == i) {
            for n in ["key_proj", "value_proj", "norm_key", "norm_query", "norm_conv", "conv1d", "ngram_embedding"] {
                v.push(format!("{p}.ple.{n}.weight"));
            }
        }
        v.push(format!("{p}.mlp.gate.weight"));
        for n in ["gate_proj", "up_proj", "down_proj"] {
            v.push(format!("{p}.mlp.experts.{n}"));
            v.push(format!("{p}.mlp.shared_expert.{n}.weight"));
        }
        v.push(format!("{p}.mlp.shared_expert_gate.weight"));
    }
    v
}

/// Grouped `[hc * D]` norm weights are stored `[hc, D]`.
pub(crate) fn is_grouped_norm(name: &str) -> bool {
    name.ends_with(".hc_norm.weight") || name.contains(".ple.norm_")
}

/// Required metadata + the `"qwen4exp"` object + the single-component declaration.
pub(crate) fn metadata(name: &str, dtype: OutDType, source: Value, cfg: &Qwen4ExpConfig, a: &Assets) -> Value {
    let arch = "qwen4exp";
    let mut m = lapid_convert::writer::metadata(name, arch, dtype, LICENSE, source);
    let mut q = cfg.to_json();
    q["special_tokens"] = a.special_tokens.clone();
    q["generation"] = a.generation.clone();
    q["tokenizer_blob"] = json!(TOKENIZER_BLOB);
    q["chat_template_blob"] = json!(TEMPLATE_BLOB);
    q["weight_layout"] = json!({"rms_norm": "effective_scale", "deltanet_A": "neg_exp", "v_heads": "tiled", "grouped_norms": "[hc, D]"});
    m[META] = q;
    m["components"] = json!({ META: {"architecture": arch, "config": META, "license": LICENSE, "tensor_prefix": ""} });
    m
}

/// Write raw ggml bytes of a row-major `shape` tensor (a whole GGUF tensor or a row range of
/// one). Matmul weights in a format lapid runs natively are kept bit-exact when a quantized
/// dtype is requested; float tensors are copied exactly; the rest is dequantized and goes
/// through the policy (expert stacks per expert).
pub(crate) fn write_ggml(w: &mut ModelWriter, dtype: OutDType, name: &str, ty: GgmlType, shape: &[usize], bytes: &[u8]) -> Result<()> {
    let numel: usize = shape.iter().product();
    let quant_out = !matches!(dtype, OutDType::F32 | OutDType::F16);
    let matmul = matches!(role(name, shape), Role::Linear | Role::LinearInt8);
    let cols = shape.last().copied().unwrap_or(0);
    if let (true, true, Some(fmt)) = (quant_out, matmul, native_format(ty))
        && matches!(shape.len(), 2 | 3)
        && cols.is_multiple_of(fmt.block_elems())
    {
        let mats: usize = shape[..shape.len() - 2].iter().product::<usize>().max(1);
        let rows = shape[shape.len() - 2];
        let per = bytes.len() / mats;
        if per * mats != bytes.len() || per as u64 != ty.nbytes((rows * cols) as u64)? {
            bail!("format", "{name}: {} bytes for {mats} x [{rows}, {cols}] {ty:?}", bytes.len());
        }
        let mut out = Vec::with_capacity(bytes.len());
        for m in bytes.chunks_exact(per) {
            out.extend(lapid::quant::from_ggml_blocks(fmt, m, rows, cols));
        }
        return w.add_raw(name, format_dtype(fmt), shape, &out);
    }
    if quant_out && matmul {
        match ty {
            GgmlType::F32 => return w.add_raw(name, DType::F32, shape, bytes),
            GgmlType::F16 => return w.add_raw(name, DType::F16, shape, bytes),
            GgmlType::BF16 => {
                let data = lapid_convert::ggml::dequantize(ty, bytes, numel)?;
                let raw: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
                return w.add_raw(name, DType::F32, shape, &raw);
            }
            _ => {}
        }
    }
    let data = lapid_convert::ggml::dequantize(ty, bytes, numel)?;
    add_f32(w, dtype, name, shape, &data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Qwen4ExpConfig {
        Qwen4ExpConfig::from_json(&json!({"model_type": "qwen4_exp_text", "vocab_size": 1000, "hidden_size": 64,
            "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
            "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32, "hc_count": 4, "hc_lowrank": 8,
            "eos_token_id": 7, "indexer_n_heads": 2, "indexer_head_dim": 32, "indexer_budget": 8, "indexer_compress_ratio": 4,
            "ple_layer_ids": [2], "ngram_vocab_size_base": 50}))
        .expect("cfg")
    }

    #[test]
    fn names() {
        let n = expected_tensors(&tiny());
        // embed + head + 3 final mixer; per layer 8 hc + 8 moe; 3 linear * 9; 1 attention * (6 + 4); ple 7
        assert_eq!(n.len(), 5 + 4 * 16 + 27 + 10 + 7);
        assert!(n.contains(&"model.layers.1.ple.ngram_embedding.weight".to_string()));
        assert!(is_grouped_norm("model.layers.1.ple.norm_key.weight") && is_grouped_norm("model.hyper_connection_mixer.hc_norm.weight"));
        assert_eq!(role("model.layers.1.ple.ngram_embedding.weight", &[1024, 16]), Role::Linear);
        assert_eq!(role("model.layers.0.attn_hyper_connection.hc_norm.weight", &[4, 64]), Role::Param);
    }
}
