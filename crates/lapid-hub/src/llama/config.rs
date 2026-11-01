//! Hyper-parameters of the plain-transformer families: Llama (2 / 3.x, Mistral-style configs),
//! Qwen2 / Qwen2.5, Qwen3 dense and Qwen3-MoE. One struct, family differences as flags:
//!
//! | family | `model_type` | q/k/v bias | o bias | q/k RMSNorm | FFN |
//! |---|---|---|---|---|---|
//! | Llama / Mistral | `llama`, `mistral` | `attention_bias` | `attention_bias` | no | SwiGLU |
//! | Qwen2 / 2.5 | `qwen2` | yes | no | no | SwiGLU |
//! | Qwen3 | `qwen3` | `attention_bias` | `attention_bias` | yes | SwiGLU |
//! | Qwen3-MoE | `qwen3_moe` | `attention_bias` | `attention_bias` | yes | sparse MoE (+ `mlp_only_layers`) |
//!
//! The `.lapid` metadata object `"llama"` keeps the Hugging Face keys (plus the resolved
//! `qkv_bias` / `o_bias` / `qk_norm` flags, `layer_types` and RoPE frequency factors), so one
//! parser reads both.

use lapid::{Error, Result, bail};
pub use lapid_nn::llm::rope::RopeScaling;
use serde_json::{Value, json};

/// Sparse MoE FFN (`qwen3_moe`): softmax router → top-k (renormalized when `norm_topk_prob`),
/// SwiGLU experts, no shared expert.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeConfig {
    pub n_experts: usize,
    pub top_k: usize,
    pub expert_intermediate: usize,
    pub norm_topk_prob: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LlamaConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Dense FFN width (unused by MoE layers).
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: RopeScaling,
    pub qkv_bias: bool,
    pub o_bias: bool,
    pub mlp_bias: bool,
    pub qk_norm: bool,
    /// Sliding-window size per layer (`None` = full causal attention).
    pub windows: Vec<Option<usize>>,
    pub tie_embeddings: bool,
    pub moe: Option<MoeConfig>,
    pub mlp_only_layers: Vec<usize>,
    pub decoder_sparse_step: usize,
    pub max_positions: usize,
    pub eos: Vec<u32>,
    pub bos: Option<u32>,
    pub pad: Option<u32>,
}

/// Families this module implements (HF `model_type`).
pub const MODEL_TYPES: &[&str] = &["llama", "mistral", "qwen2", "qwen3", "qwen3_moe"];

fn get_usize(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| Error::new("metadata", format!("llama config: missing integer {key:?}")))
}

fn opt_usize(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(Value::as_u64).map(|x| x as usize)
}

fn opt_f32(v: &Value, key: &str) -> Option<f32> {
    v.get(key).and_then(Value::as_f64).map(|x| x as f32)
}

fn opt_bool(v: &Value, key: &str) -> Option<bool> {
    v.get(key).and_then(Value::as_bool)
}

fn parse_scaling(v: &Value) -> Result<RopeScaling> {
    if let Some(f) = v.get("rope_freq_factors").and_then(Value::as_array) {
        return Ok(RopeScaling::FreqFactors(f.iter().filter_map(Value::as_f64).map(|x| x as f32).collect()));
    }
    let r = match v.get("rope_scaling").filter(|r| r.is_object()).or_else(|| v.get("rope_parameters").filter(|r| r.is_object())) {
        Some(r) => r,
        None => return Ok(RopeScaling::None),
    };
    let kind = r.get("rope_type").or_else(|| r.get("type")).and_then(Value::as_str).unwrap_or("default");
    Ok(match kind {
        "default" => RopeScaling::None,
        "linear" => RopeScaling::Linear { factor: opt_f32(r, "factor").unwrap_or(1.0) },
        "llama3" => RopeScaling::Llama3 {
            factor: opt_f32(r, "factor").unwrap_or(8.0),
            low_freq_factor: opt_f32(r, "low_freq_factor").unwrap_or(1.0),
            high_freq_factor: opt_f32(r, "high_freq_factor").unwrap_or(4.0),
            original_max_positions: opt_usize(r, "original_max_position_embeddings").unwrap_or(8192),
        },
        other => bail!("unsupported_model", "llama: rope scaling {other:?} is not supported"),
    })
}

impl LlamaConfig {
    /// Parse an HF `config.json` or the `.lapid` `"llama"` metadata object.
    pub fn from_json(v: &Value) -> Result<LlamaConfig> {
        let model_type = v.get("model_type").and_then(Value::as_str).unwrap_or("llama").to_string();
        if !MODEL_TYPES.contains(&model_type.as_str()) {
            bail!("unsupported_model", "llama: model_type {model_type:?} is not one of {}", MODEL_TYPES.join(", "));
        }
        let n_layers = get_usize(v, "num_hidden_layers")?;
        let hidden = get_usize(v, "hidden_size")?;
        let n_head = get_usize(v, "num_attention_heads")?;
        let qwen = model_type.starts_with("qwen");
        let attention_bias = opt_bool(v, "attention_bias").unwrap_or(false);
        let qwen2 = model_type == "qwen2";
        let rope_theta = v
            .get("rope_theta")
            .or_else(|| v.pointer("/rope_parameters/rope_theta"))
            .and_then(Value::as_f64)
            .unwrap_or(10_000.0) as f32;
        // Sliding windows: explicit `layer_types`, else Mistral's global window, else Qwen's
        // `use_sliding_window` (layers >= max_window_layers).
        let window = opt_usize(v, "sliding_window");
        let windows: Vec<Option<usize>> = match v.get("layer_types").and_then(Value::as_array) {
            Some(a) => a.iter().map(|t| if t.as_str() == Some("sliding_attention") { window } else { None }).collect(),
            None if model_type == "mistral" => vec![window; n_layers],
            None if qwen && opt_bool(v, "use_sliding_window").unwrap_or(false) => {
                let from = opt_usize(v, "max_window_layers").unwrap_or(n_layers);
                (0..n_layers).map(|i| if i >= from { window } else { None }).collect()
            }
            None => vec![None; n_layers],
        };
        if windows.len() != n_layers {
            bail!("metadata", "llama: {} layer types for {n_layers} layers", windows.len());
        }
        let moe = match opt_usize(v, "num_experts").or_else(|| opt_usize(v, "num_local_experts")).filter(|&e| e > 0) {
            Some(e) => Some(MoeConfig {
                n_experts: e,
                top_k: get_usize(v, "num_experts_per_tok")?,
                expert_intermediate: get_usize(v, "moe_intermediate_size")?,
                norm_topk_prob: opt_bool(v, "norm_topk_prob").unwrap_or(true),
            }),
            None if model_type == "qwen3_moe" => bail!("metadata", "llama: qwen3_moe config without num_experts"),
            None => None,
        };
        let cfg = LlamaConfig {
            vocab_size: get_usize(v, "vocab_size")?,
            hidden_size: hidden,
            intermediate_size: opt_usize(v, "intermediate_size").unwrap_or(0),
            n_layers,
            n_head,
            n_kv_head: opt_usize(v, "num_key_value_heads").unwrap_or(n_head),
            head_dim: opt_usize(v, "head_dim").unwrap_or(hidden / n_head.max(1)),
            norm_eps: v.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
            rope_theta,
            rope_scaling: parse_scaling(v)?,
            qkv_bias: opt_bool(v, "qkv_bias").unwrap_or(qwen2 || attention_bias),
            o_bias: opt_bool(v, "o_bias").unwrap_or(!qwen2 && attention_bias),
            mlp_bias: opt_bool(v, "mlp_bias").unwrap_or(false),
            qk_norm: opt_bool(v, "qk_norm").unwrap_or(matches!(model_type.as_str(), "qwen3" | "qwen3_moe")),
            windows,
            tie_embeddings: opt_bool(v, "tie_word_embeddings").unwrap_or(false),
            moe,
            mlp_only_layers: v
                .get("mlp_only_layers")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect())
                .unwrap_or_default(),
            decoder_sparse_step: opt_usize(v, "decoder_sparse_step").unwrap_or(1).max(1),
            max_positions: opt_usize(v, "max_position_embeddings").unwrap_or(32_768),
            eos: crate::runtime::chat::id_list(v.get("eos_token_id")),
            bos: v.get("bos_token_id").and_then(Value::as_u64).map(|x| x as u32),
            pad: v.get("pad_token_id").and_then(Value::as_u64).map(|x| x as u32),
            model_type,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.n_head == 0 || self.n_kv_head == 0 || !self.n_head.is_multiple_of(self.n_kv_head) {
            bail!("metadata", "llama: {} heads / {} kv heads", self.n_head, self.n_kv_head);
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            bail!("metadata", "llama: head_dim {} must be even", self.head_dim);
        }
        if self.moe.is_none() && self.intermediate_size == 0 {
            bail!("metadata", "llama: dense model without intermediate_size");
        }
        if let RopeScaling::FreqFactors(f) = &self.rope_scaling
            && f.len() != self.head_dim / 2
        {
            bail!("metadata", "llama: {} rope frequency factors for head_dim {}", f.len(), self.head_dim);
        }
        Ok(())
    }

    pub fn is_moe(&self) -> bool {
        self.moe.is_some()
    }

    /// Whether layer `i` uses the sparse MoE block (HF Qwen3-MoE rule).
    pub fn layer_is_moe(&self, i: usize) -> bool {
        self.moe.is_some() && !self.mlp_only_layers.contains(&i) && (i + 1).is_multiple_of(self.decoder_sparse_step)
    }

    /// Family label used for the `.lapid` architecture (llama.cpp naming).
    pub fn arch(&self) -> &'static str {
        match self.model_type.as_str() {
            "qwen2" => "qwen2",
            "qwen3" => "qwen3",
            "qwen3_moe" => "qwen3moe",
            "mistral" => "mistral",
            _ => "llama",
        }
    }

    /// Inverse frequencies of the rotary embedding (f32, computed like HF).
    pub fn inv_freq(&self) -> Vec<f32> {
        lapid_nn::llm::rope::scaled_inv_freq(self.head_dim, self.rope_theta, &self.rope_scaling)
    }

    /// HF-style JSON (stored in the `.lapid` metadata).
    pub fn to_json(&self) -> Value {
        let mut v = json!({
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.n_layers,
            "num_attention_heads": self.n_head,
            "num_key_value_heads": self.n_kv_head,
            "head_dim": self.head_dim,
            "rms_norm_eps": self.norm_eps,
            "rope_theta": self.rope_theta,
            "qkv_bias": self.qkv_bias,
            "o_bias": self.o_bias,
            "mlp_bias": self.mlp_bias,
            "qk_norm": self.qk_norm,
            "layer_types": self.windows.iter().map(|w| if w.is_some() { "sliding_attention" } else { "full_attention" }).collect::<Vec<_>>(),
            "sliding_window": self.windows.iter().flatten().next(),
            "tie_word_embeddings": self.tie_embeddings,
            "mlp_only_layers": self.mlp_only_layers,
            "decoder_sparse_step": self.decoder_sparse_step,
            "max_position_embeddings": self.max_positions,
            "eos_token_id": self.eos,
            "bos_token_id": self.bos,
            "pad_token_id": self.pad,
        });
        match &self.rope_scaling {
            RopeScaling::None => {}
            RopeScaling::Linear { factor } => v["rope_scaling"] = json!({"rope_type": "linear", "factor": factor}),
            RopeScaling::Llama3 { factor, low_freq_factor, high_freq_factor, original_max_positions } => {
                v["rope_scaling"] = json!({"rope_type": "llama3", "factor": factor, "low_freq_factor": low_freq_factor,
                    "high_freq_factor": high_freq_factor, "original_max_position_embeddings": original_max_positions})
            }
            RopeScaling::FreqFactors(f) => v["rope_freq_factors"] = json!(f),
        }
        if let Some(m) = &self.moe {
            v["num_experts"] = json!(m.n_experts);
            v["num_experts_per_tok"] = json!(m.top_k);
            v["moe_intermediate_size"] = json!(m.expert_intermediate);
            v["norm_topk_prob"] = json!(m.norm_topk_prob);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama32() -> Value {
        json!({"model_type": "llama", "vocab_size": 128256, "hidden_size": 2048, "intermediate_size": 8192,
            "num_hidden_layers": 16, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 64,
            "rms_norm_eps": 1e-5, "rope_theta": 500000.0, "tie_word_embeddings": true, "attention_bias": false,
            "eos_token_id": [128001, 128008, 128009], "bos_token_id": 128000,
            "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192, "rope_type": "llama3"}})
    }

    #[test]
    fn parse_llama_roundtrip() {
        let c = LlamaConfig::from_json(&llama32()).expect("cfg");
        assert!(!c.qkv_bias && !c.qk_norm && c.tie_embeddings);
        assert_eq!(c.eos, vec![128001, 128008, 128009]);
        assert!(matches!(c.rope_scaling, RopeScaling::Llama3 { .. }));
        assert_eq!(c.inv_freq().len(), 32);
        assert_eq!(LlamaConfig::from_json(&c.to_json()).expect("again"), c);
    }

    #[test]
    fn parse_qwen_families() {
        let q2 = json!({"model_type": "qwen2", "vocab_size": 151936, "hidden_size": 896, "intermediate_size": 4864,
            "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2, "rope_theta": 1000000.0,
            "sliding_window": 32768, "use_sliding_window": false, "max_window_layers": 21, "tie_word_embeddings": true});
        let c = LlamaConfig::from_json(&q2).expect("qwen2");
        assert!(c.qkv_bias && !c.o_bias && !c.qk_norm);
        assert_eq!(c.head_dim, 64);
        assert!(c.windows.iter().all(Option::is_none));
        assert_eq!(LlamaConfig::from_json(&c.to_json()).expect("again"), c);

        let moe = json!({"model_type": "qwen3_moe", "vocab_size": 151936, "hidden_size": 2048, "intermediate_size": 6144,
            "num_hidden_layers": 48, "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
            "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 768, "norm_topk_prob": true,
            "decoder_sparse_step": 1, "mlp_only_layers": [], "attention_bias": false});
        let c = LlamaConfig::from_json(&moe).expect("moe");
        assert!(c.qk_norm && c.layer_is_moe(0) && c.arch() == "qwen3moe");
        assert_eq!(LlamaConfig::from_json(&c.to_json()).expect("again"), c);
    }
}
