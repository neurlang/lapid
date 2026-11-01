//! Qwen3.5 hyper-parameters (dense `qwen3_5` and MoE `qwen3_5_moe`). The `.lapid` metadata
//! object `"qwen35"` keeps the Hugging Face `text_config` keys (plus `special_tokens`,
//! `generation`, blob names), so one parser reads both. Top-level HF configs
//! (`Qwen3_5ForConditionalGeneration`) are accepted too: their `text_config` is used.

use lapid::{Error, Result, bail};
use serde_json::{Value, json};

/// Token mixer of a decoder layer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LayerType {
    /// Gated DeltaNet (linear attention with a recurrent state).
    LinearAttention,
    /// Gated full attention (GQA, q/k RMSNorm, partial RoPE, sigmoid output gate).
    FullAttention,
}

impl LayerType {
    pub fn parse(s: &str) -> Result<LayerType> {
        Ok(match s {
            "linear_attention" => LayerType::LinearAttention,
            "full_attention" => LayerType::FullAttention,
            other => bail!("unsupported_model", "qwen35: unknown layer type {other:?}"),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            LayerType::LinearAttention => "linear_attention",
            LayerType::FullAttention => "full_attention",
        }
    }
}

/// Sparse mixture-of-experts FFN (`qwen3_5_moe`): softmax router → top-k (renormalized),
/// SwiGLU experts, plus a sigmoid-gated shared expert.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeConfig {
    pub n_experts: usize,
    pub top_k: usize,
    pub expert_intermediate: usize,
    pub shared_intermediate: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35Config {
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
    pub layer_types: Vec<LayerType>,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    // Gated DeltaNet
    pub conv_kernel: usize,
    pub linear_k_heads: usize,
    pub linear_v_heads: usize,
    pub linear_k_dim: usize,
    pub linear_v_dim: usize,
    pub tie_embeddings: bool,
    pub moe: Option<MoeConfig>,
    /// MoE models: layers that use the dense MLP instead (HF `mlp_only_layers`).
    pub mlp_only_layers: Vec<usize>,
    pub max_positions: usize,
    pub eos: Vec<u32>,
    pub pad: Option<u32>,
}

fn get_usize(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| Error::new("metadata", format!("qwen35 config: missing integer {key:?}")))
}

fn opt_usize(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(Value::as_u64).map(|x| x as usize)
}

use crate::runtime::chat::id_list as ids;

impl Qwen35Config {
    /// Parse an HF config (top level with `text_config`, or the text config itself) or the
    /// `.lapid` `"qwen35"` metadata object.
    pub fn from_json(root: &Value) -> Result<Qwen35Config> {
        let v = root.get("text_config").filter(|t| t.is_object()).unwrap_or(root);
        let model_type = root
            .get("model_type")
            .and_then(Value::as_str)
            .map(|m| m.trim_end_matches("_text").to_string())
            .unwrap_or_else(|| "qwen3_5".into());
        if !matches!(model_type.as_str(), "qwen3_5" | "qwen3_5_moe") {
            bail!("unsupported_model", "qwen35: model_type {model_type:?} is not qwen3_5 / qwen3_5_moe");
        }
        let n_layers = get_usize(v, "num_hidden_layers")?;
        let layer_types = match v.get("layer_types").and_then(Value::as_array) {
            Some(a) => a
                .iter()
                .map(|t| t.as_str().ok_or_else(|| Error::new("metadata", "qwen35: layer_types holds a non-string")).and_then(LayerType::parse))
                .collect::<Result<Vec<_>>>()?,
            None => {
                let every = opt_usize(v, "full_attention_interval").unwrap_or(4).max(1);
                (0..n_layers)
                    .map(|i| if (i + 1) % every == 0 { LayerType::FullAttention } else { LayerType::LinearAttention })
                    .collect()
            }
        };
        if layer_types.len() != n_layers {
            bail!("metadata", "qwen35: {} layer types for {n_layers} layers", layer_types.len());
        }
        let hidden = get_usize(v, "hidden_size")?;
        let n_head = get_usize(v, "num_attention_heads")?;
        let rope = v.get("rope_parameters").or_else(|| v.get("rope_scaling")).cloned().unwrap_or(Value::Null);
        let rope_theta = rope
            .get("rope_theta")
            .or_else(|| v.get("rope_theta"))
            .and_then(Value::as_f64)
            .unwrap_or(1e7) as f32;
        let partial = rope
            .get("partial_rotary_factor")
            .or_else(|| v.get("partial_rotary_factor"))
            .and_then(Value::as_f64)
            .unwrap_or(0.25) as f32;
        let moe = if model_type == "qwen3_5_moe" || v.get("num_experts").and_then(Value::as_u64).is_some_and(|e| e > 0) {
            let inter = get_usize(v, "moe_intermediate_size")?;
            Some(MoeConfig {
                n_experts: get_usize(v, "num_experts")?,
                top_k: get_usize(v, "num_experts_per_tok")?,
                expert_intermediate: inter,
                shared_intermediate: opt_usize(v, "shared_expert_intermediate_size").unwrap_or(inter),
            })
        } else {
            None
        };
        let tie = root
            .get("tie_word_embeddings")
            .or_else(|| v.get("tie_word_embeddings"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let mut eos = ids(v.get("eos_token_id"));
        if eos.is_empty() {
            eos = ids(root.get("eos_token_id"));
        }
        let cfg = Qwen35Config {
            model_type,
            vocab_size: get_usize(v, "vocab_size")?,
            hidden_size: hidden,
            intermediate_size: opt_usize(v, "intermediate_size").unwrap_or(0),
            n_layers,
            n_head,
            n_kv_head: get_usize(v, "num_key_value_heads")?,
            head_dim: opt_usize(v, "head_dim").unwrap_or(hidden / n_head.max(1)),
            norm_eps: v.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
            layer_types,
            rope_theta,
            partial_rotary_factor: partial,
            conv_kernel: opt_usize(v, "linear_conv_kernel_dim").unwrap_or(4),
            linear_k_heads: get_usize(v, "linear_num_key_heads")?,
            linear_v_heads: get_usize(v, "linear_num_value_heads")?,
            linear_k_dim: get_usize(v, "linear_key_head_dim")?,
            linear_v_dim: get_usize(v, "linear_value_head_dim")?,
            tie_embeddings: tie,
            moe,
            mlp_only_layers: v
                .get("mlp_only_layers")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect())
                .unwrap_or_default(),
            max_positions: opt_usize(v, "max_position_embeddings").unwrap_or(262_144),
            eos,
            pad: v.get("pad_token_id").and_then(Value::as_u64).map(|x| x as u32),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.n_head == 0 || self.n_kv_head == 0 || !self.n_head.is_multiple_of(self.n_kv_head) {
            bail!("metadata", "qwen35: {} heads / {} kv heads", self.n_head, self.n_kv_head);
        }
        if self.linear_k_heads == 0 || !self.linear_v_heads.is_multiple_of(self.linear_k_heads) {
            bail!("metadata", "qwen35: linear v heads {} not a multiple of k heads {}", self.linear_v_heads, self.linear_k_heads);
        }
        if self.moe.is_none() && self.intermediate_size == 0 {
            bail!("metadata", "qwen35: dense model without intermediate_size");
        }
        Ok(())
    }

    pub fn is_moe(&self) -> bool {
        self.moe.is_some()
    }

    /// Whether layer `i` uses the sparse MoE block (else the dense SwiGLU MLP).
    pub fn layer_is_moe(&self, i: usize) -> bool {
        self.moe.is_some() && !self.mlp_only_layers.contains(&i)
    }

    /// Channels of the DeltaNet causal conv (q, k and v concatenated).
    pub fn conv_dim(&self) -> usize {
        2 * self.linear_k_heads * self.linear_k_dim + self.linear_v_heads * self.linear_v_dim
    }

    /// Rotated dims of each attention head.
    pub fn rotary_dim(&self) -> usize {
        ((self.head_dim as f32 * self.partial_rotary_factor) as usize) & !1
    }

    /// HF `text_config`-style JSON (stored in the `.lapid` metadata).
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
            "layer_types": self.layer_types.iter().map(|t| t.name()).collect::<Vec<_>>(),
            "rope_parameters": {"rope_type": "default", "rope_theta": self.rope_theta, "partial_rotary_factor": self.partial_rotary_factor},
            "linear_conv_kernel_dim": self.conv_kernel,
            "linear_num_key_heads": self.linear_k_heads,
            "linear_num_value_heads": self.linear_v_heads,
            "linear_key_head_dim": self.linear_k_dim,
            "linear_value_head_dim": self.linear_v_dim,
            "tie_word_embeddings": self.tie_embeddings,
            "mlp_only_layers": self.mlp_only_layers,
            "max_position_embeddings": self.max_positions,
            "eos_token_id": self.eos,
            "pad_token_id": self.pad,
        });
        if let Some(m) = &self.moe {
            v["num_experts"] = json!(m.n_experts);
            v["num_experts_per_tok"] = json!(m.top_k);
            v["moe_intermediate_size"] = json!(m.expert_intermediate);
            v["shared_expert_intermediate_size"] = json!(m.shared_intermediate);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense() -> Value {
        json!({"model_type": "qwen3_5", "tie_word_embeddings": true, "text_config": {
            "vocab_size": 248320, "hidden_size": 1024, "intermediate_size": 3584, "num_hidden_layers": 8,
            "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 256, "full_attention_interval": 4,
            "linear_num_key_heads": 16, "linear_num_value_heads": 16, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128, "eos_token_id": 248044,
            "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}}})
    }

    #[test]
    fn parse_dense_and_roundtrip() {
        let c = Qwen35Config::from_json(&dense()).expect("cfg");
        assert!(c.tie_embeddings && !c.is_moe());
        assert_eq!(c.layer_types[3], LayerType::FullAttention);
        assert_eq!(c.layer_types[2], LayerType::LinearAttention);
        assert_eq!(c.rotary_dim(), 64);
        assert_eq!(c.conv_dim(), 6144);
        assert_eq!(c.eos, vec![248044]);
        let again = Qwen35Config::from_json(&c.to_json()).expect("again");
        assert_eq!(c, again);
    }

    #[test]
    fn parse_moe() {
        let mut v = dense();
        v["model_type"] = json!("qwen3_5_moe");
        let t = &mut v["text_config"];
        t["num_experts"] = json!(8);
        t["num_experts_per_tok"] = json!(2);
        t["moe_intermediate_size"] = json!(64);
        t["shared_expert_intermediate_size"] = json!(64);
        let c = Qwen35Config::from_json(&v).expect("cfg");
        assert_eq!(c.moe.as_ref().map(|m| m.top_k), Some(2));
        assert!(c.layer_is_moe(0));
        assert_eq!(Qwen35Config::from_json(&c.to_json()).expect("again"), c);
    }
}
