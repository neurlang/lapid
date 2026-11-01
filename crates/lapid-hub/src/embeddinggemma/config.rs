//! The `"embeddinggemma"` metadata object of a `.lapid` file (written by [`super::convert`]).

use std::collections::BTreeMap;

use lapid::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Activation after a sentence-transformers `Dense` layer.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DenseActivation {
    Identity,
    Tanh,
}

impl DenseActivation {
    /// From a sentence-transformers `activation_function` class path.
    pub fn from_torch(name: &str) -> Option<DenseActivation> {
        match name.rsplit('.').next().unwrap_or(name) {
            "Identity" => Some(DenseActivation::Identity),
            "Tanh" => Some(DenseActivation::Tanh),
            _ => None,
        }
    }
}

/// One `Dense` module of the sentence-transformers head (weights `{prefix}weight`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DenseConfig {
    pub prefix: String,
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
    pub activation: DenseActivation,
}

/// Sentence-transformers pooling (only mean pooling is implemented).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoolingConfig {
    pub mode: String,
    pub include_prompt: bool,
}

/// Everything needed to rebuild the model from a `.lapid` file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingGemmaConfig {
    pub hidden_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    /// RoPE base of full-attention layers.
    pub rope_theta: f32,
    /// RoPE base of sliding-window layers.
    pub rope_local_theta: f32,
    /// `sliding_window` from config.json (the HF bidirectional mask uses `window / 2 + 1`).
    pub sliding_window: usize,
    /// Exclusive bound of the bidirectional window: key j visible to query i iff
    /// `|i - j| < bidirectional_window`.
    pub bidirectional_window: usize,
    /// Per layer: true = sliding-window attention, false = full attention.
    pub sliding_layers: Vec<bool>,
    pub query_pre_attn_scalar: f32,
    pub hidden_activation: String,
    pub max_seq_length: usize,
    pub pad_token_id: u32,
    pub pooling: PoolingConfig,
    pub dense: Vec<DenseConfig>,
    pub normalize: bool,
    /// Prompt name -> prefix string (sentence-transformers `prompts`).
    pub prompts: BTreeMap<String, String>,
    pub default_prompt_name: Option<String>,
    /// Matryoshka output sizes (truncate, then re-normalize).
    pub matryoshka_dims: Vec<usize>,
    /// Blob holding the Hugging Face `tokenizer.json`.
    pub tokenizer_blob: String,
}

/// Metadata key of the config object.
pub const META_KEY: &str = "embeddinggemma";

impl EmbeddingGemmaConfig {
    pub fn from_metadata(meta: &Value) -> Result<EmbeddingGemmaConfig> {
        let v = meta
            .get(META_KEY)
            .ok_or_else(|| Error::new("missing_metadata", format!("no {META_KEY:?} metadata object (not an EmbeddingGemma file?)")))?;
        let cfg: EmbeddingGemmaConfig = serde_json::from_value(v.clone())
            .map_err(|e| Error::new("metadata", format!("{META_KEY}: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(Error::new("metadata", format!("{META_KEY}: {m}")));
        if self.n_head == 0 || self.n_kv_head == 0 || !self.n_head.is_multiple_of(self.n_kv_head) {
            return bad(format!("n_head {} must be a multiple of n_kv_head {}", self.n_head, self.n_kv_head));
        }
        if self.sliding_layers.len() != self.n_layer {
            return bad(format!("{} layer types for {} layers", self.sliding_layers.len(), self.n_layer));
        }
        if self.bidirectional_window == 0 {
            return bad("bidirectional_window must be >= 1".into());
        }
        if self.pooling.mode != "mean" {
            return bad(format!("pooling mode {:?} is not supported (mean)", self.pooling.mode));
        }
        let mut d = self.hidden_size;
        for (i, l) in self.dense.iter().enumerate() {
            if l.in_features != d {
                return bad(format!("dense {i}: in_features {} != previous width {d}", l.in_features));
            }
            d = l.out_features;
        }
        if self.max_seq_length < 2 {
            return bad("max_seq_length must be >= 2".into());
        }
        Ok(())
    }

    /// Width of the final embedding.
    pub fn output_dim(&self) -> usize {
        self.dense.last().map_or(self.hidden_size, |l| l.out_features)
    }

    /// `query_pre_attn_scalar^-0.5`.
    pub fn attn_scale(&self) -> f32 {
        1.0 / self.query_pre_attn_scalar.sqrt()
    }

    /// `sqrt(hidden_size)` rounded to f32 like HF's f32 `embed_scale` buffer.
    pub fn embed_scale(&self) -> f32 {
        (self.hidden_size as f64).sqrt() as f32
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    pub(crate) fn sample() -> EmbeddingGemmaConfig {
        EmbeddingGemmaConfig {
            hidden_size: 64,
            n_layer: 2,
            n_head: 2,
            n_kv_head: 1,
            head_dim: 32,
            intermediate_size: 96,
            vocab_size: 300,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            rope_local_theta: 1e4,
            sliding_window: 8,
            bidirectional_window: 5,
            sliding_layers: vec![true, false],
            query_pre_attn_scalar: 32.0,
            hidden_activation: "gelu_pytorch_tanh".into(),
            max_seq_length: 64,
            pad_token_id: 0,
            pooling: PoolingConfig { mode: "mean".into(), include_prompt: true },
            dense: vec![
                DenseConfig { prefix: "dense.0.".into(), in_features: 64, out_features: 128, bias: false, activation: DenseActivation::Identity },
                DenseConfig { prefix: "dense.1.".into(), in_features: 128, out_features: 64, bias: false, activation: DenseActivation::Identity },
            ],
            normalize: true,
            prompts: [("query".to_string(), "task: search result | query: ".to_string())].into(),
            default_prompt_name: None,
            matryoshka_dims: vec![64, 32],
            tokenizer_blob: "tokenizer/tokenizer.json".into(),
        }
    }

    #[test]
    fn roundtrip_and_validation() {
        let c = sample();
        let meta = json!({ META_KEY: serde_json::to_value(&c).expect("json") });
        assert_eq!(EmbeddingGemmaConfig::from_metadata(&meta).expect("parse"), c);
        assert_eq!(c.output_dim(), 64);
        let mut bad = c.clone();
        bad.dense[1].in_features = 7;
        assert_eq!(bad.validate().unwrap_err().code, "metadata");
        assert_eq!(EmbeddingGemmaConfig::from_metadata(&json!({})).unwrap_err().code, "missing_metadata");
        assert_eq!(DenseActivation::from_torch("torch.nn.modules.linear.Identity"), Some(DenseActivation::Identity));
    }
}
