//! LFM3 hyper-parameters, read from HF `config.json` keys (the `.lapid` metadata object
//! `"lfm3"` stores them as converted).

use lapid::{Error, Result, bail};
use serde_json::Value;

/// Operator of a decoder layer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LayerType {
    /// Gated short causal convolution (LFM2 style).
    Conv,
    FullAttention,
    SlidingAttention,
}

/// RoPE of one attention layer type (`partial`: rotated fraction, zero-tail layout).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RopeParams {
    pub theta: f32,
    pub partial: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Lfm3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub norm_eps: f32,
    pub conv_kernel: usize,
    pub conv_bias: bool,
    pub layer_types: Vec<LayerType>,
    pub sliding_window: usize,
    pub rope_full: RopeParams,
    pub rope_sliding: RopeParams,
    pub sandwich_norm: bool,
}

/// `rope_parameters.<kind>` (defaults of `Lfm3Config.__post_init__` when absent).
fn rope(v: &Value, kind: &str, default: RopeParams) -> Result<RopeParams> {
    let Some(p) = v.pointer(&format!("/rope_parameters/{kind}")) else { return Ok(default) };
    if let Some(t) = p.get("rope_type").and_then(Value::as_str).filter(|t| *t != "default") {
        bail!("unsupported_model", "lfm3: rope_type {t:?} on {kind} layers is not supported");
    }
    let theta = p.get("rope_theta").and_then(Value::as_f64).ok_or_else(|| Error::new("missing_metadata", format!("lfm3: no rope_theta for {kind}")))?;
    Ok(RopeParams { theta: theta as f32, partial: p.get("partial_rotary_factor").and_then(Value::as_f64).unwrap_or(1.0) as f32 })
}

impl Lfm3Config {
    pub fn from_json(v: &Value) -> Result<Lfm3Config> {
        let req = |k: &str| v.get(k).and_then(Value::as_u64).map(|x| x as usize).ok_or_else(|| Error::new("missing_metadata", format!("lfm3 config: missing {k:?}")));
        let opt = |k: &str, d: usize| v.get(k).and_then(Value::as_u64).map_or(d, |x| x as usize);
        let (hidden_size, n_head, n_layers) = (req("hidden_size")?, req("num_attention_heads")?, req("num_hidden_layers")?);
        let layer_types = match v.get("layer_types").and_then(Value::as_array) {
            Some(a) => a
                .iter()
                .map(|t| match t.as_str() {
                    Some("conv") => Ok(LayerType::Conv),
                    Some("full_attention") => Ok(LayerType::FullAttention),
                    Some("sliding_attention") => Ok(LayerType::SlidingAttention),
                    other => Err(Error::new("unsupported_model", format!("lfm3: unknown layer type {other:?}"))),
                })
                .collect::<Result<Vec<_>>>()?,
            None => (0..n_layers)
                .map(|i| match i % 4 {
                    2 => LayerType::FullAttention,
                    0 if i > 0 => LayerType::SlidingAttention,
                    _ => LayerType::Conv,
                })
                .collect(),
        };
        if v.get("tie_word_embeddings").and_then(Value::as_bool) == Some(false) {
            bail!("unsupported_model", "lfm3: untied lm_head is not supported");
        }
        let c = Lfm3Config {
            vocab_size: req("vocab_size")?,
            hidden_size,
            intermediate_size: req("intermediate_size")?,
            n_head,
            n_kv_head: req("num_key_value_heads")?,
            head_dim: opt("head_dim", hidden_size / n_head.max(1)),
            norm_eps: v.get("norm_eps").and_then(Value::as_f64).unwrap_or(1e-5) as f32,
            conv_kernel: opt("conv_L_cache", 7),
            conv_bias: v.get("conv_bias").and_then(Value::as_bool).unwrap_or(false),
            layer_types,
            sliding_window: opt("sliding_window", 256),
            rope_full: rope(v, "full_attention", RopeParams { theta: 1e6, partial: 0.5 })?,
            rope_sliding: rope(v, "sliding_attention", RopeParams { theta: 1e4, partial: 1.0 })?,
            sandwich_norm: v.get("use_sandwich_norm").and_then(Value::as_bool).unwrap_or(true),
        };
        if c.layer_types.len() != n_layers || c.n_kv_head == 0 || !n_head.is_multiple_of(c.n_kv_head) {
            bail!("metadata", "lfm3: {} layer types for {n_layers} layers, {n_head} / {} heads", c.layer_types.len(), c.n_kv_head);
        }
        if c.head_dim == 0 || !c.head_dim.is_multiple_of(2) || c.conv_kernel == 0 || c.sliding_window == 0 {
            bail!("metadata", "lfm3: bad head_dim {} / conv_L_cache / sliding_window", c.head_dim);
        }
        Ok(c)
    }

    /// Per-layer KV window (`Some(sliding_window)` on sliding layers).
    pub fn kv_windows(&self) -> Vec<Option<usize>> {
        self.layer_types.iter().map(|t| (*t == LayerType::SlidingAttention).then_some(self.sliding_window)).collect()
    }
}
