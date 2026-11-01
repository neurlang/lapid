//! LFM3 hyper-parameters. The `.lapid` metadata object `"lfm3"` keeps the Hugging Face
//! `config.json` keys (plus `special`, `generation` and blob names), so one parser reads both.

use lapid::{Error, Result, bail};
use serde_json::{Value, json};

/// Operator of a decoder layer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LayerType {
    /// Gated short causal convolution (LFM2 style).
    Conv,
    FullAttention,
    SlidingAttention,
}

impl LayerType {
    pub fn parse(s: &str) -> Result<LayerType> {
        Ok(match s {
            "conv" => LayerType::Conv,
            "full_attention" => LayerType::FullAttention,
            "sliding_attention" => LayerType::SlidingAttention,
            other => bail!("unsupported_model", "lfm3: unknown layer type {other:?}"),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            LayerType::Conv => "conv",
            LayerType::FullAttention => "full_attention",
            LayerType::SlidingAttention => "sliding_attention",
        }
    }

    pub fn is_attention(self) -> bool {
        self != LayerType::Conv
    }
}

/// Default RoPE of one attention layer type: `theta`, and the rotated fraction of each head
/// (LFM3 "zero-tail" partial rotation: the frequencies follow the full head dimension and
/// the tail frequencies are zero).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RopeParams {
    pub theta: f32,
    pub partial_rotary_factor: f32,
}

/// Special token ids.
#[derive(Clone, Debug, PartialEq)]
pub struct Specials {
    pub bos: Option<u32>,
    pub eos: Vec<u32>,
    pub pad: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Lfm3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
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
    pub max_positions: usize,
    pub special: Specials,
}

fn get_usize(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| Error::new("missing_metadata", format!("lfm3 config: missing integer {key:?}")))
}

fn opt_u32(v: Option<&Value>) -> Option<u32> {
    v.and_then(Value::as_u64).and_then(|x| u32::try_from(x).ok())
}

/// `eos_token_id` as int or list.
fn id_list(v: Option<&Value>) -> Vec<u32> {
    match v {
        Some(Value::Array(a)) => a.iter().filter_map(|x| opt_u32(Some(x))).collect(),
        other => opt_u32(other).into_iter().collect(),
    }
}

fn rope(v: &Value, layer_type: &str) -> Result<RopeParams> {
    let p = v.pointer(&format!("/rope_parameters/{layer_type}"));
    let Some(p) = p else {
        // Defaults of `Lfm3Config.__post_init__`.
        return Ok(match layer_type {
            "full_attention" => RopeParams { theta: 1_000_000.0, partial_rotary_factor: 0.5 },
            _ => RopeParams { theta: 10_000.0, partial_rotary_factor: 1.0 },
        });
    };
    let kind = p.get("rope_type").and_then(Value::as_str).unwrap_or("default");
    if kind != "default" {
        bail!("unsupported_model", "lfm3: rope_type {kind:?} on {layer_type} layers is not supported");
    }
    let theta = p
        .get("rope_theta")
        .and_then(Value::as_f64)
        .ok_or_else(|| Error::new("missing_metadata", format!("lfm3 config: no rope_theta for {layer_type}")))?;
    let partial = p.get("partial_rotary_factor").and_then(Value::as_f64).unwrap_or(1.0);
    Ok(RopeParams { theta: theta as f32, partial_rotary_factor: partial as f32 })
}

impl Lfm3Config {
    /// From an HF `config.json` (model_type "lfm3") or the `.lapid` metadata object.
    pub fn from_json(v: &Value) -> Result<Lfm3Config> {
        let hidden_size = get_usize(v, "hidden_size")?;
        let n_head = get_usize(v, "num_attention_heads")?;
        let n_layers = get_usize(v, "num_hidden_layers")?;
        let head_dim = v.get("head_dim").and_then(Value::as_u64).map_or(hidden_size / n_head.max(1), |x| x as usize);
        let layer_types = match v.get("layer_types").and_then(Value::as_array) {
            Some(a) => a
                .iter()
                .map(|t| LayerType::parse(t.as_str().unwrap_or("?")))
                .collect::<Result<Vec<_>>>()?,
            None => (0..n_layers)
                .map(|i| match i % 4 {
                    2 => LayerType::FullAttention,
                    0 if i > 0 => LayerType::SlidingAttention,
                    _ => LayerType::Conv,
                })
                .collect(),
        };
        if layer_types.len() != n_layers {
            bail!("metadata", "lfm3: {} layer_types for {n_layers} layers", layer_types.len());
        }
        let use_sandwich = v.get("use_sandwich_norm").and_then(Value::as_bool).unwrap_or(true);
        if v.get("tie_word_embeddings").and_then(Value::as_bool) == Some(false) {
            bail!("unsupported_model", "lfm3: untied lm_head is not supported");
        }
        let cfg = Lfm3Config {
            vocab_size: get_usize(v, "vocab_size")?,
            hidden_size,
            intermediate_size: get_usize(v, "intermediate_size")?,
            n_layers,
            n_head,
            n_kv_head: get_usize(v, "num_key_value_heads")?,
            head_dim,
            norm_eps: v.get("norm_eps").and_then(Value::as_f64).unwrap_or(1e-5) as f32,
            conv_kernel: v.get("conv_L_cache").and_then(Value::as_u64).unwrap_or(7) as usize,
            conv_bias: v.get("conv_bias").and_then(Value::as_bool).unwrap_or(false),
            layer_types,
            sliding_window: v.get("sliding_window").and_then(Value::as_u64).unwrap_or(256) as usize,
            rope_full: rope(v, "full_attention")?,
            rope_sliding: rope(v, "sliding_attention")?,
            sandwich_norm: use_sandwich,
            max_positions: v.get("max_position_embeddings").and_then(Value::as_u64).unwrap_or(128_000) as usize,
            special: Specials {
                bos: opt_u32(v.get("bos_token_id")),
                eos: id_list(v.get("eos_token_id")),
                pad: opt_u32(v.get("pad_token_id")),
            },
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(Error::new("metadata", format!("lfm3: {m}")));
        if self.n_head == 0 || self.n_kv_head == 0 || !self.n_head.is_multiple_of(self.n_kv_head) {
            return bad(format!("{} heads not a multiple of {} kv heads", self.n_head, self.n_kv_head));
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return bad(format!("head_dim {} must be even", self.head_dim));
        }
        if self.conv_kernel < 1 || self.sliding_window == 0 {
            return bad("conv_L_cache and sliding_window must be >= 1".into());
        }
        Ok(())
    }

    /// The `"lfm3"` metadata object (HF keys).
    pub fn to_json(&self) -> Value {
        let rope = |r: RopeParams| json!({"rope_type": "default", "rope_theta": r.theta, "partial_rotary_factor": r.partial_rotary_factor});
        json!({
            "model_type": "lfm3",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.n_layers,
            "num_attention_heads": self.n_head,
            "num_key_value_heads": self.n_kv_head,
            "head_dim": self.head_dim,
            "norm_eps": self.norm_eps,
            "conv_L_cache": self.conv_kernel,
            "conv_bias": self.conv_bias,
            "layer_types": self.layer_types.iter().map(|t| t.name()).collect::<Vec<_>>(),
            "sliding_window": self.sliding_window,
            "rope_parameters": {"full_attention": rope(self.rope_full), "sliding_attention": rope(self.rope_sliding)},
            "use_sandwich_norm": self.sandwich_norm,
            "max_position_embeddings": self.max_positions,
            "tie_word_embeddings": true,
            "bos_token_id": self.special.bos,
            "eos_token_id": self.special.eos,
            "pad_token_id": self.special.pad,
        })
    }

    /// Per-layer KV window (`Some(sliding_window)` on sliding layers) for `KvCache`.
    pub fn kv_windows(&self) -> Vec<Option<usize>> {
        self.layer_types
            .iter()
            .map(|t| if *t == LayerType::SlidingAttention { Some(self.sliding_window) } else { None })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_config_roundtrip() {
        let v = json!({
            "model_type": "lfm3", "vocab_size": 128000, "hidden_size": 2048, "intermediate_size": 7680,
            "num_hidden_layers": 24, "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 128,
            "norm_eps": 1e-5, "conv_L_cache": 7, "bos_token_id": 124894, "eos_token_id": 124900,
            "rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 1e6, "partial_rotary_factor": 0.5},
                                "sliding_attention": {"rope_type": "default", "rope_theta": 10000}},
        });
        let c = Lfm3Config::from_json(&v).expect("config");
        assert_eq!(c.layer_types[2], LayerType::FullAttention);
        assert_eq!(c.layer_types[4], LayerType::SlidingAttention);
        assert_eq!(c.layer_types[0], LayerType::Conv);
        assert_eq!(c.layer_types.iter().filter(|t| **t == LayerType::Conv).count(), 13);
        assert_eq!(c.rope_full.partial_rotary_factor, 0.5);
        assert_eq!(c.special.eos, vec![124900]);
        assert_eq!(Lfm3Config::from_json(&c.to_json()).expect("again"), c);
        assert_eq!(c.kv_windows()[4], Some(256));
    }
}
