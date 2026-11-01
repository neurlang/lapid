//! Gemma 4 text-decoder hyper-parameters. The `.lapid` metadata object `"gemma4"` keeps the
//! Hugging Face `text_config` keys (plus `special_tokens`, `generation`, blob names), so one
//! parser reads both. A full multimodal `config.json` (with `text_config`) is accepted too.

use lapid::{Error, Result, bail};
use serde_json::{Value, json};

/// Attention kind of a decoder layer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

impl LayerType {
    pub fn parse(s: &str) -> Result<LayerType> {
        Ok(match s {
            "sliding_attention" => LayerType::SlidingAttention,
            "full_attention" => LayerType::FullAttention,
            other => bail!("unsupported_model", "gemma4: unknown layer type {other:?}"),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            LayerType::SlidingAttention => "sliding_attention",
            LayerType::FullAttention => "full_attention",
        }
    }
}

/// RoPE of one layer type: `theta` and the rotated fraction of each head ("proportional"
/// RoPE = zero-tail: frequencies follow the full head dimension, the tail is not rotated).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RopeParams {
    pub theta: f32,
    pub partial_rotary_factor: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Specials {
    pub bos: Option<u32>,
    pub eos: Vec<u32>,
    pub pad: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_head: usize,
    /// KV heads of sliding layers.
    pub n_kv_head: usize,
    /// KV heads of full-attention layers.
    pub n_kv_head_full: usize,
    /// Head dim of sliding layers.
    pub head_dim: usize,
    /// Head dim of full-attention layers.
    pub global_head_dim: usize,
    pub norm_eps: f32,
    pub layer_types: Vec<LayerType>,
    pub sliding_window: usize,
    pub rope_sliding: RopeParams,
    pub rope_full: RopeParams,
    /// The last `n_kv_shared` layers reuse the K/V of the last earlier layer of their type.
    pub n_kv_shared: usize,
    pub double_wide_mlp: bool,
    /// Per-layer-embedding width (0 = no PLE).
    pub ple_dim: usize,
    pub ple_vocab: usize,
    pub final_logit_softcapping: Option<f32>,
    /// Full-attention layers have no v_proj (V = K before k_norm).
    pub attention_k_eq_v: bool,
    pub max_positions: usize,
    pub special: Specials,
}

fn get_usize(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| Error::new("missing_metadata", format!("gemma4 config: missing integer {key:?}")))
}

fn opt_usize(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(Value::as_u64).map(|x| x as usize)
}

fn opt_u32(v: Option<&Value>) -> Option<u32> {
    v.and_then(Value::as_u64).and_then(|x| u32::try_from(x).ok())
}

fn id_list(v: Option<&Value>) -> Vec<u32> {
    match v {
        Some(Value::Array(a)) => a.iter().filter_map(|x| opt_u32(Some(x))).collect(),
        other => opt_u32(other).into_iter().collect(),
    }
}

fn rope(v: &Value, layer_type: &str) -> Result<RopeParams> {
    let Some(p) = v.pointer(&format!("/rope_parameters/{layer_type}")) else {
        return Ok(match layer_type {
            "full_attention" => RopeParams { theta: 1_000_000.0, partial_rotary_factor: 0.25 },
            _ => RopeParams { theta: 10_000.0, partial_rotary_factor: 1.0 },
        });
    };
    let kind = p.get("rope_type").and_then(Value::as_str).unwrap_or("default");
    let theta = p
        .get("rope_theta")
        .and_then(Value::as_f64)
        .ok_or_else(|| Error::new("missing_metadata", format!("gemma4 config: no rope_theta for {layer_type}")))?;
    let partial = match kind {
        // HF "default" ignores partial_rotary_factor for Gemma 4 (rotates the whole head).
        "default" => 1.0,
        "proportional" => p.get("partial_rotary_factor").and_then(Value::as_f64).unwrap_or(1.0),
        other => bail!("unsupported_model", "gemma4: rope_type {other:?} on {layer_type} layers is not supported"),
    };
    Ok(RopeParams { theta: theta as f32, partial_rotary_factor: partial as f32 })
}

impl Gemma4Config {
    /// From an HF `config.json` (`gemma4` with `text_config`, or `gemma4_text`) or the
    /// `.lapid` metadata object.
    pub fn from_json(root: &Value) -> Result<Gemma4Config> {
        let v = root.get("text_config").unwrap_or(root);
        if v.get("enable_moe_block").and_then(Value::as_bool) == Some(true) {
            bail!("unsupported_model", "gemma4: MoE variants (enable_moe_block) are not supported yet");
        }
        if v.get("tie_word_embeddings").or_else(|| root.get("tie_word_embeddings")).and_then(Value::as_bool) == Some(false) {
            bail!("unsupported_model", "gemma4: untied lm_head is not supported");
        }
        let hidden_size = get_usize(v, "hidden_size")?;
        let n_head = get_usize(v, "num_attention_heads")?;
        let n_layers = get_usize(v, "num_hidden_layers")?;
        let head_dim = opt_usize(v, "head_dim").unwrap_or(hidden_size / n_head.max(1));
        let layer_types = match v.get("layer_types").and_then(Value::as_array) {
            Some(a) => a.iter().map(|t| LayerType::parse(t.as_str().unwrap_or("?"))).collect::<Result<Vec<_>>>()?,
            None => (0..n_layers)
                .map(|i| if (i + 1) % 5 == 0 { LayerType::FullAttention } else { LayerType::SlidingAttention })
                .collect(),
        };
        if layer_types.len() != n_layers {
            bail!("metadata", "gemma4: {} layer_types for {n_layers} layers", layer_types.len());
        }
        let n_kv_head = get_usize(v, "num_key_value_heads")?;
        let k_eq_v = v.get("attention_k_eq_v").and_then(Value::as_bool).unwrap_or(false);
        // Newer transformers serialize the full-attention overrides as `per_layer_config`
        // {"<layer>": {"head_dim": .., "num_key_value_heads": ..}} instead of global_* keys.
        let full_override = |key: &str| -> Option<usize> {
            let plc = v.get("per_layer_config")?.as_object()?;
            layer_types
                .iter()
                .enumerate()
                .filter(|(_, t)| **t == LayerType::FullAttention)
                .find_map(|(i, _)| plc.get(&i.to_string())?.get(key)?.as_u64().map(|x| x as usize))
        };
        let n_kv_head_full = opt_usize(v, "num_global_key_value_heads")
            .or_else(|| full_override("num_key_value_heads"))
            .unwrap_or(n_kv_head);
        let global_head_dim = opt_usize(v, "global_head_dim").or_else(|| full_override("head_dim")).unwrap_or(512);
        let special_src = |k: &str| v.get(k).or_else(|| root.get(k));
        let cfg = Gemma4Config {
            vocab_size: get_usize(v, "vocab_size")?,
            hidden_size,
            intermediate_size: get_usize(v, "intermediate_size")?,
            n_layers,
            n_head,
            n_kv_head,
            n_kv_head_full,
            head_dim,
            global_head_dim,
            norm_eps: v.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
            layer_types,
            sliding_window: opt_usize(v, "sliding_window").unwrap_or(512),
            rope_sliding: rope(v, "sliding_attention")?,
            rope_full: rope(v, "full_attention")?,
            n_kv_shared: opt_usize(v, "num_kv_shared_layers").unwrap_or(0),
            double_wide_mlp: v.get("use_double_wide_mlp").and_then(Value::as_bool).unwrap_or(false),
            ple_dim: opt_usize(v, "hidden_size_per_layer_input").unwrap_or(0),
            ple_vocab: opt_usize(v, "vocab_size_per_layer_input").unwrap_or(0),
            final_logit_softcapping: v.get("final_logit_softcapping").and_then(Value::as_f64).map(|x| x as f32),
            attention_k_eq_v: k_eq_v,
            max_positions: opt_usize(v, "max_position_embeddings").unwrap_or(131_072),
            special: Specials {
                bos: opt_u32(special_src("bos_token_id")),
                eos: id_list(special_src("eos_token_id")),
                pad: opt_u32(special_src("pad_token_id")),
            },
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(Error::new("metadata", format!("gemma4: {m}")));
        for kv in [self.n_kv_head, self.n_kv_head_full] {
            if self.n_head == 0 || kv == 0 || !self.n_head.is_multiple_of(kv) {
                return bad(format!("{} heads not a multiple of {kv} kv heads", self.n_head));
            }
        }
        if self.head_dim % 2 != 0 || self.global_head_dim % 2 != 0 || self.head_dim == 0 {
            return bad("head dims must be even".into());
        }
        if self.n_kv_shared >= self.n_layers && self.n_kv_shared > 0 {
            return bad(format!("{} kv-shared layers of {}", self.n_kv_shared, self.n_layers));
        }
        for i in self.first_shared()..self.n_layers {
            if self.kv_source(i).is_none() {
                return bad(format!("kv-shared layer {i} has no earlier layer of its type"));
            }
        }
        if self.ple_dim > 0 && self.ple_vocab == 0 {
            return bad("vocab_size_per_layer_input missing".into());
        }
        Ok(())
    }

    /// Index of the first kv-shared layer (`n_layers` when none).
    pub fn first_shared(&self) -> usize {
        self.n_layers - self.n_kv_shared
    }

    pub fn is_shared(&self, layer: usize) -> bool {
        self.n_kv_shared > 0 && layer >= self.first_shared()
    }

    /// For a kv-shared layer: the layer whose K/V it reads (the last non-shared layer of the
    /// same type). `Some(layer)` for layers with their own K/V.
    pub fn kv_source(&self, layer: usize) -> Option<usize> {
        if !self.is_shared(layer) {
            return Some(layer);
        }
        let t = self.layer_types[layer];
        (0..self.first_shared()).rev().find(|&j| self.layer_types[j] == t)
    }

    pub fn layer_head_dim(&self, layer: usize) -> usize {
        match self.layer_types[layer] {
            LayerType::SlidingAttention => self.head_dim,
            LayerType::FullAttention => self.global_head_dim,
        }
    }

    pub fn layer_kv_heads(&self, layer: usize) -> usize {
        match self.layer_types[layer] {
            LayerType::SlidingAttention => self.n_kv_head,
            LayerType::FullAttention => self.n_kv_head_full,
        }
    }

    pub fn layer_intermediate(&self, layer: usize) -> usize {
        if self.double_wide_mlp && self.is_shared(layer) { 2 * self.intermediate_size } else { self.intermediate_size }
    }

    /// Whether a full-attention layer takes V from k_proj (no v_proj).
    pub fn k_eq_v(&self, layer: usize) -> bool {
        self.attention_k_eq_v && self.layer_types[layer] == LayerType::FullAttention
    }

    /// Per-layer KV window for `KvCache` (kv-shared layers never append).
    pub fn kv_windows(&self) -> Vec<Option<usize>> {
        self.layer_types
            .iter()
            .map(|t| if *t == LayerType::SlidingAttention { Some(self.sliding_window) } else { None })
            .collect()
    }

    /// The `"gemma4"` metadata object (HF text_config keys).
    pub fn to_json(&self) -> Value {
        let rope = |r: RopeParams, kind: &str| {
            json!({"rope_type": kind, "rope_theta": r.theta, "partial_rotary_factor": r.partial_rotary_factor})
        };
        json!({
            "model_type": "gemma4_text",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.n_layers,
            "num_attention_heads": self.n_head,
            "num_key_value_heads": self.n_kv_head,
            "num_global_key_value_heads": self.n_kv_head_full,
            "head_dim": self.head_dim,
            "global_head_dim": self.global_head_dim,
            "rms_norm_eps": self.norm_eps,
            "layer_types": self.layer_types.iter().map(|t| t.name()).collect::<Vec<_>>(),
            "sliding_window": self.sliding_window,
            "rope_parameters": {
                "sliding_attention": rope(self.rope_sliding, "default"),
                "full_attention": rope(self.rope_full, "proportional"),
            },
            "num_kv_shared_layers": self.n_kv_shared,
            "use_double_wide_mlp": self.double_wide_mlp,
            "hidden_size_per_layer_input": self.ple_dim,
            "vocab_size_per_layer_input": self.ple_vocab,
            "final_logit_softcapping": self.final_logit_softcapping,
            "attention_k_eq_v": self.attention_k_eq_v,
            "enable_moe_block": false,
            "max_position_embeddings": self.max_positions,
            "tie_word_embeddings": true,
            "bos_token_id": self.special.bos,
            "eos_token_id": self.special.eos,
            "pad_token_id": self.special.pad,
        })
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use serde_json::{Value, json};

    /// The google/gemma-4-E2B-it config.json (text part).
    pub(crate) fn e2b() -> Value {
        let types: Vec<&str> = (0..35).map(|i| if (i + 1) % 5 == 0 { "full_attention" } else { "sliding_attention" }).collect();
        json!({"model_type": "gemma4", "eos_token_id": [1, 106], "text_config": {
            "vocab_size": 262144, "hidden_size": 1536, "intermediate_size": 6144, "num_hidden_layers": 35,
            "num_attention_heads": 8, "num_key_value_heads": 1, "head_dim": 256, "global_head_dim": 512,
            "rms_norm_eps": 1e-6, "layer_types": types, "sliding_window": 512, "num_kv_shared_layers": 20,
            "use_double_wide_mlp": true, "hidden_size_per_layer_input": 256, "vocab_size_per_layer_input": 262144,
            "final_logit_softcapping": 30.0, "attention_k_eq_v": false, "enable_moe_block": false, "bos_token_id": 2,
            "eos_token_id": 1, "pad_token_id": 0,
            "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}}})
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::e2b;
    use super::*;

    #[test]
    fn e2b_config() {
        let c = Gemma4Config::from_json(&e2b()).expect("config");
        assert_eq!(c.first_shared(), 15);
        assert_eq!(c.kv_source(15), Some(13));
        assert_eq!(c.kv_source(19), Some(14));
        assert_eq!(c.kv_source(34), Some(14));
        assert_eq!(c.layer_intermediate(20), 12288);
        assert_eq!(c.layer_intermediate(14), 6144);
        assert_eq!(c.layer_head_dim(4), 512);
        assert_eq!(c.rope_full.partial_rotary_factor, 0.25);
        assert_eq!(c.special.eos, vec![1]);
        assert_eq!(Gemma4Config::from_json(&c.to_json()).expect("again"), c);
    }
}
