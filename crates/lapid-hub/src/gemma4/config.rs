//! Gemma 4 hyper-parameters: the text decoder (dense or MoE), the vision tower and the image
//! processor. One parser reads an HF `config.json` (`text_config` + `vision_config`, with
//! `image_processor` from `processor_config.json`) and the `.lapid` `"gemma4"` metadata
//! object (flat HF text keys + `vision_config`, `image_processor`, image token ids).

use lapid::{Error, Result, bail};
use serde_json::Value;

use crate::vision::Filter;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

/// RoPE of one layer type: `theta` and the rotated fraction of each head (HF
/// "proportional" RoPE = zero tail: frequencies follow the full head dim).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RopeParams {
    pub theta: f32,
    pub partial: f32,
}

/// Sparse expert block of every layer (26B-A4B).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MoeParams {
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate: usize,
}

/// HF `Gemma4VisionConfig`.
#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub position_embedding_size: usize,
    pub rope_theta: f32,
    pub standardize: bool,
}

/// HF `Gemma4ImageProcessor` settings.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageProcessorConfig {
    pub patch_size: usize,
    /// Soft tokens per image (70, 140, 280, 560 or 1120; more = higher resolution).
    pub max_soft_tokens: usize,
    pub pooling_kernel_size: usize,
    pub resample: Filter,
    pub do_resize: bool,
    pub rescale_factor: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_head: usize,
    /// KV heads / head dim of sliding layers, and of full-attention layers.
    pub n_kv_head: usize,
    pub n_kv_head_full: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub norm_eps: f32,
    pub layer_types: Vec<LayerType>,
    pub sliding_window: usize,
    pub rope_sliding: RopeParams,
    pub rope_full: RopeParams,
    /// The last `n_kv_shared` layers reuse the K/V of the last earlier layer of their type.
    pub n_kv_shared: usize,
    pub double_wide_mlp: bool,
    /// Per-layer-embedding width (0 = no PLE) and its vocabulary.
    pub ple_dim: usize,
    pub ple_vocab: usize,
    pub final_logit_softcapping: Option<f32>,
    /// Full-attention layers have no v_proj (V = K before k_norm).
    pub attention_k_eq_v: bool,
    pub max_positions: usize,
    pub bos: Option<u32>,
    pub eos: Vec<u32>,
    pub pad: Option<u32>,
    pub moe: Option<MoeParams>,
    /// `use_bidirectional_attention == "vision"`: image tokens see each other on sliding layers.
    pub bidirectional_vision: bool,
    pub vision: Option<VisionConfig>,
    pub processor: ImageProcessorConfig,
    /// `<|image|>`, `<|image>`, `<image|>` ids.
    pub image_tokens: [Option<u32>; 3],
}

fn need(v: &Value, key: &str) -> Result<usize> {
    v.get(key).and_then(Value::as_u64).map(|x| x as usize).ok_or_else(|| Error::new("missing_metadata", format!("gemma4 config: missing integer {key:?}")))
}

fn opt(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(Value::as_u64).map(|x| x as usize)
}

fn opt_u32(v: Option<&Value>) -> Option<u32> {
    v.and_then(Value::as_u64).and_then(|x| u32::try_from(x).ok())
}

fn rope(v: &Value, layer_type: &str) -> Result<RopeParams> {
    let Some(p) = v.pointer(&format!("/rope_parameters/{layer_type}")) else {
        return Ok(if layer_type == "full_attention" { RopeParams { theta: 1e6, partial: 0.25 } } else { RopeParams { theta: 1e4, partial: 1.0 } });
    };
    let theta = p.get("rope_theta").and_then(Value::as_f64).ok_or_else(|| Error::new("missing_metadata", format!("gemma4: no rope_theta for {layer_type}")))?;
    let partial = match p.get("rope_type").and_then(Value::as_str).unwrap_or("default") {
        // HF "default" ignores partial_rotary_factor for Gemma 4 (rotates the whole head).
        "default" => 1.0,
        "proportional" => p.get("partial_rotary_factor").and_then(Value::as_f64).unwrap_or(1.0),
        other => bail!("unsupported_model", "gemma4: rope_type {other:?} on {layer_type} layers is not supported"),
    };
    Ok(RopeParams { theta: theta as f32, partial: partial as f32 })
}

impl VisionConfig {
    pub fn from_json(v: &Value) -> Result<VisionConfig> {
        let (hidden_size, n_head) = (need(v, "hidden_size")?, need(v, "num_attention_heads")?);
        let act = v.get("hidden_activation").and_then(Value::as_str).unwrap_or("gelu_pytorch_tanh");
        if act != "gelu_pytorch_tanh" {
            bail!("unsupported_model", "gemma4 vision: activation {act:?} is not supported");
        }
        let c = VisionConfig {
            hidden_size,
            intermediate_size: need(v, "intermediate_size")?,
            n_layers: need(v, "num_hidden_layers")?,
            n_head,
            n_kv_head: opt(v, "num_key_value_heads").unwrap_or(n_head),
            head_dim: opt(v, "head_dim").unwrap_or(hidden_size / n_head.max(1)),
            eps: v.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
            patch_size: opt(v, "patch_size").unwrap_or(16),
            pooling_kernel_size: opt(v, "pooling_kernel_size").unwrap_or(3),
            position_embedding_size: opt(v, "position_embedding_size").unwrap_or(10240),
            rope_theta: v.pointer("/rope_parameters/rope_theta").and_then(Value::as_f64).unwrap_or(100.0) as f32,
            standardize: v.get("standardize").and_then(Value::as_bool).unwrap_or(false),
        };
        if c.n_head == 0 || c.n_kv_head == 0 || c.n_head % c.n_kv_head != 0 || c.head_dim == 0 || c.head_dim % 4 != 0 {
            bail!("metadata", "gemma4 vision: {} heads / {} kv heads / head dim {}", c.n_head, c.n_kv_head, c.head_dim);
        }
        Ok(c)
    }

}

impl ImageProcessorConfig {
    /// From the `image_processor` object (missing keys: Gemma 4 defaults).
    pub fn from_json(v: Option<&Value>) -> ImageProcessorConfig {
        let v = v.unwrap_or(&Value::Null);
        let f = |k: &str, d: f64| v.get(k).and_then(Value::as_f64).unwrap_or(d) as f32;
        let triple = |k: &str, d: f32| match v.get(k).and_then(Value::as_array) {
            Some(a) if a.len() == 3 => [0, 1, 2].map(|i| a[i].as_f64().map_or(d, |x| x as f32)),
            _ => [d; 3],
        };
        let norm = v.get("do_normalize").and_then(Value::as_bool).unwrap_or(false);
        ImageProcessorConfig {
            patch_size: opt(v, "patch_size").unwrap_or(16),
            max_soft_tokens: opt(v, "max_soft_tokens").unwrap_or(280),
            pooling_kernel_size: opt(v, "pooling_kernel_size").unwrap_or(3),
            resample: v.get("resample").and_then(Value::as_u64).map_or(Filter::Bicubic, Filter::from_pil),
            do_resize: v.get("do_resize").and_then(Value::as_bool).unwrap_or(true),
            rescale_factor: f("rescale_factor", 1.0 / 255.0),
            mean: if norm { triple("image_mean", 0.0) } else { [0.0; 3] },
            std: if norm { triple("image_std", 1.0) } else { [1.0; 3] },
        }
    }
}

impl Gemma4Config {
    /// From an HF `config.json` (`gemma4` with `text_config`, or `gemma4_text`) or the
    /// `.lapid` metadata object.
    pub fn from_json(root: &Value) -> Result<Gemma4Config> {
        let v = root.get("text_config").unwrap_or(root);
        let get = |k: &str| v.get(k).or_else(|| root.get(k));
        if get("tie_word_embeddings").and_then(Value::as_bool) == Some(false) {
            bail!("unsupported_model", "gemma4: untied lm_head is not supported");
        }
        let (hidden_size, n_head, n_layers) = (need(v, "hidden_size")?, need(v, "num_attention_heads")?, need(v, "num_hidden_layers")?);
        let layer_types = match v.get("layer_types").and_then(Value::as_array) {
            Some(a) => a
                .iter()
                .map(|t| match t.as_str() {
                    Some("sliding_attention") => Ok(LayerType::SlidingAttention),
                    Some("full_attention") => Ok(LayerType::FullAttention),
                    other => Err(Error::new("unsupported_model", format!("gemma4: unknown layer type {other:?}"))),
                })
                .collect::<Result<Vec<_>>>()?,
            None => (0..n_layers).map(|i| if (i + 1) % 5 == 0 { LayerType::FullAttention } else { LayerType::SlidingAttention }).collect(),
        };
        // Newer transformers write the full-attention overrides as `per_layer_config`.
        let full_override = |key: &str| -> Option<usize> {
            let plc = v.get("per_layer_config")?.as_object()?;
            let i = layer_types.iter().position(|t| *t == LayerType::FullAttention)?;
            plc.get(&i.to_string())?.get(key)?.as_u64().map(|x| x as usize)
        };
        let n_kv_head = need(v, "num_key_value_heads")?;
        let moe = if v.get("enable_moe_block").and_then(Value::as_bool) == Some(true) {
            Some(MoeParams { num_experts: need(v, "num_experts")?, top_k: need(v, "top_k_experts")?, intermediate: need(v, "moe_intermediate_size")? })
        } else {
            None
        };
        let cfg = Gemma4Config {
            vocab_size: need(v, "vocab_size")?,
            hidden_size,
            intermediate_size: need(v, "intermediate_size")?,
            n_layers,
            n_head,
            n_kv_head,
            n_kv_head_full: opt(v, "num_global_key_value_heads").or_else(|| full_override("num_key_value_heads")).unwrap_or(n_kv_head),
            head_dim: opt(v, "head_dim").unwrap_or(hidden_size / n_head.max(1)),
            global_head_dim: opt(v, "global_head_dim").or_else(|| full_override("head_dim")).unwrap_or(512),
            norm_eps: v.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
            sliding_window: opt(v, "sliding_window").unwrap_or(512),
            rope_sliding: rope(v, "sliding_attention")?,
            rope_full: rope(v, "full_attention")?,
            n_kv_shared: opt(v, "num_kv_shared_layers").unwrap_or(0),
            double_wide_mlp: v.get("use_double_wide_mlp").and_then(Value::as_bool).unwrap_or(false),
            ple_dim: opt(v, "hidden_size_per_layer_input").unwrap_or(0),
            ple_vocab: opt(v, "vocab_size_per_layer_input").unwrap_or(0),
            final_logit_softcapping: v.get("final_logit_softcapping").and_then(Value::as_f64).map(|x| x as f32),
            attention_k_eq_v: v.get("attention_k_eq_v").and_then(Value::as_bool).unwrap_or(false),
            max_positions: opt(v, "max_position_embeddings").unwrap_or(131_072),
            bos: opt_u32(get("bos_token_id")),
            eos: crate::runtime::chat::id_list(get("eos_token_id")),
            pad: opt_u32(get("pad_token_id")),
            layer_types,
            moe,
            bidirectional_vision: get("use_bidirectional_attention").and_then(Value::as_str) == Some("vision"),
            vision: root.get("vision_config").filter(|x| x.is_object()).map(VisionConfig::from_json).transpose()?,
            processor: ImageProcessorConfig::from_json(root.get("image_processor")),
            image_tokens: ["image_token_id", "boi_token_id", "eoi_token_id"].map(|k| opt_u32(root.get(k))),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(Error::new("metadata", format!("gemma4: {m}")));
        if self.layer_types.len() != self.n_layers {
            return bad(format!("{} layer_types for {} layers", self.layer_types.len(), self.n_layers));
        }
        for kv in [self.n_kv_head, self.n_kv_head_full] {
            if self.n_head == 0 || kv == 0 || !self.n_head.is_multiple_of(kv) {
                return bad(format!("{} heads not a multiple of {kv} kv heads", self.n_head));
            }
        }
        if self.head_dim == 0 || self.head_dim % 2 != 0 || self.global_head_dim % 2 != 0 {
            return bad("head dims must be even".into());
        }
        if self.n_kv_shared > 0 && self.n_kv_shared >= self.n_layers || (self.first_shared()..self.n_layers).any(|i| self.kv_source(i).is_none()) {
            return bad(format!("{} kv-shared layers of {} without a source", self.n_kv_shared, self.n_layers));
        }
        if self.ple_dim > 0 && self.ple_vocab == 0 {
            return bad("vocab_size_per_layer_input missing".into());
        }
        if let Some(m) = self.moe.filter(|m| m.top_k == 0 || m.top_k > m.num_experts) {
            return bad(format!("top_k {} of {} experts", m.top_k, m.num_experts));
        }
        Ok(())
    }

    /// Index of the first kv-shared layer (`n_layers` when none).
    pub fn first_shared(&self) -> usize {
        self.n_layers - self.n_kv_shared
    }

    /// The layer whose K/V `layer` attends to: itself, or for a kv-shared layer the last
    /// non-shared layer of the same type.
    pub fn kv_source(&self, layer: usize) -> Option<usize> {
        if layer < self.first_shared() {
            return Some(layer);
        }
        (0..self.first_shared()).rev().find(|&j| self.layer_types[j] == self.layer_types[layer])
    }

    /// (head dim, kv heads) of a layer.
    pub fn attn_dims(&self, layer: usize) -> (usize, usize) {
        match self.layer_types[layer] {
            LayerType::SlidingAttention => (self.head_dim, self.n_kv_head),
            LayerType::FullAttention => (self.global_head_dim, self.n_kv_head_full),
        }
    }

    pub fn layer_intermediate(&self, layer: usize) -> usize {
        if self.double_wide_mlp && layer >= self.first_shared() { 2 * self.intermediate_size } else { self.intermediate_size }
    }

    /// Whether a full-attention layer takes V from k_proj (no v_proj).
    pub fn k_eq_v(&self, layer: usize) -> bool {
        self.attention_k_eq_v && self.layer_types[layer] == LayerType::FullAttention
    }

    /// Per-layer KV window for `KvCache` (kv-shared layers never append).
    pub fn kv_windows(&self) -> Vec<Option<usize>> {
        self.layer_types.iter().map(|t| (*t == LayerType::SlidingAttention).then_some(self.sliding_window)).collect()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// google/gemma-4-E2B-it (text part) and the 26B-A4B differences.
    fn cfg(moe: bool) -> Value {
        let full = if moe { 6 } else { 5 };
        let types: Vec<&str> = (0..35).map(|i| if (i + 1) % full == 0 { "full_attention" } else { "sliding_attention" }).collect();
        let mut t = json!({"vocab_size": 262144, "hidden_size": 1536, "intermediate_size": 6144, "num_hidden_layers": 35,
            "num_attention_heads": 8, "num_key_value_heads": 1, "head_dim": 256, "global_head_dim": 512, "layer_types": types,
            "num_kv_shared_layers": 20, "use_double_wide_mlp": true, "hidden_size_per_layer_input": 256, "vocab_size_per_layer_input": 262144,
            "final_logit_softcapping": 30.0, "eos_token_id": 1, "pad_token_id": 0,
            "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}});
        if moe {
            for (k, v) in [("enable_moe_block", json!(true)), ("num_experts", json!(128)), ("top_k_experts", json!(8)), ("moe_intermediate_size", json!(704)),
                ("attention_k_eq_v", json!(true)), ("num_kv_shared_layers", json!(0)), ("num_global_key_value_heads", json!(2)), ("use_bidirectional_attention", json!("vision"))] {
                t[k] = v;
            }
        }
        json!({"model_type": "gemma4", "eos_token_id": [1, 106], "image_token_id": 258880, "text_config": t,
               "vision_config": {"hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 16, "num_attention_heads": 12}})
    }

    #[test]
    fn configs() {
        let c = Gemma4Config::from_json(&cfg(false)).expect("e2b");
        assert_eq!((c.first_shared(), c.kv_source(15), c.kv_source(34)), (15, Some(13), Some(14)));
        assert_eq!((c.layer_intermediate(20), c.layer_intermediate(14), c.attn_dims(4)), (12288, 6144, (512, 1)));
        assert_eq!((c.rope_full.partial, c.eos.clone(), c.moe), (0.25, vec![1], None));
        assert_eq!(c.image_tokens[0], Some(258880));
        let m = Gemma4Config::from_json(&cfg(true)).expect("a4b");
        assert_eq!(m.moe, Some(MoeParams { num_experts: 128, top_k: 8, intermediate: 704 }));
        assert!(m.bidirectional_vision && m.k_eq_v(5) && !m.k_eq_v(4));
        assert_eq!(m.attn_dims(5), (512, 2));
    }
}
