//! Gemma 4 vision hyper-parameters: the HF `vision_config` (vision tower) and the processor's
//! `image_processor` object (resize budget, patching), both stored in the `.lapid` `"gemma4"`
//! metadata object by the gemma4moe converter.

use lapid::{Error, Result, bail};
use serde_json::{Value, json};

use crate::vision::Filter;

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
    pub use_clipped_linears: bool,
}

fn get(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| Error::new("missing_metadata", format!("gemma4 vision_config: missing integer {key:?}")))
}

fn opt(v: &Value, key: &str, default: usize) -> usize {
    v.get(key).and_then(Value::as_u64).map_or(default, |x| x as usize)
}

impl VisionConfig {
    pub fn from_json(v: &Value) -> Result<VisionConfig> {
        let hidden_size = get(v, "hidden_size")?;
        let n_head = get(v, "num_attention_heads")?;
        let act = v.get("hidden_activation").and_then(Value::as_str).unwrap_or("gelu_pytorch_tanh");
        if act != "gelu_pytorch_tanh" {
            bail!("unsupported_model", "gemma4 vision: activation {act:?} is not supported");
        }
        let c = VisionConfig {
            hidden_size,
            intermediate_size: get(v, "intermediate_size")?,
            n_layers: get(v, "num_hidden_layers")?,
            n_head,
            n_kv_head: opt(v, "num_key_value_heads", n_head),
            head_dim: opt(v, "head_dim", hidden_size / n_head.max(1)),
            eps: v.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
            patch_size: opt(v, "patch_size", 16),
            pooling_kernel_size: opt(v, "pooling_kernel_size", 3),
            position_embedding_size: opt(v, "position_embedding_size", 10240),
            rope_theta: v.pointer("/rope_parameters/rope_theta").and_then(Value::as_f64).unwrap_or(100.0) as f32,
            standardize: v.get("standardize").and_then(Value::as_bool).unwrap_or(false),
            use_clipped_linears: v.get("use_clipped_linears").and_then(Value::as_bool).unwrap_or(false),
        };
        if c.n_head == 0 || c.n_kv_head == 0 || c.n_head % c.n_kv_head != 0 {
            bail!("metadata", "gemma4 vision: {} heads / {} kv heads", c.n_head, c.n_kv_head);
        }
        if c.head_dim % 4 != 0 || c.head_dim == 0 {
            bail!("metadata", "gemma4 vision: head dim {} must be a multiple of 4 (2-D RoPE)", c.head_dim);
        }
        Ok(c)
    }

    pub fn to_json(&self) -> Value {
        json!({
            "model_type": "gemma4_vision",
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.n_layers,
            "num_attention_heads": self.n_head,
            "num_key_value_heads": self.n_kv_head,
            "head_dim": self.head_dim,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": self.eps,
            "patch_size": self.patch_size,
            "pooling_kernel_size": self.pooling_kernel_size,
            "position_embedding_size": self.position_embedding_size,
            "rope_parameters": {"rope_type": "default", "rope_theta": self.rope_theta},
            "standardize": self.standardize,
            "use_clipped_linears": self.use_clipped_linears,
        })
    }
}

/// HF `Gemma4ImageProcessor` settings.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageProcessorConfig {
    pub patch_size: usize,
    pub max_soft_tokens: usize,
    pub pooling_kernel_size: usize,
    pub resample: Filter,
    pub do_resize: bool,
    pub rescale_factor: f32,
    pub do_normalize: bool,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Default for ImageProcessorConfig {
    fn default() -> Self {
        ImageProcessorConfig {
            patch_size: 16,
            max_soft_tokens: 280,
            pooling_kernel_size: 3,
            resample: Filter::Bicubic,
            do_resize: true,
            rescale_factor: 1.0 / 255.0,
            do_normalize: false,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
        }
    }
}

fn triple(v: Option<&Value>, default: [f32; 3]) -> [f32; 3] {
    match v.and_then(Value::as_array) {
        Some(a) if a.len() == 3 => {
            let f = |i: usize| a[i].as_f64().map_or(default[i], |x| x as f32);
            [f(0), f(1), f(2)]
        }
        _ => default,
    }
}

impl ImageProcessorConfig {
    /// From the `image_processor` object (missing keys: Gemma 4 defaults).
    pub fn from_json(v: Option<&Value>) -> ImageProcessorConfig {
        let d = ImageProcessorConfig::default();
        let Some(v) = v else { return d };
        ImageProcessorConfig {
            patch_size: opt(v, "patch_size", d.patch_size),
            max_soft_tokens: opt(v, "max_soft_tokens", d.max_soft_tokens),
            pooling_kernel_size: opt(v, "pooling_kernel_size", d.pooling_kernel_size),
            resample: v.get("resample").and_then(Value::as_u64).map_or(d.resample, Filter::from_pil),
            do_resize: v.get("do_resize").and_then(Value::as_bool).unwrap_or(true),
            rescale_factor: v.get("rescale_factor").and_then(Value::as_f64).map_or(d.rescale_factor, |x| x as f32),
            do_normalize: v.get("do_normalize").and_then(Value::as_bool).unwrap_or(false),
            image_mean: triple(v.get("image_mean"), d.image_mean),
            image_std: triple(v.get("image_std"), d.image_std),
        }
    }

    pub fn to_json(&self) -> Value {
        let resample = match self.resample {
            Filter::Lanczos => 1,
            Filter::Bilinear => 2,
            Filter::Bicubic => 3,
        };
        json!({
            "image_processor_type": "Gemma4ImageProcessor",
            "patch_size": self.patch_size,
            "max_soft_tokens": self.max_soft_tokens,
            "pooling_kernel_size": self.pooling_kernel_size,
            "resample": resample,
            "do_resize": self.do_resize,
            "do_rescale": true,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2b_vision() {
        let v = json!({"hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 16, "num_attention_heads": 12,
            "num_key_value_heads": 12, "head_dim": 64, "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
            "use_clipped_linears": true, "standardize": false, "position_embedding_size": 10240});
        let c = VisionConfig::from_json(&v).expect("cfg");
        assert_eq!(VisionConfig::from_json(&c.to_json()).expect("again"), c);
        let p = ImageProcessorConfig::from_json(Some(&json!({"max_soft_tokens": 70, "resample": 3})));
        assert_eq!(p.max_soft_tokens, 70);
        assert_eq!(ImageProcessorConfig::from_json(Some(&p.to_json())), p);
    }
}
