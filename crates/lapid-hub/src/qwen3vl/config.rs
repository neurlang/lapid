//! Qwen3-VL hyper-parameters: the Qwen3 text decoder (`text_config`), the vision encoder
//! (`vision::VisionConfig`), the image processor and the special token ids. Parsed from an
//! HF `config.json` (`vision_config` in HF keys) or from the `.lapid` `"qwen3vl"` metadata
//! object (`text_config` in HF keys, `vision` / `processor` in lapid's generic form).

use lapid::{Error, Result, bail};
use serde_json::{Value, json};

use crate::runtime::chat::id_list;
use crate::vision::encoder::{NormKind, VisionConfig};
use crate::vision::processor::ImageProcessor;

#[derive(Clone, Debug, PartialEq)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    /// M-RoPE frequency sections (t, h, w) over head_dim / 2.
    pub mrope_section: [usize; 3],
    /// Qwen3-VL interleaves the h / w frequencies (THWTHW...); Qwen2-VL uses contiguous blocks.
    pub mrope_interleaved: bool,
    pub tie_embeddings: bool,
    pub attention_bias: bool,
    pub max_positions: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3VlConfig {
    pub text: TextConfig,
    pub vision: VisionConfig,
    pub processor: ImageProcessor,
    pub image_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub eos: Vec<u32>,
}

fn need(v: &Value, key: &str) -> Result<usize> {
    v.get(key).and_then(Value::as_u64).map(|x| x as usize).ok_or_else(|| Error::new("missing_metadata", format!("qwen3vl config: missing integer {key:?}")))
}

impl TextConfig {
    /// HF `text_config` (old `rope_scaling` / top-level `rope_theta` or new `rope_parameters`).
    pub fn from_json(v: &Value, tie_default: bool) -> Result<TextConfig> {
        let (hidden, n_head) = (need(v, "hidden_size")?, need(v, "num_attention_heads")?);
        let rope = v.get("rope_parameters").or_else(|| v.get("rope_scaling"));
        let theta = rope.and_then(|r| r.get("rope_theta")).or_else(|| v.get("rope_theta")).and_then(Value::as_f64).unwrap_or(5_000_000.0);
        let sec: Vec<usize> = rope
            .and_then(|r| r.get("mrope_section"))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|y| y as usize)).collect())
            .unwrap_or_else(|| vec![24, 20, 20]);
        if sec.len() < 3 {
            bail!("metadata", "qwen3vl: mrope_section {sec:?} needs 3 entries");
        }
        let c = TextConfig {
            vocab_size: need(v, "vocab_size")?,
            hidden_size: hidden,
            intermediate_size: need(v, "intermediate_size")?,
            n_layers: need(v, "num_hidden_layers")?,
            n_head,
            n_kv_head: need(v, "num_key_value_heads")?,
            head_dim: v.get("head_dim").and_then(Value::as_u64).map_or(hidden / n_head.max(1), |x| x as usize),
            norm_eps: v.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
            rope_theta: theta as f32,
            mrope_section: [sec[0], sec[1], sec[2]],
            mrope_interleaved: rope.and_then(|r| r.get("mrope_interleaved")).and_then(Value::as_bool).unwrap_or(true),
            tie_embeddings: v.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(tie_default),
            attention_bias: v.get("attention_bias").and_then(Value::as_bool).unwrap_or(false),
            max_positions: v.get("max_position_embeddings").and_then(Value::as_u64).unwrap_or(262_144) as usize,
        };
        if c.n_head == 0 || c.n_kv_head == 0 || !c.n_head.is_multiple_of(c.n_kv_head) {
            bail!("metadata", "qwen3vl: {} heads vs {} kv heads", c.n_head, c.n_kv_head);
        }
        if c.mrope_section.iter().sum::<usize>() * 2 != c.head_dim {
            bail!("metadata", "qwen3vl: mrope_section {:?} does not cover head dim {}", c.mrope_section, c.head_dim);
        }
        Ok(c)
    }

    pub fn to_json(&self) -> Value {
        json!({"vocab_size": self.vocab_size, "hidden_size": self.hidden_size, "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.n_layers, "num_attention_heads": self.n_head, "num_key_value_heads": self.n_kv_head,
            "head_dim": self.head_dim, "rms_norm_eps": self.norm_eps,
            "rope_parameters": {"rope_type": "default", "rope_theta": self.rope_theta, "mrope_section": self.mrope_section, "mrope_interleaved": self.mrope_interleaved},
            "tie_word_embeddings": self.tie_embeddings, "attention_bias": self.attention_bias, "max_position_embeddings": self.max_positions})
    }

    /// Which position stream (0 = t, 1 = h, 2 = w) drives frequency `i` of head_dim / 2.
    pub fn mrope_stream(&self, i: usize) -> usize {
        let [t, h, w] = self.mrope_section;
        match self.mrope_interleaved {
            true if i % 3 == 1 && i < 3 * h => 1,
            true if i % 3 == 2 && i < 3 * w => 2,
            true => 0,
            false if i < t => 0,
            false if i < t + h => 1,
            false => 2,
        }
    }
}

/// [`VisionConfig`] of an HF Qwen3-VL `vision_config`.
pub fn vision_from_hf(v: &Value) -> Result<VisionConfig> {
    let c = VisionConfig {
        hidden_size: need(v, "hidden_size")?,
        num_heads: need(v, "num_heads")?,
        depth: need(v, "depth")?,
        intermediate_size: need(v, "intermediate_size")?,
        patch_size: need(v, "patch_size")?,
        in_channels: v.get("in_channels").and_then(Value::as_u64).unwrap_or(3) as usize,
        norm: NormKind::LayerNorm,
        eps: 1e-6,
        hidden_act: v.get("hidden_act").and_then(Value::as_str).unwrap_or("gelu_pytorch_tanh").to_string(),
        position_grid: (need(v, "num_position_embeddings")? as f64).sqrt().round() as usize,
        rope_theta: Some(v.pointer("/rope_parameters/rope_theta").and_then(Value::as_f64).unwrap_or(10_000.0) as f32),
        spatial_merge_size: v.get("spatial_merge_size").and_then(Value::as_u64).unwrap_or(2) as usize,
        out_hidden_size: need(v, "out_hidden_size")?,
        deepstack_layers: v.get("deepstack_visual_indexes").and_then(Value::as_array).map(|a| a.iter().filter_map(|x| x.as_u64().map(|y| y as usize)).collect()).unwrap_or_default(),
    };
    c.validate()?;
    Ok(c)
}

impl Qwen3VlConfig {
    /// From an HF `config.json` plus its preprocessor config, or from the `.lapid` object.
    pub fn from_json(root: &Value, preprocessor: Option<&Value>) -> Result<Qwen3VlConfig> {
        let tie = root.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(false);
        let tv = root.get("text_config").ok_or_else(|| Error::new("missing_metadata", "qwen3vl config: no text_config"))?;
        let text = TextConfig::from_json(tv, tie)?;
        let vision = match (root.get("vision"), root.get("vision_config")) {
            (Some(v), _) => VisionConfig::from_json(v)?,
            (None, Some(v)) => vision_from_hf(v)?,
            _ => bail!("missing_metadata", "qwen3vl config: no vision config"),
        };
        let processor = match (root.get("processor"), preprocessor) {
            (Some(p), _) => ImageProcessor::from_json(p)?,
            (None, Some(p)) => ImageProcessor::from_hf(p),
            _ => ImageProcessor::default(),
        };
        if vision.out_hidden_size != text.hidden_size || vision.deepstack_layers.len() > text.n_layers {
            bail!("metadata", "qwen3vl: vision output {} / {} deepstack features for text {} / {} layers", vision.out_hidden_size, vision.deepstack_layers.len(), text.hidden_size, text.n_layers);
        }
        let id = |k: &str, d: u32| root.get(k).and_then(Value::as_u64).map_or(d, |x| x as u32);
        let mut eos = id_list(tv.get("eos_token_id"));
        for e in id_list(root.get("eos_token_id")) {
            if !eos.contains(&e) {
                eos.push(e);
            }
        }
        Ok(Qwen3VlConfig {
            text,
            vision,
            processor,
            image_token_id: id("image_token_id", 151_655),
            vision_start_token_id: id("vision_start_token_id", 151_652),
            vision_end_token_id: id("vision_end_token_id", 151_653),
            eos,
        })
    }

    /// The `"qwen3vl"` metadata object.
    pub fn to_json(&self) -> Value {
        json!({"model_type": "qwen3_vl", "text_config": self.text.to_json(), "vision": self.vision.to_json(), "processor": self.processor.to_json(),
            "image_token_id": self.image_token_id, "vision_start_token_id": self.vision_start_token_id,
            "vision_end_token_id": self.vision_end_token_id, "eos_token_id": self.eos, "tie_word_embeddings": self.text.tie_embeddings})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q2b_config() {
        let q2b = json!({"image_token_id": 151655, "model_type": "qwen3_vl", "tie_word_embeddings": true,
            "text_config": {"eos_token_id": 151645, "head_dim": 128, "hidden_size": 2048, "intermediate_size": 6144, "num_attention_heads": 16,
                "num_hidden_layers": 28, "num_key_value_heads": 8, "rope_scaling": {"mrope_interleaved": true, "mrope_section": [24, 20, 20]},
                "rope_theta": 5000000, "vocab_size": 151936},
            "vision_config": {"deepstack_visual_indexes": [5, 11, 17], "depth": 24, "hidden_size": 1024, "intermediate_size": 4096,
                "num_heads": 16, "num_position_embeddings": 2304, "out_hidden_size": 2048, "patch_size": 16}});
        let c = Qwen3VlConfig::from_json(&q2b, None).expect("config");
        assert_eq!((c.text.rope_theta, c.text.tie_embeddings, c.vision.position_grid, c.eos.clone()), (5_000_000.0, true, 48, vec![151645]));
        // interleaved: 0 t, 1 h, 2 w, ..., 58 h (< 60), 59 w, 60.. t
        assert_eq!([0, 1, 2, 58, 59, 61].map(|i| c.text.mrope_stream(i)), [0, 1, 2, 1, 2, 0]);
        assert_eq!(Qwen3VlConfig::from_json(&c.to_json(), None).expect("again"), c);
    }
}
