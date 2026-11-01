//! Qwen3-VL hyper-parameters: the Qwen3 text decoder (`text_config`), the vision encoder
//! (`vision::VisionConfig`), the image processor and the special token ids. Parsed from an
//! HF `config.json` (`vision_config` in HF keys) or from the `.lapid` `"qwen3vl"` metadata
//! object (`text_config` in HF keys, `vision` / `processor` in lapid's generic form).

use lapid::{Error, Result, bail};
use lapid_nn::llm::rope::MRope;
use serde_json::{Value, json};

use crate::llama::config::LlamaConfig;
use crate::runtime::chat::id_list;
use crate::vision::encoder::{NormKind, VisionConfig};
use crate::vision::processor::ImageProcessor;

/// The text decoder: a Qwen3 decoder (`LlamaConfig`, model type `qwen3`) with M-RoPE.
#[derive(Clone, Debug, PartialEq)]
pub struct TextConfig {
    pub base: LlamaConfig,
    pub mrope: MRope,
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
        let rope = v.get("rope_parameters").or_else(|| v.get("rope_scaling")).cloned().unwrap_or(Value::Null);
        let sec: Vec<usize> = rope.get("mrope_section").and_then(Value::as_array).map_or_else(|| vec![24, 20, 20], |a| a.iter().filter_map(|x| x.as_u64().map(|y| y as usize)).collect());
        let [t, h, w, ..] = sec[..] else { bail!("metadata", "qwen3vl: mrope_section {sec:?} needs 3 entries") };
        let mut b = v.clone();
        let o = b.as_object_mut().ok_or_else(|| Error::new("metadata", "qwen3vl: text_config is not an object"))?;
        o.remove("rope_parameters");
        o.remove("rope_scaling");
        let theta = rope.get("rope_theta").or_else(|| v.get("rope_theta")).and_then(Value::as_f64).unwrap_or(5_000_000.0);
        o.insert("model_type".into(), json!("qwen3"));
        o.insert("rope_theta".into(), json!(theta));
        o.entry("tie_word_embeddings").or_insert(json!(tie_default));
        o.entry("max_position_embeddings").or_insert(json!(262_144));
        let c = TextConfig {
            base: LlamaConfig::from_json(&b)?,
            mrope: MRope { section: [t, h, w], interleaved: rope.get("mrope_interleaved").and_then(Value::as_bool).unwrap_or(true) },
        };
        if c.mrope.section.iter().sum::<usize>() * 2 != c.base.head_dim {
            bail!("metadata", "qwen3vl: mrope_section {:?} does not cover head dim {}", c.mrope.section, c.base.head_dim);
        }
        Ok(c)
    }

    pub fn to_json(&self) -> Value {
        let mut v = self.base.to_json();
        v["rope_parameters"] = json!({"rope_type": "default", "rope_theta": self.base.rope_theta, "mrope_section": self.mrope.section, "mrope_interleaved": self.mrope.interleaved});
        v
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
        if vision.out_hidden_size != text.base.hidden_size || vision.deepstack_layers.len() > text.base.n_layers {
            bail!("metadata", "qwen3vl: vision output {} / {} deepstack features for text {} / {} layers", vision.out_hidden_size, vision.deepstack_layers.len(), text.base.hidden_size, text.base.n_layers);
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
            "vision_end_token_id": self.vision_end_token_id, "eos_token_id": self.eos, "tie_word_embeddings": self.text.base.tie_embeddings})
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
        assert_eq!((c.text.base.rope_theta, c.text.base.tie_embeddings, c.vision.position_grid, c.eos.clone()), (5_000_000.0, true, 48, vec![151645]));
        assert!(c.text.base.qk_norm && !c.text.base.qkv_bias && c.text.mrope.interleaved);
        assert_eq!(Qwen3VlConfig::from_json(&c.to_json(), None).expect("again"), c);
    }
}
