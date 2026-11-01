//! Gemma 4 hyper-parameters for the dense *and* MoE text decoders (`gemma4moe`), plus the
//! multimodal extras the vision path needs. The text part is parsed by
//! [`Gemma4Config::from_json`] (on a copy without `enable_moe_block`, which that parser
//! rejects); the MoE keys (`num_experts`, `top_k_experts`, `moe_intermediate_size`),
//! `use_bidirectional_attention`, `vision_config`, `image_processor` and the image token ids
//! are read here. One parser reads an HF `config.json` (with `text_config`) and the `.lapid`
//! `"gemma4"` metadata object (flat HF text keys + the extras).

use lapid::{Error, Result, bail};
use serde_json::{Value, json};

use crate::gemma4::config::Gemma4Config;

/// Sparse expert block of every decoder layer (HF `Gemma4TextRouter` + `Gemma4TextExperts`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MoeParams {
    pub num_experts: usize,
    pub top_k: usize,
    pub moe_intermediate_size: usize,
}

/// Image token ids of a multimodal checkpoint.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ImageTokens {
    /// The soft-token placeholder (`<|image|>`), repeated once per image embedding.
    pub image: Option<u32>,
    /// Begin / end of image (`<|image>` / `<image|>`).
    pub boi: Option<u32>,
    pub eoi: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4MoeConfig {
    pub text: Gemma4Config,
    /// `None` = dense Gemma 4 (E2B, E4B, 31B, ...).
    pub moe: Option<MoeParams>,
    /// `use_bidirectional_attention == "vision"`: image tokens attend to each other in
    /// both directions on sliding layers.
    pub bidirectional_vision: bool,
    /// HF `vision_config` (when the file carries a vision tower).
    pub vision: Option<Value>,
    /// HF processor `image_processor` object.
    pub image_processor: Option<Value>,
    pub image_tokens: ImageTokens,
}

fn opt_u32(v: Option<&Value>) -> Option<u32> {
    v.and_then(Value::as_u64).and_then(|x| u32::try_from(x).ok())
}

fn need(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| Error::new("missing_metadata", format!("gemma4moe config: missing integer {key:?}")))
}

impl Gemma4MoeConfig {
    /// From an HF `config.json` (`gemma4` with `text_config`, or `gemma4_text`) or the `.lapid`
    /// `"gemma4"` metadata object.
    pub fn from_json(root: &Value) -> Result<Gemma4MoeConfig> {
        let tc = root.get("text_config").unwrap_or(root);
        let moe = if tc.get("enable_moe_block").and_then(Value::as_bool) == Some(true) {
            let p = MoeParams {
                num_experts: need(tc, "num_experts")?,
                top_k: need(tc, "top_k_experts")?,
                moe_intermediate_size: need(tc, "moe_intermediate_size")?,
            };
            if p.top_k == 0 || p.top_k > p.num_experts {
                bail!("metadata", "gemma4moe: top_k {} of {} experts", p.top_k, p.num_experts);
            }
            Some(p)
        } else {
            None
        };
        // The dense parser rejects MoE configs; hand it a copy without the flag.
        let mut stripped = root.clone();
        let target = match stripped.get_mut("text_config") {
            Some(t) => t,
            None => &mut stripped,
        };
        if let Some(o) = target.as_object_mut() {
            o.remove("enable_moe_block");
        }
        let text = Gemma4Config::from_json(&stripped)?;
        let bidirectional_vision = tc.get("use_bidirectional_attention").and_then(Value::as_str) == Some("vision");
        let get = |k: &str| root.get(k).or_else(|| tc.get(k));
        Ok(Gemma4MoeConfig {
            text,
            moe,
            bidirectional_vision,
            vision: get("vision_config").filter(|v| v.is_object()).cloned(),
            image_processor: get("image_processor").filter(|v| v.is_object()).cloned(),
            image_tokens: ImageTokens {
                image: opt_u32(get("image_token_id")),
                boi: opt_u32(get("boi_token_id")),
                eoi: opt_u32(get("eoi_token_id")),
            },
        })
    }

    /// The `"gemma4"` metadata object: HF text_config keys (MoE keys included) + extras.
    pub fn to_json(&self) -> Value {
        let mut g = self.text.to_json();
        match self.moe {
            Some(m) => {
                g["enable_moe_block"] = json!(true);
                g["num_experts"] = json!(m.num_experts);
                g["top_k_experts"] = json!(m.top_k);
                g["moe_intermediate_size"] = json!(m.moe_intermediate_size);
            }
            None => g["enable_moe_block"] = json!(false),
        }
        if self.bidirectional_vision {
            g["use_bidirectional_attention"] = json!("vision");
        }
        if let Some(v) = &self.vision {
            g["vision_config"] = v.clone();
        }
        if let Some(p) = &self.image_processor {
            g["image_processor"] = p.clone();
        }
        for (k, v) in [("image_token_id", self.image_tokens.image), ("boi_token_id", self.image_tokens.boi), ("eoi_token_id", self.image_tokens.eoi)] {
            if let Some(id) = v {
                g[k] = json!(id);
            }
        }
        g
    }

    /// Pad token id (image positions use it for the per-layer-embedding lookup).
    pub fn pad_id(&self) -> u32 {
        self.text.special.pad.unwrap_or(0)
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use serde_json::{Value, json};

    /// The google/gemma-4-26B-A4B-it config.json (text part + multimodal ids).
    pub(crate) fn a4b() -> Value {
        let types: Vec<&str> = (0..30).map(|i| if (i + 1) % 6 == 0 { "full_attention" } else { "sliding_attention" }).collect();
        json!({"model_type": "gemma4", "eos_token_id": [1, 106], "image_token_id": 258880, "boi_token_id": 255999,
            "eoi_token_id": 258882, "tie_word_embeddings": true,
            "vision_config": {"hidden_size": 1152, "num_hidden_layers": 27},
            "text_config": {
            "attention_k_eq_v": true, "bos_token_id": 2, "enable_moe_block": true, "eos_token_id": 1,
            "final_logit_softcapping": 30.0, "global_head_dim": 512, "head_dim": 256, "hidden_size": 2816,
            "hidden_size_per_layer_input": 0, "intermediate_size": 2112, "layer_types": types,
            "max_position_embeddings": 262144, "model_type": "gemma4_text", "moe_intermediate_size": 704,
            "num_attention_heads": 16, "num_experts": 128, "num_global_key_value_heads": 2, "num_hidden_layers": 30,
            "num_key_value_heads": 8, "num_kv_shared_layers": 0, "pad_token_id": 0, "rms_norm_eps": 1e-6,
            "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
            "sliding_window": 1024, "tie_word_embeddings": true, "top_k_experts": 8,
            "use_bidirectional_attention": "vision", "use_double_wide_mlp": false, "vocab_size": 262144,
            "vocab_size_per_layer_input": 262144}})
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::a4b;
    use super::*;

    #[test]
    fn a4b_config() {
        let c = Gemma4MoeConfig::from_json(&a4b()).expect("config");
        assert_eq!(c.moe, Some(MoeParams { num_experts: 128, top_k: 8, moe_intermediate_size: 704 }));
        assert!(c.bidirectional_vision);
        assert!(c.text.k_eq_v(5));
        assert!(!c.text.k_eq_v(4));
        assert_eq!(c.text.ple_dim, 0);
        assert_eq!(c.text.layer_kv_heads(5), 2);
        assert_eq!(c.image_tokens.image, Some(258880));
        assert!(c.vision.is_some());
        let again = Gemma4MoeConfig::from_json(&c.to_json()).expect("again");
        assert_eq!(again, c);
    }

    #[test]
    fn dense_e2b() {
        let c = Gemma4MoeConfig::from_json(&crate::gemma4::config::test_support::e2b()).expect("e2b");
        assert_eq!(c.moe, None);
        assert!(!c.bidirectional_vision);
        assert_eq!(c.text.ple_dim, 256);
    }
}
