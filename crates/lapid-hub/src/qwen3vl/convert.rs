//! Qwen3-VL checkpoints -> `.lapid`: HF safetensors (`Qwen3VLForConditionalGeneration`) and a
//! GGUF pair (llama.cpp `qwen3vl` text model + its `clip` mmproj, projector `qwen3vl`).
//! Text tensors use HF names without the `language_model` level; vision tensors use the
//! mmproj names of `vision::encoder` (`v.blk.{i}.attn_qkv.weight`, `mm.0.weight`,
//! `v.deepstack.{layer}.fc1.weight`, ...), so an mmproj is copied as is. The Conv3d patch
//! kernel is summed over its temporal axis (a still image repeats its frame) to [D, C*P*P].
//! Precision: text Linear weights quantized, the tied embedding at most int8, vision weights
//! f16 (never quantized: ViT activation outliers), norms / biases f32.

use std::path::{Path, PathBuf};

use lapid::{Error, Result, bail};
use lapid_convert::assets::HfFiles;
use lapid_convert::engine::{ConvertHooks, ConvertSpec, Emit, HfExtras, SourceKind, TensorSource, sibling_mmproj};
use lapid_convert::gguf::{GgufFile, GgufValue};
use lapid_convert::mapping::{RoleRule, Rule};
use lapid_convert::policy::Role;
use lapid_convert::writer::read_json;
use serde_json::{Value, json};

use super::config::Qwen3VlConfig;
use super::model::{EMBED, LM_HEAD};
use crate::vision::encoder::{NormKind, VisionConfig};
use crate::vision::processor::ImageProcessor;

macro_rules! vis {
    ($h:literal, $g:literal) => {
        Rule::new(concat!("model.visual.", $h, ".{w}"), concat!($g, ".{w}"))
    };
}
const PATCH: &str = "v.patch_embd.weight";

pub static CONVERT: ConvertSpec = ConvertSpec {
    architecture: "qwen3vl",
    license: "apache-2.0",
    hf_model_types: &["qwen3_vl"],
    gguf_architectures: &["qwen3vl"],
    hf_names: &[
        Rule::new("model.language_model.{a}.{b}.{c}.{d}.{e}", "model.{a}.{b}.{c}.{d}.{e}"),
        Rule::new("model.language_model.{a}.{b}.{c}.{d}", "model.{a}.{b}.{c}.{d}"),
        Rule::new("model.language_model.{a}.{b}", "model.{a}.{b}"),
        vis!("patch_embed.proj", "v.patch_embd"),
        vis!("pos_embed", "v.position_embd"),
        vis!("merger.norm", "v.post_ln"),
        vis!("merger.linear_fc1", "mm.0"),
        vis!("merger.linear_fc2", "mm.2"),
        vis!("blocks.{i}.norm1", "v.blk.{i}.ln1"),
        vis!("blocks.{i}.norm2", "v.blk.{i}.ln2"),
        vis!("blocks.{i}.attn.qkv", "v.blk.{i}.attn_qkv"),
        vis!("blocks.{i}.attn.proj", "v.blk.{i}.attn_out"),
        vis!("blocks.{i}.mlp.linear_fc1", "v.blk.{i}.ffn_up"),
        vis!("blocks.{i}.mlp.linear_fc2", "v.blk.{i}.ffn_down"),
        // By merger index; `transform` names them by encoder layer.
        vis!("deepstack_merger_list.{j}.norm", "v.deepstack_list.{j}.norm"),
        vis!("deepstack_merger_list.{j}.linear_fc1", "v.deepstack_list.{j}.fc1"),
        vis!("deepstack_merger_list.{j}.linear_fc2", "v.deepstack_list.{j}.fc2"),
    ],
    gguf_names: crate::llama::convert::GGUF_NAMES,
    roles: &[RoleRule::new("v.*", Role::Dense), RoleRule::new("mm.*", Role::Dense), RoleRule::new(EMBED, Role::LinearInt8)],
    text_assets: true,
    hooks: &Hooks,
};

/// mmproj tensors keep their names.
static MMPROJ: &[Rule] = &[
    Rule::new("v.{a}.{b}", "v.{a}.{b}"),
    Rule::new("v.{a}.{b}.{c}", "v.{a}.{b}.{c}"),
    Rule::new("v.{a}.{b}.{c}.{d}", "v.{a}.{b}.{c}.{d}"),
    Rule::new("mm.{a}.{b}", "mm.{a}.{b}"),
];

struct Hooks;

impl ConvertHooks for Hooks {
    fn hf_config(&self, c: &Value) -> Result<Value> {
        Ok(Qwen3VlConfig::from_json(c, None)?.to_json())
    }

    /// The image processor of `preprocessor_config.json`.
    fn hf_extras(&self, files: &HfFiles, cfg: &mut Value) -> Result<HfExtras> {
        if let Some(p) = files.optional("preprocessor_config.json")? {
            cfg["processor"] = ImageProcessor::from_hf(&read_json(&p)?).to_json();
        }
        Ok(HfExtras::default())
    }

    /// Text part (the vision part comes from the mmproj, see `companion_config`).
    fn gguf_config(&self, f: &GgufFile) -> Result<Value> {
        let mut text = crate::llama::convert::gguf_config(f)?;
        let sec: Vec<u64> = f.get("qwen3vl.rope.dimension_sections").and_then(GgufValue::as_array).map(|a| a.iter().filter_map(GgufValue::as_u64).take(3).collect()).unwrap_or_else(|| vec![24, 20, 20]);
        text["rope_parameters"] = json!({"rope_type": "default", "rope_theta": text["rope_theta"], "mrope_section": sec, "mrope_interleaved": true});
        text["max_position_embeddings"] = json!(f.get("qwen3vl.context_length").and_then(GgufValue::as_u64).unwrap_or(262_144));
        let tokens = f.get("tokenizer.ggml.tokens").and_then(GgufValue::as_array);
        let id = |s: &str| tokens.and_then(|t| t.iter().position(|v| v.as_str() == Some(s))).ok_or_else(|| Error::new("missing_metadata", format!("GGUF vocabulary has no {s}")));
        Ok(json!({
            "model_type": "qwen3_vl", "tie_word_embeddings": text["tie_word_embeddings"], "text_config": text,
            "image_token_id": id("<|image_pad|>")?, "vision_start_token_id": id("<|vision_start|>")?, "vision_end_token_id": id("<|vision_end|>")?,
            "eos_token_id": f.get("tokenizer.ggml.eos_token_id").and_then(GgufValue::as_u64).into_iter().collect::<Vec<_>>(),
        }))
    }

    /// The mmproj is required: `$LAPID_MMPROJ` or a `mmproj*.gguf` next to the model.
    fn gguf_companion(&self, model: &Path) -> Result<Option<PathBuf>> {
        let p = sibling_mmproj(model).ok_or_else(|| Error::new("not_found", format!("{}: no mmproj*.gguf next to the model (set LAPID_MMPROJ)", model.display())))?;
        Ok(Some(p))
    }

    fn companion_names(&self) -> &'static [Rule] {
        MMPROJ
    }

    fn companion_config(&self, mm: &GgufFile, _model: &GgufFile, cfg: &mut Value) -> Result<()> {
        let (vision, processor) = vision_config(mm)?;
        cfg["vision"] = vision.to_json();
        cfg["processor"] = processor.to_json();
        *cfg = Qwen3VlConfig::from_json(cfg, None)?.to_json();
        Ok(())
    }

    fn expected(&self, c: &Value) -> Result<Vec<String>> {
        let cfg = Qwen3VlConfig::from_json(c, None)?;
        let mut v = crate::llama::convert::expected(&cfg.text.base);
        v.extend(cfg.vision.expected_tensors(false, false));
        Ok(v)
    }

    fn transform(&self, name: &str, src: &dyn TensorSource, c: &Value) -> Result<Emit> {
        if name == "v.patch_embd.weight.1" {
            return Ok(Emit::Drop);
        }
        if name == LM_HEAD && c.get("tie_word_embeddings").and_then(Value::as_bool) == Some(true) {
            return Ok(Emit::Drop);
        }
        // Conv3d [D, C, T, P, P] (HF) or two temporal halves [D, C, P, P] (mmproj) -> [D, C*P*P].
        if name == PATCH {
            let shape = src.shape(name).unwrap_or_default();
            let mut data = src.f32(name)?;
            let (d, c) = (shape[0], shape[1]);
            if src.kind() == SourceKind::Hf {
                let &[_, _, t, p, q] = shape.as_slice() else { bail!("shape_mismatch", "patch_embed.proj.weight: {shape:?}") };
                let pq = p * q;
                data = (0..d * c).flat_map(|oc| (0..pq).map(move |k| (oc, k))).map(|(oc, k)| (0..t).map(|ti| data[(oc * t + ti) * pq + k]).sum()).collect();
            } else if src.names().iter().any(|n| n == "v.patch_embd.weight.1") {
                data.iter_mut().zip(src.f32("v.patch_embd.weight.1")?).for_each(|(a, b)| *a += b);
            }
            return Ok(Emit::Replace(vec![(name.to_string(), vec![d, data.len() / d.max(1)], data)]));
        }
        if let Some(rest) = name.strip_prefix("v.deepstack_list.") {
            let (j, part) = rest.split_once('.').unwrap_or_default();
            let layers = Qwen3VlConfig::from_json(c, None)?.vision.deepstack_layers;
            let layer = j.parse::<usize>().ok().and_then(|j| layers.get(j).copied()).ok_or_else(|| Error::new("format", format!("qwen3vl: no deepstack layer for {name}")))?;
            return Ok(Emit::Replace(vec![(format!("v.deepstack.{layer}.{part}"), src.shape(name).unwrap_or_default(), src.f32(name)?)]));
        }
        Ok(Emit::Keep)
    }
}

/// Encoder + preprocessing of a llama.cpp `clip` mmproj with projector `qwen3vl`.
fn vision_config(f: &GgufFile) -> Result<(VisionConfig, ImageProcessor)> {
    let proj = f.get("clip.projector_type").or_else(|| f.get("clip.vision.projector_type")).and_then(GgufValue::as_str).unwrap_or("");
    if !matches!(proj, "qwen3vl" | "qwen3vl_merger") {
        bail!("unsupported_model", "mmproj projector type {proj:?} is not Qwen3-VL");
    }
    let uint = |k: &str| f.get(k).and_then(GgufValue::as_u64).map(|v| v as usize).ok_or_else(|| Error::new("missing_metadata", format!("mmproj: missing {k:?}")));
    let truthy = |v: &GgufValue| matches!(v, GgufValue::Bool(true)) || v.as_u64().is_some_and(|x| x != 0);
    let cfg = VisionConfig {
        hidden_size: uint("clip.vision.embedding_length")?,
        num_heads: uint("clip.vision.attention.head_count")?,
        depth: uint("clip.vision.block_count")?,
        intermediate_size: uint("clip.vision.feed_forward_length")?,
        patch_size: uint("clip.vision.patch_size")?,
        in_channels: 3,
        norm: if f.tensor("v.blk.0.ln1.bias").is_some() { NormKind::LayerNorm } else { NormKind::RmsNorm },
        eps: f.get("clip.vision.attention.layer_norm_epsilon").and_then(GgufValue::as_f64).unwrap_or(1e-6) as f32,
        hidden_act: "gelu_pytorch_tanh".into(),
        position_grid: f.tensor("v.position_embd.weight").map_or(0, |t| (t.shape[0] as f64).sqrt().round() as usize),
        rope_theta: Some(10_000.0),
        spatial_merge_size: f.get("clip.vision.spatial_merge_size").and_then(GgufValue::as_u64).unwrap_or(2) as usize,
        out_hidden_size: match f.tensor("mm.2.weight") {
            Some(t) => t.shape[0],
            None => uint("clip.vision.projection_dim")?,
        },
        deepstack_layers: f.get("clip.vision.is_deepstack_layers").and_then(GgufValue::as_array).map(|a| a.iter().enumerate().filter(|(_, v)| truthy(v)).map(|(i, _)| i).collect()).unwrap_or_default(),
    };
    cfg.validate()?;
    let d = ImageProcessor::default();
    let opt = |k: &str| f.get(k).and_then(GgufValue::as_u64).map(|v| v as usize);
    let f3 = |k: &str, dv: [f32; 3]| match f.get(k).and_then(GgufValue::as_array) {
        Some(a) if a.len() == 3 => [0, 1, 2].map(|i| a[i].as_f64().map_or(dv[i], |x| x as f32)),
        _ => dv,
    };
    let processor = ImageProcessor {
        patch_size: cfg.patch_size,
        merge_size: cfg.spatial_merge_size,
        min_pixels: opt("clip.vision.image_min_pixels").unwrap_or(d.min_pixels),
        max_pixels: opt("clip.vision.image_max_pixels").unwrap_or(d.max_pixels),
        mean: f3("clip.vision.image_mean", d.mean),
        std: f3("clip.vision.image_std", d.std),
        ..d
    };
    Ok((cfg, processor))
}

#[cfg(test)]
mod tests {
    use lapid_convert::mapping::{Renamed, rename};

    use super::*;

    #[test]
    fn names_and_roles() {
        let r = |rules, n| match rename(rules, n) {
            Renamed::To(s) => s,
            other => format!("{other:?}"),
        };
        assert_eq!(r(CONVERT.hf_names, "model.language_model.layers.3.mlp.up_proj.weight"), "model.layers.3.mlp.up_proj.weight");
        assert_eq!(r(CONVERT.hf_names, "model.visual.blocks.2.attn.qkv.bias"), "v.blk.2.attn_qkv.bias");
        assert_eq!(r(CONVERT.hf_names, "model.visual.merger.linear_fc2.weight"), "mm.2.weight");
        assert_eq!(r(CONVERT.gguf_names, "blk.0.ffn_norm.weight"), "model.layers.0.post_attention_layernorm.weight");
        assert_eq!(r(MMPROJ, "v.patch_embd.weight.1"), "v.patch_embd.weight.1");
        assert_eq!(CONVERT.role("v.blk.0.attn_qkv.weight", &[3072, 1024]), Role::Dense);
        assert_eq!(CONVERT.role("v.blk.0.ln1.weight", &[1024]), Role::Param);
        assert_eq!(CONVERT.role(EMBED, &[151936, 2048]), Role::LinearInt8);
        assert_eq!(CONVERT.role("model.layers.0.self_attn.q_proj.weight", &[2048, 2048]), Role::Linear);
    }
}
