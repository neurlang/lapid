//! llama.cpp `mmproj` GGUF (`clip.vision.projector_type` "gemma4v") → the Gemma 4 vision
//! tower under Hugging Face names, written into the open `.lapid` writer of the gemma4moe
//! GGUF converter (which pairs the model GGUF with its mmproj).
//!
//! Name map (llama.cpp `conversion/gemma.py` `Gemma4VisionAudioModel`, reversed):
//! `v.patch_embd.weight` [D, 3, p, p] → `patch_embedder.input_proj.weight` [D, p*p*3]
//! (columns back to the HF (row, col, channel) order), `v.position_embd.weight` →
//! `patch_embedder.position_embedding_table` [2, P, D], `v.blk.N.{ln1, attn_post_norm, ln2,
//! ffn_post_norm}` → `encoder.layers.N.{input, post_attention, pre_feedforward,
//! post_feedforward}_layernorm`, `attn_{q,k,v,out}` / `ffn_{gate,up,down}` (+ the clamp scalars
//! `input_min` ...) → `self_attn.{q,k,v,o}_proj` / `mlp.{gate,up,down}_proj` (`.linear.weight`),
//! `attn_{q,k}_norm` → `self_attn.{q,k}_norm`, `v.std_{bias,scale}` → `std_{bias,scale}`,
//! `mm.input_projection.weight` → `model.embed_vision.embedding_projection.weight`.
//! Audio tensors (`a.*`, `mm.a.*`) are skipped.

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::gguf::GgufFile;
use lapid_convert::writer::ModelWriter;
use serde_json::{Value, json};

use super::config::{ImageProcessorConfig, VisionConfig};
use super::vision::{EMBED_PROJ, TOWER};
use crate::gemma4::convert::write_gguf_native;

/// HF name of an mmproj vision tensor (`None` = not part of the vision tower).
pub(crate) fn hf_name(gguf: &str) -> Option<String> {
    match gguf {
        "v.patch_embd.weight" => return Some(format!("{TOWER}.patch_embedder.input_proj.weight")),
        "v.position_embd.weight" => return Some(format!("{TOWER}.patch_embedder.position_embedding_table")),
        "v.std_bias" => return Some(format!("{TOWER}.std_bias")),
        "v.std_scale" => return Some(format!("{TOWER}.std_scale")),
        "mm.input_projection.weight" => return Some(format!("{EMBED_PROJ}.weight")),
        _ => {}
    }
    let rest = gguf.strip_prefix("v.blk.")?;
    let (layer, rest) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let (module, suffix) = rest.split_once('.')?;
    let p = format!("{TOWER}.encoder.layers.{layer}");
    let norm = |n: &str| (suffix == "weight").then(|| format!("{p}.{n}.weight"));
    let clip = |n: &str| match suffix {
        "weight" => Some(format!("{p}.{n}.linear.weight")),
        "input_min" | "input_max" | "output_min" | "output_max" => Some(format!("{p}.{n}.{suffix}")),
        _ => None,
    };
    match module {
        "ln1" => norm("input_layernorm"),
        "attn_post_norm" => norm("post_attention_layernorm"),
        "ln2" => norm("pre_feedforward_layernorm"),
        "ffn_post_norm" => norm("post_feedforward_layernorm"),
        "attn_q_norm" => norm("self_attn.q_norm"),
        "attn_k_norm" => norm("self_attn.k_norm"),
        "attn_q" => clip("self_attn.q_proj"),
        "attn_k" => clip("self_attn.k_proj"),
        "attn_v" => clip("self_attn.v_proj"),
        "attn_out" => clip("self_attn.o_proj"),
        "ffn_gate" => clip("mlp.gate_proj"),
        "ffn_up" => clip("mlp.up_proj"),
        "ffn_down" => clip("mlp.down_proj"),
        _ => None,
    }
}

fn get_u(f: &GgufFile, key: &str) -> Result<usize> {
    f.get(key)
        .and_then(|v| v.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| Error::new("missing_metadata", format!("mmproj: missing {key:?}")))
}

/// `[D, 3, p, p]` (ggml conv kernel, numpy order) → HF `[D, p * p * 3]` (row, col, channel).
fn patch_kernel_to_hf(data: &[f32], d: usize, p: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for o in 0..d {
        for c in 0..3 {
            for y in 0..p {
                for x in 0..p {
                    out[o * 3 * p * p + (y * p + x) * 3 + c] = data[((o * 3 + c) * p + y) * p + x];
                }
            }
        }
    }
    out
}

/// Add the vision tower of a llama.cpp `mmproj` GGUF (`clip.projector_type` "gemma4v")
/// to `w` under Hugging Face names (`model.vision_tower.*`, `model.embed_vision.*`).
/// Returns `{"vision_config": {HF keys}, "image_processor": {HF processor keys}}` for the
/// `"gemma4"` metadata object. Called by the gemma4moe GGUF converter.
pub(crate) fn add_mmproj(mmproj: &GgufFile, w: &mut ModelWriter, dtype: OutDType) -> Result<Value> {
    let kind = mmproj
        .get("clip.vision.projector_type")
        .or_else(|| mmproj.get("clip.projector_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if kind != "gemma4v" {
        bail!("unsupported_model", "mmproj: projector type {kind:?} is not Gemma 4 vision (\"gemma4v\")");
    }
    let d = get_u(mmproj, "clip.vision.embedding_length")?;
    let n_head = get_u(mmproj, "clip.vision.attention.head_count")?;
    let patch = get_u(mmproj, "clip.vision.patch_size")?;
    let pos = mmproj
        .tensor("v.position_embd.weight")
        .ok_or_else(|| Error::new("missing_tensor", "mmproj: no v.position_embd.weight"))?;
    let head_dim = mmproj.tensor("v.blk.0.attn_q_norm.weight").map_or(d / n_head.max(1), |t| t.numel());
    let pooling = mmproj.get("clip.vision.projector.scale_factor").and_then(|v| v.as_u64()).map_or(3, |x| x as usize);
    let vc = VisionConfig {
        hidden_size: d,
        intermediate_size: get_u(mmproj, "clip.vision.feed_forward_length")?,
        n_layers: get_u(mmproj, "clip.vision.block_count")?,
        n_head,
        n_kv_head: mmproj.get("clip.vision.attention.head_count_kv").and_then(|v| v.as_u64()).map_or(n_head, |x| x as usize),
        head_dim,
        eps: mmproj.get("clip.vision.attention.layer_norm_epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32,
        patch_size: patch,
        pooling_kernel_size: pooling,
        position_embedding_size: pos.numel() / (2 * d).max(1),
        rope_theta: 100.0,
        standardize: mmproj.tensor("v.std_bias").is_some(),
        use_clipped_linears: mmproj.tensor("v.blk.0.attn_q.input_max").is_some(),
    };
    let names: Vec<String> = mmproj.tensors.iter().map(|t| t.name.clone()).collect();
    let mut n = 0;
    for gname in &names {
        let Some(name) = hf_name(gname) else { continue };
        let t = mmproj.tensor(gname).ok_or_else(|| Error::new("internal", format!("mmproj: lost tensor {gname}")))?;
        if gname == "v.patch_embd.weight" {
            let data = patch_kernel_to_hf(&mmproj.to_f32(t)?, d, patch);
            w.add_f32(&name, &[d, 3 * patch * patch], &data)?;
        } else if gname == "v.position_embd.weight" {
            w.add_f32(&name, &[2, vc.position_embedding_size, d], &mmproj.to_f32(t)?)?;
        } else if t.numel() == 1 {
            w.add_f32(&name, &[1], &mmproj.to_f32(t)?)?;
        } else {
            write_gguf_native(mmproj, w, &name, t, None, dtype)?;
        }
        n += 1;
    }
    if n == 0 {
        bail!("missing_tensor", "mmproj: no vision tensors");
    }
    let pc = ImageProcessorConfig { patch_size: patch, pooling_kernel_size: pooling, ..Default::default() };
    Ok(json!({"vision_config": vc.to_json(), "image_processor": pc.to_json()}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(hf_name("v.blk.3.attn_q.weight").as_deref(), Some("model.vision_tower.encoder.layers.3.self_attn.q_proj.linear.weight"));
        assert_eq!(hf_name("v.blk.0.ffn_down.output_min").as_deref(), Some("model.vision_tower.encoder.layers.0.mlp.down_proj.output_min"));
        assert_eq!(hf_name("v.blk.15.ln2.weight").as_deref(), Some("model.vision_tower.encoder.layers.15.pre_feedforward_layernorm.weight"));
        assert_eq!(hf_name("mm.input_projection.weight").as_deref(), Some("model.embed_vision.embedding_projection.weight"));
        assert_eq!(hf_name("mm.a.input_projection.weight"), None);
        assert_eq!(hf_name("a.blk.0.ln1.weight"), None);
    }

    #[test]
    fn patch_kernel_order() {
        // D=1, p=2: numpy [1, 3, 2, 2] value = c*100 + y*10 + x.
        let data: Vec<f32> = (0..12).map(|i| ((i / 4) * 100 + ((i / 2) % 2) * 10 + i % 2) as f32).collect();
        let hf = patch_kernel_to_hf(&data, 1, 2);
        // HF index (y*2 + x)*3 + c.
        assert_eq!(hf[(2 + 1) * 3 + 2], 211.0);
        assert_eq!(hf[3 + 1], 101.0);
    }
}
