//! llama.cpp `mmproj` (`general.architecture = "clip"`) GGUF files: hyper-parameters
//! (`clip.vision.*`) → [`VisionConfig`] + [`ImageProcessor`], and the vision tensors copied
//! into a `.lapid` writer under the same `v.*` / `mm.*` names (see [`super::encoder`]).
//!
//! The only tensor rewrite: a Conv3d patch embedding split by llama.cpp into two temporal
//! halves (`v.patch_embd.weight`, `v.patch_embd.weight.1`, each [D, C, P, P]) is summed
//! (identical frames for a still image) and flattened to [D, C*P*P].

use lapid::quant::Format;
use lapid::{DType, Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::ggml::GgmlType;
use lapid_convert::gguf::{GgufFile, GgufTensor, GgufValue};
use lapid_convert::writer::ModelWriter;

use super::encoder::{NormKind, VisionConfig};
use super::processor::ImageProcessor;

pub fn is_mmproj(f: &GgufFile) -> bool {
    f.architecture() == Some("clip") && f.get("clip.has_vision_encoder").is_none_or(|v| matches!(v, GgufValue::Bool(true)))
}

/// `clip.projector_type` (or `clip.vision.projector_type`).
pub fn projector_type(f: &GgufFile) -> Option<&str> {
    f.get("clip.projector_type").or_else(|| f.get("clip.vision.projector_type")).and_then(GgufValue::as_str)
}

fn uint(f: &GgufFile, k: &str) -> Result<usize> {
    f.get(k)
        .and_then(GgufValue::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| Error::new("missing_metadata", format!("mmproj: missing integer {k:?}")))
}

fn floats3(f: &GgufFile, k: &str, d: [f32; 3]) -> [f32; 3] {
    match f.get(k).and_then(GgufValue::as_array) {
        Some(a) if a.len() == 3 => {
            let mut o = d;
            for (i, v) in a.iter().enumerate() {
                o[i] = v.as_f64().unwrap_or(f64::from(d[i])) as f32;
            }
            o
        }
        _ => d,
    }
}

/// Encoder + preprocessing configuration of an mmproj file. Projector-specific defaults
/// (RoPE, activation) come from `projector` (`qwen3vl`, `qwen2vl`, `qwen25vl` known).
pub fn vision_config(f: &GgufFile) -> Result<(VisionConfig, ImageProcessor)> {
    let proj = projector_type(f).unwrap_or("");
    let (rope_theta, act) = match proj {
        "qwen3vl" | "qwen3vl_merger" => (Some(10_000.0), "gelu_pytorch_tanh"),
        other => bail!("unsupported_model", "mmproj projector type {other:?} is not supported yet"),
    };
    let hidden = uint(f, "clip.vision.embedding_length")?;
    let depth = uint(f, "clip.vision.block_count")?;
    let deepstack: Vec<usize> = f
        .get("clip.vision.is_deepstack_layers")
        .and_then(GgufValue::as_array)
        .map(|a| {
            a.iter()
                .enumerate()
                .filter(|(_, v)| match v {
                    GgufValue::Bool(b) => *b,
                    o => o.as_u64().is_some_and(|x| x != 0),
                })
                .map(|(i, _)| i)
                .collect()
        })
        .unwrap_or_default();
    let position_grid = match f.tensor("v.position_embd.weight") {
        Some(t) => (t.shape[0] as f64).sqrt().round() as usize,
        None => 0,
    };
    let out = match f.tensor("mm.2.weight") {
        Some(t) => t.shape[0],
        None => uint(f, "clip.vision.projection_dim")?,
    };
    let norm = if f.tensor("v.blk.0.ln1.bias").is_some() { NormKind::LayerNorm } else { NormKind::RmsNorm };
    let cfg = VisionConfig {
        hidden_size: hidden,
        num_heads: uint(f, "clip.vision.attention.head_count")?,
        depth,
        intermediate_size: uint(f, "clip.vision.feed_forward_length")?,
        patch_size: uint(f, "clip.vision.patch_size")?,
        in_channels: 3,
        norm,
        eps: f.get("clip.vision.attention.layer_norm_epsilon").and_then(GgufValue::as_f64).unwrap_or(1e-6) as f32,
        hidden_act: act.to_string(),
        position_grid,
        rope_theta,
        spatial_merge_size: f.get("clip.vision.spatial_merge_size").and_then(GgufValue::as_u64).unwrap_or(2) as usize,
        out_hidden_size: out,
        deepstack_layers: deepstack,
    };
    cfg.validate()?;
    let d = ImageProcessor::default();
    let opt = |k: &str| f.get(k).and_then(GgufValue::as_u64).map(|v| v as usize);
    let processor = ImageProcessor {
        patch_size: cfg.patch_size,
        merge_size: cfg.spatial_merge_size,
        min_pixels: opt("clip.vision.image_min_pixels").unwrap_or(d.min_pixels),
        max_pixels: opt("clip.vision.image_max_pixels").unwrap_or(d.max_pixels),
        mean: floats3(f, "clip.vision.image_mean", d.mean),
        std: floats3(f, "clip.vision.image_std", d.std),
        rescale: d.rescale,
        resample: d.resample,
    };
    Ok((cfg, processor))
}

/// Native lapid format of a ggml quantization type.
pub fn native_format(t: GgmlType) -> Option<Format> {
    match t {
        GgmlType::Q8_0 => Some(Format::Int8),
        GgmlType::Q4_0 => Some(Format::Int4),
        GgmlType::Q4K => Some(Format::Q4K),
        GgmlType::Q5K => Some(Format::Q5K),
        GgmlType::Q6K => Some(Format::Q6K),
        _ => None,
    }
}

fn format_dtype(f: Format) -> DType {
    match f {
        Format::Int8 => DType::Int8,
        Format::Int4 => DType::Int4,
        Format::Q4K => DType::Q4K,
        Format::Q5K => DType::Q5K,
        Format::Q6K => DType::Q6K,
    }
}

/// Write GGUF tensor `t` as `name` with row-major `shape`. A 2-D matmul weight (`matmul`)
/// stored in a ggml format lapid runs natively is kept bit-exact when the output dtype is
/// quantized (Q8_0 / Q4_0 repacked, K-quants copied); everything else is dequantized and goes
/// through the writer's dtype policy.
pub fn write_gguf_tensor(f: &GgufFile, w: &mut ModelWriter, dtype: OutDType, name: &str, t: &GgufTensor, shape: &[usize], matmul: bool) -> Result<()> {
    if shape.iter().product::<usize>() != t.numel() {
        bail!("shape_mismatch", "tensor {:?}: shape {shape:?} does not match {:?}", t.name, t.shape);
    }
    let quant_out = !matches!(dtype, OutDType::F32 | OutDType::F16);
    let cols = shape.last().copied().unwrap_or(0);
    if let (true, true, Some(fmt)) = (quant_out, matmul, native_format(t.ggml_type))
        && shape.len() == 2
        && cols == t.shape.last().copied().unwrap_or(0)
        && cols % fmt.block_elems() == 0
    {
        let bytes = lapid::quant::from_ggml_blocks(fmt, f.tensor_data(t), shape[0], cols);
        return w.add_raw(name, format_dtype(fmt), shape, &bytes);
    }
    w.add_f32(name, shape, &f.to_f32(t)?)
}

/// Copy the vision tensors of `f` into `w` (names kept; the split patch embedding folded).
/// Returns the written names.
pub fn write_vision(f: &GgufFile, w: &mut ModelWriter, dtype: OutDType, cfg: &VisionConfig) -> Result<Vec<String>> {
    let mut written = Vec::new();
    for t in &f.tensors {
        let n = t.name.as_str();
        if !(n.starts_with("v.") || n.starts_with("mm.")) || n == "v.patch_embd.weight.1" {
            continue;
        }
        if n == "v.patch_embd.weight" {
            let mut data = f.to_f32(t)?;
            if let Some(t1) = f.tensor("v.patch_embd.weight.1") {
                for (a, b) in data.iter_mut().zip(f.to_f32(t1)?) {
                    *a += b;
                }
            }
            let d = t.shape[0];
            let shape = [d, t.numel() / d.max(1)];
            if shape[1] != cfg.patch_dim() {
                bail!("shape_mismatch", "mmproj patch embedding {:?} does not match patch dim {}", t.shape, cfg.patch_dim());
            }
            w.add_f32(n, &shape, &data)?;
        } else {
            let matmul = t.shape.len() == 2 && n.ends_with(".weight") && n != "v.position_embd.weight";
            write_gguf_tensor(f, w, dtype, n, t, &t.shape.clone(), matmul)?;
        }
        written.push(n.to_string());
    }
    Ok(written)
}

/// Open an mmproj file and check it carries a vision encoder.
pub fn open(path: &std::path::Path) -> Result<GgufFile> {
    let f = GgufFile::open(path)?;
    if !is_mmproj(&f) {
        bail!("unsupported_model", "{}: not an mmproj (clip) GGUF with a vision encoder", path.display());
    }
    Ok(f)
}
