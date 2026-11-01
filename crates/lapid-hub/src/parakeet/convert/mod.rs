//! Parakeet TDT checkpoints → `.lapid`: Hugging Face (`ParakeetForTDT` safetensors, repo id
//! or local directory) and the NeMo-named GGUF export (`general.architecture = "asr"`,
//! `asr.head_type = "tdt"`, e.g. `parakeet-tdt-0.6b-v3.q8_0.gguf`).
//!
//! Both emit the same tensor names (HF naming, with these load-time transforms baked in):
//! * conv-module BatchNorm folded into the depthwise conv: `conv.depthwise_conv.weight`
//!   [C, K] and a new `conv.depthwise_conv.bias` [C]; the `conv.norm.*` tensors are dropped;
//! * 1×1 convolutions stored as 2-D matmul weights (`pointwise_conv{1,2}.weight` [out, in],
//!   subsampling point-wise convs [C, C]) so they quantize like Linear weights;
//! * subsampling 3×3 convs stored as [C, Cin·9].
//!
//! Precision (generic `lapid_convert::policy`): matmul weights quantized; the joint head at
//! most int8 (also the subsampling matmuls); depthwise / 3×3 conv weights and the decoder embedding f16; norms, biases and
//! the relative-position biases `bias_u` / `bias_v` f32. The HF `tokenizer.json` is stored as
//! the blob `tokenizer/tokenizer.json` (synthesized from the vocabulary for GGUF sources).

mod gguf;
mod hf;

use std::path::Path;

use lapid::Result;
use lapid_convert::OutDType;
use lapid_convert::policy::{Role, quantizable};
use serde_json::{Value, json};

use super::config::{ParakeetConfig, TOKENIZER_BLOB};

pub(crate) use gguf::is_parakeet_gguf;

/// BatchNorm1d epsilon (PyTorch / NeMo default).
const BN_EPS: f64 = 1e-5;

/// Convert a Parakeet TDT model in Hugging Face format (repo id or local directory).
pub fn convert_hf(repo_or_dir: &str, dtype: OutDType, out: &Path) -> Result<()> {
    hf::convert(repo_or_dir, dtype, out)
}

/// Convert a NeMo-named Parakeet TDT GGUF file.
pub fn convert_gguf(path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    gguf::convert(path, dtype, out)
}

/// Dtype-policy role of a Parakeet tensor (`.lapid` naming).
pub(crate) fn role(name: &str, shape: &[usize]) -> Role {
    if shape.len() <= 1 || name.ends_with(".bias") || name.contains("norm") || name.ends_with("bias_u") || name.ends_with("bias_v") {
        Role::Param
    } else if name == "joint.head.weight" || (name.starts_with("encoder.subsampling.") && quantizable(shape)) {
        // The joint head and the subsampling (first layers, error compounds through the
        // 24 blocks) stay int8 in int4 files.
        Role::LinearInt8
    } else if name.ends_with("depthwise_conv.weight") || name.ends_with("embedding.weight") || is_subsampling_spatial(name) {
        Role::Dense
    } else if shape.len() == 2 && quantizable(shape) {
        Role::Linear
    } else {
        Role::Dense
    }
}

/// Subsampling 3×3 convs: index 0 (full conv) and every depthwise stage (3s - 1).
fn is_subsampling_spatial(name: &str) -> bool {
    subsampling_index(name).is_some_and(|i| i == 0 || i % 3 == 2)
}

/// `n` of `encoder.subsampling.layers.{n}.weight`.
fn subsampling_index(name: &str) -> Option<usize> {
    name.strip_prefix("encoder.subsampling.layers.")?.strip_suffix(".weight")?.parse().ok()
}

/// Indices of the subsampling convs: 0, then (depthwise, pointwise) = (3s-1, 3s) per stage.
pub(crate) fn subsampling_layers(stages: usize) -> Vec<(usize, bool)> {
    let mut v = vec![(0, false)];
    for s in 1..stages {
        v.push((3 * s - 1, true));
        v.push((3 * s, false));
    }
    v
}

/// Every tensor a converted file must contain.
pub(crate) fn expected_tensors(cfg: &ParakeetConfig) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    for (i, _) in subsampling_layers(cfg.subsampling_stages()) {
        v.push(format!("encoder.subsampling.layers.{i}.weight"));
        v.push(format!("encoder.subsampling.layers.{i}.bias"));
    }
    v.push("encoder.subsampling.linear.weight".into());
    v.push("encoder.subsampling.linear.bias".into());
    for i in 0..cfg.n_layers {
        let p = format!("encoder.layers.{i}");
        for n in ["norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"] {
            v.push(format!("{p}.{n}.weight"));
            v.push(format!("{p}.{n}.bias"));
        }
        for f in ["feed_forward1", "feed_forward2"] {
            v.push(format!("{p}.{f}.linear1.weight"));
            v.push(format!("{p}.{f}.linear2.weight"));
        }
        for a in ["q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"] {
            v.push(format!("{p}.self_attn.{a}.weight"));
        }
        v.push(format!("{p}.self_attn.bias_u"));
        v.push(format!("{p}.self_attn.bias_v"));
        for c in ["pointwise_conv1.weight", "depthwise_conv.weight", "depthwise_conv.bias", "pointwise_conv2.weight"] {
            v.push(format!("{p}.conv.{c}"));
        }
    }
    v.push("encoder_projector.weight".into());
    v.push("encoder_projector.bias".into());
    v.push("decoder.embedding.weight".into());
    for l in 0..cfg.decoder_layers {
        for n in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"] {
            v.push(format!("decoder.lstm.{n}_l{l}"));
        }
    }
    for n in ["decoder.decoder_projector.weight", "decoder.decoder_projector.bias", "joint.head.weight", "joint.head.bias"] {
        v.push(n.into());
    }
    v
}

/// Required `.lapid` metadata with the `"parakeet"` object and one component.
fn metadata(name: &str, dtype: OutDType, license: &str, source: Value, cfg: &ParakeetConfig) -> Value {
    let mut m = lapid_convert::writer::metadata(name, "parakeet", dtype, license, source);
    m["parakeet"] = cfg.to_json();
    m["components"] = json!({
        "parakeet": {"architecture": "parakeet_tdt", "config": "parakeet", "license": license, "tensor_prefix": ""}
    });
    m
}

fn add_tokenizer_blob(w: &mut lapid_convert::writer::ModelWriter, tokenizer_json: &[u8]) -> Result<()> {
    w.add_blob(TOKENIZER_BLOB, "application/json", tokenizer_json)
}

/// Fold an inference BatchNorm into the preceding depthwise conv (weight [C, K], optional
/// bias [C]): `s = gamma / sqrt(var + eps)`, `w' = w s`, `b' = (b - mean) s + beta`.
pub(crate) fn fold_batch_norm(
    weight: &[f32],
    bias: Option<&[f32]>,
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let c = gamma.len();
    if c == 0 || !weight.len().is_multiple_of(c) || [beta.len(), mean.len(), var.len()].iter().any(|&n| n != c) || bias.is_some_and(|b| b.len() != c) {
        lapid::bail!("shape_mismatch", "batch norm fold: {} weights, {c} channels", weight.len());
    }
    let k = weight.len() / c;
    let mut w = vec![0.0f32; weight.len()];
    let mut b = vec![0.0f32; c];
    for ch in 0..c {
        let s = gamma[ch] as f64 / (var[ch] as f64 + BN_EPS).sqrt();
        for t in 0..k {
            w[ch * k + t] = (weight[ch * k + t] as f64 * s) as f32;
        }
        let b0 = bias.map_or(0.0, |b| b[ch] as f64);
        b[ch] = ((b0 - mean[ch] as f64) * s + beta[ch] as f64) as f32;
    }
    Ok((w, b))
}

/// Shape a source weight is written with (`.lapid` layout) given its source shape.
pub(crate) fn target_shape(name: &str, shape: &[usize]) -> Vec<usize> {
    let flat2 = |s: &[usize]| vec![s[0], s[1..].iter().product()];
    if shape.len() >= 3 && (name.contains("pointwise_conv") || name.contains("depthwise_conv") || subsampling_index(name).is_some()) {
        return flat2(shape);
    }
    shape.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lapid::quant::Format;
    use lapid_convert::policy::{Target, target};

    fn t(name: &str, shape: &[usize], dtype: OutDType) -> Target {
        target(role(name, shape), shape, dtype)
    }

    #[test]
    fn policy_table() {
        let q = "encoder.layers.0.self_attn.q_proj.weight";
        assert_eq!(t(q, &[1024, 1024], OutDType::Int4), Target::Quant(Format::Int4));
        assert_eq!(t("joint.head.weight", &[8198, 640], OutDType::Int4), Target::Quant(Format::Int8));
        assert_eq!(t("encoder.layers.0.self_attn.bias_u", &[8, 128], OutDType::Int8), Target::F32);
        assert_eq!(t("encoder.layers.0.conv.depthwise_conv.weight", &[1024, 9], OutDType::Int8), Target::F16);
        assert_eq!(t("encoder.subsampling.layers.3.weight", &[256, 256], OutDType::Int4), Target::Quant(Format::Int8));
        assert_eq!(t("encoder.subsampling.linear.weight", &[1024, 4096], OutDType::Int4), Target::Quant(Format::Int8));
        assert_eq!(t("encoder.subsampling.layers.2.weight", &[256, 9], OutDType::Int8), Target::F16);
        assert_eq!(t("decoder.embedding.weight", &[8193, 640], OutDType::Int8), Target::F16);
        assert_eq!(t("decoder.lstm.weight_ih_l0", &[2560, 640], OutDType::Int8), Target::Quant(Format::Int8));
        assert_eq!(t("decoder.lstm.bias_ih_l0", &[2560], OutDType::Int8), Target::F32);
    }

    #[test]
    fn names_and_shapes() {
        let n = expected_tensors(&ParakeetConfig::v3());
        assert_eq!(n.len(), 10 + 2 + 24 * (10 + 4 + 7 + 4) + 2 + 1 + 8 + 4);
        assert_eq!(subsampling_layers(3), vec![(0, false), (2, true), (3, false), (5, true), (6, false)]);
        assert_eq!(target_shape("encoder.layers.1.conv.pointwise_conv1.weight", &[2048, 1024, 1]), vec![2048, 1024]);
        assert_eq!(target_shape("encoder.subsampling.layers.0.weight", &[256, 1, 3, 3]), vec![256, 9]);
        assert_eq!(target_shape("encoder.subsampling.linear.weight", &[1024, 4096]), vec![1024, 4096]);
    }

    #[test]
    fn bn_fold() {
        let (w, b) = fold_batch_norm(&[1.0, 2.0, 3.0, 4.0], None, &[2.0, 1.0], &[0.5, 0.0], &[1.0, 0.0], &[3.0 - 1e-5, 1.0 - 1e-5]).expect("fold");
        let s0 = 2.0 / 3f32.sqrt();
        assert!((w[0] - s0).abs() < 1e-6 && (w[1] - 2.0 * s0).abs() < 1e-6);
        assert!((b[0] - (0.5 - s0)).abs() < 1e-6);
        assert!((w[3] - 4.0).abs() < 1e-5 && b[1].abs() < 1e-6);
    }
}
