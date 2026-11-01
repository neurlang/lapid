//! Exported icefall Zipformer2 transducer → `.lapid`.
//!
//! Source: a directory (or HF repo) written by `tools/reference/zipformer_ref.py export`:
//! `config.json` (`"model_type": "zipformer2_transducer"`, icefall's hyper-parameters, default
//! chunk / left context), `model.safetensors` (icefall `AsrModel` state dict, f32 or f16),
//! `tokens.txt` and optionally `bpe.model`. Names stay icefall's; load-time transforms are baked in:
//! * `encoder_embed.conv.{0,4,7}.weight` [Cout, Cin, 3, 3] → [Cout, (kt, kf, Cin)] (im2col);
//! * ConvNeXt depthwise 7×7 → [C, 49]; 1×1 convs → [out, in]; causal / chunkwise depthwise
//!   kernels [C, 1, K] → [C, K];
//! * `decoder.conv.weight` (grouped, [D, 4, context]) → dense block-diagonal [D, context · D];
//! * scalar `log_scale`s → [1]; the unused per-layer `bypass_scale` and the training-only
//!   `simple_am_proj` / `simple_lm_proj` are dropped.
//!
//! Precision: matmul weights quantized (`encoder_embed.out`, `joiner.output_linear` and the
//! decoder conv at most int8); conv kernels and the decoder embedding f16; norms, biases,
//! bypass / chunk scales f32.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::policy::{Role, quantizable};
use lapid_convert::safetensors::SafeTensors;
use lapid_convert::writer::{ModelWriter, file_hash, read_json};
use lapid_convert::{download_hf, download_hf_optional};
use serde_json::{Value, json};

use super::config::{BPE_BLOB, TOKENS_BLOB, ZipformerConfig};
use super::model::decoder::dense_group_conv;

/// `config.json` `model_type` of an exported model.
pub const MODEL_TYPE: &str = "zipformer2_transducer";

struct Files {
    config: PathBuf,
    weights: PathBuf,
    tokens: PathBuf,
    bpe: Option<PathBuf>,
    repo: String,
}

fn locate(src: &str) -> Result<Files> {
    let dir = Path::new(src);
    if dir.is_dir() {
        let need = |f: &str| Some(dir.join(f)).filter(|p| p.is_file()).ok_or_else(|| Error::new("not_found", format!("{}: no {f}", dir.display())));
        return Ok(Files {
            config: need("config.json")?,
            weights: need("model.safetensors")?,
            tokens: need("tokens.txt")?,
            bpe: Some(dir.join("bpe.model")).filter(|p| p.is_file()),
            repo: dir.display().to_string(),
        });
    }
    let repo = src.trim_matches('/');
    Ok(Files {
        config: download_hf(repo, "config.json")?,
        weights: download_hf(repo, "model.safetensors")?,
        tokens: download_hf(repo, "tokens.txt")?,
        bpe: download_hf_optional(repo, "bpe.model")?,
        repo: repo.to_string(),
    })
}

/// Dtype-policy role of a (converted) tensor.
pub(crate) fn role(name: &str, shape: &[usize]) -> Role {
    let param = shape.len() <= 1
        || name.ends_with(".bias")
        || name.ends_with("log_scale")
        || name.ends_with("bypass_scale")
        || name.ends_with("chunkwise_conv_scale");
    if param {
        Role::Param
    } else if name.starts_with("encoder_embed.conv.") || name.contains("depthwise_conv.") || name.ends_with("embedding.weight") {
        Role::Dense
    } else if name == "encoder_embed.out.weight" || name == "joiner.output_linear.weight" || name == "decoder.conv.weight" {
        if quantizable(shape) { Role::LinearInt8 } else { Role::Dense }
    } else if shape.len() == 2 && quantizable(shape) {
        Role::Linear
    } else {
        Role::Dense
    }
}

/// Prefix of layer `j` of stack `i`.
pub(crate) fn layer_prefix(cfg: &ZipformerConfig, i: usize, j: usize) -> String {
    if cfg.downsampling_factor[i] > 1 { format!("encoder.encoders.{i}.encoder.layers.{j}") } else { format!("encoder.encoders.{i}.layers.{j}") }
}

/// Every tensor a converted file contains.
pub(crate) fn expected_tensors(cfg: &ZipformerConfig) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    let wb = |v: &mut Vec<String>, p: &str| {
        v.push(format!("{p}.weight"));
        v.push(format!("{p}.bias"));
    };
    for c in ["conv.0", "conv.4", "conv.7", "convnext.depthwise_conv", "convnext.pointwise_conv1", "convnext.pointwise_conv2", "out"] {
        wb(&mut v, &format!("encoder_embed.{c}"));
    }
    v.push("encoder_embed.out_norm.log_scale".into());
    v.push("encoder_embed.out_norm.bias".into());
    for i in 0..cfg.num_stacks() {
        if cfg.downsampling_factor[i] > 1 {
            v.push(format!("encoder.encoders.{i}.downsample.bias"));
            v.push(format!("encoder.encoders.{i}.out_combiner.bypass_scale"));
        }
        for j in 0..cfg.num_encoder_layers[i] {
            let p = layer_prefix(cfg, i, j);
            v.push(format!("{p}.bypass.bypass_scale"));
            v.push(format!("{p}.bypass_mid.bypass_scale"));
            wb(&mut v, &format!("{p}.self_attn_weights.in_proj"));
            v.push(format!("{p}.self_attn_weights.linear_pos.weight"));
            for m in ["self_attn1", "self_attn2", "feed_forward1", "feed_forward2", "feed_forward3", "nonlin_attention"] {
                wb(&mut v, &format!("{p}.{m}.in_proj"));
                wb(&mut v, &format!("{p}.{m}.out_proj"));
            }
            for m in ["conv_module1", "conv_module2"] {
                wb(&mut v, &format!("{p}.{m}.in_proj"));
                wb(&mut v, &format!("{p}.{m}.out_proj"));
                v.push(format!("{p}.{m}.depthwise_conv.chunkwise_conv_scale"));
                wb(&mut v, &format!("{p}.{m}.depthwise_conv.causal_conv"));
                wb(&mut v, &format!("{p}.{m}.depthwise_conv.chunkwise_conv"));
            }
            v.push(format!("{p}.norm.log_scale"));
            v.push(format!("{p}.norm.bias"));
        }
    }
    v.push("encoder.downsample_output.bias".into());
    v.push("decoder.embedding.weight".into());
    if cfg.context_size > 1 {
        v.push("decoder.conv.weight".into());
    }
    for m in ["encoder_proj", "decoder_proj", "output_linear"] {
        wb(&mut v, &format!("joiner.{m}"));
    }
    v
}

/// (shape, data) of `name` as stored in `.lapid`, from its source shape and values.
fn transform(name: &str, shape: &[usize], data: Vec<f32>, cfg: &ZipformerConfig) -> Result<(Vec<usize>, Vec<f32>)> {
    if name.starts_with("encoder_embed.conv.") && name.ends_with(".weight") {
        let [o, i, kh, kw] = shape else { bail!("shape_mismatch", "{name}: expected 4-D, got {shape:?}") };
        let (o, i, kh, kw) = (*o, *i, *kh, *kw);
        let mut out = vec![0.0f32; data.len()];
        for oc in 0..o {
            for ic in 0..i {
                for a in 0..kh {
                    for b in 0..kw {
                        out[oc * kh * kw * i + (a * kw + b) * i + ic] = data[((oc * i + ic) * kh + a) * kw + b];
                    }
                }
            }
        }
        return Ok((vec![o, kh * kw * i], out));
    }
    if name == "decoder.conv.weight" {
        let [d, per, k] = shape else { bail!("shape_mismatch", "{name}: expected 3-D, got {shape:?}") };
        if *d != cfg.decoder_dim || *k != cfg.context_size {
            bail!("shape_mismatch", "{name}: {shape:?} does not match decoder dim {} / context {}", cfg.decoder_dim, cfg.context_size);
        }
        return Ok((vec![*d, k * d], dense_group_conv(&data, *d, *per, *k)?));
    }
    if shape.is_empty() {
        return Ok((vec![1], data));
    }
    if shape.len() >= 3 && (name.contains("conv") || name.contains("depthwise")) && !name.ends_with("chunkwise_conv_scale") {
        return Ok((vec![shape[0], shape[1..].iter().product()], data));
    }
    Ok((shape.to_vec(), data))
}

/// `ZipformerConfig` of an exported `config.json`.
pub(crate) fn config_from_export(raw: &Value) -> Result<ZipformerConfig> {
    if raw.get("model_type").and_then(Value::as_str) != Some(MODEL_TYPE) {
        bail!("unsupported_model", "config.json model_type is not {MODEL_TYPE:?}");
    }
    if raw.get("causal").and_then(Value::as_bool) == Some(false) {
        bail!("unsupported_model", "non-causal Zipformer2 checkpoints cannot stream (causal = false)");
    }
    ZipformerConfig::from_json(raw)
}

/// Convert an exported Zipformer2 transducer (directory or HF repo).
pub fn convert_hf(src: &str, dtype: OutDType, out: &Path) -> Result<()> {
    let files = locate(src)?;
    let raw = read_json(&files.config)?;
    let cfg = config_from_export(&raw)?;
    let tokens = std::fs::read(&files.tokens).map_err(|e| Error::new("io", format!("{}: {e}", files.tokens.display())))?;
    let st = SafeTensors::open(&files.weights)?;
    let have: BTreeSet<String> = st.names().into_iter().collect();
    let expected = expected_tensors(&cfg);

    let name = raw
        .get("name")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| files.repo.trim_end_matches('/').rsplit('/').next().unwrap_or("zipformer").to_string());
    let license = raw.get("license").and_then(Value::as_str).unwrap_or("unknown");
    let mut source = json!({ "repo": raw.get("source").and_then(Value::as_str).unwrap_or(&files.repo), "format": "icefall-safetensors" });
    source["file_hash"] = json!(file_hash(&files.weights)?);
    let mut meta = lapid_convert::writer::metadata(&name, "zipformer", dtype, license, source);
    meta["zipformer"] = cfg.to_json();
    meta["components"] = json!({
        "zipformer": {"architecture": "zipformer2_transducer", "config": "zipformer", "license": license, "tensor_prefix": ""}
    });

    let mut w = ModelWriter::create(out, meta, dtype, role, expected.len())?;
    w.add_blob(TOKENS_BLOB, "text/plain", &tokens)?;
    if let Some(bpe) = &files.bpe {
        let b = std::fs::read(bpe).map_err(|e| Error::new("io", format!("{}: {e}", bpe.display())))?;
        w.add_blob(BPE_BLOB, "application/x-sentencepiece", &b)?;
    }
    for n in &expected {
        if !have.contains(n) {
            bail!("missing_tensor", "{}: checkpoint has no {n:?}", files.repo);
        }
        let (shape, data) = transform(n, &st.shape(n)?, st.to_f32(n)?, &cfg)?;
        w.add_f32(n, &shape, &data)?;
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zipformer::config::tests::medium;
    use lapid::quant::Format;
    use lapid_convert::policy::{Target, target};

    #[test]
    fn names_and_roles() {
        let cfg = medium();
        let n = expected_tensors(&cfg);
        // embed 14 + 2, per layer 49, downsampled stacks 2 each, out bias, decoder 2, joiner 6
        assert_eq!(n.len(), 16 + 16 * 49 + 5 * 2 + 1 + 2 + 6); // 819 = 839 - 16 bypass_scale - 4 simple_*
        let t = |name: &str, shape: &[usize], d| target(role(name, shape), shape, d);
        let p = "encoder.encoders.0.layers.0";
        assert_eq!(t(&format!("{p}.feed_forward1.in_proj.weight"), &[384, 192], OutDType::Int4), Target::Quant(Format::Int4));
        assert_eq!(t("joiner.output_linear.weight", &[500, 512], OutDType::Int4), Target::Quant(Format::Int8));
        assert_eq!(t(&format!("{p}.conv_module1.depthwise_conv.causal_conv.weight"), &[192, 16], OutDType::Int8), Target::F16);
        assert_eq!(t(&format!("{p}.conv_module1.depthwise_conv.chunkwise_conv_scale"), &[2, 192, 31], OutDType::Int8), Target::F32);
        assert_eq!(t(&format!("{p}.norm.log_scale"), &[1], OutDType::Int8), Target::F32);
        assert_eq!(t("decoder.embedding.weight", &[500, 512], OutDType::Int8), Target::F16);
    }

    #[test]
    fn im2col_weight_order() {
        let cfg = medium();
        // [o=1, i=2, 3, 3]: value = 100 i + 10 kh + kw
        let data: Vec<f32> = (0..18).map(|x| (100 * (x / 9) + 10 * ((x % 9) / 3) + x % 3) as f32).collect();
        let (shape, out) = transform("encoder_embed.conv.4.weight", &[1, 2, 3, 3], data, &cfg).expect("t");
        assert_eq!(shape, vec![1, 18]);
        // column (kt, kf, c) = (1, 2, 1) -> 100 + 12
        assert_eq!(out[(3 + 2) * 2 + 1], 112.0);
        let (s, _) = transform("encoder.encoders.0.layers.0.norm.log_scale", &[], vec![1.0], &cfg).expect("t");
        assert_eq!(s, vec![1]);
    }
}
