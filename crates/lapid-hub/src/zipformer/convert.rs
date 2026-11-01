//! Exported icefall Zipformer2 transducer -> `.lapid` through the generic engine.
//!
//! Source: a directory (or HF repo) written by `tools/reference/zipformer_ref.py export`:
//! `config.json` (`"model_type": "zipformer2_transducer"`, icefall's hyper-parameters, default
//! chunk / left context), `model.safetensors` (icefall names, kept), `tokens.txt` and
//! optionally `bpe.model` (stored as blobs). Load-time transforms baked in:
//! * `encoder_embed.conv.{0,4,7}.weight` `[Cout, Cin, 3, 3]` -> `[Cout, (kt, kf, Cin)]` (im2col);
//! * ConvNeXt depthwise 7x7 -> `[C, 49]`; 1x1 convs -> `[out, in]`; causal / chunkwise
//!   depthwise kernels `[C, 1, K]` -> `[C, K]`; scalar `log_scale`s -> `[1]`;
//! * `decoder.conv.weight` (grouped `[D, 4, context]`) -> dense block-diagonal `[D, context D]`;
//! * everything the model does not use (per-layer `bypass_scale`, `simple_*_proj`) dropped.
//!
//! Precision: matmul weights quantized (`encoder_embed.out`, `joiner.output_linear` and the
//! decoder conv at most int8); conv kernels and the decoder embedding f16; norms, biases,
//! bypass / chunk scales f32.

use lapid::{Error, Result, bail};
use lapid_convert::assets::HfFiles;
use lapid_convert::engine::{ConvertHooks, ConvertSpec, Emit, HfExtras, SourceKind, TensorSource};
use lapid_convert::mapping::RoleRule;
use lapid_convert::policy::Role;
use lapid_convert::writer::read_json;
use serde_json::{Map, Value, json};

use super::config::{BPE_BLOB, TOKENS_BLOB, ZipformerConfig};
use super::model::decoder::dense_group_conv;

/// `config.json` `model_type` of an exported model.
pub const MODEL_TYPE: &str = "zipformer2_transducer";

pub static CONVERT: ConvertSpec = ConvertSpec {
    architecture: "zipformer",
    license: "unknown",
    hf_model_types: &[MODEL_TYPE],
    gguf_architectures: &[],
    hf_names: &[],
    gguf_names: &[],
    roles: &[
        RoleRule::new("*chunkwise_conv_scale", Role::Param),
        RoleRule::new("*log_scale", Role::Param),
        RoleRule::new("*bypass_scale", Role::Param),
        RoleRule::new("encoder_embed.conv.*", Role::Dense),
        RoleRule::new("*depthwise_conv.*", Role::Dense),
        RoleRule::new("*embedding.weight", Role::Dense),
        RoleRule::new("encoder_embed.out.weight", Role::LinearInt8),
        RoleRule::new("joiner.output_linear.weight", Role::LinearInt8),
        RoleRule::new("decoder.conv.weight", Role::LinearInt8),
    ],
    text_assets: false,
    hooks: &Hooks,
};

struct Hooks;

impl ConvertHooks for Hooks {
    fn hf_config(&self, raw: &Value) -> Result<Value> {
        if raw.get("causal").and_then(Value::as_bool) == Some(false) {
            bail!("unsupported_model", "non-causal Zipformer2 checkpoints cannot stream (causal = false)");
        }
        Ok(ZipformerConfig::from_json(raw)?.to_json())
    }

    /// `tokens.txt` (+ `bpe.model`) blobs; name and license from the exported `config.json`.
    fn hf_extras(&self, files: &HfFiles, _cfg: &mut Value) -> Result<HfExtras> {
        let raw = read_json(&files.config)?;
        let read = |p: std::path::PathBuf| std::fs::read(&p).map_err(|e| Error::new("io", format!("{}: {e}", p.display())));
        let mut blobs = vec![(TOKENS_BLOB.to_string(), "text/plain".to_string(), read(files.required("tokens.txt")?)?)];
        if let Some(bpe) = files.optional("bpe.model")? {
            blobs.push((BPE_BLOB.to_string(), "application/x-sentencepiece".to_string(), read(bpe)?));
        }
        let license = raw["license"].as_str().unwrap_or("unknown");
        let mut metadata = Map::new();
        metadata.insert("name".into(), json!(raw["name"].as_str().map_or_else(|| files.name("zipformer"), str::to_string)));
        metadata.insert("license".into(), json!(license));
        metadata.insert("components".into(), json!({"zipformer": {"architecture": MODEL_TYPE, "config": "zipformer", "license": license, "tensor_prefix": ""}}));
        Ok(HfExtras { blobs, metadata, ..HfExtras::default() })
    }

    fn expected(&self, cfg: &Value) -> Result<Vec<String>> {
        Ok(expected_tensors(&ZipformerConfig::from_json(cfg)?))
    }

    fn reshape(&self, name: &str, shape: &[usize], _cfg: &Value, _src: SourceKind) -> Option<Vec<usize>> {
        if shape.is_empty() {
            return Some(vec![1]);
        }
        let transformed = name.starts_with("encoder_embed.conv.") || name == "decoder.conv.weight" || name.ends_with("chunkwise_conv_scale");
        (shape.len() >= 3 && !transformed && (name.contains("conv") || name.contains("depthwise"))).then(|| vec![shape[0], shape[1..].iter().product()])
    }

    fn transform(&self, name: &str, src: &dyn TensorSource, cfg: &Value) -> Result<Emit> {
        let shape = src.shape(name).unwrap_or_default();
        if name.starts_with("encoder_embed.conv.") && name.ends_with(".weight") {
            let [o, i, kh, kw] = shape[..] else { bail!("shape_mismatch", "{name}: expected 4-D, got {shape:?}") };
            let data = src.f32(name)?;
            let mut out = vec![0.0f32; data.len()];
            for (oc, ic, a, b) in (0..o).flat_map(|oc| (0..i).flat_map(move |ic| (0..kh).flat_map(move |a| (0..kw).map(move |b| (oc, ic, a, b))))) {
                out[oc * kh * kw * i + (a * kw + b) * i + ic] = data[((oc * i + ic) * kh + a) * kw + b];
            }
            return Ok(Emit::Replace(vec![(name.to_string(), vec![o, kh * kw * i], out)]));
        }
        if name == "decoder.conv.weight" {
            let [d, per, k] = shape[..] else { bail!("shape_mismatch", "{name}: expected 3-D, got {shape:?}") };
            return Ok(Emit::Replace(vec![(name.to_string(), vec![d, k * d], dense_group_conv(&src.f32(name)?, d, per, k)?)]));
        }
        // only what the model uses
        let used = expected_tensors(&ZipformerConfig::from_json(cfg)?);
        Ok(if used.iter().any(|n| n == name) { Emit::Keep } else { Emit::Drop })
    }
}

/// Prefix of layer `j` of stack `i`.
pub(crate) fn layer_prefix(cfg: &ZipformerConfig, i: usize, j: usize) -> String {
    if cfg.downsampling_factor[i] > 1 { format!("encoder.encoders.{i}.encoder.layers.{j}") } else { format!("encoder.encoders.{i}.layers.{j}") }
}

/// Every tensor a converted file contains.
pub(crate) fn expected_tensors(cfg: &ZipformerConfig) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    let wb = |v: &mut Vec<String>, p: &str| v.extend([format!("{p}.weight"), format!("{p}.bias")]);
    for c in ["conv.0", "conv.4", "conv.7", "convnext.depthwise_conv", "convnext.pointwise_conv1", "convnext.pointwise_conv2", "out"] {
        wb(&mut v, &format!("encoder_embed.{c}"));
    }
    v.extend(["encoder_embed.out_norm.log_scale".into(), "encoder_embed.out_norm.bias".into()]);
    for i in 0..cfg.num_stacks() {
        if cfg.downsampling_factor[i] > 1 {
            v.extend([format!("encoder.encoders.{i}.downsample.bias"), format!("encoder.encoders.{i}.out_combiner.bypass_scale")]);
        }
        for j in 0..cfg.num_encoder_layers[i] {
            let p = layer_prefix(cfg, i, j);
            v.extend([format!("{p}.bypass.bypass_scale"), format!("{p}.bypass_mid.bypass_scale")]);
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
            v.extend([format!("{p}.norm.log_scale"), format!("{p}.norm.bias")]);
        }
    }
    v.extend(["encoder.downsample_output.bias".into(), "decoder.embedding.weight".into()]);
    if cfg.context_size > 1 {
        v.push("decoder.conv.weight".into());
    }
    for m in ["encoder_proj", "decoder_proj", "output_linear"] {
        wb(&mut v, &format!("joiner.{m}"));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zipformer::config::tests::medium;
    use lapid::quant::Format;
    use lapid_convert::OutDType;
    use lapid_convert::policy::{Target, target};

    #[test]
    fn names_and_roles() {
        let n = expected_tensors(&medium());
        // embed 14 + 2, per layer 49, downsampled stacks 2 each, out bias, decoder 2, joiner 6
        assert_eq!(n.len(), 16 + 16 * 49 + 5 * 2 + 1 + 2 + 6);
        let t = |name: &str, shape: &[usize], d| target(CONVERT.role(name, shape), shape, d);
        let p = "encoder.encoders.0.layers.0";
        assert_eq!(t(&format!("{p}.feed_forward1.in_proj.weight"), &[384, 192], OutDType::Int4), Target::Quant(Format::Int4));
        assert_eq!(t("joiner.output_linear.weight", &[500, 512], OutDType::Int4), Target::Quant(Format::Int8));
        assert_eq!(t(&format!("{p}.conv_module1.depthwise_conv.causal_conv.weight"), &[192, 16], OutDType::Int8), Target::F16);
        assert_eq!(t(&format!("{p}.conv_module1.depthwise_conv.chunkwise_conv_scale"), &[2, 192, 31], OutDType::Int8), Target::F32);
        assert_eq!(t(&format!("{p}.norm.log_scale"), &[1], OutDType::Int8), Target::F32);
        assert_eq!(t("decoder.embedding.weight", &[500, 512], OutDType::Int8), Target::F16);
        assert_eq!(t("encoder_embed.convnext.pointwise_conv1.weight", &[384, 128], OutDType::Int4), Target::Quant(Format::Int4));
    }
}
