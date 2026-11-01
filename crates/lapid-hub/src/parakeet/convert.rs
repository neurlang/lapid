//! Parakeet TDT checkpoints -> `.lapid` through the generic engine: Hugging Face
//! (`ParakeetForTDT`: `config.json` + `processor_config.json`, `tokenizer.json`,
//! `model.safetensors`) and the NeMo-named GGUF export (`general.architecture = "asr"`,
//! `asr.head_type = "tdt"`, e.g. `parakeet-tdt-0.6b-v3.q8_0.gguf`; Q8_0 / Q4_0 repacked).
//!
//! Both write HF names with these load-time transforms baked in: the conv-module BatchNorm
//! folded into the depthwise conv (`conv.depthwise_conv.{weight [C, K], bias}`), 1x1 convs
//! as matmul weights `[out, in]`, subsampling 3x3 convs as `[C, Cin 9]`. Precision: matmul
//! weights quantized, the joint head and the subsampling at most int8, depthwise / 3x3 conv
//! weights and the decoder embedding f16, norms, biases and `bias_u` / `bias_v` f32. The
//! tokenizer is the blob `tokenizer/tokenizer.json` (synthesized from the GGUF vocabulary).

use lapid::{Error, Result, bail};
use lapid_convert::assets::HfFiles;
use lapid_convert::engine::{ConvertHooks, ConvertSpec, Emit, HfExtras, SourceKind, TensorSource};
use lapid_convert::gguf::{GgufFile, GgufValue};
use lapid_convert::mapping::{RoleRule, Rule};
use lapid_convert::policy::Role;
use lapid_convert::transforms::fold_batchnorm;
use lapid_convert::writer::read_json;
use serde_json::{Map, Value, json};

use super::config::{ParakeetConfig, TOKENIZER_BLOB};

macro_rules! layer {
    ($g:literal, $h:literal) => {
        Rule::new(concat!("encoder.layers.{i}.", $g), concat!("encoder.layers.{i}.", $h))
    };
}

pub static CONVERT: ConvertSpec = ConvertSpec {
    architecture: "parakeet",
    license: "unknown",
    hf_model_types: &["parakeet_tdt"],
    gguf_architectures: &["asr"],
    hf_names: &[],
    gguf_names: &[
        Rule::drop("encoder.pos_enc.pe"),
        Rule::drop("preprocessor.*"),
        Rule::new("encoder.pre_encode.out.{w}", "encoder.subsampling.linear.{w}"),
        Rule::new("encoder.pre_encode.conv.{n}.{w}", "encoder.subsampling.layers.{n}.{w}"),
        layer!("self_attn.pos_bias_u", "self_attn.bias_u"),
        layer!("self_attn.pos_bias_v", "self_attn.bias_v"),
        layer!("self_attn.linear_q.{w}", "self_attn.q_proj.{w}"),
        layer!("self_attn.linear_k.{w}", "self_attn.k_proj.{w}"),
        layer!("self_attn.linear_v.{w}", "self_attn.v_proj.{w}"),
        layer!("self_attn.linear_out.{w}", "self_attn.o_proj.{w}"),
        layer!("self_attn.linear_pos.{w}", "self_attn.relative_k_proj.{w}"),
        layer!("conv.batch_norm.{w}", "conv.norm.{w}"),
        layer!("{m}.{w}", "{m}.{w}"),
        layer!("{m}.{s}.{w}", "{m}.{s}.{w}"),
        Rule::new("decoder.prediction.embed.weight", "decoder.embedding.weight"),
        Rule::new("decoder.prediction.dec_rnn.lstm.{g}.{k}", "decoder.lstm.{k}_{g}"),
        Rule::new("joint.pred.{w}", "decoder.decoder_projector.{w}"),
        Rule::new("joint.enc.{w}", "encoder_projector.{w}"),
        Rule::new("joint.joint_net.2.{w}", "joint.head.{w}"),
    ],
    roles: &[
        RoleRule::new("*.bias_u", Role::Param),
        RoleRule::new("*.bias_v", Role::Param),
        RoleRule::new("joint.head.weight", Role::LinearInt8),
        // first layers (error compounds through the blocks): at most int8; 3x3 convs f16
        RoleRule::new("encoder.subsampling.*", Role::LinearInt8),
        RoleRule::new("*depthwise_conv.weight", Role::Dense),
        RoleRule::new("*embedding.weight", Role::Dense),
    ],
    text_assets: false,
    hooks: &Hooks,
};

struct Hooks;

fn extras(license: &str, tokenizer: Vec<u8>) -> HfExtras {
    let mut metadata = Map::new();
    metadata.insert("license".into(), json!(license));
    metadata.insert("components".into(), json!({"parakeet": {"architecture": "parakeet_tdt", "config": "parakeet", "license": license, "tensor_prefix": ""}}));
    HfExtras { blobs: vec![(TOKENIZER_BLOB.into(), "application/json".into(), tokenizer)], metadata, ..HfExtras::default() }
}

fn config(cfg: &Value) -> Result<ParakeetConfig> {
    ParakeetConfig::from_metadata(&json!({ "parakeet": cfg }))
}

impl ConvertHooks for Hooks {
    /// `processor_config.json` completes the config; `tokenizer.json` is stored as is.
    fn hf_extras(&self, files: &HfFiles, cfg: &mut Value) -> Result<HfExtras> {
        let processor = files.optional("processor_config.json")?.as_deref().map(read_json).transpose()?;
        *cfg = config_from_hf(cfg, processor.as_ref())?.to_json();
        let tok = files.required("tokenizer.json")?;
        let bytes = std::fs::read(&tok).map_err(|e| Error::new("io", format!("{}: {e}", tok.display())))?;
        Ok(extras(if files.repo.starts_with("nvidia/parakeet") { "cc-by-4.0" } else { "unknown" }, bytes))
    }

    fn gguf_config(&self, f: &GgufFile) -> Result<Value> {
        if f.get("asr.head_type").and_then(GgufValue::as_str) != Some("tdt") {
            bail!("unsupported_model", "GGUF asr model is not a Parakeet TDT (asr.head_type != \"tdt\")");
        }
        Ok(config_from_gguf(f)?.to_json())
    }

    fn gguf_extras(&self, f: &GgufFile, cfg: &mut Value) -> Result<HfExtras> {
        Ok(extras("cc-by-4.0", synth_tokenizer(f, config(cfg)?.blank_id)?))
    }

    fn expected(&self, cfg: &Value) -> Result<Vec<String>> {
        Ok(expected_tensors(&config(cfg)?))
    }

    fn reshape(&self, name: &str, shape: &[usize], _cfg: &Value, _src: SourceKind) -> Option<Vec<usize>> {
        let flat = shape.len() >= 3 && (name.contains("pointwise_conv") || name.contains("depthwise_conv") || name.starts_with("encoder.subsampling.layers."));
        flat.then(|| vec![shape[0], shape[1..].iter().product()])
    }

    /// BatchNorm folded into the depthwise conv (weight and a new bias).
    fn transform(&self, name: &str, src: &dyn TensorSource, _cfg: &Value) -> Result<Emit> {
        if name.contains(".conv.norm.") || name.ends_with(".conv.depthwise_conv.bias") {
            return Ok(Emit::Drop);
        }
        let Some(p) = name.strip_suffix(".depthwise_conv.weight").filter(|p| p.starts_with("encoder.layers.")) else { return Ok(Emit::Keep) };
        let shape = src.shape(name).ok_or_else(|| Error::new("missing_tensor", format!("no {name}")))?;
        let bias_name = format!("{p}.depthwise_conv.bias");
        let bias = if src.shape(&bias_name).is_some() { Some(src.f32(&bias_name)?) } else { None };
        let bn = |k: &str| src.f32(&format!("{p}.norm.{k}"));
        let (w, b) = fold_batchnorm(&src.f32(name)?, bias.as_deref(), &bn("weight")?, &bn("bias")?, &bn("running_mean")?, &bn("running_var")?, 1e-5)?;
        Ok(Emit::Replace(vec![(name.to_string(), shape, w), (bias_name, vec![b.len()], b)]))
    }
}

/// Indices of the subsampling convs: 0, then (depthwise, pointwise) = (3s-1, 3s) per stage.
pub(crate) fn subsampling_layers(stages: usize) -> Vec<(usize, bool)> {
    let mut v = vec![(0, false)];
    for s in 1..stages {
        v.extend([(3 * s - 1, true), (3 * s, false)]);
    }
    v
}

/// Every tensor a converted file must contain.
pub(crate) fn expected_tensors(cfg: &ParakeetConfig) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    let wb = |v: &mut Vec<String>, p: &str| v.extend([format!("{p}.weight"), format!("{p}.bias")]);
    for (i, _) in subsampling_layers(cfg.subsampling_stages()) {
        wb(&mut v, &format!("encoder.subsampling.layers.{i}"));
    }
    wb(&mut v, "encoder.subsampling.linear");
    for i in 0..cfg.n_layers {
        let p = format!("encoder.layers.{i}");
        for n in ["norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"] {
            wb(&mut v, &format!("{p}.{n}"));
        }
        v.extend(["feed_forward1.linear1", "feed_forward1.linear2", "feed_forward2.linear1", "feed_forward2.linear2"].map(|f| format!("{p}.{f}.weight")));
        v.extend(["q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"].map(|a| format!("{p}.self_attn.{a}.weight")));
        v.extend([format!("{p}.self_attn.bias_u"), format!("{p}.self_attn.bias_v")]);
        v.extend(["pointwise_conv1.weight", "depthwise_conv.weight", "depthwise_conv.bias", "pointwise_conv2.weight"].map(|c| format!("{p}.conv.{c}")));
    }
    wb(&mut v, "encoder_projector");
    v.push("decoder.embedding.weight".into());
    for l in 0..cfg.decoder_layers {
        v.extend(["weight_ih", "weight_hh", "bias_ih", "bias_hh"].map(|n| format!("decoder.lstm.{n}_l{l}")));
    }
    wb(&mut v, "decoder.decoder_projector");
    wb(&mut v, "joint.head");
    v
}

/// `ParakeetConfig` from `config.json` (+ the `processor_config.json` feature extractor).
pub(crate) fn config_from_hf(cfg: &Value, processor: Option<&Value>) -> Result<ParakeetConfig> {
    let enc = cfg.get("encoder_config").ok_or_else(|| Error::new("missing_metadata", "config.json: no encoder_config"))?;
    let mut c = ParakeetConfig::v3();
    let u = |v: &Value, k: &str, d: usize| v.get(k).and_then(Value::as_u64).map_or(d, |x| x as usize);
    c.d_model = u(enc, "hidden_size", c.d_model);
    c.n_layers = u(enc, "num_hidden_layers", c.n_layers);
    c.n_heads = u(enc, "num_attention_heads", c.n_heads);
    c.ff_dim = u(enc, "intermediate_size", c.ff_dim);
    c.conv_kernel = u(enc, "conv_kernel_size", c.conv_kernel);
    c.subsampling_factor = u(enc, "subsampling_factor", c.subsampling_factor);
    c.subsampling_channels = u(enc, "subsampling_conv_channels", c.subsampling_channels);
    c.n_mels = u(enc, "num_mel_bins", c.n_mels);
    c.decoder_hidden = u(cfg, "decoder_hidden_size", c.decoder_hidden);
    c.decoder_layers = u(cfg, "num_decoder_layers", c.decoder_layers);
    c.vocab_size = u(cfg, "vocab_size", c.vocab_size);
    c.blank_id = u(cfg, "blank_token_id", c.blank_id as usize) as u32;
    c.max_symbols_per_step = u(cfg, "max_symbols_per_step", c.max_symbols_per_step);
    if let Some(d) = cfg.get("durations").and_then(Value::as_array) {
        c.durations = d.iter().filter_map(Value::as_u64).map(|x| x as usize).collect();
    }
    if let Some(a) = enc.get("hidden_act").and_then(Value::as_str).filter(|a| *a != "silu") {
        bail!("unsupported_model", "encoder hidden_act {a:?} (only \"silu\" is implemented)");
    }
    if let Some(a) = cfg.get("hidden_act").and_then(Value::as_str).filter(|a| *a != "relu") {
        bail!("unsupported_model", "joint activation {a:?} (only \"relu\" is implemented)");
    }
    if enc.get("scale_input").and_then(Value::as_bool) == Some(true) {
        bail!("unsupported_model", "encoder scale_input = true is not implemented");
    }
    if let Some(fe) = processor.and_then(|p| p.get("feature_extractor")) {
        c.sample_rate = u(fe, "sampling_rate", c.sample_rate as usize) as u32;
        c.n_fft = u(fe, "n_fft", c.n_fft);
        c.hop_length = u(fe, "hop_length", c.hop_length);
        c.win_length = u(fe, "win_length", c.win_length);
        c.n_mels = u(fe, "feature_size", c.n_mels);
        c.preemphasis = fe.get("preemphasis").and_then(Value::as_f64).unwrap_or(c.preemphasis);
    }
    c.validate()?;
    Ok(c)
}

/// `ParakeetConfig` from the GGUF `asr.*` metadata.
pub(crate) fn config_from_gguf(f: &GgufFile) -> Result<ParakeetConfig> {
    let mut c = ParakeetConfig::v3();
    let u = |key: &str, d: usize| f.get(key).and_then(GgufValue::as_u64).map_or(d, |v| v as usize);
    c.d_model = u("asr.encoder.d_model", c.d_model);
    c.n_layers = u("asr.encoder.n_layers", c.n_layers);
    c.n_heads = u("asr.encoder.n_heads", c.n_heads);
    c.ff_dim = u("asr.encoder.d_ff", c.ff_dim);
    c.conv_kernel = u("asr.encoder.conv_kernel_size", c.conv_kernel);
    c.subsampling_factor = u("asr.encoder.subsampling_factor", c.subsampling_factor);
    c.subsampling_channels = u("asr.encoder.subsampling_conv_channels", c.subsampling_channels);
    c.n_mels = u("asr.encoder.feat_in", c.n_mels);
    c.sample_rate = u("asr.preprocessor.sample_rate", c.sample_rate as usize) as u32;
    c.n_fft = u("asr.preprocessor.n_fft", c.n_fft);
    let secs = |k: &str| f.get(k).and_then(GgufValue::as_f64);
    if let Some(w) = secs("asr.preprocessor.window_size") {
        c.win_length = (w * f64::from(c.sample_rate)).round() as usize;
    }
    if let Some(h) = secs("asr.preprocessor.window_stride") {
        c.hop_length = (h * f64::from(c.sample_rate)).round() as usize;
    }
    if let Some(p) = secs("asr.preprocessor.preemph") {
        c.preemphasis = (p * 1e6).round() / 1e6; // stored as f32
    }
    c.vocab_size = u("asr.rnnt.vocab_size", c.vocab_size);
    c.blank_id = u("asr.rnnt.blank_id", c.blank_id as usize) as u32;
    c.decoder_hidden = u("asr.rnnt.pred_hidden", c.decoder_hidden);
    c.decoder_layers = u("asr.rnnt.pred_num_layers", c.decoder_layers);
    c.max_symbols_per_step = u("asr.rnnt.max_symbols_per_step", c.max_symbols_per_step);
    if let Some(d) = f.get("asr.tdt.durations").and_then(GgufValue::as_array) {
        c.durations = d.iter().filter_map(GgufValue::as_u64).map(|x| x as usize).collect();
    }
    if f.get("asr.encoder.xscaling").and_then(GgufValue::as_u64) == Some(1) {
        bail!("unsupported_model", "GGUF: encoder xscaling is not implemented");
    }
    if let Some(n) = f.get("asr.preprocessor.normalize").and_then(GgufValue::as_str).filter(|n| *n != "per_feature") {
        bail!("unsupported_model", "GGUF: feature normalization {n:?} (only per_feature)");
    }
    c.validate()?;
    Ok(c)
}

/// A decode-capable `tokenizer.json` (SentencePiece-style BPE, Metaspace) from the GGUF
/// vocabulary (no merges are stored, so encoding falls back to whole pieces).
fn synth_tokenizer(f: &GgufFile, blank_id: u32) -> Result<Vec<u8>> {
    let vocab = f.get("asr.tokenizer.vocab").and_then(GgufValue::as_array).ok_or_else(|| Error::new("tokenizer", "GGUF has no asr.tokenizer.vocab"))?;
    let added_token = |id: u64, content: &str, special: bool| {
        json!({"id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": special})
    };
    let (mut map, mut added) = (Map::new(), Vec::new());
    for (id, v) in vocab.iter().enumerate() {
        let piece = v.as_str().ok_or_else(|| Error::new("tokenizer", format!("vocab entry {id} is not a string")))?;
        map.insert(piece.to_string(), json!(id));
        if piece.starts_with('<') && piece.ends_with('>') && piece.len() > 2 {
            added.push(added_token(id as u64, piece, matches!(piece, "<unk>" | "<pad>")));
        }
    }
    added.push(added_token(u64::from(blank_id), "<blank>", true));
    let meta = json!({"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true});
    let tj = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": null, "pre_tokenizer": meta, "post_processor": null, "decoder": meta,
        "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null,
                  "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": false, "ignore_merges": false,
                  "vocab": Value::Object(map), "merges": []},
    });
    Ok(serde_json::to_vec(&tj)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lapid::quant::Format;
    use lapid_convert::OutDType;
    use lapid_convert::mapping::{Renamed, rename};
    use lapid_convert::policy::{Target, target};

    #[test]
    fn policy_names_and_config() {
        let t = |n: &str, s: &[usize], d| target(CONVERT.role(n, s), s, d);
        assert_eq!(t("encoder.layers.0.self_attn.q_proj.weight", &[1024, 1024], OutDType::Int4), Target::Quant(Format::Int4));
        assert_eq!(t("joint.head.weight", &[8198, 640], OutDType::Int4), Target::Quant(Format::Int8));
        assert_eq!(t("encoder.layers.0.self_attn.bias_u", &[8, 128], OutDType::Int8), Target::F32);
        assert_eq!(t("encoder.layers.0.conv.depthwise_conv.weight", &[1024, 9], OutDType::Int8), Target::F16);
        assert_eq!(t("encoder.subsampling.layers.3.weight", &[256, 256], OutDType::Int4), Target::Quant(Format::Int8));
        assert_eq!(t("encoder.subsampling.layers.2.weight", &[256, 9], OutDType::Int8), Target::F16);
        assert_eq!(t("decoder.embedding.weight", &[8193, 640], OutDType::Int8), Target::F16);
        assert_eq!(t("decoder.lstm.weight_ih_l0", &[2560, 640], OutDType::Int8), Target::Quant(Format::Int8));
        let n = expected_tensors(&ParakeetConfig::v3());
        assert_eq!(n.len(), 10 + 2 + 24 * (10 + 4 + 7 + 4) + 2 + 1 + 8 + 4);
        assert_eq!(subsampling_layers(3), vec![(0, false), (2, true), (3, false), (5, true), (6, false)]);
        let m = |g: &str| rename(CONVERT.gguf_names, g);
        assert_eq!(m("encoder.pre_encode.conv.5.bias"), Renamed::To("encoder.subsampling.layers.5.bias".into()));
        assert_eq!(m("encoder.layers.3.self_attn.linear_pos.weight"), Renamed::To("encoder.layers.3.self_attn.relative_k_proj.weight".into()));
        assert_eq!(m("encoder.layers.3.self_attn.pos_bias_v"), Renamed::To("encoder.layers.3.self_attn.bias_v".into()));
        assert_eq!(m("encoder.layers.3.conv.batch_norm.running_var"), Renamed::To("encoder.layers.3.conv.norm.running_var".into()));
        assert_eq!(m("encoder.layers.3.norm_out.bias"), Renamed::To("encoder.layers.3.norm_out.bias".into()));
        assert_eq!(m("encoder.layers.3.feed_forward1.linear1.weight"), Renamed::To("encoder.layers.3.feed_forward1.linear1.weight".into()));
        assert_eq!(m("decoder.prediction.dec_rnn.lstm.hh_l1.bias"), Renamed::To("decoder.lstm.bias_hh_l1".into()));
        assert_eq!(m("joint.joint_net.2.weight"), Renamed::To("joint.head.weight".into()));
        assert_eq!((m("encoder.pos_enc.pe"), m("preprocessor.fb")), (Renamed::Drop, Renamed::Drop));
        let cfg = json!({
            "model_type": "parakeet_tdt", "blank_token_id": 8192, "decoder_hidden_size": 640, "durations": [0, 1, 2, 3, 4],
            "hidden_act": "relu", "max_symbols_per_step": 10, "num_decoder_layers": 2, "vocab_size": 8193,
            "encoder_config": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 8, "intermediate_size": 4096,
                "conv_kernel_size": 9, "subsampling_factor": 8, "subsampling_conv_channels": 256, "num_mel_bins": 128, "hidden_act": "silu"}
        });
        let proc = json!({"feature_extractor": {"feature_size": 128, "hop_length": 160, "n_fft": 512, "preemphasis": 0.97, "sampling_rate": 16000, "win_length": 400}});
        assert_eq!(config_from_hf(&cfg, Some(&proc)).expect("cfg"), ParakeetConfig::v3());
    }
}
