//! NeMo-named Parakeet TDT GGUF (`general.architecture = "asr"`, `asr.head_type = "tdt"`,
//! as published next to `nvidia/parakeet-tdt-0.6b-v3`) → `.lapid`.
//!
//! Tensor names follow NeMo (`encoder.pre_encode.*`, `self_attn.linear_q`, `conv.batch_norm`,
//! `decoder.prediction.*`, `joint.{enc,pred,joint_net.2}`) and are mapped to the HF names
//! the model loads. Q8_0 / Q4_0 matmul weights are repacked losslessly when the output
//! dtype matches (`int8` for a q8_0 file); the precomputed positional table
//! (`encoder.pos_enc.pe`) and mel filterbank (`preprocessor.fb`) are not needed. The GGUF
//! carries only the vocabulary, so a decode-capable `tokenizer.json` is synthesized from it.

use std::path::Path;

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::gguf::{GgufFile, GgufTensor, GgufValue};
use lapid_convert::writer::{ModelWriter, file_hash};
use serde_json::{Map, Value, json};

use super::{add_tokenizer_blob, expected_tensors, fold_batch_norm, metadata, role, target_shape};
use crate::parakeet::config::ParakeetConfig;

/// True for a GGUF Parakeet TDT export (`general.architecture = "asr"`, head type "tdt").
pub(crate) fn is_parakeet_gguf(f: &GgufFile) -> bool {
    f.architecture() == Some("asr") && f.get("asr.head_type").and_then(GgufValue::as_str) == Some("tdt")
}

/// NeMo GGUF name → `.lapid` (HF) name; `None` for tensors that are not carried over.
pub(crate) fn map_name(n: &str) -> Option<String> {
    if n == "encoder.pos_enc.pe" || n.starts_with("preprocessor.") || n.contains(".batch_norm.") {
        return None;
    }
    if let Some(r) = n.strip_prefix("encoder.pre_encode.out.") {
        return Some(format!("encoder.subsampling.linear.{r}"));
    }
    if let Some(r) = n.strip_prefix("encoder.pre_encode.conv.") {
        return Some(format!("encoder.subsampling.layers.{r}"));
    }
    if n.starts_with("encoder.layers.") {
        let pairs = [
            (".self_attn.pos_bias_u", ".self_attn.bias_u"),
            (".self_attn.pos_bias_v", ".self_attn.bias_v"),
            (".self_attn.linear_q.", ".self_attn.q_proj."),
            (".self_attn.linear_k.", ".self_attn.k_proj."),
            (".self_attn.linear_v.", ".self_attn.v_proj."),
            (".self_attn.linear_out.", ".self_attn.o_proj."),
            (".self_attn.linear_pos.", ".self_attn.relative_k_proj."),
        ];
        for (from, to) in pairs {
            if n.contains(from) {
                return Some(n.replacen(from, to, 1));
            }
        }
        return Some(n.to_string());
    }
    if n == "decoder.prediction.embed.weight" {
        return Some("decoder.embedding.weight".into());
    }
    if let Some(r) = n.strip_prefix("decoder.prediction.dec_rnn.lstm.") {
        // {ih,hh}_l{L}.{weight,bias} -> {weight,bias}_{ih,hh}_l{L}
        let (gate, kind) = r.split_once('.')?;
        return Some(format!("decoder.lstm.{kind}_{gate}"));
    }
    let fixed = [
        ("joint.pred.", "decoder.decoder_projector."),
        ("joint.enc.", "encoder_projector."),
        ("joint.joint_net.2.", "joint.head."),
    ];
    for (from, to) in fixed {
        if let Some(r) = n.strip_prefix(from) {
            return Some(format!("{to}{r}"));
        }
    }
    Some(n.to_string())
}

fn meta_u(f: &GgufFile, key: &str, default: usize) -> usize {
    f.get(key).and_then(GgufValue::as_u64).map_or(default, |v| v as usize)
}

/// `ParakeetConfig` from the `asr.*` metadata.
pub(crate) fn config_from_gguf(f: &GgufFile) -> Result<ParakeetConfig> {
    let mut c = ParakeetConfig::v3();
    c.d_model = meta_u(f, "asr.encoder.d_model", c.d_model);
    c.n_layers = meta_u(f, "asr.encoder.n_layers", c.n_layers);
    c.n_heads = meta_u(f, "asr.encoder.n_heads", c.n_heads);
    c.ff_dim = meta_u(f, "asr.encoder.d_ff", c.ff_dim);
    c.conv_kernel = meta_u(f, "asr.encoder.conv_kernel_size", c.conv_kernel);
    c.subsampling_factor = meta_u(f, "asr.encoder.subsampling_factor", c.subsampling_factor);
    c.subsampling_channels = meta_u(f, "asr.encoder.subsampling_conv_channels", c.subsampling_channels);
    c.n_mels = meta_u(f, "asr.encoder.feat_in", c.n_mels);
    c.sample_rate = meta_u(f, "asr.preprocessor.sample_rate", c.sample_rate as usize) as u32;
    c.n_fft = meta_u(f, "asr.preprocessor.n_fft", c.n_fft);
    let secs = |k: &str| f.get(k).and_then(GgufValue::as_f64);
    if let Some(w) = secs("asr.preprocessor.window_size") {
        c.win_length = (w * c.sample_rate as f64).round() as usize;
    }
    if let Some(h) = secs("asr.preprocessor.window_stride") {
        c.hop_length = (h * c.sample_rate as f64).round() as usize;
    }
    if let Some(p) = secs("asr.preprocessor.preemph") {
        c.preemphasis = (p * 1e6).round() / 1e6; // stored as f32
    }
    c.vocab_size = meta_u(f, "asr.rnnt.vocab_size", c.vocab_size);
    c.blank_id = meta_u(f, "asr.rnnt.blank_id", c.blank_id as usize) as u32;
    c.decoder_hidden = meta_u(f, "asr.rnnt.pred_hidden", c.decoder_hidden);
    c.decoder_layers = meta_u(f, "asr.rnnt.pred_num_layers", c.decoder_layers);
    c.max_symbols_per_step = meta_u(f, "asr.rnnt.max_symbols_per_step", c.max_symbols_per_step);
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

/// A `tokenizer.json` (SentencePiece-style BPE, Metaspace) built from the GGUF vocabulary:
/// enough to decode ids; encoding falls back to whole pieces (no merges are stored).
pub(crate) fn synth_tokenizer(f: &GgufFile, blank_id: u32) -> Result<Vec<u8>> {
    let vocab = f
        .get("asr.tokenizer.vocab")
        .and_then(GgufValue::as_array)
        .ok_or_else(|| Error::new("tokenizer", "GGUF has no asr.tokenizer.vocab"))?;
    let mut map = Map::new();
    let mut added = Vec::new();
    for (id, v) in vocab.iter().enumerate() {
        let piece = v.as_str().ok_or_else(|| Error::new("tokenizer", format!("vocab entry {id} is not a string")))?;
        map.insert(piece.to_string(), json!(id));
        if piece.starts_with('<') && piece.ends_with('>') && piece.len() > 2 {
            let special = matches!(piece, "<unk>" | "<pad>");
            added.push(json!({"id": id, "content": piece, "single_word": false, "lstrip": false, "rstrip": false,
                              "normalized": false, "special": special}));
        }
    }
    added.push(json!({"id": blank_id, "content": "<blank>", "single_word": false, "lstrip": false, "rstrip": false,
                      "normalized": false, "special": true}));
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

pub(crate) fn convert(path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    let f = GgufFile::open(path)?;
    if !is_parakeet_gguf(&f) {
        bail!("unsupported_model", "{}: not a Parakeet TDT GGUF (asr / tdt)", path.display());
    }
    let cfg = config_from_gguf(&f)?;
    let expected = expected_tensors(&cfg);
    let tensor = |n: &str| -> Result<&GgufTensor> {
        f.tensor(n).ok_or_else(|| Error::new("missing_tensor", format!("{}: GGUF has no {n:?}", path.display())))
    };

    let gname = f.get("general.name").and_then(GgufValue::as_str).unwrap_or("parakeet-tdt").to_string();
    let source = json!({"repo": path.display().to_string(), "format": "gguf", "file_hash": file_hash(path)?, "gguf_name": gname});
    let meta = metadata(&gname, dtype, "cc-by-4.0", source, &cfg);
    let mut w = ModelWriter::create(out, meta, dtype, role, expected.len())?;
    add_tokenizer_blob(&mut w, &synth_tokenizer(&f, cfg.blank_id)?)?;

    for t in &f.tensors {
        let Some(name) = map_name(&t.name) else { continue };
        if let Some(p) = t.name.strip_suffix(".conv.depthwise_conv.weight") {
            let bn = |k: &str| -> Result<Vec<f32>> { f.to_f32(tensor(&format!("{p}.conv.batch_norm.{k}"))?) };
            let bias = match f.tensor(&format!("{p}.conv.depthwise_conv.bias")) {
                Some(b) => Some(f.to_f32(b)?),
                None => None,
            };
            let (fw, fb) = fold_batch_norm(&f.to_f32(t)?, bias.as_deref(), &bn("weight")?, &bn("bias")?, &bn("running_mean")?, &bn("running_var")?)?;
            w.add_f32(&name, &target_shape(&name, &t.shape), &fw)?;
            w.add_f32(&name.replace(".weight", ".bias"), &[fb.len()], &fb)?;
            continue;
        }
        if t.name.ends_with(".conv.depthwise_conv.bias") {
            continue; // folded above
        }
        let shape = target_shape(&name, &t.shape);
        f.write_tensor(&mut w, &name, t, Some(&shape))?;
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let m = |n: &str| map_name(n);
        assert_eq!(m("encoder.pre_encode.out.weight").as_deref(), Some("encoder.subsampling.linear.weight"));
        assert_eq!(m("encoder.pre_encode.conv.5.bias").as_deref(), Some("encoder.subsampling.layers.5.bias"));
        assert_eq!(m("encoder.layers.3.self_attn.linear_pos.weight").as_deref(), Some("encoder.layers.3.self_attn.relative_k_proj.weight"));
        assert_eq!(m("encoder.layers.3.self_attn.pos_bias_v").as_deref(), Some("encoder.layers.3.self_attn.bias_v"));
        assert_eq!(m("encoder.layers.3.conv.batch_norm.running_var"), None);
        assert_eq!(m("encoder.layers.3.norm_out.bias").as_deref(), Some("encoder.layers.3.norm_out.bias"));
        assert_eq!(m("decoder.prediction.dec_rnn.lstm.hh_l1.bias").as_deref(), Some("decoder.lstm.bias_hh_l1"));
        assert_eq!(m("decoder.prediction.embed.weight").as_deref(), Some("decoder.embedding.weight"));
        assert_eq!(m("joint.joint_net.2.weight").as_deref(), Some("joint.head.weight"));
        assert_eq!(m("joint.pred.bias").as_deref(), Some("decoder.decoder_projector.bias"));
        assert_eq!(m("encoder.pos_enc.pe"), None);
        assert_eq!(m("preprocessor.fb"), None);
    }
}
