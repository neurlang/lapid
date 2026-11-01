//! Whisper checkpoints -> `.lapid`: Hugging Face (`WhisperForConditionalGeneration`) through
//! the generic engine ([`CONVERT`]), and whisper.cpp `ggml-*.bin` ([`convert_ggml`], the
//! legacy ggml container only whisper.cpp uses).
//!
//! Both emit HF tensor names and the same `"whisper"` metadata object (hyper-parameters,
//! special tokens, languages, suppress lists) plus the id-ordered token strings as the blob
//! [`TOKENS_BLOB`] (a JSON array, GPT-2 byte-level form, specials verbatim). Precision:
//! Linear weights quantized; `model.decoder.embed_tokens.weight` (also the output projection)
//! at most int8; conv weights and positional embeddings f16; norms and biases f32.

mod cpp;
mod ggml;
mod tokens;

use std::collections::BTreeMap;
use std::path::Path;

use lapid::{Error, LapidFile, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::assets::HfFiles;
use lapid_convert::engine::{ConvertHooks, ConvertSpec, HfExtras};
use lapid_convert::mapping::{RoleRule, Rule};
use lapid_convert::policy::Role;
use lapid_convert::writer::read_json;
use serde_json::{Map, Value, json};

pub(crate) use ggml::is_ggml_file;
use tokens::Specials;

/// Token embedding; also the (tied) output projection.
pub(crate) const EMBED_TOKENS: &str = "model.decoder.embed_tokens.weight";
/// Blob holding the id-ordered token strings.
pub(crate) const TOKENS_BLOB: &str = "tokenizer/tokens.json";

pub static CONVERT: ConvertSpec = ConvertSpec {
    architecture: "whisper",
    license: "unknown",
    hf_model_types: &["whisper"],
    gguf_architectures: &[],
    hf_names: &[
        Rule::drop("proj_out.weight"),
        Rule::new("encoder.{a}.{b}", "model.encoder.{a}.{b}"),
        Rule::new("encoder.{a}.{b}.{c}", "model.encoder.{a}.{b}.{c}"),
        Rule::new("encoder.{a}.{b}.{c}.{d}", "model.encoder.{a}.{b}.{c}.{d}"),
        Rule::new("encoder.{a}.{b}.{c}.{d}.{e}", "model.encoder.{a}.{b}.{c}.{d}.{e}"),
        Rule::new("decoder.{a}.{b}", "model.decoder.{a}.{b}"),
        Rule::new("decoder.{a}.{b}.{c}", "model.decoder.{a}.{b}.{c}"),
        Rule::new("decoder.{a}.{b}.{c}.{d}", "model.decoder.{a}.{b}.{c}.{d}"),
        Rule::new("decoder.{a}.{b}.{c}.{d}.{e}", "model.decoder.{a}.{b}.{c}.{d}.{e}"),
    ],
    gguf_names: &[],
    roles: &[RoleRule::new(EMBED_TOKENS, Role::LinearInt8), RoleRule::new("*embed_positions*", Role::Dense), RoleRule::new("*conv*", Role::Dense)],
    text_assets: false,
    hooks: &Hooks,
};

struct Hooks;

impl ConvertHooks for Hooks {
    /// Tokens and generation config complete the `"whisper"` object.
    fn hf_extras(&self, files: &HfFiles, cfg: &mut Value) -> Result<HfExtras> {
        let gen_cfg = files.optional("generation_config.json")?.as_deref().map(read_json).transpose()?;
        let tokens = load_tokens(files, cfg_usize(cfg, "vocab_size")?)?;
        let ident = if files.remote { files.repo.clone() } else { cfg["_name_or_path"].as_str().unwrap_or("").to_string() };
        let license = if ident.starts_with("openai/whisper") || files.repo.starts_with("openai/whisper") { "mit" } else { "unknown" };
        *cfg = whisper_meta(cfg, gen_cfg.as_ref(), &tokens)?;
        let mut metadata = Map::new();
        metadata.insert("license".into(), json!(license));
        metadata.insert("components".into(), components(license));
        Ok(HfExtras { blobs: vec![(TOKENS_BLOB.into(), "application/json".into(), serde_json::to_vec(&tokens)?)], metadata, ..HfExtras::default() })
    }

    fn expected(&self, cfg: &Value) -> Result<Vec<String>> {
        Ok(expected_tensors(cfg_usize(cfg, "n_audio_layer")?, cfg_usize(cfg, "n_text_layer")?))
    }
}

fn components(license: &str) -> Value {
    json!({"whisper": {"architecture": "whisper", "config": "whisper", "license": license, "tensor_prefix": ""}})
}

/// Convert a whisper.cpp `ggml-*.bin` model.
pub fn convert_ggml(ggml_bin: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    cpp::convert(ggml_bin, dtype, out)
}

/// Required `.lapid` metadata with the `"whisper"` object (ggml path).
fn metadata(name: &str, dtype: OutDType, license: &str, source: Value, mut whisper: Value) -> Value {
    let mut m = lapid_convert::writer::metadata(name, "whisper", dtype, license, source);
    whisper["tokens_blob"] = json!(TOKENS_BLOB);
    m["whisper"] = whisper;
    m["components"] = components(license);
    m
}

/// All tensor names a Whisper checkpoint must provide (HF naming).
pub(crate) fn expected_tensors(n_audio_layer: usize, n_text_layer: usize) -> Vec<String> {
    let mut v: Vec<String> = ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "embed_positions.weight", "layer_norm.weight", "layer_norm.bias"]
        .iter()
        .map(|s| format!("model.encoder.{s}"))
        .collect();
    v.extend([EMBED_TOKENS, "model.decoder.embed_positions.weight", "model.decoder.layer_norm.weight", "model.decoder.layer_norm.bias"].map(String::from));
    let attn = |p: &str, v: &mut Vec<String>| {
        for (proj, bias) in [("q_proj", true), ("k_proj", false), ("v_proj", true), ("out_proj", true)] {
            v.push(format!("{p}.{proj}.weight"));
            if bias {
                v.push(format!("{p}.{proj}.bias"));
            }
        }
        v.extend([format!("{p}_layer_norm.weight"), format!("{p}_layer_norm.bias")]);
    };
    let mlp = |p: &str, v: &mut Vec<String>| v.extend(["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "final_layer_norm.weight", "final_layer_norm.bias"].map(|s| format!("{p}.{s}")));
    for i in 0..n_audio_layer {
        let p = format!("model.encoder.layers.{i}");
        attn(&format!("{p}.self_attn"), &mut v);
        mlp(&p, &mut v);
    }
    for i in 0..n_text_layer {
        let p = format!("model.decoder.layers.{i}");
        attn(&format!("{p}.self_attn"), &mut v);
        attn(&format!("{p}.encoder_attn"), &mut v);
        mlp(&p, &mut v);
    }
    v
}

fn cfg_usize(cfg: &Value, key: &str) -> Result<usize> {
    cfg.get(key).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| Error::new("missing_metadata", format!("config: missing integer {key:?}")))
}

fn id_map(v: &Value, what: &str) -> Result<Vec<(String, u64)>> {
    let m = v.as_object().ok_or_else(|| Error::new("tokenizer", format!("{what}: expected a JSON object")))?;
    m.iter().map(|(k, id)| id.as_u64().map(|id| (k.clone(), id)).ok_or_else(|| Error::new("tokenizer", format!("{what}: non-integer id for {k:?}")))).collect()
}

/// Id-ordered token strings (byte-level BPE form, specials verbatim) from `vocab.json` +
/// `added_tokens.json`, or `tokenizer.json`.
fn load_tokens(files: &HfFiles, n_vocab: usize) -> Result<Vec<String>> {
    let mut pairs: Vec<(String, u64)> = Vec::new();
    if let Some(v) = files.optional("vocab.json")? {
        pairs.extend(id_map(&read_json(&v)?, "vocab.json")?);
        if let Some(a) = files.optional("added_tokens.json")? {
            pairs.extend(id_map(&read_json(&a)?, "added_tokens.json")?);
        }
    } else {
        let tj = read_json(&files.required("tokenizer.json")?)?;
        pairs.extend(id_map(&tj["model"]["vocab"], "tokenizer.json model.vocab")?);
        for a in tj["added_tokens"].as_array().into_iter().flatten() {
            if let (Some(id), Some(c)) = (a["id"].as_u64(), a["content"].as_str()) {
                pairs.push((c.to_string(), id));
            }
        }
    }
    let mut table: Vec<Option<String>> = vec![None; n_vocab];
    for (s, id) in pairs {
        if let Some(slot) = table.get_mut(id as usize) {
            *slot = Some(s);
        }
    }
    Ok(table.into_iter().enumerate().map(|(i, t)| t.unwrap_or_else(|| format!("<|unused_{i}|>"))).collect())
}

fn u32_list(v: Option<&Value>) -> Option<Vec<u32>> {
    v?.as_array()?.iter().map(|x| x.as_u64().map(|x| x as u32)).collect()
}

/// The `"whisper"` metadata object from HF `config.json`, `generation_config.json` and tokens.
fn whisper_meta(cfg: &Value, gen_cfg: Option<&Value>, tokens: &[String]) -> Result<Value> {
    let n_vocab = cfg_usize(cfg, "vocab_size")?;
    let lookup: BTreeMap<&str, u32> = tokens.iter().enumerate().map(|(i, s)| (s.as_str(), i as u32)).collect();
    let tok = |s: &str| lookup.get(s).copied();
    let gen_u32 = |k: &str| gen_cfg.and_then(|g| g.get(k)).and_then(Value::as_u64).map(|v| v as u32);
    let cfg_u32 = |k: &str| cfg.get(k).and_then(Value::as_u64).map(|v| v as u32);
    let need = |v: Option<u32>, what: &str| v.ok_or_else(|| Error::new("tokenizer", format!("cannot determine {what} token id")));
    let task = |t: &str| gen_cfg.and_then(|g| g.pointer(&format!("/task_to_id/{t}"))).and_then(Value::as_u64).map(|v| v as u32);

    let sot = need(gen_u32("decoder_start_token_id").or(cfg_u32("decoder_start_token_id")).or(tok("<|startoftranscript|>")), "sot")?;
    let eot = need(tok("<|endoftext|>").or(gen_u32("eos_token_id")).or(cfg_u32("eos_token_id")), "eot")?;
    let transcribe = need(task("transcribe").or(tok("<|transcribe|>")), "transcribe")?;
    let translate = need(task("translate").or(tok("<|translate|>")), "translate")?;
    let no_timestamps = need(gen_u32("no_timestamps_token_id").or(tok("<|notimestamps|>")), "no_timestamps")?;
    let prev = need(gen_u32("prev_sot_token_id").or(tok("<|startofprev|>")), "prev")?;
    let no_speech = tok("<|nocaptions|>").or(tok("<|nospeech|>")).unwrap_or(no_timestamps.saturating_sub(1));
    let mut special = json!({
        "sot": sot, "eot": eot, "transcribe": transcribe, "translate": translate,
        "no_timestamps": no_timestamps, "timestamp_begin": no_timestamps + 1, "no_speech": no_speech, "prev": prev,
    });
    if let Some(lm) = tok("<|startoflm|>") {
        special["sot_lm"] = json!(lm);
    }
    let multilingual = gen_cfg.and_then(|g| g.get("is_multilingual")).and_then(Value::as_bool).unwrap_or(n_vocab >= 51865);
    let mut languages = Map::new();
    if let Some(l2i) = gen_cfg.and_then(|g| g.get("lang_to_id")).and_then(Value::as_object) {
        for (k, v) in l2i {
            if let Some(id) = v.as_u64() {
                languages.insert(k.trim_start_matches("<|").trim_end_matches("|>").to_string(), json!(id));
            }
        }
    } else if multilingual {
        // language tokens sit between sot and translate
        for id in sot + 1..translate.min(transcribe) {
            let Some(s) = tokens.get(id as usize) else { break };
            if let Some(code) = s.strip_prefix("<|").and_then(|c| c.strip_suffix("|>"))
                && (2..=3).contains(&code.len())
                && code.bytes().all(|b| b.is_ascii_lowercase())
            {
                languages.insert(code.to_string(), json!(id));
            }
        }
    }
    let pick = |k: &str| u32_list(gen_cfg.and_then(|g| g.get(k))).or_else(|| u32_list(cfg.get(k)));
    let suppress = pick("suppress_tokens").unwrap_or_else(|| Specials::from_n_vocab(n_vocab as u32).suppress_tokens());
    let begin_suppress = pick("begin_suppress_tokens").unwrap_or_else(|| vec![220, eot]);
    let d_model = cfg_usize(cfg, "d_model")?;
    let mut w = json!({
        "n_mels": cfg_usize(cfg, "num_mel_bins")?,
        "n_audio_ctx": cfg_usize(cfg, "max_source_positions")?,
        "n_audio_state": d_model,
        "n_audio_head": cfg_usize(cfg, "encoder_attention_heads")?,
        "n_audio_layer": cfg_usize(cfg, "encoder_layers")?,
        "n_audio_ffn": cfg_usize(cfg, "encoder_ffn_dim").unwrap_or(4 * d_model),
        "n_text_ctx": cfg_usize(cfg, "max_target_positions")?,
        "n_text_state": d_model,
        "n_text_head": cfg_usize(cfg, "decoder_attention_heads")?,
        "n_text_layer": cfg_usize(cfg, "decoder_layers")?,
        "n_text_ffn": cfg_usize(cfg, "decoder_ffn_dim").unwrap_or(4 * d_model),
        "n_vocab": n_vocab,
        "multilingual": multilingual,
        "activation": cfg.get("activation_function").and_then(Value::as_str).unwrap_or("gelu"),
        "special": special,
        "languages": languages,
        "suppress_tokens": suppress,
        "begin_suppress_tokens": begin_suppress,
        "tokens_blob": TOKENS_BLOB,
    });
    for key in ["alignment_heads", "max_initial_timestamp_index"] {
        if let Some(v) = gen_cfg.and_then(|g| g.get(key)).filter(|v| v.is_array() || v.is_u64()) {
            w[key] = v.clone();
        }
    }
    Ok(w)
}

/// Id-ordered token strings of a Whisper `.lapid` file: the `whisper.tokens_blob` blob, or
/// for format 1.0 files the metadata array `whisper.tokens`; checked against `n_vocab` / eot.
pub fn load_token_strings(file: &LapidFile) -> Result<Vec<String>> {
    let w = file.metadata.get("whisper");
    let meta_err = |m: String| Error::new("metadata", format!("whisper: {m}"));
    let tokens: Vec<String> = match w.and_then(|w| w.get("tokens")) {
        Some(Value::Array(arr)) => arr.iter().map(|v| v.as_str().map(str::to_string).ok_or_else(|| meta_err("token is not a string".into()))).collect::<Result<_>>()?,
        Some(_) => bail!("metadata", "whisper: metadata 'tokens' is not an array"),
        None => {
            let name = w.and_then(|w| w.get("tokens_blob")).and_then(Value::as_str).unwrap_or(TOKENS_BLOB);
            let bytes = file.blob(name).ok_or_else(|| meta_err(format!("no token list (blob {name:?} missing)")))?;
            serde_json::from_slice(bytes).map_err(|e| meta_err(format!("blob {name:?}: {e}")))?
        }
    };
    if let Some(n) = w.and_then(|w| w.get("n_vocab")).and_then(Value::as_u64).filter(|&n| tokens.len() as u64 != n) {
        bail!("metadata", "whisper: {} token strings, n_vocab is {n}", tokens.len());
    }
    if let Some(eot) = w.and_then(|w| w.pointer("/special/eot")).and_then(Value::as_u64).filter(|&e| e as usize > tokens.len()) {
        bail!("metadata", "whisper: eot id {eot} outside the token list ({} tokens)", tokens.len());
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lapid::quant::Format;
    use lapid_convert::policy::Target;

    #[test]
    fn policy_and_names() {
        let t = |name: &str, shape: &[usize], d| lapid_convert::policy::target(CONVERT.role(name, shape), shape, d);
        let q = "model.encoder.layers.0.self_attn.q_proj.weight";
        assert_eq!(t(q, &[768, 768], OutDType::Int4), Target::Quant(Format::Int4));
        assert_eq!(t(EMBED_TOKENS, &[51865, 768], OutDType::Int4), Target::Quant(Format::Int8));
        assert_eq!(t("model.encoder.conv1.weight", &[768, 80, 3], OutDType::Int8), Target::F16);
        assert_eq!(t("model.encoder.embed_positions.weight", &[1500, 768], OutDType::Int4), Target::F16);
        assert_eq!(t("model.decoder.layers.0.fc1.bias", &[3072], OutDType::Int8), Target::F32);
        let n = expected_tensors(12, 12);
        assert_eq!(n.len(), 11 + 12 * 15 + 12 * 24);
        assert!(!n.contains(&"model.encoder.layers.0.self_attn.k_proj.bias".to_string()));
        use lapid_convert::mapping::{Renamed, rename};
        assert_eq!(rename(CONVERT.hf_names, "encoder.layers.0.fc1.weight"), Renamed::To("model.encoder.layers.0.fc1.weight".into()));
        assert_eq!(rename(CONVERT.hf_names, "model.encoder.conv1.weight"), Renamed::NoMatch);
    }

    #[test]
    fn meta_without_generation_config() {
        let cfg = json!({
            "model_type": "whisper", "vocab_size": 51865, "d_model": 768, "num_mel_bins": 80,
            "max_source_positions": 1500, "max_target_positions": 448, "encoder_attention_heads": 12,
            "decoder_attention_heads": 12, "encoder_layers": 12, "decoder_layers": 12,
            "decoder_start_token_id": 50258, "eos_token_id": 50257, "suppress_tokens": [1, 2]
        });
        let s = Specials::from_n_vocab(51865);
        let mut tokens: Vec<String> = (0..51865).map(|i| format!("t{i}")).collect();
        for (id, st) in s.token_strings(51865) {
            tokens[id as usize] = st;
        }
        let w = whisper_meta(&cfg, None, &tokens).expect("meta");
        assert_eq!((w["special"]["sot"].clone(), w["special"]["timestamp_begin"].clone()), (json!(50258), json!(50364)));
        assert_eq!((w["languages"]["en"].clone(), w["languages"]["su"].clone()), (json!(50259), json!(50357)));
        assert_eq!(w["suppress_tokens"], json!([1, 2]));
        assert_eq!(w["begin_suppress_tokens"], json!([220, 50257]));
    }
}
