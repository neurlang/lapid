//! Hugging Face Whisper (`WhisperForConditionalGeneration`) → `.lapid`.
//!
//! Reads `config.json`, `generation_config.json` (optional), the tokenizer
//! (`vocab.json` + `added_tokens.json`, or `tokenizer.json`) and the safetensors weights
//! (`model.safetensors` or a sharded `model.safetensors.index.json`), then streams the
//! tensors one by one through the dtype policy into the writer.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::safetensors::SafeTensors;
use lapid_convert::writer::{ModelWriter, check_complete, file_hash, read_json};
use lapid_convert::{download_hf, download_hf_optional as download_optional};
use serde_json::{Map, Value, json};

use super::tokens::Specials;
use super::{EMBED_TOKENS, add_token_blob, expected_tensors, metadata, role};

struct Files {
    config: PathBuf,
    generation: Option<PathBuf>,
    vocab: Option<PathBuf>,
    added: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    weights: Vec<PathBuf>,
    /// Repo id, or the directory for local sources.
    repo: String,
    remote: bool,
}

fn shard_files(index: &Value) -> Result<Vec<String>> {
    let map = index
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| Error::new("format", "model.safetensors.index.json has no weight_map"))?;
    let files: BTreeSet<String> = map.values().filter_map(|v| v.as_str().map(str::to_string)).collect();
    if files.is_empty() {
        bail!("format", "model.safetensors.index.json lists no shards");
    }
    Ok(files.into_iter().collect())
}

fn locate(src: &str) -> Result<Files> {
    let dir = Path::new(src);
    if dir.is_dir() {
        let opt = |f: &str| Some(dir.join(f)).filter(|p| p.is_file());
        let config = opt("config.json").ok_or_else(|| Error::new("not_found", format!("{}: no config.json", dir.display())))?;
        let weights = if let Some(p) = opt("model.safetensors") {
            vec![p]
        } else if let Some(idx) = opt("model.safetensors.index.json") {
            shard_files(&read_json(&idx)?)?.iter().map(|f| dir.join(f)).collect()
        } else {
            bail!("not_found", "{}: no model.safetensors (PyTorch .bin checkpoints are not supported)", dir.display());
        };
        return Ok(Files {
            config,
            generation: opt("generation_config.json"),
            vocab: opt("vocab.json"),
            added: opt("added_tokens.json"),
            tokenizer: opt("tokenizer.json"),
            weights,
            repo: dir.display().to_string(),
            remote: false,
        });
    }
    let repo = src.trim_matches('/');
    let config = download_hf(repo, "config.json")?;
    let generation = download_optional(repo, "generation_config.json")?;
    let vocab = download_optional(repo, "vocab.json")?;
    let (added, tokenizer) = if vocab.is_some() {
        (download_optional(repo, "added_tokens.json")?, None)
    } else {
        (None, Some(download_hf(repo, "tokenizer.json")?))
    };
    let weights = match download_optional(repo, "model.safetensors")? {
        Some(p) => vec![p],
        None => {
            let idx = download_optional(repo, "model.safetensors.index.json")?
                .ok_or_else(|| Error::new("not_found", format!("{repo}: no model.safetensors in repo")))?;
            let mut v = Vec::new();
            for f in shard_files(&read_json(&idx)?)? {
                v.push(download_hf(repo, &f)?);
            }
            v
        }
    };
    Ok(Files { config, generation, vocab, added, tokenizer, weights, repo: repo.to_string(), remote: true })
}

fn cfg_usize(cfg: &Value, key: &str) -> Result<usize> {
    cfg.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| Error::new("missing_metadata", format!("config.json: missing integer {key:?}")))
}

fn id_map(v: &Value, what: &str) -> Result<Vec<(String, u64)>> {
    let m = v.as_object().ok_or_else(|| Error::new("tokenizer", format!("{what}: expected a JSON object")))?;
    m.iter()
        .map(|(k, id)| {
            id.as_u64()
                .map(|id| (k.clone(), id))
                .ok_or_else(|| Error::new("tokenizer", format!("{what}: non-integer id for {k:?}")))
        })
        .collect()
}

/// Id-ordered token strings (byte-level BPE form, specials verbatim).
fn load_tokens(f: &Files, n_vocab: usize) -> Result<Vec<String>> {
    let mut pairs: Vec<(String, u64)> = Vec::new();
    if let Some(v) = &f.vocab {
        pairs.extend(id_map(&read_json(v)?, "vocab.json")?);
        if let Some(a) = &f.added {
            pairs.extend(id_map(&read_json(a)?, "added_tokens.json")?);
        }
    } else if let Some(t) = &f.tokenizer {
        let tj = read_json(t)?;
        let vocab = tj.get("model").and_then(|m| m.get("vocab")).cloned().unwrap_or(Value::Null);
        pairs.extend(id_map(&vocab, "tokenizer.json model.vocab")?);
        for a in tj.get("added_tokens").and_then(|a| a.as_array()).into_iter().flatten() {
            if let (Some(id), Some(c)) = (a.get("id").and_then(|i| i.as_u64()), a.get("content").and_then(|c| c.as_str())) {
                pairs.push((c.to_string(), id));
            }
        }
    } else {
        bail!("tokenizer", "{}: no vocab.json or tokenizer.json", f.repo);
    }
    let mut table: Vec<Option<String>> = vec![None; n_vocab];
    for (s, id) in pairs {
        if let Some(slot) = table.get_mut(id as usize) {
            *slot = Some(s);
        }
    }
    let missing = table.iter().filter(|t| t.is_none()).count();
    if missing > 0 {
        eprintln!("warning: {missing} of {n_vocab} token ids have no string; using <|unused_ID|>");
    }
    Ok(table.into_iter().enumerate().map(|(i, t)| t.unwrap_or_else(|| format!("<|unused_{i}|>"))).collect())
}

fn u32_list(v: Option<&Value>) -> Option<Vec<u32>> {
    v?.as_array()?.iter().map(|x| x.as_u64().map(|x| x as u32)).collect()
}

/// Builds the `"whisper"` metadata object (see CONTRACT / docs/spec/format.md).
fn whisper_meta(cfg: &Value, gen_cfg: Option<&Value>, tokens: &[String]) -> Result<Value> {
    let n_vocab = cfg_usize(cfg, "vocab_size")?;
    let lookup: BTreeMap<&str, u32> = tokens.iter().enumerate().map(|(i, s)| (s.as_str(), i as u32)).collect();
    let tok = |s: &str| lookup.get(s).copied();
    let gen_u32 = |k: &str| gen_cfg.and_then(|g| g.get(k)).and_then(|v| v.as_u64()).map(|v| v as u32);
    let cfg_u32 = |k: &str| cfg.get(k).and_then(|v| v.as_u64()).map(|v| v as u32);
    let need = |v: Option<u32>, what: &str| v.ok_or_else(|| Error::new("tokenizer", format!("cannot determine {what} token id")));
    let task = |t: &str| gen_cfg.and_then(|g| g.get("task_to_id")).and_then(|m| m.get(t)).and_then(|v| v.as_u64()).map(|v| v as u32);

    let sot = need(gen_u32("decoder_start_token_id").or(cfg_u32("decoder_start_token_id")).or(tok("<|startoftranscript|>")), "sot")?;
    let eot = need(tok("<|endoftext|>").or(gen_u32("eos_token_id")).or(cfg_u32("eos_token_id")), "eot")?;
    let transcribe = need(task("transcribe").or(tok("<|transcribe|>")), "transcribe")?;
    let translate = need(task("translate").or(tok("<|translate|>")), "translate")?;
    let no_timestamps = need(gen_u32("no_timestamps_token_id").or(tok("<|notimestamps|>")), "no_timestamps")?;
    let prev = need(gen_u32("prev_sot_token_id").or(tok("<|startofprev|>")), "prev")?;
    let no_speech = tok("<|nocaptions|>").or(tok("<|nospeech|>")).unwrap_or(no_timestamps.saturating_sub(1));
    let mut special = json!({
        "sot": sot, "eot": eot, "transcribe": transcribe, "translate": translate,
        "no_timestamps": no_timestamps, "timestamp_begin": no_timestamps + 1,
        "no_speech": no_speech, "prev": prev,
    });
    if let Some(lm) = tok("<|startoflm|>") {
        special["sot_lm"] = json!(lm);
    }

    let multilingual = gen_cfg
        .and_then(|g| g.get("is_multilingual"))
        .and_then(|v| v.as_bool())
        .unwrap_or(n_vocab >= 51865);
    let mut languages = Map::new();
    if let Some(l2i) = gen_cfg.and_then(|g| g.get("lang_to_id")).and_then(|m| m.as_object()) {
        for (k, v) in l2i {
            if let Some(id) = v.as_u64() {
                languages.insert(k.trim_start_matches("<|").trim_end_matches("|>").to_string(), json!(id));
            }
        }
    } else if multilingual {
        // Language tokens sit between sot and translate.
        for id in sot + 1..translate.min(transcribe) {
            let Some(s) = tokens.get(id as usize) else { break };
            if let Some(code) = s.strip_prefix("<|").and_then(|c| c.strip_suffix("|>"))
                && (2..=3).contains(&code.len()) && code.bytes().all(|b| b.is_ascii_lowercase()) {
                    languages.insert(code.to_string(), json!(id));
                }
        }
    }

    let defaults = Specials::from_n_vocab(n_vocab as u32);
    let pick = |k: &str| u32_list(gen_cfg.and_then(|g| g.get(k))).or_else(|| u32_list(cfg.get(k)));
    let suppress = pick("suppress_tokens").unwrap_or_else(|| defaults.suppress_tokens());
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
        "activation": cfg.get("activation_function").and_then(|v| v.as_str()).unwrap_or("lapid"),
        "special": special,
        "languages": languages,
        "suppress_tokens": suppress,
        "begin_suppress_tokens": begin_suppress,
    });
    if let Some(ah) = gen_cfg.and_then(|g| g.get("alignment_heads")).filter(|a| a.is_array()) {
        w["alignment_heads"] = ah.clone();
    }
    if let Some(m) = gen_cfg.and_then(|g| g.get("max_initial_timestamp_index")).filter(|m| m.is_u64()) {
        w["max_initial_timestamp_index"] = m.clone();
    }
    Ok(w)
}

/// Normalizes checkpoint names to `WhisperForConditionalGeneration` naming.
fn normalize_name(n: &str) -> String {
    if n.starts_with("encoder.") || n.starts_with("decoder.") { format!("model.{n}") } else { n.to_string() }
}

pub(crate) fn convert(src: &str, dtype: OutDType, out: &Path) -> Result<()> {
    let files = locate(src)?;
    let cfg = read_json(&files.config)?;
    if cfg.get("model_type").and_then(|m| m.as_str()) != Some("whisper") {
        bail!("unsupported_model", "{}: config.json model_type is not \"whisper\"", files.repo);
    }
    let gen_cfg = files.generation.as_deref().map(read_json).transpose()?;
    let n_vocab = cfg_usize(&cfg, "vocab_size")?;
    let tokens = load_tokens(&files, n_vocab)?;
    let whisper = whisper_meta(&cfg, gen_cfg.as_ref(), &tokens)?;

    // Open all shards and map (normalized) tensor names to (shard, original name).
    let shards: Vec<SafeTensors> = files.weights.iter().map(|p| SafeTensors::open(p)).collect::<Result<_>>()?;
    let tied = cfg.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(true);
    let mut by_name: BTreeMap<String, (usize, String)> = BTreeMap::new();
    for (i, st) in shards.iter().enumerate() {
        for n in st.names() {
            let norm = normalize_name(&n);
            if tied && norm == "proj_out.weight" {
                continue;
            }
            by_name.insert(norm, (i, n));
        }
    }
    let n_audio_layer = cfg_usize(&cfg, "encoder_layers")?;
    let n_text_layer = cfg_usize(&cfg, "decoder_layers")?;
    let expected = expected_tensors(n_audio_layer, n_text_layer);
    let have: BTreeSet<String> = by_name.keys().cloned().collect();
    check_complete(&have, &expected)?;
    let (ei, en) = &by_name[EMBED_TOKENS];
    let embed_shape = shards[*ei].shape(en)?;
    if embed_shape.first() != Some(&n_vocab) {
        bail!("shape_mismatch", "{EMBED_TOKENS} has shape {embed_shape:?}, config vocab_size is {n_vocab}");
    }

    let name_or_path = cfg.get("_name_or_path").and_then(|v| v.as_str()).unwrap_or("");
    let ident = if files.remote { files.repo.as_str() } else { name_or_path };
    let license = if ident.starts_with("openai/whisper") || files.repo.starts_with("openai/whisper") { "mit" } else { "unknown" };
    let name = [ident, files.repo.as_str()]
        .iter()
        .filter_map(|s| s.trim_end_matches('/').rsplit('/').next())
        .find(|s| !s.is_empty())
        .unwrap_or("whisper")
        .to_string();
    let mut source = json!({ "repo": files.repo });
    if files.remote {
        source["revision"] = json!("main");
    }
    if files.weights.len() == 1 {
        source["file_hash"] = json!(file_hash(&files.weights[0])?);
    }
    let meta = metadata(&name, dtype, license, source, whisper);

    let mut order: Vec<String> = expected.clone();
    let expected_set: BTreeSet<&String> = expected.iter().collect();
    order.extend(by_name.keys().filter(|k| !expected_set.contains(k)).cloned());
    let mut w = ModelWriter::create(out, meta, dtype, role, order.len())?;
    add_token_blob(&mut w, &tokens)?;
    for n in &order {
        let (si, orig) = &by_name[n];
        let st = &shards[*si];
        let shape = st.shape(orig)?;
        let data = st.to_f32(orig)?;
        w.add_f32(n, &shape, &data)?;
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Value {
        json!({
            "model_type": "whisper", "vocab_size": 51865, "d_model": 768, "num_mel_bins": 80,
            "max_source_positions": 1500, "max_target_positions": 448, "encoder_attention_heads": 12,
            "decoder_attention_heads": 12, "encoder_layers": 12, "decoder_layers": 12,
            "encoder_ffn_dim": 3072, "decoder_ffn_dim": 3072, "decoder_start_token_id": 50258,
            "eos_token_id": 50257, "begin_suppress_tokens": [220, 50257], "suppress_tokens": [1, 2]
        })
    }

    fn small_tokens() -> Vec<String> {
        let s = Specials::from_n_vocab(51865);
        let mut t: Vec<String> = (0..51865).map(|i| format!("t{i}")).collect();
        for (id, st) in s.token_strings(51865) {
            t[id as usize] = st;
        }
        t
    }

    #[test]
    fn meta_without_generation_config() {
        let w = whisper_meta(&cfg(), None, &small_tokens()).expect("meta");
        assert_eq!(w["special"]["sot"], 50258);
        assert_eq!(w["special"]["eot"], 50257);
        assert_eq!(w["special"]["transcribe"], 50359);
        assert_eq!(w["special"]["timestamp_begin"], 50364);
        assert_eq!(w["special"]["no_speech"], 50362);
        assert_eq!(w["special"]["sot_lm"], 50360);
        assert_eq!(w["languages"]["en"], 50259);
        assert_eq!(w["languages"]["su"], 50357);
        assert_eq!(w["suppress_tokens"], json!([1, 2]));
        assert_eq!(w["n_audio_state"], 768);
        assert!(w.get("tokens").is_none(), "tokens live in a blob");
    }

    #[test]
    fn meta_prefers_generation_config() {
        let g = json!({
            "lang_to_id": {"<|en|>": 50259}, "task_to_id": {"transcribe": 50359, "translate": 50358},
            "no_timestamps_token_id": 50363, "prev_sot_token_id": 50361, "suppress_tokens": [7],
            "alignment_heads": [[5, 3]], "is_multilingual": true
        });
        let w = whisper_meta(&cfg(), Some(&g), &small_tokens()).expect("meta");
        assert_eq!(w["languages"], json!({"en": 50259}));
        assert_eq!(w["suppress_tokens"], json!([7]));
        assert_eq!(w["alignment_heads"], json!([[5, 3]]));
        assert_eq!(w["multilingual"], true);
    }

    #[test]
    fn names_normalized() {
        assert_eq!(normalize_name("encoder.conv1.weight"), "model.encoder.conv1.weight");
        assert_eq!(normalize_name("model.decoder.layer_norm.bias"), "model.decoder.layer_norm.bias");
    }
}
