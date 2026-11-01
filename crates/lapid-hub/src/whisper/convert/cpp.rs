//! whisper.cpp `ggml-*.bin` → `.lapid` (HF tensor naming, same metadata as the HF path).
//!
//! q8_0 → int8 and q4_0 → int4 are copied block-for-block when the policy target
//! matches; every other combination is dequantized to f32 and re-encoded by the policy.

use std::collections::BTreeSet;
use std::path::Path;

use lapid::quant::Format;
use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::policy::Target;
use lapid_convert::writer::{ModelWriter, check_complete, file_hash};
use serde_json::{Map, Value, json};

use super::ggml::{self, GgmlModel, GgmlTensor, GgmlType, Hparams};
use super::tokens::{Specials, byte_to_char_table, bytes_to_bpe_string};
use super::{EMBED_TOKENS, add_token_blob, expected_tensors, metadata, role};

/// Maps a whisper.cpp (OpenAI checkpoint) tensor name to the HF name.
pub(crate) fn map_name(n: &str) -> Option<String> {
    let fixed = match n {
        "encoder.positional_embedding" => Some("model.encoder.embed_positions.weight"),
        "encoder.conv1.weight" => Some("model.encoder.conv1.weight"),
        "encoder.conv1.bias" => Some("model.encoder.conv1.bias"),
        "encoder.conv2.weight" => Some("model.encoder.conv2.weight"),
        "encoder.conv2.bias" => Some("model.encoder.conv2.bias"),
        "encoder.ln_post.weight" => Some("model.encoder.layer_norm.weight"),
        "encoder.ln_post.bias" => Some("model.encoder.layer_norm.bias"),
        "decoder.positional_embedding" => Some("model.decoder.embed_positions.weight"),
        "decoder.token_embedding.weight" => Some("model.decoder.embed_tokens.weight"),
        "decoder.ln.weight" => Some("model.decoder.layer_norm.weight"),
        "decoder.ln.bias" => Some("model.decoder.layer_norm.bias"),
        _ => None,
    };
    if let Some(f) = fixed {
        return Some(f.to_string());
    }
    let (side, rest) = if let Some(r) = n.strip_prefix("encoder.blocks.") {
        ("encoder", r)
    } else {
        ("decoder", n.strip_prefix("decoder.blocks.")?)
    };
    let (layer, rest) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let (module, param) = rest.rsplit_once('.')?;
    if param != "weight" && param != "bias" {
        return None;
    }
    let module = match module {
        "attn.query" => "self_attn.q_proj",
        "attn.key" => "self_attn.k_proj",
        "attn.value" => "self_attn.v_proj",
        "attn.out" => "self_attn.out_proj",
        "attn_ln" => "self_attn_layer_norm",
        "cross_attn.query" if side == "decoder" => "encoder_attn.q_proj",
        "cross_attn.key" if side == "decoder" => "encoder_attn.k_proj",
        "cross_attn.value" if side == "decoder" => "encoder_attn.v_proj",
        "cross_attn.out" if side == "decoder" => "encoder_attn.out_proj",
        "cross_attn_ln" if side == "decoder" => "encoder_attn_layer_norm",
        "mlp.0" => "fc1",
        "mlp.2" => "fc2",
        "mlp_ln" => "final_layer_norm",
        _ => return None,
    };
    Some(format!("model.{side}.layers.{layer}.{module}.{param}"))
}

/// HF shape for a ggml tensor: ne reversed, biases flattened to 1-D (whisper.cpp stores
/// conv biases as [n, 1]).
fn hf_shape(hf_name: &str, t: &GgmlTensor<'_>) -> Vec<usize> {
    if hf_name.ends_with(".bias") { vec![t.numel()] } else { t.shape() }
}

fn tokens(h: &Hparams, vocab: &[Vec<u8>], sp: &Specials) -> Result<Vec<String>> {
    let n = h.n_vocab;
    let table = byte_to_char_table();
    let mut out: Vec<Option<String>> = vec![None; n];
    for (i, b) in vocab.iter().enumerate().take(n) {
        out[i] = Some(bytes_to_bpe_string(&table, b));
    }
    for (id, s) in sp.token_strings(n as u32) {
        if let Some(slot) = out.get_mut(id as usize) {
            *slot = Some(s);
        }
    }
    Ok(out.into_iter().enumerate().map(|(i, t)| t.unwrap_or_else(|| format!("<|unused_{i}|>"))).collect())
}

fn whisper_meta(h: &Hparams) -> Result<Value> {
    if h.n_audio_state != h.n_text_state {
        bail!("unsupported_model", "ggml: n_audio_state {} != n_text_state {}", h.n_audio_state, h.n_text_state);
    }
    let sp = Specials::from_n_vocab(h.n_vocab as u32);
    if (sp.timestamp_begin as usize) > h.n_vocab {
        bail!("unsupported_model", "ggml: n_vocab {} too small for a Whisper vocabulary", h.n_vocab);
    }
    let mut languages = Map::new();
    if sp.is_multilingual() {
        for (code, id) in sp.languages() {
            languages.insert(code.to_string(), json!(id));
        }
    }
    let mut special = json!({
        "sot": sp.sot, "eot": sp.eot, "transcribe": sp.transcribe, "translate": sp.translate,
        "no_timestamps": sp.no_timestamps, "timestamp_begin": sp.timestamp_begin,
        "no_speech": sp.no_speech, "prev": sp.prev,
    });
    special["sot_lm"] = json!(sp.sot_lm);
    Ok(json!({
        "n_mels": h.n_mels,
        "n_audio_ctx": h.n_audio_ctx,
        "n_audio_state": h.n_audio_state,
        "n_audio_head": h.n_audio_head,
        "n_audio_layer": h.n_audio_layer,
        "n_audio_ffn": 4 * h.n_audio_state,
        "n_text_ctx": h.n_text_ctx,
        "n_text_state": h.n_text_state,
        "n_text_head": h.n_text_head,
        "n_text_layer": h.n_text_layer,
        "n_text_ffn": 4 * h.n_text_state,
        "n_vocab": h.n_vocab,
        "multilingual": sp.is_multilingual(),
        "activation": "gelu",
        "special": special,
        "languages": languages,
        "suppress_tokens": sp.suppress_tokens(),
        "begin_suppress_tokens": sp.begin_suppress_tokens(),
        "max_initial_timestamp_index": 50,
    }))
}

/// Model name from the file name: `ggml-small.en-q8_0.bin` → `whisper-small.en`.
fn model_name(path: &Path) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let base = stem.strip_prefix("ggml-").unwrap_or(stem);
    let base = ["-q8_0", "-q5_0", "-q5_1", "-q4_0", "-q4_1", "-f16", "-f32"]
        .iter()
        .find_map(|s| base.strip_suffix(s))
        .unwrap_or(base);
    if base.starts_with("whisper") { base.to_string() } else { format!("whisper-{base}") }
}

/// Encode one ggml tensor according to the policy target.
fn convert_tensor(w: &mut ModelWriter, hf_name: &str, t: &GgmlTensor<'_>) -> Result<()> {
    let shape = hf_shape(hf_name, t);
    let target = w.target(hf_name, &shape);
    let direct = match (t.ttype, target) {
        (GgmlType::Q8_0, Target::Quant(Format::Int8)) => Some(Format::Int8),
        (GgmlType::Q4_0, Target::Quant(Format::Int4)) => Some(Format::Int4),
        _ => None,
    };
    if let Some(f) = direct {
        // ggml rows are ne[0] long; the policy only quantizes 2-D [rows, cols] with cols == ne[0].
        let (rows, cols) = (shape[0], shape[1]);
        let bytes = lapid::quant::from_ggml_blocks(f, t.data, rows, cols);
        return w.add_raw(hf_name, target.dtype(), &shape, &bytes);
    }
    let data = ggml::dequantize(t.ttype, t.data, t.numel())?;
    w.add_f32(hf_name, &shape, &data)
}

pub(crate) fn convert(path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    let model = GgmlModel::open(path)?;
    let h = model.hparams.clone();
    let whisper = whisper_meta(&h)?;
    let token_list = tokens(&h, &model.vocab, &Specials::from_n_vocab(h.n_vocab as u32))?;
    let tensors = model.tensors()?;
    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
    let source = json!({ "repo": "ggerganov/whisper.cpp", "file": file_name, "file_hash": file_hash(path)?, "ggml_ftype": h.ftype % 1000 });
    let meta = metadata(&model_name(path), dtype, "mit", source, whisper);

    // Map + validate names before writing anything.
    let mut mapped = Vec::with_capacity(tensors.len());
    let mut seen = BTreeSet::new();
    for t in &tensors {
        let hf = map_name(&t.name)
            .ok_or_else(|| Error::new("unsupported_tensor", format!("ggml: unknown whisper.cpp tensor {:?}", t.name)))?;
        if !seen.insert(hf.clone()) {
            bail!("format", "ggml: duplicate tensor {:?}", t.name);
        }
        mapped.push((hf, t));
    }
    let expected = expected_tensors(h.n_audio_layer, h.n_text_layer);
    check_complete(&seen, &expected)?;
    if let Some((_, t)) = mapped.iter().find(|(n, _)| n == EMBED_TOKENS) {
        let shape = t.shape();
        if shape != [h.n_vocab, h.n_text_state] {
            bail!("shape_mismatch", "ggml: token embedding has shape {shape:?}, expected [{}, {}]", h.n_vocab, h.n_text_state);
        }
    }

    let mut w = ModelWriter::create(out, meta, dtype, role, mapped.len())?;
    add_token_blob(&mut w, &token_list)?;
    for (hf, t) in &mapped {
        convert_tensor(&mut w, hf, t)?;
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(map_name("encoder.conv1.bias").as_deref(), Some("model.encoder.conv1.bias"));
        assert_eq!(
            map_name("encoder.blocks.3.attn.query.weight").as_deref(),
            Some("model.encoder.layers.3.self_attn.q_proj.weight")
        );
        assert_eq!(
            map_name("decoder.blocks.11.cross_attn.value.bias").as_deref(),
            Some("model.decoder.layers.11.encoder_attn.v_proj.bias")
        );
        assert_eq!(
            map_name("decoder.blocks.0.cross_attn_ln.weight").as_deref(),
            Some("model.decoder.layers.0.encoder_attn_layer_norm.weight")
        );
        assert_eq!(map_name("encoder.blocks.0.mlp.2.weight").as_deref(), Some("model.encoder.layers.0.fc2.weight"));
        assert_eq!(map_name("encoder.blocks.0.cross_attn.query.weight"), None);
        assert_eq!(map_name("bogus"), None);
    }

    #[test]
    fn every_expected_name_has_a_source() {
        // Build the whisper.cpp name list for a 2+2 layer model and check the mapping covers
        // the expected HF set exactly.
        let mut src = vec![
            "encoder.positional_embedding", "encoder.conv1.weight", "encoder.conv1.bias", "encoder.conv2.weight",
            "encoder.conv2.bias", "encoder.ln_post.weight", "encoder.ln_post.bias", "decoder.positional_embedding",
            "decoder.token_embedding.weight", "decoder.ln.weight", "decoder.ln.bias",
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
        let attn = |p: &str, v: &mut Vec<String>| {
            for m in ["query", "value", "out"] {
                v.push(format!("{p}.{m}.weight"));
                v.push(format!("{p}.{m}.bias"));
            }
            v.push(format!("{p}.key.weight"));
            v.push(format!("{p}_ln.weight"));
            v.push(format!("{p}_ln.bias"));
        };
        for side in ["encoder", "decoder"] {
            for i in 0..2 {
                let b = format!("{side}.blocks.{i}");
                attn(&format!("{b}.attn"), &mut src);
                if side == "decoder" {
                    attn(&format!("{b}.cross_attn"), &mut src);
                }
                for s in ["mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias", "mlp_ln.weight", "mlp_ln.bias"] {
                    src.push(format!("{b}.{s}"));
                }
            }
        }
        let mapped: BTreeSet<String> = src.iter().map(|n| map_name(n).expect(n)).collect();
        let expected: BTreeSet<String> = expected_tensors(2, 2).into_iter().collect();
        assert_eq!(mapped, expected);
    }

    #[test]
    fn names_from_files() {
        assert_eq!(model_name(Path::new("/x/ggml-small.bin")), "whisper-small");
        assert_eq!(model_name(Path::new("ggml-base.en-q8_0.bin")), "whisper-base.en");
    }

    #[test]
    fn token_table() {
        let h = Hparams {
            n_vocab: 51865, n_audio_ctx: 1500, n_audio_state: 768, n_audio_head: 12, n_audio_layer: 12,
            n_text_ctx: 448, n_text_state: 768, n_text_head: 12, n_text_layer: 12, n_mels: 80, ftype: 1,
        };
        let vocab: Vec<Vec<u8>> = (0..50257u32).map(|i| if i == 0 { b" the".to_vec() } else { vec![b'a'] }).collect();
        let m = whisper_meta(&h).expect("meta");
        let t = tokens(&h, &vocab, &Specials::from_n_vocab(51865)).expect("tokens");
        assert_eq!(t.len(), 51865);
        assert_eq!(t[0], "Ġthe");
        assert_eq!(t[50257], "<|endoftext|>");
        assert_eq!(t[50259], "<|en|>");
        assert_eq!(t[50363], "<|notimestamps|>");
        assert_eq!(m["special"]["timestamp_begin"], 50364);
        assert_eq!(m["languages"]["en"], 50259);
    }
}
