//! `lapid verify whisper`: compare lapid's Whisper against the HF reference, stage by stage
//! (reference data written by `tools/reference/whisper_ref.py`).
//!
//! Every stage is fed the *reference* input so errors do not compound across stages: the
//! encoder runs on the reference mel, the decoder on our encoding of the reference mel. The
//! full transcription runs end to end from the audio file.
//!
//! Two questions are answered separately for quantized models (int8 / int4):
//! * **kernel correctness** — the model is compared with a fake-quant reference: the same
//!   file with every quantized weight dequantized to f32, run on the CPU with f32 kernels.
//!   Both see identical weights, so only kernel error remains (tight tolerance).
//! * **quality loss** — the model is compared with the PyTorch fp32 reference
//!   (informational; int8 must still reproduce the reference transcript exactly).
//!
//! For fp32 / fp16 models the PyTorch comparison is the correctness check.

use std::path::{Path, PathBuf};
use std::time::Instant;

use lapid::{Device, Error, Result};
use lapid_convert::{OutDType, default_model_path};
use lapid_verify::timing::ms;
use lapid_verify::{Activations, Comparison, KERNEL_TOLERANCE, LayerReport, Stages, argmax_rows, tolerance};
use serde::Serialize;
use serde_json::{Value, json};

use super::{Precision, Task, TranscribeOptions, Whisper};

/// Tolerance of our log-mel spectrogram (independent of the model dtype).
pub const MEL_TOLERANCE: f32 = 1e-3;

/// Contract summary of a verification run (the JSON report adds diagnostics).
#[derive(Debug, Clone, Serialize)]
pub struct VerifyReport {
    pub device: String,
    pub model: String,
    /// Correctness rows: vs the fake-quant reference for quantized models, else vs PyTorch.
    pub layers: Vec<LayerReport>,
    pub logits_max_abs: f32,
    pub tokens_match: bool,
    pub text: String,
    pub reference_text: String,
    pub ok: bool,
}

/// Whether the transcript must match the reference exactly for this dtype.
pub fn text_must_match(dtype: &str) -> bool {
    matches!(dtype, "fp32" | "fp16" | "int8")
}

/// True when `model` names a local `.lapid` file rather than a hub repo / HF directory.
pub(crate) fn is_lapid_file(model: &str) -> bool {
    let p = Path::new(model);
    p.extension().is_some_and(|e| e == "lapid") || p.is_file()
}

/// The `.lapid` file for `model` (a `.lapid` path, or a hub repo converted into the cache on
/// first use). Returns the path and the conversion time if one was needed.
pub(crate) fn resolve_model(model: &str, dtype: &str) -> Result<(PathBuf, Option<f64>)> {
    if is_lapid_file(model) {
        return Ok((PathBuf::from(model), None));
    }
    let path = default_model_path(model, OutDType::parse(dtype)?);
    if path.is_file() {
        return Ok((path, None));
    }
    let t = Instant::now();
    super::convert::convert_hf(model, OutDType::parse(dtype)?, &path)?;
    Ok((path, Some(ms(t))))
}

/// Weight dtype recorded in the file ("fp16", "int8", ...).
fn file_dtype(path: &Path) -> Result<String> {
    let r = lapid::format::Reader::open(path)?;
    r.metadata
        .get("dtype")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| Error::new("missing_metadata", format!("{}: no \"dtype\"", path.display())))
}

/// Parsed `result.json` of a reference directory.
struct Reference {
    prompt_tokens: Vec<u32>,
    tokens: Vec<u32>,
    text: String,
    eot: u32,
    audio: PathBuf,
}

fn u32_list(v: &Value, key: &str) -> Result<Vec<u32>> {
    let arr = v
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("reference", format!("result.json: missing array {key:?}")))?;
    arr.iter()
        .map(|x| {
            x.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| Error::new("reference", format!("result.json: {key:?} holds a non-token value {x}")))
        })
        .collect()
}

fn read_reference(dir: &Path) -> Result<Reference> {
    let path = dir.join("result.json");
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| Error::new("io", format!("{}: {e} (run tools/reference/whisper_ref.py)", path.display())))?;
    let v: Value = serde_json::from_str(&raw)?;
    let prompt_tokens = u32_list(&v, "prompt_tokens")?;
    if prompt_tokens.is_empty() {
        return Err(Error::new("reference", "result.json: empty prompt_tokens"));
    }
    let tokens = u32_list(&v, "tokens")?;
    let text = v
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::new("reference", "result.json: missing \"text\""))?
        .trim()
        .to_string();
    // <|endoftext|> sits right before <|startoftranscript|> in every Whisper vocabulary.
    let eot = v
        .pointer("/special/eot")
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(prompt_tokens[0].saturating_sub(1));
    // "audio" is relative to the reference dir; default: data/jfk.wav two levels up.
    let audio = match v.get("audio").and_then(Value::as_str) {
        Some(rel) => dir.join(rel),
        None => dir.join("..").join("..").join("jfk.wav"),
    };
    Ok(Reference { prompt_tokens, tokens, text, eot, audio })
}

/// Encoder intermediates and prompt logits of one model on the reference mel.
struct ModelOutputs {
    traced: Vec<(String, Vec<f32>)>,
    logits: Vec<f32>,
    /// The encoding the logits were computed from (host copy) and its shape.
    encoded: Option<(Vec<f32>, Vec<usize>)>,
}

/// Run `whisper` on `mel`. The decoder runs on `encoded` when given (so a reference decoder
/// sees exactly the encoding of the model under test), else on its own encoding.
fn run_model(whisper: &Whisper, mel: &[f32], prompt: &[u32], encoded: Option<&(Vec<f32>, Vec<usize>)>) -> Result<ModelOutputs> {
    let traced = whisper.encode_traced(mel)?;
    let (enc, host) = match encoded {
        Some((data, shape)) => (whisper.device().upload(&lapid::Tensor::from_f32(data, shape)?)?, None),
        None => {
            let enc = whisper.encode(mel)?;
            let host = (whisper.device().to_vec_f32(&enc)?, enc.shape().to_vec());
            (enc, Some(host))
        }
    };
    let logits = whisper.decoder_logits(&enc, prompt)?;
    Ok(ModelOutputs { traced, logits, encoded: host })
}

/// Compare encoder layers + prompt logits of `ours` with `reference`; returns the
/// comparison and the per-position argmax agreement.
fn compare_model(
    expected: &[String],
    ours: &ModelOutputs,
    reference: &ModelOutputs,
    n_vocab: usize,
    tol: f32,
) -> Result<(Comparison, Vec<bool>)> {
    let mut c = Comparison::new();
    c.compare_named(
        expected,
        &ours.traced,
        |name| {
            reference
                .traced
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| v.clone())
                .ok_or_else(|| Error::new("missing_tensor", format!("reference has no {name:?}")))
        },
        tol,
    )?;
    c.compare("decoder.logits.prompt", &ours.logits, &reference.logits, tol);
    Ok((c, argmax_rows(&ours.logits, &reference.logits, n_vocab)))
}

/// Per tensor, the closer of the two fake-quant references (f32 activations, or int8
/// activations per 32-block): the kernel contract allows either arithmetic. Returns the
/// merged comparison and, per row, which reference matched.
fn closest(weights: &Comparison, activations: &Comparison) -> (Comparison, Vec<&'static str>) {
    let mut merged = Comparison::new();
    let mut which = Vec::new();
    for (w, a) in weights.layers.iter().zip(&activations.layers) {
        let use_a = a.rel < w.rel || (w.rel.is_nan() && !a.rel.is_nan());
        merged.layers.push(if use_a { a.clone() } else { w.clone() });
        which.push(if use_a { "int8_activations" } else { "f32_activations" });
    }
    merged.notes = weights.notes.clone();
    (merged, which)
}

fn all_true(v: &[bool]) -> bool {
    !v.is_empty() && v.iter().all(|&b| b)
}

/// Full verification as JSON: the `VerifyReport` fields plus `kernel` / `quality`
/// sections and diagnostics. Check `"ok"`.
pub fn verify(reference_dir: &Path, model: &str, dtype: &str, device: Option<&str>) -> Result<Value> {
    let reference = read_reference(reference_dir)?;
    let acts = Activations::open(&reference_dir.join("activations.safetensors"))?;
    let dev = lapid_verify::open_device(device)?;
    let mut stages = Stages::new();

    let (path, convert_ms) = resolve_model(model, dtype)?;
    if let Some(t) = convert_ms {
        stages.record("convert", t);
    }
    let dtype = file_dtype(&path)?;
    let quantized = matches!(dtype.as_str(), "int8" | "int4");
    let quality_tol = tolerance(&dtype)?;
    let whisper = stages.time("load", || Whisper::load(&path, &dev))?;

    // ---- PyTorch fp32 reference
    let mut expected = vec!["encoder.embed".to_string()];
    expected.extend(acts.numbered("encoder.layers."));
    expected.push("encoder.out".to_string());
    let ref_mel = acts.get("mel")?;
    let pytorch = ModelOutputs {
        traced: expected.iter().map(|n| Ok((n.clone(), acts.get(n)?))).collect::<Result<_>>()?,
        logits: acts.get("decoder.logits.prompt")?,
        encoded: None,
    };
    let n_vocab = acts.shape("decoder.logits.prompt")?.last().copied().unwrap_or(0);

    // ---- mel (from the audio file)
    let samples = crate::audio::load_audio_16k_mono(&reference.audio)?;
    let mut mel = Comparison::new();
    mel.compare("mel", &whisper.log_mel(&samples), &ref_mel, MEL_TOLERANCE);

    // ---- the model under test, on the reference mel
    let ours = stages.time("run", || run_model(&whisper, &ref_mel, &reference.prompt_tokens, None))?;
    let (quality, quality_argmax) = compare_model(&expected, &ours, &pytorch, n_vocab, quality_tol)?;

    // ---- fake-quant reference (quantized models only)
    let mut kernel_json = Value::Null;
    let mut weight_loss_json = Value::Null;
    let mut kernel = None;
    if quantized {
        let cpu = Device::cpu();
        let prompt = &reference.prompt_tokens;
        let our_enc = ours.encoded.as_ref();
        let (fq_w, fq_w_own, fq_a) = stages.time("fake_quant", || -> Result<_> {
            let w = Whisper::load_with(&path, &cpu, Precision::FakeQuantWeights)?;
            let w_out = run_model(&w, &ref_mel, prompt, our_enc)?;
            // Same weights on its own encoding: weight-quantization loss alone.
            let w_own = ModelOutputs {
                traced: w_out.traced.clone(),
                logits: w.decoder_logits(&w.encode(&ref_mel)?, prompt)?,
                encoded: None,
            };
            drop(w);
            let a = Whisper::load_with(&path, &cpu, Precision::FakeQuantActivations)?;
            let a_out = run_model(&a, &ref_mel, prompt, our_enc)?;
            Ok((w_out, w_own, a_out))
        })?;
        let (kw, kw_argmax) = compare_model(&expected, &ours, &fq_w, n_vocab, KERNEL_TOLERANCE)?;
        let (ka, ka_argmax) = compare_model(&expected, &ours, &fq_a, n_vocab, KERNEL_TOLERANCE)?;
        let (k, matched) = closest(&kw, &ka);
        let logits_ref = matched.last().copied().unwrap_or("f32_activations");
        let k_argmax = if logits_ref == "int8_activations" { ka_argmax } else { kw_argmax };
        let (w, w_argmax) = compare_model(&expected, &fq_w_own, &pytorch, n_vocab, quality_tol)?;
        kernel_json = json!({
            "reference": "fake-quant: the same weights dequantized to f32, run on cpu; per tensor the closer of \
                          f32 activations and int8-per-32-block activations (both allowed by the kernel contract); \
                          the reference decoders run on the encoding of the model under test",
            "tolerance": KERNEL_TOLERANCE,
            "matched_reference": k.layers.iter().map(|l| l.name.clone()).zip(matched).collect::<std::collections::BTreeMap<_, _>>(),
            "logits_argmax_match": k_argmax,
            "comparison": k.to_json(),
            "vs_f32_activations": kw.to_json(),
            "vs_int8_activations": ka.to_json(),
        });
        weight_loss_json = json!({
            "description": "fake-quant (f32 activations) vs PyTorch fp32: loss from weight quantization alone",
            "max_rel": w.max_rel(),
            "logits_argmax_match": w_argmax,
            "layers": w.layers,
        });
        kernel = Some((k, k_argmax));
    }

    // ---- end-to-end transcription
    let opts = TranscribeOptions { language: Some("en".to_string()), task: Task::Transcribe, timestamps: true };
    let transcript = stages.time("transcribe", || whisper.transcribe(&samples, &opts))?;
    let eot = reference.eot;
    let flat = |t: &[u32]| -> Vec<u32> { t.iter().copied().filter(|&t| t < eot).collect() };
    let our_tokens = flat(&transcript.segments.iter().flat_map(|s| s.tokens.iter().copied()).collect::<Vec<_>>());
    let ref_tokens = flat(&reference.tokens);
    let tokens_match = our_tokens == ref_tokens;
    let text = transcript.text.trim().to_string();
    let text_match = text == reference.text;
    let strict = text_must_match(&dtype);
    let text_ok = !strict || (text_match && tokens_match);

    // ---- verdict: kernels (or fp reference) + mel + text
    let (correctness, argmax) = match &kernel {
        Some((k, a)) => (k, a),
        None => (&quality, &quality_argmax),
    };
    let ok = mel.ok() && correctness.ok() && all_true(argmax) && text_ok;

    let mut layers = mel.layers.clone();
    layers.extend(correctness.layers.iter().cloned());
    let logits_max_abs = correctness.row("decoder.logits.prompt").map(|r| r.max_abs).unwrap_or(f32::NAN);
    let report = VerifyReport {
        device: dev.description(),
        model: model.to_string(),
        layers,
        logits_max_abs,
        tokens_match,
        text,
        reference_text: reference.text.clone(),
        ok,
    };
    let mut notes = mel.notes.clone();
    notes.extend(correctness.notes.iter().cloned());
    let mut value = serde_json::to_value(&report)?;
    lapid_verify::merge_object(
        &mut value,
        json!({
            "dtype": dtype,
            "correctness_reference": if quantized { "fake-quant" } else { "pytorch-fp32" },
            "tolerance": if quantized { KERNEL_TOLERANCE } else { quality_tol },
            "mel_tolerance": MEL_TOLERANCE,
            "logits_argmax_match": argmax,
            "first_failing_layer": mel.first_failing().or(correctness.first_failing()),
            "kernel": kernel_json,
            "quality": {
                "reference": "PyTorch fp32 (HF WhisperForConditionalGeneration)",
                "informational": quantized,
                "tolerance": quality_tol,
                "within_tolerance": quality.ok(),
                "max_rel": quality.max_rel(),
                "logits_argmax_match": quality_argmax,
                "layers": quality.layers,
                "weight_quantization_only": weight_loss_json,
            },
            "text_match": text_match,
            "text_must_match": strict,
            "tokens": our_tokens,
            "reference_tokens": ref_tokens,
            "language": transcript.language,
            "reference_dir": reference_dir.display().to_string(),
            "model_file": path.display().to_string(),
            "audio": reference.audio.display().to_string(),
            "timings_ms": stages.to_json(),
            "notes": notes,
        }),
    );
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_text() {
        assert!(text_must_match("int8"));
        assert!(text_must_match("fp16"));
        assert!(!text_must_match("int4"));
    }

    #[test]
    fn gelu_paths() {
        assert!(is_lapid_file("x/model.lapid"));
        assert!(!is_lapid_file("openai/whisper-small"));
    }
}
