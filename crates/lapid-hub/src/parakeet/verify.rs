//! `verify parakeet`: compare lapid's Parakeet TDT with the HF reference written by
//! `tools/reference/parakeet_ref.py`, stage by stage, for one reference directory (with
//! `result.json`) or every sub-directory that has one (e.g. `data/reference/parakeet/{en,de}`).
//!
//! Every stage is fed the reference input: the encoder runs on the reference mel, the TDT
//! decoder replays the reference step path (frames, tokens, durations) on the reference
//! projected encoding and its joint logits are compared per step. The full transcription
//! runs end to end from the audio file and must reproduce the reference tokens exactly
//! (fp32 / fp16 / int8; int4 is reported).
//!
//! Quantized models are checked twice, as for Whisper: kernels against the fake-quant
//! reference (same weights dequantized to f32, on the CPU; per tensor the closer of f32
//! and int8-per-block activations) at `KERNEL_TOLERANCE`, and quality against PyTorch
//! (informational).

use std::path::{Path, PathBuf};
use std::time::Instant;

use lapid::{Device, Error, Result, Tensor};
use lapid_convert::{OutDType, default_model_path};
use lapid_verify::timing::ms;
use lapid_verify::{Activations, Comparison, KERNEL_TOLERANCE, Stages, tolerance};
use serde_json::{Value, json};

use super::{Parakeet, Precision, Step, TranscribeOptions};
use crate::model::Model;
use super::decode::Path as DecodePath;

/// Tolerance of our log-mel features (independent of the model dtype).
pub const MEL_TOLERANCE: f32 = 1e-3;

/// Whether the transcript must match the reference exactly for this dtype.
pub fn text_must_match(dtype: &str) -> bool {
    matches!(dtype, "fp32" | "fp16" | "int8")
}

/// The `.lapid` file for `model`: a path, or a hub repo / GGUF converted into the cache.
pub(crate) fn resolve_model(model: &str, dtype: &str) -> Result<(PathBuf, Option<f64>)> {
    let p = Path::new(model);
    if p.extension().is_some_and(|e| e == "lapid") {
        return Ok((p.to_path_buf(), None));
    }
    let out_dtype = OutDType::parse(dtype)?;
    let is_gguf = p.is_file() && lapid_convert::gguf::is_gguf(p)?;
    let key = if is_gguf {
        format!("gguf/{}", p.file_stem().and_then(|s| s.to_str()).unwrap_or("parakeet"))
    } else {
        model.to_string()
    };
    let path = default_model_path(&key, out_dtype);
    if path.is_file() {
        return Ok((path, None));
    }
    let t = Instant::now();
    if is_gguf {
        super::convert::convert_gguf(p, out_dtype, &path)?;
    } else {
        super::convert::convert_hf(model, out_dtype, &path)?;
    }
    Ok((path, Some(ms(t))))
}

fn file_dtype(path: &Path) -> Result<String> {
    let r = lapid::format::Reader::open(path)?;
    r.metadata
        .get("dtype")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| Error::new("missing_metadata", format!("{}: no \"dtype\"", path.display())))
}

/// Parsed `result.json`.
struct Reference {
    tokens: Vec<u32>,
    steps: Vec<Step>,
    text: String,
    words: Value,
    audio: PathBuf,
}

fn usize_list(v: &Value, ptr: &str) -> Result<Vec<usize>> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("reference", format!("result.json: missing array {ptr}")))?
        .iter()
        .map(|x| x.as_u64().map(|x| x as usize).ok_or_else(|| Error::new("reference", format!("result.json: bad value in {ptr}"))))
        .collect()
}

fn read_reference(dir: &Path) -> Result<Reference> {
    let path = dir.join("result.json");
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| Error::new("io", format!("{}: {e} (run tools/reference/parakeet_ref.py)", path.display())))?;
    let v: Value = serde_json::from_str(&raw)?;
    let (frames, toks, durs) = (usize_list(&v, "/steps/frames")?, usize_list(&v, "/steps/tokens")?, usize_list(&v, "/steps/durations")?);
    if frames.len() != toks.len() || frames.len() != durs.len() {
        return Err(Error::new("reference", "result.json: steps arrays differ in length"));
    }
    let steps = frames.iter().zip(&toks).zip(&durs).map(|((&f, &t), &d)| Step { frame: f, token: t as u32, duration: d }).collect();
    let audio = v.get("audio").and_then(Value::as_str).ok_or_else(|| Error::new("reference", "result.json: no \"audio\""))?;
    Ok(Reference {
        tokens: usize_list(&v, "/tokens")?.into_iter().map(|t| t as u32).collect(),
        steps,
        text: v.get("text").and_then(Value::as_str).unwrap_or("").trim().to_string(),
        words: v.get("words").cloned().unwrap_or(Value::Null),
        audio: dir.join(audio),
    })
}

/// Encoder intermediates and replayed step logits of one model.
struct Outputs {
    traced: Vec<(String, Vec<f32>)>,
    logits: Vec<f32>,
}

fn run(p: &Parakeet, mel: &[f32], frames: usize, enc: &(Vec<f32>, Vec<usize>), steps: &[Step]) -> Result<Outputs> {
    let traced = p.encode_traced(mel, frames)?;
    let enc_t = p.device().upload(&Tensor::from_f32(&enc.0, &enc.1)?)?;
    let n = enc.1.get(1).copied().unwrap_or(0);
    let d = p.decode_frames(&enc_t, n, DecodePath::Replay(steps), true)?;
    Ok(Outputs { traced, logits: d.logits.concat() })
}

fn compare(expected: &[String], ours: &Outputs, reference: &Outputs, tol: f32) -> Result<Comparison> {
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
    c.compare("decoder.step_logits", &ours.logits, &reference.logits, tol);
    Ok(c)
}

/// Per step: token argmax and duration argmax agree.
fn argmax_agree(ours: &[f32], reference: &[f32], vocab: usize, width: usize) -> Vec<bool> {
    let am = |v: &[f32]| v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b });
    ours.chunks_exact(width)
        .zip(reference.chunks_exact(width))
        .map(|(a, b)| am(&a[..vocab]) == am(&b[..vocab]) && am(&a[vocab..]) == am(&b[vocab..]))
        .collect()
}

fn closest(w: &Comparison, a: &Comparison) -> (Comparison, Vec<&'static str>) {
    let mut merged = Comparison::new();
    let mut which = Vec::new();
    for (lw, la) in w.layers.iter().zip(&a.layers) {
        let use_a = la.rel < lw.rel || (lw.rel.is_nan() && !la.rel.is_nan());
        merged.layers.push(if use_a { la.clone() } else { lw.clone() });
        which.push(if use_a { "int8_activations" } else { "f32_activations" });
    }
    merged.notes = w.notes.clone();
    (merged, which)
}

/// Verify one sample directory (model already loaded).
fn verify_sample(dir: &Path, p: &Parakeet, path: &Path, dtype: &str, stages: &mut Stages) -> Result<Value> {
    let reference = read_reference(dir)?;
    let acts = Activations::open(&dir.join("activations.safetensors"))?;
    let quantized = matches!(dtype, "int8" | "int4");
    let quality_tol = tolerance(dtype)?;
    let cfg = p.config().clone();

    let mut expected = vec!["encoder.subsampling".to_string()];
    expected.extend(acts.numbered("encoder.layers."));
    expected.push("encoder.projected".to_string());
    let ref_mel = acts.get("mel")?;
    let mel_frames = acts.shape("mel")?.first().copied().unwrap_or(0);
    let enc_shape = acts.shape("encoder.projected")?;
    let ref_enc = (acts.get("encoder.projected")?, vec![1, enc_shape[0], enc_shape.get(1).copied().unwrap_or(0)]);
    let pytorch = Outputs {
        traced: expected.iter().map(|n| Ok((n.clone(), acts.get(n)?))).collect::<Result<_>>()?,
        logits: acts.get("decoder.step_logits")?,
    };

    let samples = lapid_nn::audio::load_audio(&reference.audio, cfg.sample_rate)?;
    let mut mel = Comparison::new();
    mel.compare("mel", &p.features(&samples).data, &ref_mel, MEL_TOLERANCE);

    let ours = stages.time("run", || run(p, &ref_mel, mel_frames, &ref_enc, &reference.steps))?;
    let quality = compare(&expected, &ours, &pytorch, quality_tol)?;
    let width = cfg.joint_dim();
    let quality_argmax = argmax_agree(&ours.logits, &pytorch.logits, cfg.vocab_size, width);

    let mut kernel_json = Value::Null;
    let mut kernel = None;
    if quantized {
        let cpu = Device::cpu();
        let (fw, fa) = stages.time("fake_quant", || -> Result<_> {
            let w = run(&Parakeet::load_with(path, &cpu, Precision::FakeQuantWeights)?, &ref_mel, mel_frames, &ref_enc, &reference.steps)?;
            let a = run(&Parakeet::load_with(path, &cpu, Precision::FakeQuantActivations)?, &ref_mel, mel_frames, &ref_enc, &reference.steps)?;
            Ok((w, a))
        })?;
        let kw = compare(&expected, &ours, &fw, KERNEL_TOLERANCE)?;
        let ka = compare(&expected, &ours, &fa, KERNEL_TOLERANCE)?;
        let (k, matched) = closest(&kw, &ka);
        let fq_logits = if matched.last() == Some(&"int8_activations") { &fa.logits } else { &fw.logits };
        let k_argmax = argmax_agree(&ours.logits, fq_logits, cfg.vocab_size, width);
        kernel_json = json!({
            "reference": "fake-quant: same weights dequantized to f32 on cpu; per tensor the closer of f32 and int8-per-32-block activations",
            "tolerance": KERNEL_TOLERANCE,
            "matched_reference": k.layers.iter().map(|l| l.name.clone()).zip(matched).collect::<std::collections::BTreeMap<_, _>>(),
            "step_argmax_match": k_argmax,
            "comparison": k.to_json(),
        });
        kernel = Some((k, k_argmax));
    }

    // ---- end to end
    let transcript = stages.time("transcribe", || p.transcribe(&samples, &TranscribeOptions::default()))?;
    let our_tokens: Vec<u32> = transcript.tokens.iter().map(|t| t.id).collect();
    let ref_tokens: Vec<u32> = reference.tokens.iter().copied().filter(|&t| !p.tokenizer().is_special(t) && t != cfg.blank_id).collect();
    let tokens_match = our_tokens == ref_tokens;
    let text_match = transcript.text == reference.text;
    let strict = text_must_match(dtype);
    let text_ok = !strict || (tokens_match && text_match);
    let words_match = serde_json::to_value(&transcript.words)
        .ok()
        .and_then(|w| w.as_array().cloned())
        .zip(reference.words.as_array())
        .is_some_and(|(o, r)| {
            o.len() == r.len()
                && o.iter().zip(r).all(|(a, b)| {
                    let f = |v: &Value, k: &str| v.get(k).and_then(Value::as_f64).unwrap_or(f64::NAN);
                    a.get("word") == b.get("word") && (f(a, "start") - f(b, "start")).abs() < 1e-3 && (f(a, "end") - f(b, "end")).abs() < 1e-3
                })
        });

    let (correctness, argmax) = match &kernel {
        Some((k, a)) => (k, a),
        None => (&quality, &quality_argmax),
    };
    let ok = mel.ok() && correctness.ok() && !argmax.is_empty() && argmax.iter().all(|&b| b) && text_ok;
    let mut layers = mel.layers.clone();
    layers.extend(correctness.layers.iter().cloned());
    let notes = [mel.notes.clone(), correctness.notes.clone()].concat();
    Ok(json!({
        "sample": dir.display().to_string(),
        "ok": ok,
        "layers": layers,
        "first_failing_layer": mel.first_failing().or(correctness.first_failing()),
        "step_argmax_match": argmax.iter().all(|&b| b),
        "tokens_match": tokens_match,
        "text_match": text_match,
        "text_must_match": strict,
        "words_match": words_match,
        "text": transcript.text,
        "reference_text": reference.text,
        "kernel": kernel_json,
        "quality": {
            "reference": "PyTorch fp32 (HF ParakeetForTDT)",
            "informational": quantized,
            "tolerance": quality_tol,
            "within_tolerance": quality.ok(),
            "max_rel": quality.max_rel(),
            "step_argmax_match": quality_argmax.iter().all(|&b| b),
            "layers": quality.layers,
        },
        "notes": notes,
    }))
}

/// Full verification as JSON; check `"ok"`. `reference_dir` holds `result.json` or
/// sub-directories that do.
pub fn verify(reference_dir: &Path, model: &str, dtype: &str, device: Option<&str>) -> Result<Value> {
    let dev = lapid_verify::open_device(device)?;
    let mut stages = Stages::new();
    let (path, convert_ms) = resolve_model(model, dtype)?;
    if let Some(t) = convert_ms {
        stages.record("convert", t);
    }
    let dtype = file_dtype(&path)?;
    let p = stages.time("load", || Parakeet::load(&path, &dev))?;

    let mut dirs = Vec::new();
    if reference_dir.join("result.json").is_file() {
        dirs.push(reference_dir.to_path_buf());
    } else {
        let rd = std::fs::read_dir(reference_dir).map_err(|e| Error::new("io", format!("{}: {e}", reference_dir.display())))?;
        let mut subs: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|d| d.join("result.json").is_file()).collect();
        subs.sort();
        dirs.extend(subs);
    }
    if dirs.is_empty() {
        return Err(Error::new("not_found", format!("{}: no result.json (run tools/reference/parakeet_ref.py)", reference_dir.display())));
    }
    let samples: Vec<Value> = dirs.iter().map(|d| verify_sample(d, &p, &path, &dtype, &mut stages)).collect::<Result<_>>()?;
    let ok = samples.iter().all(|s| s.get("ok").and_then(Value::as_bool).unwrap_or(false));
    Ok(json!({
        "ok": ok,
        "device": dev.description(),
        "model": model,
        "model_file": path.display().to_string(),
        "dtype": dtype,
        "correctness_reference": if matches!(dtype.as_str(), "int8" | "int4") { "fake-quant" } else { "pytorch-fp32" },
        "mel_tolerance": MEL_TOLERANCE,
        "samples": samples,
        "timings_ms": stages.to_json(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_text_and_argmax() {
        assert!(text_must_match("int8") && !text_must_match("int4"));
        let a = [0.0, 1.0, 0.5, 2.0, 0.0];
        let b = [0.0, 0.9, 0.1, 3.0, 1.0];
        assert_eq!(argmax_agree(&a, &b, 3, 5), vec![true]);
        let c = [0.0, 0.9, 0.1, 0.0, 1.0];
        assert_eq!(argmax_agree(&a, &c, 3, 5), vec![false]);
    }

    /// Manual run: `PARAKEET_VERIFY_MODEL=<.lapid|repo> [PARAKEET_VERIFY_DTYPE=int8]
    /// [PARAKEET_VERIFY_DEVICE=cuda] cargo test -p lapid-hub --release parakeet_verify_env --
    /// --ignored --nocapture` (from the repo root's crate dir; reference data/reference/parakeet).
    #[test]
    #[ignore]
    fn parakeet_verify_env() {
        let model = std::env::var("PARAKEET_VERIFY_MODEL").unwrap_or_else(|_| "nvidia/parakeet-tdt-0.6b-v3".into());
        let dtype = std::env::var("PARAKEET_VERIFY_DTYPE").unwrap_or_else(|_| "int8".into());
        let device = std::env::var("PARAKEET_VERIFY_DEVICE").ok();
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/reference/parakeet");
        let v = verify(&root, &model, &dtype, device.as_deref()).expect("verify");
        println!("{}", serde_json::to_string_pretty(&v).expect("json"));
        assert_eq!(v["ok"], true);
    }
}
