//! `lapid verify` / `lapid bench` / serving for any [`SpeechDef`] model.
//!
//! Reference format (`tools/reference/<model>_ref.py`): `result.json` with `text`,
//! `text_ids` and `n_frames`; `activations.safetensors` with the stage taps the model's
//! [`SpeechDef::reference_stages`] reproduces (teacher-forced on the reference inputs) and the
//! final `audio` of a free run on the reference noise.
//!
//! Verify: tokens; every stage vs PyTorch (fp32 / fp16 judged at the dtype tolerance), or for
//! int8 / int4 vs fake-quant references (same weights dequantized on the CPU, per tensor the
//! closer of f32 and int8 activations; PyTorch reported as quality); end to end: the frame
//! count, the log-spectrogram distance of the waveform and the model's own checks must hold
//! for fp32 / fp16 (reported for quantized models). Bench: time to first audio, real-time
//! factor and time per frame over `runs` runs after one warmup (fixed seed).

use std::path::Path;
use std::time::Instant;

use lapid::{Device, Error, Result};
use lapid_nn::Precision;
use lapid_verify::harness::{Reference, ids, kernel_compare, str_of};
use lapid_verify::timing::{median, ms, stats};
use lapid_verify::{Comparison, KERNEL_TOLERANCE, LayerReport, Stages, diff, tolerance};
use serde_json::{Value, json};

use super::registry::ModelEntry;
use super::resolve::{file_dtype, resolve};
use super::serve::{AnyModel, Capability, LoadOptions, ModelMeta, TtsParams};
use super::tts::{SpeechDef, SpeechRequest, load, speak};
use crate::cmd::ModelArgs;

/// Waveforms are compared through their log-magnitude spectrograms (phase-insensitive):
/// free-running generation compounds per-step rounding through the autoregression.
pub const SPECTRAL_TOLERANCE: f32 = 0.1;

const BENCH_TEXT: &str = "Hello from lapid. This is a short benchmark of streaming speech synthesis \
     on the lapid runtime, measuring the time to the first audio chunk and the real-time factor.";

/// Relative L2 distance of the log-magnitude spectrograms (n_fft 1024, hop 256).
pub fn spectral_rel(ours: &[f32], reference: &[f32]) -> Result<f32> {
    let stft = lapid_nn::audio::Stft::new(lapid_nn::audio::StftConfig::new(1024, 256))?;
    let spec = |x: &[f32]| -> Vec<f32> {
        let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        stft.spectrogram(&x64, 1.0).iter().map(|&m| (m + 1e-4).ln() as f32).collect()
    };
    Ok(diff(&spec(ours), &spec(reference))?.rel)
}

/// Stage names may carry a `.streamed` suffix (same reference tensor, other code path).
fn reference_name(name: &str) -> &str {
    name.strip_suffix(".streamed").unwrap_or(name)
}

fn compare_pytorch(ours: &[(String, Vec<f32>)], r: &Reference, tol: f32) -> Comparison {
    let mut c = Comparison::new();
    for (name, v) in ours {
        match r.acts.get(reference_name(name)) {
            Ok(x) if x.len() == v.len() => {
                c.compare(name, v, &x, tol);
            }
            Ok(x) => {
                c.layers.push(LayerReport::failed(name.clone()));
                c.notes.push(format!("{name}: {} values vs reference {}", v.len(), x.len()));
            }
            Err(e) => {
                c.layers.push(LayerReport::failed(name.clone()));
                c.notes.push(format!("{name}: {}", e.message));
            }
        }
    }
    c
}

/// `lapid verify <model>`; check `"ok"`.
pub fn verify<M: SpeechDef>(reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    let d = &M::DESCRIPTOR;
    let model = args.model.as_deref().unwrap_or(d.default_model);
    let dir = reference.unwrap_or(Path::new(d.reference_dir));
    let r = Reference::open(dir, d.reference_script)?;
    let dev = lapid_verify::open_device(args.device.as_deref())?;
    let mut stages = Stages::new();
    let res = resolve(model, Some(args.dtype.as_deref().unwrap_or(d.default_dtype)))?;
    if let Some(t) = res.convert_ms {
        stages.record("convert", t);
    }
    let path = res.path;
    let dtype = file_dtype(&path)?;
    let quantized = matches!(dtype.as_str(), "int8" | "int4");
    let quality_tol = tolerance(&dtype)?;
    let tts = stages.time("load", || load::<M>(&path, &dev, Precision::Stored))?;

    let text = str_of(&r.result, "text")?.to_string();
    let ref_ids = ids(&r.result, "text_ids")?;
    let tokens = tts.tokens(&text)?;
    let tokens_match = tokens == ref_ids;

    // Stages without a reference tensor (e.g. raw outputs HF does not expose) are skipped.
    let ours: Vec<(String, Vec<f32>)> =
        stages.time("run", || tts.reference_stages(&r))?.into_iter().filter(|(n, _)| r.acts.has(reference_name(n))).collect();
    let quality = compare_pytorch(&ours, &r, quality_tol);
    let mut kernel_json = Value::Null;
    let kernel = if quantized {
        let cpu = Device::cpu();
        let (fw, fa) = stages.time("fake_quant", || -> Result<_> {
            let w = load::<M>(&path, &cpu, Precision::FakeQuantWeights)?.reference_stages(&r)?;
            let a = load::<M>(&path, &cpu, Precision::FakeQuantActivations)?.reference_stages(&r)?;
            Ok((w, a))
        })?;
        let names: Vec<String> = ours.iter().map(|(n, _)| n.clone()).collect();
        let (k, which) = kernel_compare(&names, &ours, &fw, &fa)?;
        kernel_json = json!({
            "reference": "fake-quant: same weights dequantized to f32 on cpu; per tensor the closer of f32 and int8 activations",
            "tolerance": KERNEL_TOLERANCE,
            "comparison": k.to_json(),
            "matched": which,
        });
        Some(k)
    } else {
        None
    };

    // ---- end to end on the reference noise
    let t = Instant::now();
    let e2e = tts.reference_speech(&r)?;
    stages.record("speak", ms(t));
    let ref_frames = r.result.get("n_frames").and_then(Value::as_u64).unwrap_or(0) as usize;
    let ref_audio = r.acts.get("audio")?;
    let frames_match = e2e.frames == ref_frames;
    let same_len = e2e.audio.len() == ref_audio.len();
    let audio_diff = if same_len { Some(diff(&e2e.audio, &ref_audio)?) } else { None };
    let spectral = if same_len { Some(spectral_rel(&e2e.audio, &ref_audio)?) } else { None };
    let strict = d.tokens_must_match(&dtype);
    let checks_ok = e2e.checks.iter().all(|(_, v, tol)| v.is_some_and(|v| v <= *tol));
    let audio_ok = !strict || (frames_match && spectral.is_some_and(|s| s <= SPECTRAL_TOLERANCE) && checks_ok);
    let checks: serde_json::Map<String, Value> =
        e2e.checks.iter().map(|(n, v, tol)| (n.clone(), json!({"rel": v, "tolerance": tol}))).collect();

    let correctness = kernel.as_ref().unwrap_or(&quality);
    let ok = tokens_match && correctness.ok() && audio_ok;
    Ok(json!({
        "ok": ok,
        "device": dev.description(),
        "model": model,
        "model_file": path.display().to_string(),
        "dtype": dtype,
        "correctness_reference": if quantized { "fake-quant" } else { "pytorch-fp32" },
        "tolerance": if quantized { KERNEL_TOLERANCE } else { quality_tol },
        "first_failing_layer": correctness.first_failing(),
        "max_rel": correctness.max_rel(),
        "layers": correctness.layers,
        "tokens_match": tokens_match,
        "tokens": tokens,
        "reference_tokens": ref_ids,
        "end_to_end": {
            "frames": e2e.frames,
            "reference_frames": ref_frames,
            "frames_match": frames_match,
            "checks": checks,
            "spectral_rel": spectral,
            "spectral_tolerance": SPECTRAL_TOLERANCE,
            "waveform_rel": audio_diff.map(|d| d.rel),
            "waveform_max_abs": audio_diff.map(|d| d.max_abs),
            "must_match": strict,
            "ok": audio_ok,
            "first_audio_ms": e2e.first_audio_ms,
        },
        "kernel": kernel_json,
        "quality": {
            "reference": d.reference_label,
            "informational": quantized,
            "tolerance": quality_tol,
            "within_tolerance": quality.ok(),
            "max_rel": quality.max_rel(),
            "layers": quality.layers,
        },
        "notes": correctness.notes,
        "timings_ms": stages.to_json(),
        "reference_dir": dir.display().to_string(),
    }))
}

/// One measured run: (first audio ms, total ms, frames, audio seconds).
fn one_run<M: SpeechDef>(tts: &M, text: &str, opts: &M::Options) -> Result<Value> {
    let t = Instant::now();
    let mut stream = speak(tts, text, opts)?;
    let mut first = None;
    let mut samples = 0usize;
    for pcm in stream.by_ref() {
        samples += pcm?.len();
        first.get_or_insert_with(|| ms(t));
    }
    let total_ms = ms(t);
    let frames = stream.stats().frames;
    let audio_s = samples as f64 / f64::from(tts.sample_rate());
    let first_audio_ms = first.unwrap_or(total_ms);
    Ok(json!({
        "first_audio_ms": first_audio_ms,
        "total_ms": total_ms,
        "frames": frames,
        "audio_s": audio_s,
        "rtf": if audio_s > 0.0 { total_ms / 1e3 / audio_s } else { 0.0 },
        "frame_ms": if frames > 1 { (total_ms - first_audio_ms) / (frames - 1) as f64 } else { 0.0 },
    }))
}

/// `lapid bench <model> [text file]`: time to first audio and real-time factor.
pub fn bench<M: SpeechDef>(input: Option<&Path>, args: &ModelArgs, runs: usize) -> Result<Value> {
    let d = &M::DESCRIPTOR;
    let model = args.model.as_deref().unwrap_or(d.default_model);
    let text = match input {
        Some(p) => std::fs::read_to_string(p).map_err(|e| Error::new("io", format!("{}: {e}", p.display())))?,
        None => BENCH_TEXT.to_string(),
    };
    let dev = lapid_verify::open_device(args.device.as_deref())?;
    let res = resolve(model, Some(args.dtype.as_deref().unwrap_or(d.default_dtype)))?;
    let t = Instant::now();
    let tts = load::<M>(&res.path, &dev, Precision::Stored)?;
    let load_ms = ms(t);
    let opts = tts.request_options(&SpeechRequest { seed: Some(0), ..Default::default() });
    let warmup = one_run(&tts, &text, &opts)?;
    let measured: Vec<Value> = (0..runs.max(1)).map(|_| one_run(&tts, &text, &opts)).collect::<Result<_>>()?;
    let col = |k: &str| measured.iter().map(|r| r[k].as_f64().unwrap_or(0.0)).collect::<Vec<f64>>();
    Ok(json!({
        "device": dev.description(),
        "model": model,
        "model_file": res.path.display().to_string(),
        "dtype": file_dtype(&res.path)?,
        "text": text,
        "convert_ms": res.convert_ms,
        "load_ms": load_ms,
        "warmup": warmup,
        "runs": measured,
        "first_audio_ms": stats(&col("first_audio_ms")),
        "rtf": stats(&col("rtf")),
        "frame_ms": stats(&col("frame_ms")),
        "median_first_audio_ms": median(col("first_audio_ms")),
        "median_rtf": median(col("rtf")),
    }))
}

/// [`AnyModel`] of a speech model (capability `speech`).
pub struct SpeechServed<M: SpeechDef> {
    model: M,
    meta: ModelMeta,
}

impl<M: SpeechDef> AnyModel for SpeechServed<M> {
    fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    fn meta_mut(&mut self) -> &mut ModelMeta {
        &mut self.meta
    }

    fn speak(&mut self, text: &str, p: &TtsParams, out: &mut dyn FnMut(&[f32]) -> bool) -> Result<()> {
        let req = SpeechRequest { voice: p.voice.clone(), seed: p.seed, temperature: p.temperature, frames_per_chunk: Some(3) };
        for pcm in speak(&self.model, text, &self.model.request_options(&req))? {
            if !out(&pcm?) {
                break;
            }
        }
        Ok(())
    }

    fn sample_rate(&self) -> u32 {
        self.model.sample_rate()
    }
}

/// Registry `serve` entry of a speech model.
pub fn serve_speech<M: SpeechDef>(path: &Path, device: &Device, mut meta: ModelMeta, _: &LoadOptions) -> Result<Box<dyn AnyModel>> {
    let model = load::<M>(path, device, Precision::Stored)?;
    meta.capabilities = vec![Capability::Speech];
    meta.extra.insert("sample_rate".into(), json!(model.sample_rate()));
    meta.extra.insert("voices".into(), json!(model.voices()));
    meta.extra.insert("default_voice".into(), json!(model.default_voice()));
    Ok(Box::new(SpeechServed { model, meta }))
}

impl ModelEntry {
    /// The entry of a speech model: generic verify, bench and serving.
    pub const fn speech<M: SpeechDef>() -> ModelEntry {
        ModelEntry { name: M::DESCRIPTOR.name, convert: M::CONVERT, verify: verify::<M>, bench: bench::<M>, serve: serve_speech::<M>, calibrate: None }
    }
}

