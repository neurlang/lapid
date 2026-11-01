//! `verify vad`: compare lapid's Silero VAD with the TorchScript reference written by
//! `tools/reference/silero_vad_ref.py` for every sample directory (`jfk`, `gaps`) under the
//! reference root.
//!
//! Per sample:
//! * per-window probabilities of the whole clip (max abs error <= [`PROB_TOLERANCE`]);
//! * intermediates of the first windows (STFT magnitude, encoder, LSTM h / c, probability;
//!   relative error <= [`LAYER_TOLERANCE`]);
//! * `get_speech_timestamps` for each reference option set: exact, both from our
//!   probabilities and (segment logic alone) from the reference probabilities;
//! * `VADIterator` events from a [`super::VadStream`] fed in odd-sized chunks: exact;
//! * stream vs whole-buffer probabilities: bit-identical.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use lapid::{Device, Error, Result};
use lapid_convert::OutDType;
use lapid_verify::timing::ms;
use lapid_verify::{Activations, Comparison, Stages, diff};
use serde_json::{Value, json};

use super::segments::{SpeechSegment, VadEvent, VadOptions, speech_timestamps};
use super::{Vad, convert};
use crate::model::Model;

/// Max abs difference of a speech probability (fp32 weights).
pub const PROB_TOLERANCE: f32 = 1e-4;
/// Relative error of the traced intermediates.
pub const LAYER_TOLERANCE: f32 = 1e-4;
/// Chunk size the stream check feeds (not a multiple of the window on purpose).
pub const STREAM_CHUNK: usize = 1000;

/// Open `model`: a `.lapid` file, an exported directory (converted in memory) or a hub repo.
pub fn open_model(model: &str, device: &Device) -> Result<(Vad, Option<f64>)> {
    let p = Path::new(model);
    if p.extension().is_some_and(|e| e == "lapid") {
        return Ok((Vad::load(p, device)?, None));
    }
    if p.is_dir() {
        let t = Instant::now();
        let tmp = std::env::temp_dir().join(format!("lapid-vad-{}-{}.lapid", std::process::id(), unique()));
        let r = convert::convert_hf(model, OutDType::F32, &tmp).and_then(|_| {
            std::fs::read(&tmp).map_err(|e| Error::new("io", format!("{}: {e}", tmp.display())))
        });
        std::fs::remove_file(&tmp).ok();
        let bytes = r?;
        let convert_ms = ms(t);
        return Ok((Vad::load_bytes(Arc::new(bytes), device)?, Some(convert_ms)));
    }
    Ok((Vad::from_hub(model, "fp32", device)?, None))
}

fn unique() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    N.fetch_add(1, Ordering::Relaxed)
}

/// silero option names (as in `result.json`) → [`VadOptions`].
pub fn options_from_json(v: &Value) -> VadOptions {
    let d = VadOptions::default();
    let f = |k: &str| v.get(k).and_then(Value::as_f64);
    let u = |k: &str, x: u32| v.get(k).and_then(Value::as_u64).map_or(x, |n| n as u32);
    VadOptions {
        threshold: f("threshold").map_or(d.threshold, |x| x as f32),
        neg_threshold: f("neg_threshold").map(|x| x as f32),
        min_speech_duration_ms: u("min_speech_duration_ms", d.min_speech_duration_ms),
        min_silence_duration_ms: u("min_silence_duration_ms", d.min_silence_duration_ms),
        speech_pad_ms: u("speech_pad_ms", d.speech_pad_ms),
        max_speech_duration_s: f("max_speech_duration_s").map_or(d.max_speech_duration_s, |x| x as f32),
    }
}

fn segments_from_json(v: &Value) -> Vec<SpeechSegment> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|s| Some(SpeechSegment { start: s.get(0)?.as_u64()? as usize, end: s.get(1)?.as_u64()? as usize }))
                .collect()
        })
        .unwrap_or_default()
}

fn events_from_json(v: &Value) -> Vec<VadEvent> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|e| {
                    if let Some(s) = e.get("start").and_then(Value::as_u64) {
                        Some(VadEvent::SpeechStart { sample: s as usize })
                    } else {
                        e.get("end").and_then(Value::as_u64).map(|s| VadEvent::SpeechEnd { sample: s as usize })
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Stream `samples` in `chunk`-sized pieces, then zeros up to a whole window (what the
/// reference `VADIterator` loop is fed), then flush: (events before flush, flush events,
/// probabilities).
pub fn run_stream(vad: &Vad, samples: &[f32], chunk: usize, opts: &VadOptions) -> Result<(Vec<VadEvent>, Vec<VadEvent>, Vec<f32>)> {
    let mut s = vad.stream(opts);
    let mut events = Vec::new();
    let mut probs = Vec::new();
    let pad = vec![0f32; (vad.window_size() - samples.len() % vad.window_size()) % vad.window_size()];
    for c in samples.chunks(chunk.max(1)).chain(std::iter::once(&pad[..])) {
        events.extend(s.accept_waveform(c)?);
        probs.extend_from_slice(s.last_probabilities());
    }
    let tail = s.flush()?;
    probs.extend_from_slice(s.last_probabilities());
    Ok((events, tail, probs))
}

fn verify_sample(dir: &Path, vad: &Vad, stages: &mut Stages) -> Result<Value> {
    let path = dir.join("result.json");
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| Error::new("io", format!("{}: {e} (run tools/reference/silero_vad_ref.py)", path.display())))?;
    let r: Value = serde_json::from_str(&raw)?;
    let audio = r.get("audio").and_then(Value::as_str).ok_or_else(|| Error::new("reference", "result.json: no \"audio\""))?;
    let samples = crate::audio::load_audio_16k_mono(&dir.join(audio))?;
    let acts = Activations::open(&dir.join("activations.safetensors"))?;
    let ref_probs = acts.get("probs")?;

    let probs = stages.time("probabilities", || vad.probabilities(&samples))?;
    let mut cmp = Comparison::new();
    cmp.compare("probs", &probs, &ref_probs, LAYER_TOLERANCE);
    let prob_max_abs = diff(&probs, &ref_probs).map(|d| d.max_abs).unwrap_or(f32::INFINITY);

    let k = acts.shape("prob")?.first().copied().unwrap_or(0);
    let tr = stages.time("trace", || vad.trace(&samples, k))?;
    for (name, ours) in [("stft", &tr.stft), ("encoder", &tr.encoder), ("lstm.h", &tr.h), ("lstm.c", &tr.c), ("prob", &tr.probs)] {
        cmp.compare(name, ours, &acts.get(name)?, LAYER_TOLERANCE);
    }

    let mut seg_rows = Vec::new();
    let mut segments_ok = true;
    for set in r.get("speech_timestamps").and_then(Value::as_array).into_iter().flatten() {
        let opts = options_from_json(set.get("options").unwrap_or(&Value::Null));
        let want = segments_from_json(set.get("segments").unwrap_or(&Value::Null));
        let got = speech_timestamps(&probs, samples.len(), vad.sample_rate(), vad.window_size(), &opts);
        let logic = speech_timestamps(&ref_probs, samples.len(), vad.sample_rate(), vad.window_size(), &opts);
        let (exact, logic_exact) = (got == want, logic == want);
        segments_ok &= exact && logic_exact;
        seg_rows.push(json!({
            "options": opts, "exact": exact, "logic_exact": logic_exact,
            "segments": got.iter().map(|s| [s.start, s.end]).collect::<Vec<_>>(),
            "reference": want.iter().map(|s| [s.start, s.end]).collect::<Vec<_>>(),
        }));
    }

    let mut it_rows = Vec::new();
    let mut events_ok = true;
    let mut stream_max_abs = 0f32;
    let mut stream_exact = true;
    for set in r.get("iterator").and_then(Value::as_array).into_iter().flatten() {
        let opts = options_from_json(set.get("options").unwrap_or(&Value::Null));
        let want = events_from_json(set.get("events").unwrap_or(&Value::Null));
        let (events, tail, sprobs) = stages.time("stream", || run_stream(vad, &samples, STREAM_CHUNK, &opts))?;
        let d = diff(&sprobs, &probs).map(|d| d.max_abs).unwrap_or(f32::INFINITY);
        stream_max_abs = stream_max_abs.max(d);
        stream_exact &= sprobs == probs;
        let exact = events == want;
        events_ok &= exact;
        it_rows.push(json!({ "options": opts, "exact": exact, "events": events, "flush": tail, "reference": want }));
    }

    let ok = cmp.ok() && prob_max_abs <= PROB_TOLERANCE && segments_ok && events_ok && stream_exact;
    Ok(json!({
        "sample": dir.display().to_string(),
        "ok": ok,
        "windows": probs.len(),
        "prob_max_abs": prob_max_abs,
        "segments_exact": segments_ok,
        "iterator_events_exact": events_ok,
        "stream_vs_batch_max_abs": stream_max_abs,
        "stream_vs_batch_exact": stream_exact,
        "first_failing_layer": cmp.first_failing(),
        "layers": cmp.layers,
        "notes": cmp.notes,
        "speech_timestamps": seg_rows,
        "iterator": it_rows,
    }))
}

/// Full verification as JSON; check `"ok"`.
pub fn verify(reference_dir: &Path, model: &str, device: Option<&str>) -> Result<Value> {
    let dev = lapid_verify::open_device(device)?;
    let mut stages = Stages::new();
    let t = Instant::now();
    let (vad, convert_ms) = open_model(model, &dev)?;
    stages.record("load", ms(t));
    if let Some(c) = convert_ms {
        stages.record("convert", c);
    }
    let rd = std::fs::read_dir(reference_dir).map_err(|e| Error::new("io", format!("{}: {e}", reference_dir.display())))?;
    let mut dirs: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|d| d.join("result.json").is_file()).collect();
    dirs.sort();
    if dirs.is_empty() {
        return Err(Error::new("not_found", format!("{}: no sample with result.json (run tools/reference/silero_vad_ref.py)", reference_dir.display())));
    }
    let samples: Vec<Value> = dirs.iter().map(|d| verify_sample(d, &vad, &mut stages)).collect::<Result<_>>()?;
    let ok = samples.iter().all(|s| s.get("ok").and_then(Value::as_bool).unwrap_or(false));
    Ok(json!({
        "ok": ok,
        "device": dev.description(),
        "model": model,
        "info": vad.info(),
        "prob_tolerance": PROB_TOLERANCE,
        "layer_tolerance": LAYER_TOLERANCE,
        "samples": samples,
        "timings_ms": stages.to_json(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/reference/vad")
    }

    /// Runs whenever the reference data exists (it is small: the real 1.2 MB model + 2 clips).
    #[test]
    fn vad_verify_reference_cpu() {
        let root = root();
        if !root.join("model/config.json").is_file() {
            eprintln!("skipping: {} missing (run tools/reference/silero_vad_ref.py)", root.display());
            return;
        }
        let model = root.join("model");
        let v = verify(&root, model.to_str().expect("utf8"), Some("cpu")).expect("verify");
        assert_eq!(v["ok"], true, "{}", serde_json::to_string_pretty(&v).unwrap_or_default());
    }

    /// Stream and whole-buffer probabilities agree bit for bit for any chunking (CPU).
    #[test]
    fn vad_stream_equals_batch() {
        let root = root();
        if !root.join("model/config.json").is_file() {
            return;
        }
        let (vad, _) = open_model(root.join("model").to_str().expect("utf8"), &Device::cpu()).expect("model");
        let samples = crate::audio::load_audio_16k_mono(&root.join("../../jfk.wav")).expect("audio");
        let samples = &samples[..512 * 40 + 300];
        let batch = vad.probabilities(samples).expect("probs");
        for chunk in [160, 512, 1000, 512 * 3, 512 * 9 + 7, 512 * 20] {
            let (_, _, p) = run_stream(&vad, samples, chunk, &VadOptions::default()).expect("stream");
            assert_eq!(p, batch, "chunk {chunk}");
        }
    }

    #[test]
    fn options_parse() {
        let o = options_from_json(&json!({"threshold": 0.6, "neg_threshold": 0.3, "max_speech_duration_s": 2.0}));
        assert_eq!(o.threshold, 0.6);
        assert_eq!(o.neg_threshold, Some(0.3));
        assert_eq!(o.min_speech_duration_ms, 250);
        assert_eq!(o.max_speech_duration_s, 2.0);
        assert_eq!(options_from_json(&json!({})), VadOptions::default());
    }
}

