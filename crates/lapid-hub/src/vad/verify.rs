//! Silero VAD on the generic audio harness (`lapid verify vad`, `lapid bench vad`).
//!
//! Per sample: per-window probabilities of the whole clip (max abs error <= 1e-4) and the
//! intermediates of the first windows (STFT magnitude, encoder, LSTM h / c, probability;
//! relative error <= 1e-4); `get_speech_timestamps` for each reference option set, exact from
//! our probabilities and (segment logic alone) from the reference ones; `VADIterator` events
//! of a stream fed in odd-sized chunks, exact; stream vs whole-buffer probabilities identical.
//! Bench: the input tiled to 60 s; whole-buffer probabilities, its device / host halves,
//! `process`, and streaming one window per call and 100 ms per call.

use std::path::Path;
use std::time::Instant;

use lapid::{Device, Result};
use lapid_convert::engine::ConvertSpec;
use lapid_nn::Precision;
use lapid_verify::{Descriptor, diff};
use serde_json::{Map, Value, json};

use super::model::LstmState;
use super::segments::{SpeechSegment, VadEvent, VadOptions, speech_timestamps};
use super::{BATCH_WINDOWS, Vad};
use crate::runtime::audio::{AudioDef, BenchRun, Check, EndToEnd, Row, Sample, Stages};

/// Max abs difference of a speech probability (fp32 weights).
const PROB_TOLERANCE: f32 = 1e-4;
/// Relative error of the probabilities and traced intermediates.
const LAYER_TOLERANCE: f32 = 1e-4;
/// Chunk size the stream check feeds (not a multiple of the window on purpose).
const STREAM_CHUNK: usize = 1000;
/// Seconds of audio benchmarked (the input is tiled to this length).
const BENCH_SECONDS: usize = 60;

/// silero option names (as in `result.json`) -> [`VadOptions`].
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

/// Stream `samples` in `chunk`-sized pieces, then zeros up to a whole window (what the
/// reference `VADIterator` loop is fed), then flush: (events before flush, flush events,
/// probabilities).
pub fn run_stream(vad: &Vad, samples: &[f32], chunk: usize, opts: &VadOptions) -> Result<(Vec<VadEvent>, Vec<VadEvent>, Vec<f32>)> {
    let mut s = vad.stream(opts);
    let (mut events, mut probs) = (Vec::new(), Vec::new());
    let w = vad.window_size();
    let pad = vec![0f32; (w - samples.len() % w) % w];
    for c in samples.chunks(chunk.max(1)).chain(std::iter::once(&pad[..])) {
        events.extend(s.accept_waveform(c)?);
        probs.extend_from_slice(s.last_probabilities());
    }
    let tail = s.flush()?;
    probs.extend_from_slice(s.last_probabilities());
    Ok((events, tail, probs))
}

fn segments_of(v: &Value) -> Vec<SpeechSegment> {
    let seg = |s: &Value| Some(SpeechSegment { start: s.get(0)?.as_u64()? as usize, end: s.get(1)?.as_u64()? as usize });
    v.as_array().map(|a| a.iter().filter_map(seg).collect()).unwrap_or_default()
}

fn events_of(v: &Value) -> Vec<VadEvent> {
    let ev = |e: &Value| match e.get("start").and_then(Value::as_u64) {
        Some(s) => Some(VadEvent::SpeechStart { sample: s as usize }),
        None => e.get("end").and_then(Value::as_u64).map(|s| VadEvent::SpeechEnd { sample: s as usize }),
    };
    v.as_array().map(|a| a.iter().filter_map(ev).collect()).unwrap_or_default()
}

/// (front-end ms, recurrence ms) of the whole-buffer path, timed separately.
fn split_ms(vad: &Vad, samples: &[f32]) -> Result<(f64, f64)> {
    let (net, cfg) = (vad.net(), vad.config());
    let (w, c, n) = (vad.window_size(), cfg.context_size, samples.len().div_ceil(vad.window_size()));
    let zeros = vec![0f32; c];
    let (mut front_ms, mut rec_ms, mut inputs) = (0.0, 0.0, Vec::new());
    let mut state = LstmState::zeros(cfg.hidden_size);
    for first in (0..n).step_by(BATCH_WINDOWS) {
        let count = BATCH_WINDOWS.min(n - first);
        inputs.clear();
        let t = Instant::now();
        for i in first..first + count {
            let s = i * w;
            net.push_input(if i == 0 { &zeros[..] } else { &samples[s - c..s] }, &samples[s..samples.len().min(s + w)], &mut inputs);
        }
        let f = net.front(crate::model::Model::device(vad), &inputs, count, false)?;
        front_ms += t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let acc: f32 = f.gates.chunks_exact(4 * cfg.hidden_size).map(|gx| net.step(&mut state, gx)).sum();
        std::hint::black_box(acc);
        rec_ms += t.elapsed().as_secs_f64() * 1e3;
    }
    Ok((front_ms, rec_ms))
}

fn stream_ms(vad: &Vad, samples: &[f32], chunk: usize) -> Result<f64> {
    let t = Instant::now();
    let mut s = vad.stream(&VadOptions::default());
    for c in samples.chunks(chunk) {
        s.accept_waveform(c)?;
    }
    s.flush()?;
    Ok(t.elapsed().as_secs_f64() * 1e3)
}

impl AudioDef for Vad {
    const DESCRIPTOR: Descriptor = Descriptor {
        name: "vad",
        default_model: "data/reference/vad/model",
        default_dtype: "fp32",
        reference_dir: "data/reference/vad",
        reference_script: "tools/reference/silero_vad_ref.py",
        reference_label: "silero-vad 5.1.2 TorchScript fp32",
        strict_dtypes: &["fp32"],
        bench_prompt_tokens: 0,
        bench_gen_tokens: 0,
    };
    const CONVERT: &'static ConvertSpec = &super::convert::CONVERT;

    fn load_with(path: &Path, device: &Device, _: Precision) -> Result<Self> {
        Vad::load(path, device)
    }

    fn sample_rate(&self) -> u32 {
        self.config().sample_rate
    }

    fn stages(&self, s: &Sample, audio: &[f32]) -> Result<Stages> {
        let probs = self.probabilities(audio)?;
        let k = s.acts.shape("prob")?.first().copied().unwrap_or(0);
        let tr = self.trace(audio, k)?;
        let mut rows = vec![Row::front("probs", probs.clone(), LAYER_TOLERANCE)];
        for (name, v) in [("stft", tr.stft), ("encoder", tr.encoder), ("lstm.h", tr.h), ("lstm.c", tr.c), ("prob", tr.probs)] {
            rows.push(Row::front(name, v, LAYER_TOLERANCE));
        }
        let max_abs = diff(&probs, &s.get("probs")?).map_or(f32::INFINITY, |d| d.max_abs);
        Ok(Stages { rows, argmax: Vec::new(), checks: vec![Check::new("prob_max_abs", max_abs <= PROB_TOLERANCE, true)] })
    }

    /// Segments and stream events for every reference option set.
    fn end_to_end(&self, s: &Sample, audio: &[f32]) -> Result<EndToEnd> {
        let (probs, ref_probs) = (self.probabilities(audio)?, s.get("probs")?);
        let (sr, w) = (self.sample_rate(), self.window_size());
        let (mut segments_ok, mut events_ok, mut stream_exact) = (true, true, true);
        let mut info = Map::new();
        let mut seg_rows = Vec::new();
        for set in s.result["speech_timestamps"].as_array().into_iter().flatten() {
            let opts = options_from_json(&set["options"]);
            let want = segments_of(&set["segments"]);
            let got = speech_timestamps(&probs, audio.len(), sr, w, &opts);
            let logic = speech_timestamps(&ref_probs, audio.len(), sr, w, &opts);
            segments_ok &= got == want && logic == want;
            seg_rows.push(json!({"options": opts, "exact": got == want, "logic_exact": logic == want}));
        }
        let mut it_rows = Vec::new();
        for set in s.result["iterator"].as_array().into_iter().flatten() {
            let opts = options_from_json(&set["options"]);
            let (events, tail, sprobs) = run_stream(self, audio, STREAM_CHUNK, &opts)?;
            stream_exact &= sprobs == probs;
            events_ok &= events == events_of(&set["events"]);
            it_rows.push(json!({"options": opts, "exact": events == events_of(&set["events"]), "events": events, "flush": tail}));
        }
        info.insert("windows".into(), json!(probs.len()));
        info.insert("speech_timestamps".into(), Value::Array(seg_rows));
        info.insert("iterator".into(), Value::Array(it_rows));
        let checks = vec![
            Check::new("segments_exact", segments_ok, true),
            Check::new("iterator_events_exact", events_ok, true),
            Check::new("stream_vs_batch_exact", stream_exact, true),
        ];
        Ok(EndToEnd { checks, info, ..EndToEnd::default() })
    }

    fn bench_input(&self, audio: Vec<f32>) -> Vec<f32> {
        audio.iter().copied().cycle().take(BENCH_SECONDS * self.sample_rate() as usize).collect()
    }

    fn bench_run(&self, audio: &[f32]) -> Result<BenchRun> {
        let t = Instant::now();
        self.probabilities(audio)?;
        let probabilities = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let segments = self.process(audio, &VadOptions::default())?;
        let process = t.elapsed().as_secs_f64() * 1e3;
        let (front, recurrence) = split_ms(self, audio)?;
        let (stream_512, stream_100ms) = (stream_ms(self, audio, 512)?, stream_ms(self, audio, 1600)?);
        Ok(BenchRun {
            times: vec![("total", probabilities), ("probabilities", probabilities), ("front", front), ("recurrence", recurrence), ("process", process), ("stream_512", stream_512), ("stream_100ms", stream_100ms)],
            values: vec![("segments", segments.len() as f64)],
            text: String::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/reference/vad")
    }

    /// Runs whenever the reference data exists (the real 1.2 MB model + 2 clips).
    #[test]
    fn vad_verify_reference_cpu() {
        if !root().join("model/config.json").is_file() {
            return;
        }
        let args = crate::cmd::ModelArgs { model: Some(root().join("model").display().to_string()), dtype: None, device: Some("cpu".into()) };
        let v = crate::runtime::audio_verify::verify::<Vad>(Some(&root()), &args).expect("verify");
        assert_eq!(v["ok"], true, "{}", serde_json::to_string_pretty(&v).unwrap_or_default());
    }

    /// Stream and whole-buffer probabilities agree bit for bit for any chunking (CPU).
    #[test]
    fn vad_stream_equals_batch() {
        if !root().join("model/config.json").is_file() {
            return;
        }
        let path = std::env::temp_dir().join(format!("lapid-vad-stream-{}.lapid", std::process::id()));
        super::super::convert::CONVERT.convert(root().join("model").to_str().expect("utf8"), lapid_convert::OutDType::F32, &path).expect("convert");
        let vad = Vad::load(&path, &Device::cpu()).expect("model");
        std::fs::remove_file(&path).ok();
        let samples = crate::audio::load_audio_16k_mono(&root().join("../../jfk.wav")).expect("audio");
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
        assert_eq!((o.threshold, o.neg_threshold, o.min_speech_duration_ms, o.max_speech_duration_s), (0.6, Some(0.3), 250, 2.0));
        assert_eq!(options_from_json(&json!({})), VadOptions::default());
    }
}
