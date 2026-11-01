//! `lapid bench phonon`: time to first audio and real-time factor over N runs after one
//! warmup run (fixed seed, so every run generates the same frames).

use std::time::Instant;

use lapid::Result;
use lapid_verify::timing::{median, ms, stats};
use serde::Serialize;
use serde_json::{Value, json};

use super::verify::resolve_model;
use super::{Phonon, SpeakOptions};

pub const DEFAULT_TEXT: &str = "Hello from Phonon. This is a short benchmark of streaming speech synthesis \
     on the lapid runtime, measuring the time to the first audio chunk and the real-time factor.";

/// One measured run.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BenchRun {
    /// Time from the call to the first PCM chunk (includes the prompt prefill).
    pub first_audio_ms: f64,
    pub total_ms: f64,
    pub frames: usize,
    pub audio_s: f64,
    /// Compute time / audio duration (< 1 = faster than real time).
    pub rtf: f64,
    /// Mean time per frame after the first one.
    pub frame_ms: f64,
}

fn one_run(tts: &Phonon, text: &str, opts: &SpeakOptions) -> Result<BenchRun> {
    let t = Instant::now();
    let mut stream = tts.speak_stream(text, opts)?;
    let mut first = None;
    let mut samples = 0usize;
    for pcm in stream.by_ref() {
        samples += pcm?.len();
        if first.is_none() {
            first = Some(ms(t));
        }
    }
    let total_ms = ms(t);
    let frames = stream.stats().frames;
    let audio_s = samples as f64 / tts.sample_rate() as f64;
    let first_audio_ms = first.unwrap_or(total_ms);
    Ok(BenchRun {
        first_audio_ms,
        total_ms,
        frames,
        audio_s,
        rtf: if audio_s > 0.0 { total_ms / 1e3 / audio_s } else { 0.0 },
        frame_ms: if frames > 1 { (total_ms - first_audio_ms) / (frames - 1) as f64 } else { 0.0 },
    })
}

/// Benchmark report as JSON.
pub fn bench(text: Option<&str>, model: &str, dtype: &str, device: Option<&str>, runs: usize) -> Result<Value> {
    let dev = lapid_verify::open_device(device)?;
    let (path, convert_ms) = resolve_model(model, dtype)?;
    let t = Instant::now();
    let tts = Phonon::load(&path, &dev)?;
    let load_ms = ms(t);
    let text = text.unwrap_or(DEFAULT_TEXT);
    let opts = SpeakOptions { seed: Some(0), ..Default::default() };
    let warmup = one_run(&tts, text, &opts)?;
    let mut measured = Vec::new();
    for _ in 0..runs.max(1) {
        measured.push(one_run(&tts, text, &opts)?);
    }
    let col = |f: fn(&BenchRun) -> f64| measured.iter().map(f).collect::<Vec<_>>();
    Ok(json!({
        "device": dev.description(),
        "model": model,
        "model_file": path.display().to_string(),
        "dtype": dtype,
        "text": text,
        "convert_ms": convert_ms,
        "load_ms": load_ms,
        "warmup": warmup,
        "runs": measured,
        "first_audio_ms": stats(&col(|r| r.first_audio_ms)),
        "rtf": stats(&col(|r| r.rtf)),
        "frame_ms": stats(&col(|r| r.frame_ms)),
        "median_first_audio_ms": median(col(|r| r.first_audio_ms)),
        "median_rtf": median(col(|r| r.rtf)),
    }))
}
