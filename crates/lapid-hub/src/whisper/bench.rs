//! `lapid bench whisper`: stage timings over N runs after one warmup run.
//!
//! Per run: `log_mel` (mel), `encode` (encode, realized), then a full `transcribe` from the
//! samples (total). Decode time is `total − mel − encode`, exact for audio ≤ 30 s (one
//! window) and an estimate for long-form audio (flagged in the JSON).

use std::path::Path;
use std::time::Instant;

use lapid::{Error, Result};
use lapid_verify::timing::{median, ms, stats};
use serde::Serialize;
use serde_json::{Value, json};

use super::verify::resolve_model;
use super::{Task, TranscribeOptions, Whisper};
const WINDOW_SECONDS: f64 = 30.0;

/// Timings of one measured run (milliseconds).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BenchRun {
    pub mel_ms: f64,
    pub encode_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    pub tokens: usize,
    pub tokens_per_s: f64,
}

fn run_stats(runs: &[BenchRun], f: impl Fn(&BenchRun) -> f64) -> Value {
    stats(&runs.iter().map(f).collect::<Vec<_>>())
}

fn options() -> TranscribeOptions {
    // whisper-cli defaults to English; match it so both engines skip language detection.
    TranscribeOptions { language: Some("en".to_string()), task: Task::Transcribe, timestamps: true }
}

fn one_run(whisper: &Whisper, samples: &[f32], opts: &TranscribeOptions) -> Result<(BenchRun, String)> {
    let t = Instant::now();
    let mel = whisper.log_mel(samples);
    let mel_ms = ms(t);

    let t = Instant::now();
    let encoded = whisper.encode(&mel)?;
    let encode_ms = ms(t);
    drop(encoded);

    let t = Instant::now();
    let transcript = whisper.transcribe(samples, opts)?;
    let total_ms = ms(t);

    let tokens: usize = transcript.segments.iter().map(|s| s.tokens.len()).sum();
    let decode_ms = (total_ms - mel_ms - encode_ms).max(0.0);
    let tokens_per_s = if decode_ms > 0.0 { tokens as f64 / (decode_ms / 1e3) } else { 0.0 };
    Ok((BenchRun { mel_ms, encode_ms, decode_ms, total_ms, tokens, tokens_per_s }, transcript.text.trim().to_string()))
}

/// Benchmark report as JSON (see module docs for the stage definitions).
pub fn bench(audio: &Path, model: &str, dtype: &str, device: Option<&str>, runs: usize) -> Result<Value> {
    if runs == 0 {
        return Err(Error::new("invalid_argument", "--runs must be at least 1"));
    }
    let dev = lapid_verify::open_device(device)?;

    let t = Instant::now();
    let samples = crate::audio::load_audio_16k_mono(audio)?;
    let audio_load_ms = ms(t);
    let audio_seconds = samples.len() as f64 / crate::audio::SAMPLE_RATE as f64;
    if samples.is_empty() {
        return Err(Error::new("invalid_argument", format!("{}: no audio samples", audio.display())));
    }

    // Convert first (if needed) so `load` measures loading, not conversion.
    let (path, convert_ms) = resolve_model(model, dtype)?;

    let t = Instant::now();
    let whisper = Whisper::load(&path, &dev)?;
    let load_ms = ms(t);

    let opts = options();
    let t = Instant::now();
    let (_, warm_text) = one_run(&whisper, &samples, &opts)?;
    let warmup_ms = ms(t);

    let mut measured = Vec::with_capacity(runs);
    let mut text = warm_text;
    for _ in 0..runs {
        let (run, t) = one_run(&whisper, &samples, &opts)?;
        measured.push(run);
        text = t;
    }

    let total_median = median(measured.iter().map(|r| r.total_ms).collect());
    let x_realtime = if total_median > 0.0 { audio_seconds / (total_median / 1e3) } else { 0.0 };
    let rtf = if audio_seconds > 0.0 { (total_median / 1e3) / audio_seconds } else { 0.0 };
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    Ok(json!({
        "engine": "lapid",
        "model": model,
        "dtype": dtype,
        "device": dev.description(),
        "backend": dev.name(),
        "host": { "arch": std::env::consts::ARCH, "os": std::env::consts::OS, "threads": threads },
        "audio": audio.display().to_string(),
        "audio_seconds": audio_seconds,
        "runs": runs,
        "text": text,
        "convert_ms": convert_ms,
        "audio_load_ms": audio_load_ms,
        "load_ms": load_ms,
        "warmup_ms": warmup_ms,
        "mel_ms": run_stats(&measured, |r| r.mel_ms),
        "encode_ms": run_stats(&measured, |r| r.encode_ms),
        "decode_ms": run_stats(&measured, |r| r.decode_ms),
        "total_ms": run_stats(&measured, |r| r.total_ms),
        "tokens": measured.last().map(|r| r.tokens).unwrap_or(0),
        "tokens_per_s": run_stats(&measured, |r| r.tokens_per_s),
        "realtime_factor": rtf,
        "x_realtime": x_realtime,
        "decode_estimated": audio_seconds > WINDOW_SECONDS,
        "per_run": measured,
    }))
}
