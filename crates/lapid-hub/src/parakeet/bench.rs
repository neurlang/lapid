//! `bench parakeet`: stage timings over N runs after one warmup run.
//!
//! Per run: `features` (log-mel of the whole signal), `encode` (all encoder windows,
//! realized), `decode` (greedy TDT host loop), `total` = the sum (what `transcribe` does).

use std::path::Path;
use std::time::Instant;

use lapid::{Error, Result};
use lapid_verify::timing::{median, ms, stats};
use serde::Serialize;
use serde_json::{Value, json};

use super::decode::{self, Path as DecodePath};
use super::transcribe::encode_long;
use super::verify::resolve_model;
use super::{Parakeet, TranscribeOptions};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct BenchRun {
    pub features_ms: f64,
    pub encode_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    pub tokens: usize,
    pub steps: usize,
}

fn one_run(p: &Parakeet, samples: &[f32], opts: &TranscribeOptions) -> Result<(BenchRun, String, usize)> {
    let t = Instant::now();
    let mel = p.features(samples);
    let features_ms = ms(t);
    if mel.n_frames == 0 {
        return Err(Error::new("invalid_argument", "audio shorter than one feature frame"));
    }
    let t = Instant::now();
    let (enc, windows) = encode_long(p, &mel, opts)?;
    let encode_ms = ms(t);
    let t = Instant::now();
    let frames = enc.shape().get(1).copied().unwrap_or(0);
    let d = p.decode_frames(&enc, frames, DecodePath::Greedy, false)?;
    let decode_ms = ms(t);
    let toks = decode::tokens(p.tokenizer(), &d.emitted(p.config().blank_id), p.config().frame_seconds(), p.config().blank_id);
    let run = BenchRun {
        features_ms,
        encode_ms,
        decode_ms,
        total_ms: features_ms + encode_ms + decode_ms,
        tokens: toks.len(),
        steps: d.steps.len(),
    };
    Ok((run, decode::text(&toks), windows))
}

/// Benchmark report as JSON.
pub fn bench(audio: &Path, model: &str, dtype: &str, device: Option<&str>, runs: usize) -> Result<Value> {
    if runs == 0 {
        return Err(Error::new("invalid_argument", "--runs must be at least 1"));
    }
    let dev = lapid_verify::open_device(device)?;
    let (path, convert_ms) = resolve_model(model, dtype)?;
    let t = Instant::now();
    let p = Parakeet::load(&path, &dev)?;
    let load_ms = ms(t);
    let samples = lapid_nn::audio::load_audio(audio, p.config().sample_rate)?;
    let audio_seconds = samples.len() as f64 / p.config().sample_rate as f64;
    let opts = TranscribeOptions::default();

    let t = Instant::now();
    let (_, mut text, windows) = one_run(&p, &samples, &opts)?;
    let warmup_ms = ms(t);
    let mut measured = Vec::with_capacity(runs);
    for _ in 0..runs {
        let (r, tx, _) = one_run(&p, &samples, &opts)?;
        measured.push(r);
        text = tx;
    }
    let col = |f: fn(&BenchRun) -> f64| stats(&measured.iter().map(f).collect::<Vec<_>>());
    let total = median(measured.iter().map(|r| r.total_ms).collect());
    let rtf = if audio_seconds > 0.0 { total / 1e3 / audio_seconds } else { 0.0 };
    Ok(json!({
        "engine": "lapid",
        "model": model,
        "model_file": path.display().to_string(),
        "dtype": dtype,
        "device": dev.description(),
        "backend": dev.name(),
        "audio": audio.display().to_string(),
        "audio_seconds": audio_seconds,
        "encoder_windows": windows,
        "runs": runs,
        "text": text,
        "convert_ms": convert_ms,
        "load_ms": load_ms,
        "warmup_ms": warmup_ms,
        "features_ms": col(|r| r.features_ms),
        "encode_ms": col(|r| r.encode_ms),
        "decode_ms": col(|r| r.decode_ms),
        "total_ms": col(|r| r.total_ms),
        "tokens": measured.last().map_or(0, |r| r.tokens),
        "decode_steps": measured.last().map_or(0, |r| r.steps),
        "realtime_factor": rtf,
        "x_realtime": if total > 0.0 { audio_seconds / (total / 1e3) } else { 0.0 },
        "per_run": measured,
    }))
}
