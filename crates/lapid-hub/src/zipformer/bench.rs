//! `bench zipformer`: streaming recognition of a WAV fed in 100 ms pieces as fast as possible,
//! one warmup run then N measured runs.
//!
//! Per run: wall time (RTF = wall / audio), per-chunk encoder / decode time, and the first
//! partial: how much audio had been fed when the first token appeared (`first_partial_audio_s`,
//! the algorithmic latency: chunk + 13 frames of look-ahead), the token's own time
//! (`first_token_time_s`) and the compute of that call (`first_partial_compute_ms`).

use std::path::Path;
use std::time::Instant;

use lapid::Result;
use lapid_verify::timing::{median, ms, stats};
use serde::Serialize;
use serde_json::{Value, json};

use super::verify::resolve_model;
use super::{StreamOptions, Zipformer};
use crate::model::Model;

/// Samples per `accept_waveform` call (100 ms at 16 kHz).
pub const PIECE: usize = 1600;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct BenchRun {
    pub total_ms: f64,
    pub fbank_ms: f64,
    pub encoder_ms: f64,
    pub decode_ms: f64,
    pub chunks: usize,
    pub encoder_ms_per_chunk: f64,
    pub first_partial_audio_s: f64,
    pub first_token_time_s: f64,
    pub first_partial_compute_ms: f64,
}

fn one_run(z: &Zipformer, samples: &[f32], opts: &StreamOptions) -> Result<(BenchRun, String)> {
    let sr = z.config().sample_rate as f64;
    let mut s = z.stream(opts)?;
    let t = Instant::now();
    let mut fed = 0usize;
    let mut first: Option<(f64, f64, f64)> = None;
    for piece in samples.chunks(PIECE) {
        let tc = Instant::now();
        let out = s.accept_waveform(piece)?;
        fed += piece.len();
        if first.is_none() {
            if let Some(r) = out.iter().find(|r| !r.tokens.is_empty()) {
                first = Some((fed as f64 / sr, f64::from(r.timestamps.first().copied().unwrap_or(0.0)), ms(tc)));
            }
        }
    }
    let tc = Instant::now();
    let out = s.input_finished()?;
    if first.is_none() {
        if let Some(r) = out.iter().find(|r| !r.tokens.is_empty()) {
            first = Some((fed as f64 / sr, f64::from(r.timestamps.first().copied().unwrap_or(0.0)), ms(tc)));
        }
    }
    let total_ms = ms(t);
    let st = s.stats();
    let (fa, ft, fc) = first.unwrap_or((f64::NAN, f64::NAN, f64::NAN));
    let run = BenchRun {
        total_ms,
        fbank_ms: st.fbank_ms,
        encoder_ms: st.encoder_ms,
        decode_ms: st.decode_ms,
        chunks: st.chunks,
        encoder_ms_per_chunk: if st.chunks > 0 { st.encoder_ms / st.chunks as f64 } else { 0.0 },
        first_partial_audio_s: fa,
        first_token_time_s: ft,
        first_partial_compute_ms: fc,
    };
    Ok((run, s.result().text))
}

/// Benchmark report as JSON.
pub fn bench(audio: &Path, model: &str, dtype: &str, device: Option<&str>, runs: usize, chunk: Option<usize>) -> Result<Value> {
    let dev = lapid_verify::open_device(device)?;
    let path = resolve_model(model, dtype)?;
    let t = Instant::now();
    let z = Zipformer::load(&path, &dev)?;
    let load_ms = ms(t);
    let samples = lapid_nn::audio::load_audio(audio, z.config().sample_rate)?;
    let audio_seconds = samples.len() as f64 / z.config().sample_rate as f64;
    let opts = StreamOptions { chunk_size: chunk, left_context_frames: None, endpoint: None };
    let t = Instant::now();
    let (_, mut text) = one_run(&z, &samples, &opts)?;
    let warmup_ms = ms(t);
    let mut measured = Vec::with_capacity(runs.max(1));
    for _ in 0..runs.max(1) {
        let (r, tx) = one_run(&z, &samples, &opts)?;
        measured.push(r);
        text = tx;
    }
    let col = |f: fn(&BenchRun) -> f64| stats(&measured.iter().map(f).collect::<Vec<_>>());
    let total = median(measured.iter().map(|r| r.total_ms).collect());
    Ok(json!({
        "engine": "lapid",
        "model": model,
        "model_file": path.display().to_string(),
        "name": z.info().name,
        "dtype": dtype,
        "device": dev.description(),
        "backend": dev.name(),
        "audio": audio.display().to_string(),
        "audio_seconds": audio_seconds,
        "chunk_size": chunk.unwrap_or(z.config().chunk_size),
        "left_context_frames": z.config().left_context_frames,
        "runs": runs,
        "text": text,
        "load_ms": load_ms,
        "warmup_ms": warmup_ms,
        "total_ms": col(|r| r.total_ms),
        "fbank_ms": col(|r| r.fbank_ms),
        "encoder_ms": col(|r| r.encoder_ms),
        "decode_ms": col(|r| r.decode_ms),
        "encoder_ms_per_chunk": col(|r| r.encoder_ms_per_chunk),
        "first_partial_audio_s": col(|r| r.first_partial_audio_s),
        "first_token_time_s": col(|r| r.first_token_time_s),
        "first_partial_compute_ms": col(|r| r.first_partial_compute_ms),
        "realtime_factor": if audio_seconds > 0.0 { total / 1e3 / audio_seconds } else { 0.0 },
        "per_run": measured,
    }))
}
