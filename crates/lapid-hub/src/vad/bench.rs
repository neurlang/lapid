//! `bench vad`: real-time factor over N runs after one warmup run, on `audio` repeated to
//! [`BENCH_SECONDS`].
//!
//! Rows: `probabilities` (whole buffer: batched device front-end + host recurrence), its two
//! halves (`front_ms`: device graphs, `recurrence_ms`: host LSTM + head), `process`
//! (probabilities + segments), and two streaming modes: `stream_512` (one window per call =
//! one device evaluation per window, the worst case) and `stream_100ms` (1600-sample chunks).

use std::path::Path;
use std::time::Instant;

use lapid::Result;
use lapid_verify::timing::{median, ms, stats};
use serde_json::{Value, json};

use super::model::LstmState;
use super::verify::open_model;
use super::{BATCH_WINDOWS, Vad, VadOptions};
use crate::model::Model;

/// Seconds of audio benchmarked (the input is tiled to this length).
pub const BENCH_SECONDS: usize = 60;

fn stream_ms(vad: &Vad, samples: &[f32], chunk: usize) -> Result<f64> {
    let t = Instant::now();
    let mut s = vad.stream(&VadOptions::default());
    for c in samples.chunks(chunk) {
        s.accept_waveform(c)?;
    }
    s.flush()?;
    Ok(ms(t))
}

/// (front-end ms, recurrence ms) of the whole-buffer path, timed separately.
fn split_ms(vad: &Vad, samples: &[f32]) -> Result<(f64, f64)> {
    let net = vad.net();
    let (w, c) = (vad.window_size(), vad.config().context_size);
    let n = samples.len().div_ceil(w);
    let zeros = vec![0f32; c];
    let (mut front_ms, mut rec_ms) = (0.0, 0.0);
    let mut state = LstmState::zeros(vad.config().hidden_size);
    let mut inputs = Vec::new();
    for first in (0..n).step_by(BATCH_WINDOWS) {
        let count = BATCH_WINDOWS.min(n - first);
        inputs.clear();
        let t = Instant::now();
        for i in first..first + count {
            let s = i * w;
            let ctx = if i == 0 { &zeros[..] } else { &samples[s - c..s] };
            net.push_input(ctx, &samples[s..samples.len().min(s + w)], &mut inputs);
        }
        let f = net.front(vad.device(), &inputs, count, false)?;
        front_ms += ms(t);
        let t = Instant::now();
        let mut acc = 0f32;
        for gx in f.gates.chunks_exact(4 * vad.config().hidden_size) {
            acc += net.step(&mut state, gx);
        }
        std::hint::black_box(acc);
        rec_ms += ms(t);
    }
    Ok((front_ms, rec_ms))
}

/// Benchmark report as JSON.
pub fn bench(audio: &Path, model: &str, device: Option<&str>, runs: usize) -> Result<Value> {
    if runs == 0 {
        return Err(lapid::Error::new("invalid_argument", "--runs must be at least 1"));
    }
    let dev = lapid_verify::open_device(device)?;
    let t = Instant::now();
    let (vad, _) = open_model(model, &dev)?;
    let load_ms = ms(t);
    let clip = crate::audio::load_audio_16k_mono(audio)?;
    if clip.is_empty() {
        return Err(lapid::Error::new("invalid_argument", format!("{}: no samples", audio.display())));
    }
    let target = BENCH_SECONDS * vad.sample_rate() as usize;
    let samples: Vec<f32> = clip.iter().copied().cycle().take(target).collect();
    let seconds = samples.len() as f64 / vad.sample_rate() as f64;

    let t = Instant::now();
    let segments = vad.process(&samples, &VadOptions::default())?;
    let warmup_ms = ms(t);
    stream_ms(&vad, &samples[..samples.len().min(16_000)], 512)?;

    let (mut probs, mut process, mut front, mut rec, mut s512, mut s1600) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..runs {
        let t = Instant::now();
        vad.probabilities(&samples)?;
        probs.push(ms(t));
        let t = Instant::now();
        vad.process(&samples, &VadOptions::default())?;
        process.push(ms(t));
        let (f, r) = split_ms(&vad, &samples)?;
        front.push(f);
        rec.push(r);
        s512.push(stream_ms(&vad, &samples, 512)?);
        s1600.push(stream_ms(&vad, &samples, 1600)?);
    }
    let rtf = |v: &[f64]| median(v.to_vec()) / 1e3 / seconds;
    Ok(json!({
        "engine": "lapid",
        "model": model,
        "device": dev.description(),
        "backend": dev.name(),
        "audio": audio.display().to_string(),
        "audio_seconds": seconds,
        "windows": samples.len().div_ceil(vad.window_size()),
        "runs": runs,
        "load_ms": load_ms,
        "warmup_ms": warmup_ms,
        "segments": segments.len(),
        "probabilities_ms": stats(&probs),
        "front_ms": stats(&front),
        "recurrence_ms": stats(&rec),
        "process_ms": stats(&process),
        "stream_512_ms": stats(&s512),
        "stream_100ms_ms": stats(&s1600),
        "rtf": {
            "probabilities": rtf(&probs),
            "process": rtf(&process),
            "stream_512": rtf(&s512),
            "stream_100ms": rtf(&s1600),
        },
    }))
}
