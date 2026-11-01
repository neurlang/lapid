//! `lapid bench <audio model>`: the generic driver over [`AudioDef`]: resolve and load the
//! model, read the input at the model's sample rate, one warmup run, then `runs` measured
//! runs of [`AudioDef::bench_run`]. Every stage time and value gets min / median / mean;
//! the real-time factor is the median `total` over the audio length.

use std::path::Path;
use std::time::Instant;

use lapid::{Result, bail};
use lapid_nn::Precision;
use lapid_verify::open_device;
use lapid_verify::timing::{median, ms, stats};
use serde_json::{Map, Value, json};

use super::audio::{AudioDef, BenchRun};
use super::resolve::{file_dtype, resolve};
use crate::cmd::ModelArgs;

fn column(runs: &[BenchRun], key: &str, times: bool) -> Vec<f64> {
    runs.iter()
        .filter_map(|r| (if times { &r.times } else { &r.values }).iter().find(|(k, _)| *k == key).map(|(_, v)| *v))
        .collect()
}

/// Benchmark `M` on `input` (default [`AudioDef::BENCH_AUDIO`]).
pub fn bench<M: AudioDef>(input: Option<&Path>, args: &ModelArgs, runs: usize) -> Result<Value> {
    let d = &M::DESCRIPTOR;
    let dev = open_device(args.device.as_deref())?;
    let model = args.model.as_deref().unwrap_or(d.default_model);
    let r = resolve(model, Some(args.dtype.as_deref().unwrap_or(d.default_dtype)))?;
    let t = Instant::now();
    let m = M::load_with(&r.path, &dev, Precision::Stored)?;
    let load_ms = ms(t);
    let audio_path = input.unwrap_or(Path::new(M::BENCH_AUDIO));
    let audio = m.bench_input(lapid_nn::audio::load_audio(audio_path, m.sample_rate())?);
    if audio.is_empty() {
        bail!("invalid_argument", "{}: no audio samples", audio_path.display());
    }
    let seconds = audio.len() as f64 / f64::from(m.sample_rate());
    let t = Instant::now();
    let mut text = m.bench_run(&audio)?.text;
    let warmup_ms = ms(t);
    let mut measured = Vec::with_capacity(runs);
    for _ in 0..runs.max(1) {
        let run = m.bench_run(&audio)?;
        text = run.text.clone();
        measured.push(run);
    }
    let total = median(column(&measured, "total", true));
    let mut out = Map::new();
    let first = &measured[0];
    for (k, _) in &first.times {
        out.insert(format!("{k}_ms"), stats(&column(&measured, k, true)));
    }
    for (k, _) in &first.values {
        out.insert((*k).to_string(), stats(&column(&measured, k, false)));
    }
    let per_run: Vec<Value> = measured
        .iter()
        .map(|r| {
            let mut o: Map<String, Value> = r.times.iter().map(|(k, v)| (format!("{k}_ms"), json!(v))).collect();
            o.extend(r.values.iter().map(|(k, v)| ((*k).to_string(), json!(v))));
            Value::Object(o)
        })
        .collect();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut v = json!({
        "engine": "lapid",
        "model": model,
        "model_file": r.path.display().to_string(),
        "dtype": file_dtype(&r.path)?,
        "device": dev.description(),
        "backend": dev.name(),
        "host": { "arch": std::env::consts::ARCH, "os": std::env::consts::OS, "threads": threads },
        "audio": audio_path.display().to_string(),
        "audio_seconds": seconds,
        "runs": runs,
        "text": text,
        "convert_ms": r.convert_ms,
        "load_ms": load_ms,
        "warmup_ms": warmup_ms,
        "realtime_factor": if seconds > 0.0 { total / 1e3 / seconds } else { 0.0 },
        "x_realtime": if total > 0.0 { seconds / (total / 1e3) } else { 0.0 },
        "per_run": per_run,
    });
    if let Value::Object(o) = &mut v {
        o.extend(out);
        o.extend(m.bench_info());
    }
    Ok(v)
}
