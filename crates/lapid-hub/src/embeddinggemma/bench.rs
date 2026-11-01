//! `lapid bench embeddinggemma`: sentences per second at batch 1 and batch 32.
//!
//! Corpus: [`CORPUS`] (or the caller's lines) cycled to `n_sentences` texts with the "document" prompt (≈ 15–30
//! tokens each). Batch 1 embeds the texts one call at a time; batch 32 embeds them all in
//! one call (sorted, chunks of 32). One warmup pass per mode, then `runs` measured passes.

use std::time::Instant;

use lapid::{Error, Result};
use lapid_verify::timing::{median, ms, stats};
use serde_json::{Value, json};

use super::verify::resolve_model;
use super::{EmbedOptions, EmbeddingGemma};

pub const CORPUS: [&str; 16] = [
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Sourdough bread needs a lively starter, patience and a very hot oven.",
    "Rust guarantees memory safety without a garbage collector.",
    "The stock market closed higher today after a volatile trading session.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "A good night's sleep improves memory, mood and concentration.",
    "The museum's new exhibition features impressionist paintings from France.",
    "Electric cars are becoming cheaper as battery prices continue to fall.",
    "The recipe calls for two cups of flour, one egg and a pinch of salt.",
    "Mount Everest is the highest mountain above sea level on Earth.",
    "Machine translation quality has improved dramatically in the last decade.",
    "The library will be closed on Monday for the public holiday.",
];

/// `n` texts cycled from `lines` (default [`CORPUS`]).
fn corpus(n: usize, lines: Option<&[String]>) -> Vec<&str> {
    match lines {
        Some(l) if !l.is_empty() => l.iter().map(String::as_str).cycle().take(n).collect(),
        _ => CORPUS.iter().copied().cycle().take(n).collect(),
    }
}

fn mode(model: &EmbeddingGemma, texts: &[&str], batch: usize, runs: usize) -> Result<Value> {
    let opts = EmbedOptions::document().with_batch_size(batch);
    let pass = || -> Result<f64> {
        let t = Instant::now();
        if batch == 1 {
            for s in texts {
                model.embed(&[s], &opts)?;
            }
        } else {
            model.embed(texts, &opts)?;
        }
        Ok(ms(t))
    };
    let warmup_ms = pass()?;
    let mut totals = Vec::with_capacity(runs);
    for _ in 0..runs {
        totals.push(pass()?);
    }
    let med = median(totals.clone());
    let n = texts.len() as f64;
    Ok(json!({
        "batch_size": batch,
        "sentences": texts.len(),
        "warmup_ms": warmup_ms,
        "total_ms": stats(&totals),
        "ms_per_sentence": med / n,
        "sentences_per_s": if med > 0.0 { n / (med / 1e3) } else { 0.0 },
    }))
}

/// Benchmark report as JSON.
pub fn bench(
    model: &str,
    dtype: &str,
    device: Option<&str>,
    runs: usize,
    n_sentences: usize,
    lines: Option<&[String]>,
) -> Result<Value> {
    if runs == 0 || n_sentences == 0 {
        return Err(Error::new("invalid_argument", "runs and sentences must be at least 1"));
    }
    let dev = lapid_verify::open_device(device)?;
    let (path, convert_ms) = resolve_model(model, dtype)?;
    let t = Instant::now();
    let eg = EmbeddingGemma::load(&path, &dev)?;
    let load_ms = ms(t);
    let texts = corpus(n_sentences, lines);
    let prompt = eg.prompt_for(&EmbedOptions::document())?;
    let tokens: usize = texts.iter().map(|s| eg.tokenize(s, &prompt).map(|v| v.len())).sum::<Result<usize>>()?;
    let b1 = mode(&eg, &texts, 1, runs)?;
    let b32 = mode(&eg, &texts, 32, runs)?;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    Ok(json!({
        "engine": "lapid",
        "model": model,
        "model_file": path.display().to_string(),
        "dtype": dtype,
        "device": dev.description(),
        "backend": dev.name(),
        "host": { "arch": std::env::consts::ARCH, "os": std::env::consts::OS, "threads": threads },
        "runs": runs,
        "convert_ms": convert_ms,
        "load_ms": load_ms,
        "mean_tokens": tokens as f64 / texts.len() as f64,
        "batch_1": b1,
        "batch_32": b32,
    }))
}

