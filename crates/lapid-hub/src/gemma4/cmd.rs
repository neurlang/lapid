//! `lapid verify gemma4` / `lapid bench gemma4` (routed from `crate::cmd`).

use std::path::Path;

use lapid::{Error, Result};
use serde_json::Value;

use crate::cmd::ModelArgs;

pub const DEFAULT_MODEL: &str = "google/gemma-4-E2B-it";
pub const DEFAULT_DTYPE: &str = "int8";
pub const DEFAULT_REFERENCE: &str = "data/reference/gemma4";
/// Bench prompt length (tokens) and generated tokens; override with
/// `LAPID_BENCH_PROMPT_TOKENS` / `LAPID_BENCH_GEN_TOKENS`.
pub const BENCH_PROMPT_TOKENS: usize = 512;
pub const BENCH_GEN_TOKENS: usize = 128;

fn model_dtype(args: &ModelArgs) -> (&str, &str) {
    (args.model.as_deref().unwrap_or(DEFAULT_MODEL), args.dtype.as_deref().unwrap_or(DEFAULT_DTYPE))
}

fn env_usize(key: &str, default: usize) -> Result<usize> {
    match std::env::var(key) {
        Ok(v) => v.trim().parse().map_err(|_| Error::new("invalid_argument", format!("{key}={v:?} is not a count"))),
        Err(_) => Ok(default),
    }
}

/// Template, tokenizer, layers, cached decoding and greedy text vs `tools/reference/gemma4_ref.py`.
pub fn verify(reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    let (model, dtype) = model_dtype(args);
    super::verify::verify(reference.unwrap_or(Path::new(DEFAULT_REFERENCE)), model, dtype, args.device.as_deref())
}

/// Prefill / decode tokens per second and time to first token. `input` (optional) is a
/// text file used as the user message instead of the synthetic prompt.
pub fn bench(input: Option<&Path>, args: &ModelArgs, runs: usize) -> Result<Value> {
    let (model, dtype) = model_dtype(args);
    let prompt_tokens = env_usize("LAPID_BENCH_PROMPT_TOKENS", BENCH_PROMPT_TOKENS)?;
    let gen_tokens = env_usize("LAPID_BENCH_GEN_TOKENS", BENCH_GEN_TOKENS)?;
    let text = match input {
        Some(p) => Some(std::fs::read_to_string(p).map_err(|e| Error::new("io", format!("{}: {e}", p.display())))?),
        None => None,
    };
    super::bench::bench(model, dtype, args.device.as_deref(), runs, prompt_tokens, gen_tokens, text.as_deref())
}
