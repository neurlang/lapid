//! `lapid verify llama` / `lapid bench llama` (routed from `crate::cmd`).

use std::path::Path;

use lapid::{Error, Result};
use serde_json::Value;

use crate::cmd::ModelArgs;

pub const DEFAULT_MODEL: &str = "Qwen/Qwen3-0.6B";
pub const DEFAULT_DTYPE: &str = "int8";
pub const DEFAULT_REFERENCE: &str = "data/reference/llama/qwen3";
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

/// Template, tokenizer, layers, cached decoding and greedy text vs `tools/reference/llama_ref.py`.
/// Without `--model` / `--dtype`, the reference's own `"model"` (repo id or tiny checkpoint
/// directory) and `"default_dtype"` from its `result.json` are used.
pub fn verify(reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    let reference = reference.unwrap_or(Path::new(DEFAULT_REFERENCE));
    let r: Value = std::fs::read_to_string(reference.join("result.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(Value::Null);
    let model = match &args.model {
        Some(m) => m.clone(),
        None => {
            let m = r.get("model").and_then(Value::as_str).unwrap_or(DEFAULT_MODEL);
            // tiny references keep their checkpoint next to result.json
            let local = reference.join(m);
            if local.exists() { local.display().to_string() } else { m.to_string() }
        }
    };
    let dtype = match &args.dtype {
        Some(d) => d.clone(),
        None => r.get("default_dtype").and_then(Value::as_str).unwrap_or(DEFAULT_DTYPE).to_string(),
    };
    super::verify::verify(reference, &model, &dtype, args.device.as_deref())
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
