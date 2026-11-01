//! `lapid verify gemma4vl` / `lapid bench gemma4vl` (routed from `crate::cmd`).

use std::path::Path;

use lapid::{Error, Result};
use serde_json::Value;

use crate::cmd::ModelArgs;

pub const DEFAULT_MODEL: &str = "google/gemma-4-E2B-it";
pub const DEFAULT_DTYPE: &str = "fp16";
pub const DEFAULT_REFERENCE: &str = "data/reference/gemma4vl";
/// Generated tokens of the bench (override with `LAPID_BENCH_GEN_TOKENS`).
pub const BENCH_GEN_TOKENS: usize = 64;

fn model_dtype(args: &ModelArgs) -> (&str, &str) {
    (args.model.as_deref().unwrap_or(DEFAULT_MODEL), args.dtype.as_deref().unwrap_or(DEFAULT_DTYPE))
}

/// Prompt, preprocessing, vision tower, decoder and greedy text vs `tools/reference/gemma4vl_ref.py`.
pub fn verify(reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    let (model, dtype) = model_dtype(args);
    super::verify::verify(reference.unwrap_or(Path::new(DEFAULT_REFERENCE)), model, dtype, args.device.as_deref())
}

/// Image encode time, TTFT and decode speed. `input` is an image file (default: the
/// reference picture).
pub fn bench(input: Option<&Path>, args: &ModelArgs, runs: usize) -> Result<Value> {
    let (model, dtype) = model_dtype(args);
    let gen_tokens = match std::env::var("LAPID_BENCH_GEN_TOKENS") {
        Ok(v) => v.trim().parse().map_err(|_| Error::new("invalid_argument", format!("LAPID_BENCH_GEN_TOKENS={v:?}")))?,
        Err(_) => BENCH_GEN_TOKENS,
    };
    let default = Path::new(DEFAULT_REFERENCE).join("image.png");
    super::bench::bench(model, dtype, args.device.as_deref(), runs, input.unwrap_or(&default), gen_tokens)
}
