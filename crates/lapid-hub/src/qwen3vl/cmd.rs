//! `lapid verify qwen3vl` / `lapid bench qwen3vl` (routed from `crate::cmd`).

use std::path::Path;

use lapid::Result;
use serde_json::Value;

use crate::cmd::ModelArgs;

pub const DEFAULT_MODEL: &str = "Qwen/Qwen3-VL-2B-Instruct";
pub const DEFAULT_DTYPE: &str = "int8";
pub const DEFAULT_REFERENCE: &str = "data/reference/qwen3vl";
pub const BENCH_GEN_TOKENS: usize = 128;

fn model_dtype(args: &ModelArgs) -> (&str, &str) {
    (args.model.as_deref().unwrap_or(DEFAULT_MODEL), args.dtype.as_deref().unwrap_or(DEFAULT_DTYPE))
}

/// Templates, preprocessing, vision encoder, text layers, cached decoding and greedy text vs
/// `tools/reference/qwen3vl_ref.py`.
pub fn verify(reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    let (model, dtype) = model_dtype(args);
    super::verify::verify(reference.unwrap_or(Path::new(DEFAULT_REFERENCE)), model, dtype, args.device.as_deref())
}

/// Vision encode time, prefill / decode tokens per second; `input` is an optional picture.
pub fn bench(input: Option<&Path>, args: &ModelArgs, runs: usize) -> Result<Value> {
    let (model, dtype) = model_dtype(args);
    let n_gen = std::env::var("LAPID_BENCH_GEN_TOKENS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(BENCH_GEN_TOKENS);
    super::bench::bench(model, dtype, args.device.as_deref(), runs, n_gen, input)
}
