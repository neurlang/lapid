//! `lapid verify embeddinggemma` / `lapid bench embeddinggemma` (routed from `crate::cmd`).

use std::path::Path;

use lapid::{Error, Result};
use serde_json::Value;

use crate::cmd::ModelArgs;

pub const DEFAULT_MODEL: &str = super::DEFAULT_REPO;
pub const DEFAULT_DTYPE: &str = "fp16";
pub const DEFAULT_REFERENCE: &str = "data/reference/embeddinggemma";
/// Sentences per benchmark pass.
pub const BENCH_SENTENCES: usize = 64;

fn model_dtype(args: &ModelArgs) -> (&str, &str) {
    (args.model.as_deref().unwrap_or(DEFAULT_MODEL), args.dtype.as_deref().unwrap_or(DEFAULT_DTYPE))
}

/// Compare tokens, every layer and the final embeddings with
/// `tools/reference/embeddinggemma_ref.py` output.
pub fn verify(reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    let (model, dtype) = model_dtype(args);
    let reference = reference.unwrap_or(Path::new(DEFAULT_REFERENCE));
    super::verify::verify(reference, model, dtype, args.device.as_deref())
}

/// Sentences/s at batch 1 and 32. `input` (optional) is a text file with one sentence per
/// line; default: the built-in corpus.
pub fn bench(input: Option<&Path>, args: &ModelArgs, runs: usize) -> Result<Value> {
    let (model, dtype) = model_dtype(args);
    let lines = match input {
        Some(p) => {
            let s = std::fs::read_to_string(p).map_err(|e| Error::new("io", format!("{}: {e}", p.display())))?;
            Some(s.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect::<Vec<_>>())
        }
        None => None,
    };
    super::bench::bench(model, dtype, args.device.as_deref(), runs, BENCH_SENTENCES, lines.as_deref())
}
