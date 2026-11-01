//! `lapid verify phonon` / `lapid bench phonon` (routed from `crate::cmd`).

use std::path::Path;

use lapid::{Error, Result};
use serde_json::Value;

use crate::cmd::ModelArgs;

pub const DEFAULT_MODEL: &str = "gradium/phonon-40m-en";
pub const DEFAULT_DTYPE: &str = "fp16";
pub const DEFAULT_REFERENCE: &str = "data/reference/phonon-40m-en";

fn model_dtype(args: &ModelArgs) -> (&str, &str) {
    (args.model.as_deref().unwrap_or(DEFAULT_MODEL), args.dtype.as_deref().unwrap_or(DEFAULT_DTYPE))
}

/// Compare tokens, backbone, flow head, codec and end-to-end audio with
/// `tools/reference/phonon_ref.py`.
pub fn verify(reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    let (model, dtype) = model_dtype(args);
    let reference = reference.unwrap_or(Path::new(DEFAULT_REFERENCE));
    super::verify::verify(reference, model, dtype, args.device.as_deref())
}

/// Time to first audio and real-time factor; `input` is an optional UTF-8 text file.
pub fn bench(input: Option<&Path>, args: &ModelArgs, runs: usize) -> Result<Value> {
    let (model, dtype) = model_dtype(args);
    let text = match input {
        Some(p) => Some(std::fs::read_to_string(p).map_err(|e| Error::new("io", format!("{}: {e}", p.display())))?),
        None => None,
    };
    super::bench::bench(text.as_deref(), model, dtype, args.device.as_deref(), runs)
}
