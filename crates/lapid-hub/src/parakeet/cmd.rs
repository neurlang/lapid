//! `lapid verify parakeet` / `lapid bench parakeet` (routed from `crate::cmd`).

use std::path::Path;

use lapid::Result;
use serde_json::Value;

use crate::cmd::ModelArgs;

pub const DEFAULT_MODEL: &str = "nvidia/parakeet-tdt-0.6b-v3";
pub const DEFAULT_DTYPE: &str = "int8";
/// Holds one sub-directory per sample (`en`: data/jfk.wav, `de`: a German clip).
pub const DEFAULT_REFERENCE: &str = "data/reference/parakeet";
pub const DEFAULT_AUDIO: &str = "data/jfk.wav";

fn model_dtype(args: &ModelArgs) -> (&str, &str) {
    (args.model.as_deref().unwrap_or(DEFAULT_MODEL), args.dtype.as_deref().unwrap_or(DEFAULT_DTYPE))
}

/// Compare mel, encoder layers, joint logits and transcripts with
/// `tools/reference/parakeet_ref.py` output. `--model` may also be a Parakeet GGUF.
pub fn verify(reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    let (model, dtype) = model_dtype(args);
    let reference = reference.unwrap_or(Path::new(DEFAULT_REFERENCE));
    super::verify::verify(reference, model, dtype, args.device.as_deref())
}

/// Time features / encode / decode on `audio` (default `data/jfk.wav`).
pub fn bench(audio: Option<&Path>, args: &ModelArgs, runs: usize) -> Result<Value> {
    let (model, dtype) = model_dtype(args);
    let audio = audio.unwrap_or(Path::new(DEFAULT_AUDIO));
    super::bench::bench(audio, model, dtype, args.device.as_deref(), runs)
}
