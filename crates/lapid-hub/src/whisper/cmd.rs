//! `lapid verify whisper` / `lapid bench whisper` (routed from `crate::cmd`).

use std::path::Path;

use lapid::Result;
use serde_json::Value;

use crate::cmd::ModelArgs;

pub const DEFAULT_MODEL: &str = "openai/whisper-small";
pub const DEFAULT_DTYPE: &str = "int8";
pub const DEFAULT_REFERENCE: &str = "data/reference/whisper-small";
pub const DEFAULT_AUDIO: &str = "data/jfk.wav";

fn model_dtype(args: &ModelArgs) -> (&str, &str) {
    (args.model.as_deref().unwrap_or(DEFAULT_MODEL), args.dtype.as_deref().unwrap_or(DEFAULT_DTYPE))
}

/// Compare mel, encoder layers, logits and transcript with `tools/reference/whisper_ref.py`.
pub fn verify(reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    let (model, dtype) = model_dtype(args);
    let reference = reference.unwrap_or(Path::new(DEFAULT_REFERENCE));
    super::verify::verify(reference, model, dtype, args.device.as_deref())
}

/// Time load / mel / encode / decode on `audio` (default `data/jfk.wav`).
pub fn bench(audio: Option<&Path>, args: &ModelArgs, runs: usize) -> Result<Value> {
    let (model, dtype) = model_dtype(args);
    let audio = audio.unwrap_or(Path::new(DEFAULT_AUDIO));
    super::bench::bench(audio, model, dtype, args.device.as_deref(), runs)
}
