//! `lapid verify vad` / `lapid bench vad` (routed from `crate::cmd`).
//!
//! `--model` is a `.lapid` file, an exported directory (`config.json` + `model.safetensors`
//! from `tools/reference/silero_vad_ref.py`, converted in memory) or an HF repo with those
//! files. `--dtype` is ignored (weights are always f32).

use std::path::Path;

use lapid::Result;
use serde_json::Value;

use crate::cmd::ModelArgs;

/// The exported real v5 weights shipped with the reference data.
pub const DEFAULT_MODEL: &str = "data/reference/vad/model";
/// Holds one sub-directory per clip (`jfk`, `gaps`).
pub const DEFAULT_REFERENCE: &str = "data/reference/vad";
pub const DEFAULT_AUDIO: &str = "data/jfk.wav";

fn model(args: &ModelArgs) -> &str {
    args.model.as_deref().unwrap_or(DEFAULT_MODEL)
}

/// Compare probabilities, intermediates, segments and stream events with the reference.
pub fn verify(reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    let reference = reference.unwrap_or(Path::new(DEFAULT_REFERENCE));
    super::verify::verify(reference, model(args), args.device.as_deref())
}

/// Real-time factor on `audio` (default `data/jfk.wav`) tiled to 60 s.
pub fn bench(audio: Option<&Path>, args: &ModelArgs, runs: usize) -> Result<Value> {
    let audio = audio.unwrap_or(Path::new(DEFAULT_AUDIO));
    super::bench::bench(audio, model(args), args.device.as_deref(), runs)
}
