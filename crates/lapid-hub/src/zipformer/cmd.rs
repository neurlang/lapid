//! `lapid verify zipformer` / `lapid bench zipformer` (routed from `crate::cmd`).
//!
//! `--model`: a `.lapid` file or a directory exported by `tools/reference/zipformer_ref.py
//! export` (converted into the cache). Default for verify: each reference's own model
//! (`result.json` `"model"`); for bench: the exported LibriSpeech model.
//! `LAPID_ZIPFORMER_CHUNK` overrides the bench chunk size (encoder frames at 50 Hz).

use std::path::Path;

use lapid::Result;
use serde_json::Value;

use crate::cmd::ModelArgs;

pub const DEFAULT_MODEL: &str = "data/hf/zipformer2-en-streaming-2023-05-17";
pub const DEFAULT_DTYPE: &str = "fp16";
/// One sub-directory per sample (`tiny`, `en`, `en-libri`, `he`).
pub const DEFAULT_REFERENCE: &str = "data/reference/zipformer";
pub const DEFAULT_AUDIO: &str = "data/jfk.wav";

/// Compare fbank, every chunk's encoder output and states, tokens and the end-to-end
/// transcript with `tools/reference/zipformer_ref.py` output.
pub fn verify(reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    let dtype = args.dtype.as_deref().unwrap_or(DEFAULT_DTYPE);
    let reference = reference.unwrap_or(Path::new(DEFAULT_REFERENCE));
    super::verify::verify(reference, args.model.as_deref(), dtype, args.device.as_deref())
}

/// Streaming RTF and first-partial latency on `audio` (default `data/jfk.wav`).
pub fn bench(audio: Option<&Path>, args: &ModelArgs, runs: usize) -> Result<Value> {
    let model = args.model.as_deref().unwrap_or(DEFAULT_MODEL);
    let dtype = args.dtype.as_deref().unwrap_or(DEFAULT_DTYPE);
    let chunk = std::env::var("LAPID_ZIPFORMER_CHUNK").ok().and_then(|v| v.parse().ok());
    super::bench::bench(audio.unwrap_or(Path::new(DEFAULT_AUDIO)), model, dtype, args.device.as_deref(), runs, chunk)
}
