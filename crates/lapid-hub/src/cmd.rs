//! Developer entry points the `lapid` binary routes to: `convert`, `verify <model>`,
//! `bench <model>`. Each model implements `verify` / `bench` in its own `<model>/cmd.rs`;
//! this module only routes by model name. Apps (transcription, ...) are examples.

use std::path::Path;

use lapid::{Result, bail};
use serde_json::Value;

pub use crate::convert::{convert, convert_with};

/// Model families `verify` / `bench` route to.
pub const MODELS: &[&str] = &["whisper", "embeddinggemma", "parakeet", "lfm3", "phonon", "qwen35", "gemma4"];

/// Common model selection of `lapid verify` / `lapid bench`; `None` = the model's default.
#[derive(Clone, Debug, Default)]
pub struct ModelArgs {
    /// HF repo id, local HF directory or `.lapid` file.
    pub model: Option<String>,
    /// Weight dtype ("fp16", "int8", ...).
    pub dtype: Option<String>,
    /// Device name ("cpu", "cuda", "vulkan:1", ...); `None` = best / `$LAPID_DEVICE`.
    pub device: Option<String>,
}

fn unknown(model: &str) -> lapid::Error {
    lapid::Error::new("invalid_argument", format!("unknown model {model:?} (expected one of {})", MODELS.join(", ")))
}

/// Verify `model` against a reference directory (default: the model's own). Check `"ok"`.
pub fn verify(model: &str, reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    match model {
        "whisper" => crate::whisper::cmd::verify(reference, args),
        "embeddinggemma" => crate::embeddinggemma::cmd::verify(reference, args),
        "parakeet" => crate::parakeet::cmd::verify(reference, args),
        "lfm3" => crate::lfm3::cmd::verify(reference, args),
        "phonon" => crate::phonon::cmd::verify(reference, args),
        "qwen35" => crate::qwen35::cmd::verify(reference, args),
        "gemma4" => crate::gemma4::cmd::verify(reference, args),
        _ => Err(unknown(model)),
    }
}

/// Benchmark `model` on `input` (default: the model's own sample) over `runs` runs.
pub fn bench(model: &str, input: Option<&Path>, args: &ModelArgs, runs: usize) -> Result<Value> {
    if runs == 0 {
        bail!("invalid_argument", "--runs must be at least 1");
    }
    match model {
        "whisper" => crate::whisper::cmd::bench(input, args, runs),
        "embeddinggemma" => crate::embeddinggemma::cmd::bench(input, args, runs),
        "parakeet" => crate::parakeet::cmd::bench(input, args, runs),
        "lfm3" => crate::lfm3::cmd::bench(input, args, runs),
        "phonon" => crate::phonon::cmd::bench(input, args, runs),
        "qwen35" => crate::qwen35::cmd::bench(input, args, runs),
        "gemma4" => crate::gemma4::cmd::bench(input, args, runs),
        _ => Err(unknown(model)),
    }
}
