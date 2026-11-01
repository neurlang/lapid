//! The uniform model surface every hub model follows (see the crate docs for the full
//! convention): the [`Model`] trait, [`ModelInfo`], and the shared hub-cache helper behind
//! every `from_hub`.

use std::path::{Path, PathBuf};

use lapid::{Device, Error, LapidFile, Result};
use lapid_convert::OutDType;
use serde::Serialize;

pub use lapid::format::Bytes;

/// What a loaded model is. Built from the `.lapid` metadata at load time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ModelInfo {
    /// Metadata `"name"`, e.g. "whisper-small".
    pub name: String,
    /// Metadata `"architecture"`, e.g. "whisper".
    pub architecture: String,
    /// Hugging Face pipeline tag of what the model does: "automatic-speech-recognition",
    /// "text-generation", "text-to-speech", "feature-extraction", ...
    pub task: String,
    /// Main weight dtype: "fp32", "fp16", "int8", "int4".
    pub dtype: String,
    /// SPDX id or "unknown".
    pub license: String,
    /// Upstream repo (metadata `"source"."repo"`).
    pub source: String,
    /// Component names (one for single-graph models).
    pub components: Vec<String>,
}

impl ModelInfo {
    /// Info of `file` for a model doing `task`.
    pub fn from_file(file: &LapidFile, task: &str) -> ModelInfo {
        let m = &file.metadata;
        let text = |v: &serde_json::Value| v.as_str().unwrap_or("unknown").to_string();
        ModelInfo {
            name: text(&m["name"]),
            architecture: text(&m["architecture"]),
            task: task.to_string(),
            dtype: text(&m["dtype"]),
            license: text(&m["license"]),
            source: text(&m["source"]["repo"]),
            components: file.components().iter().map(|c| c.name.clone()).collect(),
        }
    }
}

/// The object-safe part of every hub model (bindings hold models as `Box<dyn Model>`).
///
/// Constructors are not part of the trait (they return `Self`); every model provides the
/// same three by convention — see the crate docs.
pub trait Model: Send + Sync {
    fn info(&self) -> ModelInfo;
    fn device(&self) -> &Device;
}

/// `.lapid` path of `repo` at `dtype` in the lapid cache
/// (`lapid_convert::default_model_path`), running `convert(repo, dtype, out_path)` first
/// when it is missing. Without the `download` feature the conversion only succeeds when
/// its upstream files are cached (otherwise "download_disabled").
pub fn hub_model_path(
    repo: &str,
    dtype: &str,
    convert: impl FnOnce(&str, OutDType, &Path) -> Result<()>,
) -> Result<PathBuf> {
    let out_dtype = OutDType::parse(dtype)?;
    let path = lapid_convert::default_model_path(repo, out_dtype);
    if path.is_file() {
        return Ok(path);
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| Error::new("io", format!("cannot create {}: {e}", dir.display())))?;
    }
    if crate::debug_enabled() {
        eprintln!("[lapid-hub] converting {repo} ({dtype}) -> {}", path.display());
    }
    convert(repo, out_dtype, &path)?;
    Ok(path)
}
