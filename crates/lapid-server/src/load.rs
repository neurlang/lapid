//! Model resolution and loading, delegated to the hub: a model argument (HF repo id, local HF
//! directory, `.gguf`, ggml or `.lapid` file, optionally prefixed with `id=`) becomes a
//! `.lapid` file in the lapid cache (`lapid_hub::runtime::resolve`), whose metadata
//! `architecture` picks the model and its adapter (`lapid_hub::runtime::serve::load_path`).

use std::path::{Path, PathBuf};

use lapid::{Device, Result};

pub use lapid_hub::LoadOptions;

use crate::served::Served;

/// A parsed model argument.
#[derive(Clone, Debug)]
pub struct ModelSpec {
    /// Explicit id (`id=source`).
    pub id: Option<String>,
    pub source: String,
}

impl ModelSpec {
    pub fn parse(arg: &str) -> ModelSpec {
        // `id=source`, unless the whole argument is an existing path.
        if !Path::new(arg).exists()
            && let Some((id, src)) = arg.split_once('=')
            && !id.is_empty()
            && !src.is_empty()
        {
            return ModelSpec { id: Some(id.to_string()), source: src.to_string() };
        }
        ModelSpec { id: None, source: arg.to_string() }
    }
}

/// The `.lapid` file for `source` (converting it into the lapid cache when missing) and the
/// default model id.
pub fn resolve(source: &str, dtype: Option<&str>) -> Result<(PathBuf, String)> {
    let r = lapid_hub::runtime::resolve::resolve(source, dtype)?;
    Ok((r.path, r.id))
}

/// Load the model in `path` on `device` with the adapter its architecture needs.
pub fn load(path: &Path, id: String, device: &Device, opts: &LoadOptions) -> Result<Box<dyn Served>> {
    lapid_hub::runtime::serve::load_path(path, id, device, opts)
}

/// Resolve and load `spec` (convenience for `run`).
pub fn load_spec(spec: &ModelSpec, device: &Device, opts: &LoadOptions) -> Result<Box<dyn Served>> {
    lapid_hub::runtime::serve::load(&spec.source, spec.id.clone(), device, opts)
}
