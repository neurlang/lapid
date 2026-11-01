//! Model arguments: `[id=]source`, where the source is anything `lapid_hub::load` takes (HF
//! repo id, local HF directory, `.gguf`, ggml or `.lapid` file). Resolution (conversion into
//! the lapid cache) and loading are the hub's (`lapid_hub::load`, `runtime::resolve`,
//! `runtime::serve::load_path`); the model's `.lapid` metadata picks its `AnyModel`.

use std::path::Path;

pub use lapid_hub::LoadOptions;

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
