//! Model resolution and loading: a model argument (HF repo id, local HF directory, `.gguf`,
//! ggml or `.lapid` file, optionally prefixed with `id=`) becomes a `.lapid` file in the lapid
//! cache (converted on first use through the hub's converter routing), whose metadata
//! `architecture` picks the hub model and its [`Served`] adapter.

use std::path::{Path, PathBuf};
use std::time::Instant;

use lapid::{Device, Error, LapidFile, Result, bail};
use lapid_convert::{OutDType, default_model_path};

use crate::adapters::{EmbedServed, ParakeetServed, PhononServed, WhisperServed};
use crate::llm::LlmServed;
use crate::served::{Capability, GenParams, ModelMeta, Served};

/// Options shared by every model of one `serve` / `run`.
#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// Weight dtype for conversions ("int8", "fp16", "q4_k_m", ...); `None` = int8 for
    /// safetensors, the GGUF file's quant family for GGUF.
    pub dtype: Option<String>,
    /// Sessions kept per chat model for prefix reuse.
    pub sessions: usize,
    /// Server-side sampling defaults (used when a request leaves a field unset).
    pub defaults: GenParams,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions { dtype: None, sessions: 4, defaults: GenParams::default() }
    }
}

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

fn stem(p: &Path) -> String {
    p.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string()
}

fn is_lapid(p: &Path) -> Result<bool> {
    if !p.is_file() {
        return Ok(false);
    }
    if p.extension().is_some_and(|e| e == "lapid") {
        return Ok(true);
    }
    let mut magic = [0u8; 4];
    let mut f = std::fs::File::open(p).map_err(|e| Error::new("io", format!("{}: {e}", p.display())))?;
    use std::io::Read;
    Ok(f.read_exact(&mut magic).is_ok() && &magic == b"LPID")
}

/// The `.lapid` file for `source` (converting it into the lapid cache when missing) and the
/// default model id.
pub fn resolve(source: &str, dtype: Option<&str>) -> Result<(PathBuf, String)> {
    let p = Path::new(source);
    if is_lapid(p)? {
        // Id: the upstream repo the file was converted from, else the file name.
        let file = LapidFile::open(p)?;
        let repo = file.metadata["source"]["repo"].as_str().filter(|r| !r.is_empty() && *r != "unknown").map(str::to_string);
        return Ok((p.to_path_buf(), repo.unwrap_or_else(|| stem(p))));
    }
    let dtype = match dtype {
        Some(d) => d,
        None if source.to_ascii_lowercase().ends_with(".gguf") => crate::hf::dtype_for_gguf(&stem(p)),
        None => "int8",
    };
    let out_dtype = OutDType::parse(dtype)?;
    let (cached, id) = if p.is_file() {
        let name = stem(p);
        // Same cache names as `lapid_hub::convert` uses for GGUF / ggml sources.
        let mut magic = [0u8; 4];
        let is_gguf = std::fs::File::open(p).and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic)).is_ok() && &magic == b"GGUF";
        let prefix = if is_gguf { "gguf" } else { "whisper.cpp" };
        (default_model_path(&format!("{prefix}/{name}"), out_dtype), name)
    } else if p.is_dir() {
        let name = p.canonicalize().ok().as_deref().map(stem).unwrap_or_else(|| stem(p));
        (default_model_path(&format!("local/{name}"), out_dtype), name)
    } else {
        let repo = source.trim_matches('/');
        (default_model_path(repo, out_dtype), repo.to_string())
    };
    if cached.is_file() {
        return Ok((cached, id));
    }
    eprintln!("lapid: converting {source} ({dtype}) -> {}", cached.display());
    let path = lapid_hub::convert::convert(source, dtype, None)?;
    Ok((path, id))
}

/// Base metadata of a `.lapid` file (capabilities and extras are filled by the adapter).
fn base_meta(file: &LapidFile, id: String, path: &Path, device: &Device) -> ModelMeta {
    let m = &file.metadata;
    let s = |v: &serde_json::Value| v.as_str().unwrap_or("unknown").to_string();
    ModelMeta {
        id,
        name: s(&m["name"]),
        architecture: s(&m["architecture"]),
        task: String::new(),
        dtype: s(&m["dtype"]),
        license: s(&m["license"]),
        source: s(&m["source"]["repo"]),
        path: path.display().to_string(),
        file_size: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
        device: device.description(),
        capabilities: Vec::new(),
        load_ms: 0,
        extra: Default::default(),
    }
}

/// Load the model in `path` on `device` with the adapter its architecture needs.
pub fn load(path: &Path, id: String, device: &Device, opts: &LoadOptions) -> Result<Box<dyn Served>> {
    let t0 = Instant::now();
    let file = LapidFile::open(path)?;
    let meta = base_meta(&file, id, path, device);
    drop(file);
    let arch = meta.architecture.clone();
    let mut served: Box<dyn Served> = match arch.as_str() {
        "lfm3" => Box::new(LlmServed::new(lapid_hub::Lfm3::load(path, device)?, meta, opts)),
        a if a.starts_with("qwen35") => Box::new(LlmServed::new(lapid_hub::Qwen35::load(path, device)?, meta, opts)),
        "gemma4" => Box::new(LlmServed::new(lapid_hub::Gemma4::load(path, device)?, meta, opts)),
        "whisper" => Box::new(WhisperServed::new(lapid_hub::Whisper::load(path, device)?, meta)),
        a if a.starts_with("parakeet") => Box::new(ParakeetServed::new(lapid_hub::Parakeet::load(path, device)?, meta)),
        "phonon" => Box::new(PhononServed::new(lapid_hub::Phonon::load(path, device)?, meta)),
        "embeddinggemma" => Box::new(EmbedServed::new(lapid_hub::EmbeddingGemma::load(path, device)?, meta)),
        other => bail!("unsupported_model", "{}: architecture {other:?} cannot be served", path.display()),
    };
    set_task_and_time(served.as_mut(), t0);
    Ok(served)
}

fn set_task_and_time(s: &mut dyn Served, t0: Instant) {
    let meta = s.meta_mut();
    meta.load_ms = t0.elapsed().as_millis() as u64;
    meta.task = match meta.capabilities.first() {
        Some(Capability::Chat | Capability::Completion) => "text-generation",
        Some(Capability::Embedding) => "feature-extraction",
        Some(Capability::Transcription) => "automatic-speech-recognition",
        Some(Capability::Speech) => "text-to-speech",
        None => "unknown",
    }
    .to_string();
}

/// Resolve and load `spec` (convenience for `run`).
pub fn load_spec(spec: &ModelSpec, device: &Device, opts: &LoadOptions) -> Result<Box<dyn Served>> {
    let (path, default_id) = resolve(&spec.source, opts.dtype.as_deref())?;
    load(&path, spec.id.clone().unwrap_or(default_id), device, opts)
}
