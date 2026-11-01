//! Model sources to `.lapid` files: an HF repo id, a local HF directory, a `.gguf` / ggml file
//! or a `.lapid` file becomes a `.lapid` path in the lapid cache, converted on first use
//! through the hub's converter routing (`crate::convert`). Shared by `lapid verify`,
//! `lapid bench`, `lapid run`, `lapid serve` and [`crate::load`].

use std::path::{Path, PathBuf};
use std::time::Instant;

use lapid::{Error, LapidFile, Result};
use lapid_convert::{OutDType, default_model_path};
use serde_json::Value;

fn stem(p: &Path) -> String {
    p.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string()
}

fn magic(p: &Path) -> Option<[u8; 4]> {
    let mut m = [0u8; 4];
    let mut f = std::fs::File::open(p).ok()?;
    std::io::Read::read_exact(&mut f, &mut m).ok()?;
    Some(m)
}

/// Whether `p` is a `.lapid` file (extension or magic).
pub fn is_lapid(p: &Path) -> bool {
    p.is_file() && (p.extension().is_some_and(|e| e == "lapid") || magic(p) == Some(*b"LPID"))
}

/// Conversion dtype for a GGUF file name: its quant family ("q4_k_m", "q4_k", "fp16", ...).
pub fn dtype_for_gguf(name: &str) -> &'static str {
    let n = name.to_ascii_lowercase();
    if n.contains("q4_k_m") {
        "q4_k_m"
    } else if n.contains("q4_k") {
        "q4_k"
    } else if n.contains("q5_k") {
        "q5_k"
    } else if n.contains("q6_k") {
        "q6_k"
    } else if n.contains("f16") || n.contains("bf16") || n.contains("f32") {
        "fp16"
    } else if n.contains("q4_0") || n.contains("q4_1") || n.contains("iq4") || n.contains("q3_k") || n.contains("q2_k") {
        "int4"
    } else {
        "int8"
    }
}

/// FNV-1a 64 (stable across builds and platforms).
fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}

/// `size:mtime` of a file (empty when missing).
fn stamp(p: &Path) -> String {
    match std::fs::metadata(p) {
        Ok(m) => {
            let t = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
            format!("{}:{t}", m.len())
        }
        Err(_) => String::new(),
    }
}

/// Cache key of a local file or directory: its name plus a hash of the canonical path and the
/// size / mtime of the files that define it (`config.json` and the weights for a directory),
/// so different checkpoints with the same name never share a cache entry.
fn local_key(p: &Path) -> String {
    let canon = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    let mut key = canon.display().to_string();
    if p.is_dir() {
        for f in ["config.json", "model.safetensors", "model.safetensors.index.json"] {
            key.push('|');
            key.push_str(&stamp(&p.join(f)));
        }
    } else {
        key.push('|');
        key.push_str(&stamp(p));
    }
    format!("{}-{:08x}", stem(&canon), fnv64(key.as_bytes()) as u32)
}

/// Cache key and default id of a non-`.lapid` source (same names as `lapid convert`):
/// `gguf/<name>-<hash>`, `whisper.cpp/<name>-<hash>`, `local/<name>-<hash>` or the repo id.
fn cache_key(source: &str) -> (String, String) {
    let p = Path::new(source);
    if p.is_file() {
        let prefix = if magic(p) == Some(*b"GGUF") { "gguf" } else { "whisper.cpp" };
        (format!("{prefix}/{}", local_key(p)), stem(p))
    } else if p.is_dir() {
        let name = p.canonicalize().ok().as_deref().map(stem).unwrap_or_else(|| stem(p));
        (format!("local/{}", local_key(p)), name)
    } else {
        let repo = source.trim_matches('/');
        (repo.to_string(), repo.to_string())
    }
}

/// Cache path and default id of a non-`.lapid` source at `dtype`.
pub fn cache_path(source: &str, dtype: OutDType) -> (PathBuf, String) {
    let (key, id) = cache_key(source);
    (default_model_path(&key, dtype), id)
}

/// [`cache_path`] for a variant conversion of the same source (e.g. `"vision"`).
pub fn cache_path_tagged(source: &str, dtype: OutDType, tag: &str) -> PathBuf {
    default_model_path(&format!("{}-{tag}", cache_key(source).0), dtype)
}

/// Whether the cached `.lapid` at `cached` was converted from `source` (checked through its
/// `source` metadata for local sources; repo ids are unique cache keys).
pub fn cached_matches(cached: &Path, source: &str) -> bool {
    let Ok(r) = lapid::format::Reader::open(cached) else { return false };
    let p = Path::new(source);
    if !p.exists() {
        return true;
    }
    let src = &r.metadata["source"];
    let same = |a: &Path| a.canonicalize().ok().is_some_and(|a| Some(a) == p.canonicalize().ok());
    if p.is_dir() {
        return src["repo"].as_str().is_some_and(|repo| same(Path::new(repo)));
    }
    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
    match (src["file"].as_str(), src["repo"].as_str()) {
        (Some(f), _) if !f.is_empty() => f == name,
        (_, Some(repo)) => same(Path::new(repo)) || Path::new(repo).file_name().and_then(|n| n.to_str()) == Some(name),
        _ => false,
    }
}

/// A resolved source.
#[derive(Clone, Debug)]
pub struct Resolved {
    pub path: PathBuf,
    /// Default model id: the upstream repo, else the file name.
    pub id: String,
    /// Conversion time when it was converted now.
    pub convert_ms: Option<f64>,
}

/// The `.lapid` file for `source` at `dtype` (`None`: int8 for safetensors, the file's quant
/// family for GGUF), converting it into the cache when missing.
pub fn resolve(source: &str, dtype: Option<&str>) -> Result<Resolved> {
    let p = Path::new(source);
    if is_lapid(p) {
        let file = LapidFile::open(p)?;
        let repo = file.metadata["source"]["repo"].as_str().filter(|r| !r.is_empty() && *r != "unknown").map(str::to_string);
        return Ok(Resolved { path: p.to_path_buf(), id: repo.unwrap_or_else(|| stem(p)), convert_ms: None });
    }
    let dtype = match dtype {
        Some(d) => d,
        None if source.to_ascii_lowercase().ends_with(".gguf") => dtype_for_gguf(&stem(p)),
        None => "int8",
    };
    let (path, id) = cache_path(source, OutDType::parse(dtype)?);
    if path.is_file() && cached_matches(&path, source) {
        return Ok(Resolved { path, id, convert_ms: None });
    }
    eprintln!("lapid: converting {source} ({dtype}) -> {}", path.display());
    let t = Instant::now();
    crate::convert::convert(source, dtype, Some(&path))?;
    Ok(Resolved { path, id, convert_ms: Some(t.elapsed().as_secs_f64() * 1e3) })
}

/// Metadata `dtype` of a `.lapid` file.
pub fn file_dtype(path: &Path) -> Result<String> {
    let r = lapid::format::Reader::open(path)?;
    r.metadata
        .get("dtype")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| Error::new("missing_metadata", format!("{}: no \"dtype\"", path.display())))
}

/// Metadata `architecture` of a `.lapid` file.
pub fn file_architecture(path: &Path) -> Result<String> {
    let r = lapid::format::Reader::open(path)?;
    Ok(r.metadata.get("architecture").and_then(Value::as_str).unwrap_or("unknown").to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_keys_differ() {
        let base = std::env::temp_dir().join(format!("lapid-resolve-{}", std::process::id()));
        let (a, b) = (base.join("a/model"), base.join("b/model"));
        for d in [&a, &b] {
            std::fs::create_dir_all(d).expect("mkdir");
            std::fs::write(d.join("config.json"), b"{}").expect("write");
        }
        let (pa, _) = cache_path(a.to_str().expect("utf8"), OutDType::F32);
        let (pb, _) = cache_path(b.to_str().expect("utf8"), OutDType::F32);
        assert_ne!(pa, pb);
        assert!(pa.to_string_lossy().contains("local--model-"));
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn gguf_dtypes() {
        assert_eq!(dtype_for_gguf("LFM3-1.7B-Q4_K_M"), "q4_k_m");
        assert_eq!(dtype_for_gguf("m-Q8_0"), "int8");
        assert_eq!(dtype_for_gguf("m-F16"), "fp16");
    }
}
