//! `lapid convert`: detect the architecture of a source checkpoint and dispatch to the
//! model's converter.
//!
//! Sources: a Hugging Face repo id (`owner/name`, downloaded into the cache) or local HF
//! directory — routed by `config.json` `model_type` — a whisper.cpp `ggml-*.bin` (ggml
//! magic), or a GGUF file (magic "GGUF") — routed by `general.architecture`. Anything
//! else fails with "unsupported_model".

use std::path::{Path, PathBuf};

use lapid::{Error, Result, bail};
use lapid_convert::calibrate::{CalibrationOptions, CalibrationSource, Calibrator, convert_calibrated};
use lapid_convert::gguf::{GgufFile, is_gguf};
use lapid_convert::writer::{RoleFn, read_json};
use lapid_convert::{OutDType, default_model_path, download_hf};

use crate::whisper;

/// Where a checkpoint comes from.
enum Source {
    HfRepo(String),
    HfDir(PathBuf),
    Ggml(PathBuf),
    Gguf(PathBuf),
}

/// Model architectures with a converter.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Architecture {
    Whisper,
    EmbeddingGemma,
    Parakeet,
    Lfm3,
    Phonon,
    Qwen35,
    Deepseek4,
    Qwen4Exp,
    Gemma4,
    Llama,
    Gemma4Moe,
    Qwen3Vl,
}

fn classify(src: &str) -> Result<Source> {
    let p = Path::new(src);
    if p.is_file() {
        if whisper::convert::is_ggml_file(p)? {
            return Ok(Source::Ggml(p.to_path_buf()));
        }
        if is_gguf(p)? {
            return Ok(Source::Gguf(p.to_path_buf()));
        }
        bail!("unsupported_model", "{src}: unknown file format (pass the directory for HF safetensors models)");
    }
    if p.is_dir() {
        if !p.join("config.json").is_file() {
            bail!("not_found", "{src}: directory has no config.json");
        }
        return Ok(Source::HfDir(p.to_path_buf()));
    }
    if src.contains('/') && !src.starts_with('/') && !src.starts_with('.') {
        return Ok(Source::HfRepo(src.trim_matches('/').to_string()));
    }
    bail!("not_found", "{src}: no such file or directory, and not an HF repo id (owner/name)")
}

/// Architecture from an HF `config.json` `model_type`.
fn hf_architecture(config: &Path) -> Result<Architecture> {
    let cfg = read_json(config)?;
    match cfg.get("model_type").and_then(|m| m.as_str()) {
        Some("whisper") => Ok(Architecture::Whisper),
        Some("gemma3_text") => Ok(Architecture::EmbeddingGemma),
        Some("parakeet_tdt") => Ok(Architecture::Parakeet),
        Some("lfm3") => Ok(Architecture::Lfm3),
        Some("llama" | "mistral" | "qwen2" | "qwen3" | "qwen3_moe") => Ok(Architecture::Llama),
        Some("gemma4" | "gemma4_text") if gemma4_is_moe(&cfg) => Ok(Architecture::Gemma4Moe),
        Some("gemma4" | "gemma4_text") => Ok(Architecture::Gemma4),
        Some("qwen3_vl") => Ok(Architecture::Qwen3Vl),
        Some("gradium_phonon_tts") => Ok(Architecture::Phonon),
        Some("qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text") => Ok(Architecture::Qwen35),
        Some("deepseek_v4") => Ok(Architecture::Deepseek4),
        Some("qwen4_exp" | "qwen4_exp_text") => Ok(Architecture::Qwen4Exp),
        Some(other) => bail!("unsupported_model", "{}: model_type {other:?} has no converter", config.display()),
        None => bail!("unsupported_model", "{}: no model_type", config.display()),
    }
}

/// Gemma 4 MoE variants (`enable_moe_block`) go to `gemma4moe`.
fn gemma4_is_moe(cfg: &serde_json::Value) -> bool {
    let text = cfg.get("text_config").unwrap_or(cfg);
    text.get("enable_moe_block").and_then(serde_json::Value::as_bool) == Some(true)
}

/// Architecture of a ggml file. The legacy (unversioned) ggml container was only ever
/// used for whisper.cpp models.
fn ggml_architecture(_path: &Path) -> Architecture {
    Architecture::Whisper
}

/// Architecture of a GGUF file from `general.architecture`. No GGUF mapping exists yet
/// (whisper.cpp ships Whisper only in its legacy ggml container, so there is no Whisper
/// GGUF naming to follow); models add theirs here.
fn gguf_architecture(path: &Path) -> Result<Architecture> {
    let f = GgufFile::open(path)?;
    if crate::parakeet::convert::is_parakeet_gguf(&f) {
        return Ok(Architecture::Parakeet);
    }
    if crate::lfm3::convert::is_lfm3_gguf(&f) {
        return Ok(Architecture::Lfm3);
    }
    if crate::llama::convert::is_llama_gguf(&f) {
        return Ok(Architecture::Llama);
    }
    if crate::gemma4moe::convert::is_gemma4moe_gguf(&f) {
        return Ok(Architecture::Gemma4Moe);
    }
    if crate::qwen3vl::convert::is_qwen3vl_gguf(&f) {
        return Ok(Architecture::Qwen3Vl);
    }
    if crate::gemma4::convert::is_gemma4_gguf(&f) {
        return Ok(Architecture::Gemma4);
    }
    if crate::qwen35::convert::is_qwen35_gguf(&f) {
        return Ok(Architecture::Qwen35);
    }
    if crate::deepseek4::convert::is_deepseek4_gguf(&f) {
        return Ok(Architecture::Deepseek4);
    }
    if crate::qwen4exp::convert::is_qwen4exp_gguf(&f) {
        return Ok(Architecture::Qwen4Exp);
    }
    match f.architecture() {
        Some(arch) => bail!("unsupported_model", "{}: GGUF architecture {arch:?} has no converter", path.display()),
        None => bail!("unsupported_model", "{}: GGUF file without general.architecture", path.display()),
    }
}

fn file_label(p: &Path) -> String {
    p.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string()
}

/// Convert `src` (HF repo id, HF directory, or whisper.cpp ggml file) to `.lapid` at
/// `dtype` ("fp32", "fp16", "int8", "int4", "q4_k", "q5_k", "q6_k", "q4_k_m"). `out`
/// defaults to `lapid_convert::default_model_path(..)`. Returns the written path.
pub fn convert(src: &str, dtype: &str, out: Option<&Path>) -> Result<PathBuf> {
    convert_with(src, dtype, out, None)
}

/// Calibrated conversion (see `lapid_convert::calibrate`): convert to a temporary f32 file,
/// load the model from it on the best device, capture activations on the calibration
/// inputs, GPTQ-quantize, write `out`. `LAPID_CALIB_FLOAT=<file>` reuses (and keeps) an
/// existing f32 conversion.
fn calibrated(
    out: &Path,
    dtype: OutDType,
    source: &CalibrationSource,
    role: RoleFn,
    to_f32: impl FnOnce(&Path) -> Result<()>,
    open: impl FnOnce(&Path, &lapid::Device) -> Result<Box<dyn Calibrator>>,
) -> Result<()> {
    if dtype != OutDType::Int4 {
        bail!("invalid_argument", "--calibrate supports --dtype int4 only (got {})", dtype.name());
    }
    let keep = std::env::var_os("LAPID_CALIB_FLOAT").filter(|v| !v.is_empty()).map(PathBuf::from);
    let tmp = keep.clone().unwrap_or_else(|| out.with_file_name(format!(".{}.calib-f32.lapid", file_label(out))));
    if !(keep.is_some() && tmp.is_file()) {
        to_f32(&tmp)?;
    }
    let result = (|| {
        let device = lapid::Device::best();
        let mut cal = open(&tmp, &device)?;
        convert_calibrated(&tmp, out, dtype, role, cal.as_mut(), source, &CalibrationOptions::from_env())
    })();
    if keep.is_none() {
        std::fs::remove_file(&tmp).ok();
    }
    result
}

/// [`convert`] with optional calibration inputs for int4 (`"builtin"` or a text file);
/// supported for LFM3 and EmbeddingGemma.
pub fn convert_with(src: &str, dtype: &str, out: Option<&Path>, calibrate: Option<&str>) -> Result<PathBuf> {
    let dtype = OutDType::parse(dtype)?;
    let source = classify(src)?;
    let arch = match &source {
        Source::HfRepo(r) => hf_architecture(&download_hf(r, "config.json")?)?,
        Source::HfDir(d) => hf_architecture(&d.join("config.json"))?,
        Source::Ggml(f) => ggml_architecture(f),
        Source::Gguf(f) => gguf_architecture(f)?,
    };
    let out = match out {
        Some(o) => o.to_path_buf(),
        None => match &source {
            Source::HfRepo(r) => default_model_path(r, dtype),
            Source::HfDir(d) => {
                let name = d.canonicalize().ok().as_deref().map(file_label).unwrap_or_else(|| file_label(d));
                default_model_path(&format!("local/{name}"), dtype)
            }
            Source::Ggml(f) => default_model_path(&format!("whisper.cpp/{}", file_label(f)), dtype),
            Source::Gguf(f) => default_model_path(&format!("gguf/{}", file_label(f)), dtype),
        },
    };
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| Error::new("io", format!("create {}: {e}", dir.display())))?;
    }
    if let Some(cal) = calibrate {
        let cal = CalibrationSource::parse(cal);
        let hf_src = match &source {
            Source::HfRepo(r) => r.clone(),
            Source::HfDir(d) | Source::Ggml(d) | Source::Gguf(d) => d
                .to_str()
                .ok_or_else(|| Error::new("invalid_argument", "model path is not UTF-8"))?
                .to_string(),
        };
        let gguf = matches!(source, Source::Gguf(_));
        match arch {
            Architecture::Lfm3 => calibrated(
                &out,
                dtype,
                &cal,
                crate::lfm3::convert::role,
                |tmp| {
                    if gguf {
                        crate::lfm3::convert::convert_gguf(Path::new(&hf_src), OutDType::F32, tmp)
                    } else {
                        crate::lfm3::convert::convert_hf(&hf_src, OutDType::F32, tmp)
                    }
                },
                |p, d| Ok(Box::new(crate::lfm3::convert::calibrate::Lfm3Calibrator::open(p, d, &cal)?)),
            )?,
            Architecture::EmbeddingGemma if !gguf => calibrated(
                &out,
                dtype,
                &cal,
                crate::embeddinggemma::convert::role,
                |tmp| crate::embeddinggemma::convert::convert_hf(&hf_src, OutDType::F32, tmp),
                |p, d| Ok(Box::new(crate::embeddinggemma::convert::calibrate::GemmaCalibrator::open(p, d, &cal)?)),
            )?,
            other => bail!("unsupported_model", "calibrated quantization is not implemented for {other:?} ({src})"),
        }
        return Ok(out);
    }
    match (arch, &source) {
        (Architecture::Whisper, Source::HfRepo(r)) => whisper::convert::convert_hf(r, dtype, &out)?,
        (Architecture::Whisper, Source::HfDir(d)) => {
            let s = d.to_str().ok_or_else(|| Error::new("invalid_argument", "model directory path is not UTF-8"))?;
            whisper::convert::convert_hf(s, dtype, &out)?
        }
        (Architecture::Whisper, Source::Ggml(f)) => whisper::convert::convert_ggml(f, dtype, &out)?,
        (Architecture::EmbeddingGemma, Source::HfRepo(r)) => crate::embeddinggemma::convert::convert_hf(r, dtype, &out)?,
        (Architecture::EmbeddingGemma, Source::HfDir(d)) => {
            let s = d.to_str().ok_or_else(|| Error::new("invalid_argument", "model directory path is not UTF-8"))?;
            crate::embeddinggemma::convert::convert_hf(s, dtype, &out)?
        }
        (Architecture::EmbeddingGemma, Source::Ggml(f) | Source::Gguf(f)) => {
            bail!("unsupported_model", "{}: EmbeddingGemma converts from Hugging Face safetensors only", f.display())
        }
        (Architecture::Parakeet, Source::HfRepo(r)) => crate::parakeet::convert::convert_hf(r, dtype, &out)?,
        (Architecture::Parakeet, Source::HfDir(d)) => {
            let s = d.to_str().ok_or_else(|| Error::new("invalid_argument", "model directory path is not UTF-8"))?;
            crate::parakeet::convert::convert_hf(s, dtype, &out)?
        }
        (Architecture::Parakeet, Source::Gguf(f)) => crate::parakeet::convert::convert_gguf(f, dtype, &out)?,
        (Architecture::Parakeet, Source::Ggml(f)) => {
            bail!("unsupported_model", "{}: Parakeet has no ggml converter", f.display())
        }
        (Architecture::Lfm3, Source::HfRepo(r)) => crate::lfm3::convert::convert_hf(r, dtype, &out)?,
        (Architecture::Lfm3, Source::HfDir(d)) => {
            let s = d.to_str().ok_or_else(|| Error::new("invalid_argument", "model directory path is not UTF-8"))?;
            crate::lfm3::convert::convert_hf(s, dtype, &out)?
        }
        (Architecture::Lfm3, Source::Gguf(f)) => crate::lfm3::convert::convert_gguf(f, dtype, &out)?,
        (Architecture::Lfm3, Source::Ggml(f)) => {
            bail!("unsupported_model", "{}: LFM3 has no ggml converter", f.display())
        }
        (Architecture::Phonon, Source::HfRepo(r)) => crate::phonon::convert::convert_hf(r, dtype, &out)?,
        (Architecture::Phonon, Source::HfDir(d)) => {
            let s = d.to_str().ok_or_else(|| Error::new("invalid_argument", "model directory path is not UTF-8"))?;
            crate::phonon::convert::convert_hf(s, dtype, &out)?
        }
        (Architecture::Phonon, Source::Ggml(f) | Source::Gguf(f)) => {
            bail!("unsupported_model", "{}: Phonon converts from its Hugging Face package only", f.display())
        }
        (Architecture::Deepseek4, Source::HfRepo(r)) => crate::deepseek4::convert::convert_hf(r, dtype, &out)?,
        (Architecture::Deepseek4, Source::HfDir(d)) => {
            let s = d.to_str().ok_or_else(|| Error::new("invalid_argument", "model directory path is not UTF-8"))?;
            crate::deepseek4::convert::convert_hf(s, dtype, &out)?
        }
        (Architecture::Deepseek4, Source::Gguf(f)) => crate::deepseek4::convert::convert_gguf(f, dtype, &out)?,
        (Architecture::Deepseek4, Source::Ggml(f)) => {
            bail!("unsupported_model", "{}: DeepSeek V4 has no ggml converter", f.display())
        }
        (Architecture::Qwen35, Source::HfRepo(r)) => crate::qwen35::convert::convert_hf(r, dtype, &out)?,
        (Architecture::Qwen35, Source::HfDir(d)) => {
            let s = d.to_str().ok_or_else(|| Error::new("invalid_argument", "model directory path is not UTF-8"))?;
            crate::qwen35::convert::convert_hf(s, dtype, &out)?
        }
        (Architecture::Qwen35, Source::Gguf(f)) => crate::qwen35::convert::convert_gguf(f, dtype, &out)?,
        (Architecture::Qwen35, Source::Ggml(f)) => {
            bail!("unsupported_model", "{}: Qwen3.5 has no ggml converter", f.display())
        }
        (Architecture::Qwen4Exp, Source::HfRepo(r)) => crate::qwen4exp::convert::convert_hf(r, dtype, &out)?,
        (Architecture::Qwen4Exp, Source::HfDir(d)) => {
            let s = d.to_str().ok_or_else(|| Error::new("invalid_argument", "model directory path is not UTF-8"))?;
            crate::qwen4exp::convert::convert_hf(s, dtype, &out)?
        }
        (Architecture::Qwen4Exp, Source::Gguf(f)) => crate::qwen4exp::convert::convert_gguf(f, dtype, &out)?,
        (Architecture::Qwen4Exp, Source::Ggml(f)) => {
            bail!("unsupported_model", "{}: Qwen4-Exp has no ggml converter", f.display())
        }
        (Architecture::Gemma4, Source::HfRepo(r)) => crate::gemma4::convert::convert_hf(r, dtype, &out)?,
        (Architecture::Gemma4, Source::HfDir(d)) => {
            let s = d.to_str().ok_or_else(|| Error::new("invalid_argument", "model directory path is not UTF-8"))?;
            crate::gemma4::convert::convert_hf(s, dtype, &out)?
        }
        (Architecture::Gemma4, Source::Gguf(f)) => crate::gemma4::convert::convert_gguf(f, dtype, &out)?,
        (Architecture::Gemma4, Source::Ggml(f)) => {
            bail!("unsupported_model", "{}: Gemma 4 has no ggml converter", f.display())
        }
        (Architecture::Gemma4Moe, Source::HfRepo(r)) => crate::gemma4moe::convert::convert_hf(r, dtype, &out)?,
        (Architecture::Gemma4Moe, Source::HfDir(d)) => {
            let s = d.to_str().ok_or_else(|| Error::new("invalid_argument", "model directory path is not UTF-8"))?;
            crate::gemma4moe::convert::convert_hf(s, dtype, &out)?
        }
        (Architecture::Gemma4Moe, Source::Gguf(f)) => crate::gemma4moe::convert::convert_gguf(f, dtype, &out)?,
        (Architecture::Qwen3Vl, Source::HfRepo(r)) => crate::qwen3vl::convert::convert_hf(r, dtype, &out)?,
        (Architecture::Qwen3Vl, Source::HfDir(d)) => {
            let s = d.to_str().ok_or_else(|| Error::new("invalid_argument", "model directory path is not UTF-8"))?;
            crate::qwen3vl::convert::convert_hf(s, dtype, &out)?
        }
        (Architecture::Qwen3Vl, Source::Gguf(f)) => crate::qwen3vl::convert::convert_gguf(f, dtype, &out)?,
        (Architecture::Gemma4Moe | Architecture::Qwen3Vl, Source::Ggml(f)) => {
            bail!("unsupported_model", "{}: no ggml converter for this model", f.display())
        }
        (Architecture::Llama, Source::HfRepo(r)) => crate::llama::convert::convert_hf(r, dtype, &out)?,
        (Architecture::Llama, Source::HfDir(d)) => {
            let s = d.to_str().ok_or_else(|| Error::new("invalid_argument", "model directory path is not UTF-8"))?;
            crate::llama::convert::convert_hf(s, dtype, &out)?
        }
        (Architecture::Llama, Source::Gguf(f)) => crate::llama::convert::convert_gguf(f, dtype, &out)?,
        (Architecture::Llama, Source::Ggml(f)) => {
            bail!("unsupported_model", "{}: Llama-family models have no ggml converter", f.display())
        }
        (Architecture::Whisper, Source::Gguf(f)) => {
            bail!("unsupported_model", "{}: no GGUF converter for whisper", f.display())
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("lapid-hub-convert-{}-{tag}", std::process::id()))
    }

    #[test]
    fn classify_sources() {
        assert!(matches!(classify("openai/whisper-small"), Ok(Source::HfRepo(r)) if r == "openai/whisper-small"));
        assert_eq!(classify("whisper-small").err().map(|e| e.code), Some("not_found"));
        assert_eq!(classify("./nope/nope").err().map(|e| e.code), Some("not_found"));
        let p = tmp("x.bin");
        std::fs::write(&p, 0x6767_6d6cu32.to_le_bytes()).expect("write");
        assert!(matches!(classify(p.to_str().expect("utf8")), Ok(Source::Ggml(_))));
        // GGUF v3, no tensors, general.architecture = "mamba": routed, then unsupported.
        let mut g = b"GGUF".to_vec();
        g.extend(3u32.to_le_bytes());
        g.extend(0u64.to_le_bytes());
        g.extend(1u64.to_le_bytes());
        for s in ["general.architecture"] {
            g.extend((s.len() as u64).to_le_bytes());
            g.extend(s.as_bytes());
        }
        g.extend(8u32.to_le_bytes());
        g.extend(5u64.to_le_bytes());
        g.extend(b"mamba");
        std::fs::write(&p, &g).expect("write");
        assert!(matches!(classify(p.to_str().expect("utf8")), Ok(Source::Gguf(_))));
        let e = convert(p.to_str().expect("utf8"), "int8", Some(&tmp("o.lapid"))).err().expect("unsupported");
        assert_eq!(e.code, "unsupported_model");
        assert!(e.message.contains("mamba"), "{}", e.message);
        std::fs::write(&p, b"nope").expect("write");
        assert_eq!(classify(p.to_str().expect("utf8")).err().map(|e| e.code), Some("unsupported_model"));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn model_type_routing() {
        let dir = tmp("dir");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let cfg = dir.join("config.json");
        std::fs::write(&cfg, br#"{"model_type": "whisper"}"#).expect("write");
        assert_eq!(hf_architecture(&cfg).ok(), Some(Architecture::Whisper));
        std::fs::write(&cfg, br#"{"model_type": "mamba"}"#).expect("write");
        assert_eq!(hf_architecture(&cfg).err().map(|e| e.code), Some("unsupported_model"));
        let r = convert(dir.to_str().expect("utf8"), "int8", Some(&dir.join("o.lapid")));
        assert_eq!(r.err().map(|e| e.code), Some("unsupported_model"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
