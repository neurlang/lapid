//! `verify zipformer`: compare lapid's streaming Zipformer2 with the icefall reference written
//! by `tools/reference/zipformer_ref.py` (`run` / `tiny`), for one reference directory (with
//! `result.json`) or every sub-directory that has one (`data/reference/zipformer/{tiny,en,..}`).
//!
//! Per sample:
//! * fbank of the audio vs lhotse;
//! * chunk by chunk on the reference fbank (icefall's schedule and tail padding): the embed and
//!   stack outputs of the traced chunks, the projected encoder output of every chunk, and every
//!   cached state (keys, nonlinear-attention inputs, values, conv caches, ConvNeXt cache) after
//!   the reference's state chunks;
//! * greedy tokens of that run (overall and per chunk) vs icefall;
//! * end to end from the audio file, fed in odd-sized pieces: tokens and text must equal the
//!   reference (fp32 / fp16 / int8; int4 reported).
//!
//! Quantized files are also checked against the fake-quant model (same weights dequantized,
//! CPU) at `KERNEL_TOLERANCE` (encoder output of all chunks).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use lapid::{Error, Result};
use lapid_convert::{OutDType, default_model_path};
use lapid_verify::{Activations, Comparison, KERNEL_TOLERANCE, tolerance};
use serde_json::{Value, json};

use super::config::{LOG_EPS, PAD_FRAMES, TAIL_FRAMES};
use super::fbank::Fbank;
use super::{Precision, StreamOptions, Zipformer};

/// Tolerance of our fbank against lhotse (f64 vs f32 math).
pub const FBANK_TOLERANCE: f32 = 1e-4;

/// Whether the transcript must match the reference exactly for this dtype.
pub fn text_must_match(dtype: &str) -> bool {
    matches!(dtype, "fp32" | "fp16" | "int8")
}

/// `.lapid` for `model` (a `.lapid` path or an exported directory / repo converted into the
/// cache; re-converted when the source weights are newer).
pub(crate) fn resolve_model(model: &str, dtype: &str) -> Result<PathBuf> {
    let p = Path::new(model);
    if p.extension().is_some_and(|e| e == "lapid") {
        return Ok(p.to_path_buf());
    }
    let out_dtype = OutDType::parse(dtype)?;
    let key = if p.is_dir() {
        let canon = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
        let parts: Vec<String> = canon.components().rev().take(2).map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
        format!("local/zipformer-{}", parts.into_iter().rev().collect::<Vec<_>>().join("-"))
    } else {
        model.to_string()
    };
    let path = default_model_path(&key, out_dtype);
    let stale = |src: &Path| -> bool {
        let m = |q: &Path| std::fs::metadata(q).and_then(|m| m.modified()).ok();
        matches!((m(src), m(&path)), (Some(s), Some(o)) if s > o)
    };
    if path.is_file() && !(p.is_dir() && stale(&p.join("model.safetensors"))) {
        return Ok(path);
    }
    super::convert::convert_hf(model, out_dtype, &path)?;
    Ok(path)
}

struct Reference {
    dir: PathBuf,
    audio: PathBuf,
    model: Option<String>,
    chunk: usize,
    left: usize,
    chunks: usize,
    tokens: Vec<u32>,
    chunk_tokens: Vec<Vec<u32>>,
    text: String,
    states_chunks: Vec<usize>,
    random: bool,
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array().map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as u32).collect()).unwrap_or_default()
}

/// `rel` as given (absolute or from the working directory), else relative to `dir` or one of
/// its ancestors (reference files name repo-root-relative paths).
fn find_path(dir: &Path, rel: &str) -> Option<PathBuf> {
    let p = PathBuf::from(rel);
    if p.exists() {
        return Some(p);
    }
    dir.ancestors().map(|a| a.join(rel)).find(|q| q.exists())
}

fn read_reference(dir: &Path) -> Result<Reference> {
    let path = dir.join("result.json");
    let raw = std::fs::read_to_string(&path).map_err(|e| Error::new("io", format!("{}: {e} (run tools/reference/zipformer_ref.py)", path.display())))?;
    let v: Value = serde_json::from_str(&raw)?;
    let num = |k: &str| v.get(k).and_then(Value::as_u64).map(|x| x as usize).ok_or_else(|| Error::new("reference", format!("result.json: no {k:?}")));
    let audio = v.get("audio").and_then(Value::as_str).ok_or_else(|| Error::new("reference", "result.json: no \"audio\""))?;
    let audio = find_path(dir, audio).ok_or_else(|| Error::new("not_found", format!("{}: audio {audio:?} not found", dir.display())))?;
    Ok(Reference {
        dir: dir.to_path_buf(),
        audio,
        model: v.get("model").and_then(Value::as_str).map(|m| find_path(dir, m).map_or_else(|| m.to_string(), |p| p.display().to_string())),
        chunk: num("chunk_size")?,
        left: num("left_context_frames")?,
        chunks: num("chunks")?,
        tokens: u32s(&v["tokens"]),
        chunk_tokens: v.get("chunk_tokens").and_then(Value::as_array).map(|a| a.iter().map(u32s).collect()).unwrap_or_default(),
        text: v.get("text").and_then(Value::as_str).unwrap_or("").trim().to_string(),
        random: v.get("random").and_then(Value::as_bool).unwrap_or(false),
        states_chunks: v.get("states_chunks").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect()).unwrap_or_default(),
    })
}

/// Our run over the reference fbank: rows (name -> concatenated values) and tokens per chunk.
struct Run {
    rows: BTreeMap<String, Vec<f32>>,
    chunk_tokens: Vec<Vec<u32>>,
    tokens: Vec<u32>,
}

/// Row name of a traced / state tensor: per-kind aggregation over chunks and layers.
fn row_of(name: &str) -> String {
    match name.strip_prefix("state.") {
        Some(rest) if rest != "embed" => format!("state.{}", rest.rsplit('.').next().unwrap_or(rest)),
        _ => name.to_string(),
    }
}

fn run_chunks(z: &Zipformer, r: &Reference, acts: &Activations, fbank: &[f32]) -> Result<Run> {
    let f = z.config().feature_dim;
    let opts = StreamOptions { chunk_size: Some(r.chunk), left_context_frames: Some(r.left), endpoint: None };
    let mut s = z.stream(&opts)?;
    let mut padded = fbank.to_vec();
    padded.extend(std::iter::repeat_n(LOG_EPS, (PAD_FRAMES + TAIL_FRAMES) * f));
    let total = padded.len() / f;
    let window = 2 * r.chunk + PAD_FRAMES;
    let mut run = Run { rows: BTreeMap::new(), chunk_tokens: Vec::new(), tokens: Vec::new() };
    let (mut n, mut c) = (0, 0);
    while n < total {
        let end = (n + window).min(total);
        let mut x = padded[n * f..end * f].to_vec();
        x.resize(window * f, LOG_EPS);
        let before = s.result().tokens.len();
        let mut out = Vec::new();
        let trace = s.step(&x, true, &mut out)?.ok_or_else(|| Error::new("internal", "no chunk trace"))?;
        run.chunk_tokens.push(s.result().tokens[before..].to_vec());
        run.rows.entry("encoder_out".into()).or_default().extend(&trace.encoder_out);
        for (name, v) in &trace.traced {
            if acts.has(&format!("chunk.{c}.{name}")) {
                run.rows.entry(name.clone()).or_default().extend(v);
            }
        }
        if r.states_chunks.contains(&c) {
            for (name, v) in &trace.state {
                run.rows.entry(row_of(name)).or_default().extend(v);
            }
        }
        n += 2 * r.chunk;
        c += 1;
    }
    run.tokens = s.result().tokens;
    Ok(run)
}

/// Reference rows in the same aggregation as `run_chunks`.
fn reference_rows(acts: &Activations, r: &Reference) -> Result<BTreeMap<String, Vec<f32>>> {
    let mut rows: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for c in 0..r.chunks {
        let p = format!("chunk.{c}.");
        rows.entry("encoder_out".into()).or_default().extend(acts.get(&format!("{p}encoder_out"))?);
        for name in acts.names().iter().filter(|n| n.starts_with(&p)) {
            let rest = &name[p.len()..];
            if rest == "encoder_out" {
                continue;
            }
            if rest.starts_with("state.") && !r.states_chunks.contains(&c) {
                continue;
            }
            rows.entry(row_of(rest)).or_default().extend(acts.get(name)?);
        }
    }
    Ok(rows)
}

/// Rows in their own order (names sorted by `reference_rows`): reorder our state rows to the
/// reference's tensor order (both iterate stacks, layers, kinds in the same order).
fn compare_rows(ours: &Run, reference: &BTreeMap<String, Vec<f32>>, tol: f32) -> Comparison {
    let mut c = Comparison::new();
    for (name, want) in reference {
        match ours.rows.get(name) {
            Some(got) => {
                c.compare(name, got, want, tol);
            }
            None => {
                c.notes.push(format!("{name}: not produced"));
                c.layers.push(lapid_verify::LayerReport::failed(name.as_str()));
            }
        }
    }
    c
}

fn verify_sample(r: &Reference, model_arg: Option<&str>, dtype: &str, device: Option<&str>) -> Result<Value> {
    let Some(model) = model_arg.map(str::to_string).or_else(|| r.model.clone()) else {
        return Ok(json!({"sample": r.dir.display().to_string(), "skipped": "no model (pass --model)"}));
    };
    if !Path::new(&model).exists() {
        return Ok(json!({"sample": r.dir.display().to_string(), "skipped": format!("model {model} not found (export it with tools/reference/zipformer_ref.py)")}));
    }
    let path = resolve_model(&model, dtype)?;
    let dev = lapid_verify::open_device(device)?;
    let z = Zipformer::load(&path, &dev)?;
    let acts = Activations::open(&r.dir.join("activations.safetensors"))?;
    let ref_fbank = acts.get("fbank")?;

    let samples = lapid_nn::audio::load_audio(&r.audio, z.config().sample_rate)?;
    let mut fb = Comparison::new();
    fb.compare("fbank", &Fbank::new(z.config().feature_dim, z.config().sample_rate).compute(&samples), &ref_fbank, FBANK_TOLERANCE);

    let ours = run_chunks(&z, r, &acts, &ref_fbank)?;
    let reference = reference_rows(&acts, r)?;
    let quality = compare_rows(&ours, &reference, tolerance(dtype)?);
    let chunk_tokens_match = ours.chunk_tokens == r.chunk_tokens;
    // Relative error of the encoder output per chunk (where a drift starts).
    let per = ours.rows["encoder_out"].len() / r.chunks.max(1);
    let per_chunk: Vec<f32> = (0..r.chunks)
        .map(|c| {
            let range = c * per..(c + 1) * per;
            match (ours.rows["encoder_out"].get(range.clone()), reference["encoder_out"].get(range)) {
                (Some(a), Some(b)) => lapid_verify::diff(a, b).map_or(f32::NAN, |d| d.rel),
                _ => f32::NAN,
            }
        })
        .collect();

    // End to end from audio, odd-sized pieces.
    let opts = StreamOptions { chunk_size: Some(r.chunk), left_context_frames: Some(r.left), endpoint: None };
    let mut s = z.stream(&opts)?;
    for piece in samples.chunks(1013) {
        s.accept_waveform(piece)?;
    }
    s.input_finished()?;
    let e2e = s.result();

    let mut kernel = Value::Null;
    let mut kernel_ok = true;
    if matches!(dtype, "int8" | "int4") {
        let cpu = lapid::Device::cpu();
        let mut best: Option<Comparison> = None;
        for p in [Precision::FakeQuantWeights, Precision::FakeQuantActivations] {
            let fq = run_chunks(&Zipformer::load_with(&path, &cpu, p)?, r, &acts, &ref_fbank)?;
            let mut c = Comparison::new();
            c.compare("encoder_out", &ours.rows["encoder_out"], &fq.rows["encoder_out"], KERNEL_TOLERANCE);
            if best.as_ref().is_none_or(|b| c.max_rel() < b.max_rel()) {
                best = Some(c);
            }
        }
        if let Some(b) = best {
            kernel_ok = b.ok();
            kernel = b.to_json();
        }
    }

    let tokens_match = ours.tokens == r.tokens;
    let e2e_match = e2e.tokens == r.tokens && e2e.text == r.text;
    // A random-weight model has near-tied logits: quantization noise may flip a token, so
    // its tokens must match only at fp32 / fp16.
    let must = text_must_match(dtype) && !(r.random && dtype == "int8");
    // Per-chunk grouping is exact in fp32; lower precisions may move a token by one frame
    // across a chunk boundary (reported).
    let chunks_must = dtype == "fp32";
    let ok = fb.ok() && quality.ok() && kernel_ok && (!must || (tokens_match && e2e_match)) && (!chunks_must || chunk_tokens_match);
    Ok(json!({
        "sample": r.dir.display().to_string(),
        "ok": ok,
        "model": path.display().to_string(),
        "chunk_size": r.chunk,
        "left_context_frames": r.left,
        "chunks": r.chunks,
        "fbank": fb.to_json(),
        "quality": quality.to_json(),
        "kernel": kernel,
        "encoder_out_rel_per_chunk": per_chunk,
        "tokens_match": tokens_match,
        "chunk_tokens_match": chunk_tokens_match,
        "e2e_match": e2e_match,
        "text": e2e.text,
        "reference_text": r.text,
        "tokens": e2e.tokens.len(),
    }))
}

/// Verify one reference directory or every sub-directory with a `result.json`.
pub fn verify(reference: &Path, model: Option<&str>, dtype: &str, device: Option<&str>) -> Result<Value> {
    let dirs: Vec<PathBuf> = if reference.join("result.json").is_file() {
        vec![reference.to_path_buf()]
    } else {
        let mut v: Vec<PathBuf> = std::fs::read_dir(reference)
            .map_err(|e| Error::new("io", format!("{}: {e}", reference.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.join("result.json").is_file())
            .collect();
        v.sort();
        v
    };
    if dirs.is_empty() {
        return Err(Error::new("not_found", format!("{}: no result.json (run tools/reference/zipformer_ref.py)", reference.display())));
    }
    let mut samples = Vec::new();
    let mut ok = true;
    let mut ran = 0;
    for d in &dirs {
        let r = read_reference(d)?;
        let v = verify_sample(&r, model, dtype, device)?;
        if v.get("skipped").is_none() {
            ran += 1;
            ok &= v.get("ok").and_then(Value::as_bool).unwrap_or(false);
        }
        samples.push(v);
    }
    Ok(json!({ "ok": ok && ran > 0, "dtype": dtype, "device": device.unwrap_or("best"), "samples": samples }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_names() {
        assert_eq!(row_of("state.3.1.key"), "state.key");
        assert_eq!(row_of("state.embed"), "state.embed");
        assert_eq!(row_of("stack.2"), "stack.2");
    }

    /// The tiny random model (data/reference/zipformer/tiny) on the CPU, when present.
    #[test]
    fn tiny_reference_cpu() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/reference/zipformer/tiny");
        if !dir.join("result.json").is_file() {
            eprintln!("skipping: {} missing", dir.display());
            return;
        }
        let model = dir.join("model");
        let v = verify(&dir, model.to_str(), "fp32", Some("cpu")).expect("verify");
        assert_eq!(v["ok"], json!(true), "{v:#}");
    }
}
