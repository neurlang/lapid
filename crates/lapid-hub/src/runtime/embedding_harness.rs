//! `lapid verify` / `lapid bench` / calibrated int4 / serving for any [`EmbedderDef`] model.
//!
//! Reference format (`tools/reference/<model>_ref.py`, sentence-transformers): `result.json`
//! `items` (`text`, `prompt_name`, `prompt`, `input_ids`) and `layered` (indices of the traced
//! items); `activations.safetensors` with `s<k>.<tap>` per traced item over its real rows
//! (`embed`, `layers.{i}`, `norm`, `pooled`, `dense.{k}`, `embedding`) and `embeddings`
//! `[items, dim]` from `model.encode`.
//!
//! Verify: tokens; the traced items as *one padded batch* (exercising the padding masks) on
//! the reference ids, every tap vs PyTorch (fp32 / fp16) or vs fake-quant references (int8 /
//! int4: same weights dequantized on the CPU, per tensor the closer of f32 and int8
//! activations; PyTorch reported as quality); the public `embed` API (grouped by prompt,
//! sorted / chunked batches) vs `model.encode` by cosine, Matryoshka sizes too.
//! Bench: sentences per second at batch 1 and batch 32.

use std::path::Path;
use std::time::Instant;

use lapid::format::Reader;
use lapid::{Device, Error, LapidFile, Result, Tensor};
use lapid_convert::calibrate::{CalibrationSource, Calibrator};
use lapid_convert::policy::Role;
use lapid_nn::Precision;
use lapid_verify::harness::{Reference, ids};
use lapid_verify::timing::{median, ms, stats};
use lapid_verify::{Comparison, LayerReport, Stages, tolerance};
use serde_json::{Value, json};

use super::causal_lm::Taps;
use super::embedding::{EmbedOptions, EmbedderDef, TextEmbedder, TokenBatch, cosine_similarity, embed, embed_traced, embedding_graph, load, prompt_for, tokenize};
use super::registry::ModelEntry;
use super::resolve::{file_dtype, resolve};
use super::serve::{AnyModel, Capability, EmbedParams, Embeddings, LoadOptions, ModelMeta};
use crate::cmd::ModelArgs;

/// Kernel tolerance vs the fake-quant reference along the full chain. The harness default
/// (2e-2) was measured on Whisper's 12-layer encoder; EmbeddingGemma's chain is 24 layers
/// plus pooling and two Dense layers, where int8-activation kernels (lapid-cpu) drift from
/// 1.3e-3 at layer 2 to 2.4e-2 at the head (measured); f32-activation kernels stay at 1.2e-3.
pub const KERNEL_TOLERANCE: f32 = 3e-2;

/// Minimum cosine of the final embeddings vs the reference (`None` = report only).
pub fn min_cosine(dtype: &str) -> Option<f32> {
    match dtype {
        "fp32" | "fp16" => Some(0.9999),
        "int8" => Some(0.999),
        _ => None,
    }
}

struct Item {
    text: String,
    prompt_name: Option<String>,
    prompt: String,
    ids: Vec<u32>,
}

fn items(r: &Reference) -> Result<(Vec<Item>, Vec<usize>)> {
    let list = r.result.get("items").and_then(Value::as_array).ok_or_else(|| Error::new("reference", "result.json: missing items"))?;
    let s = |v: &Value, k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);
    let items = list
        .iter()
        .map(|it| {
            Ok(Item {
                text: s(it, "text").ok_or_else(|| Error::new("reference", "item without text"))?,
                prompt_name: s(it, "prompt_name"),
                prompt: s(it, "prompt").unwrap_or_default(),
                ids: ids(it, "input_ids")?,
            })
        })
        .collect::<Result<_>>()?;
    let layered = r.result.get("layered").and_then(Value::as_array).map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect());
    Ok((items, layered.unwrap_or_default()))
}

type Named = Vec<(String, Vec<f32>)>;

/// Per traced item, every tap over its real rows: `s<k>.<tap>`.
fn traced_items<M: TextEmbedder>(model: &M, items: &[Item], layered: &[usize]) -> Result<Named> {
    let ids: Vec<&[u32]> = layered.iter().map(|&k| items[k].ids.as_slice()).collect();
    let (traced, emb) = embed_traced(model, &ids)?;
    let mut out = Vec::new();
    for (r, &k) in layered.iter().enumerate() {
        let n = ids[r].len();
        for (name, data, shape) in &traced {
            let row = match shape.as_slice() {
                [_, s, d] => data[r * s * d..r * s * d + n * d].to_vec(),
                [_, d] => data[r * d..(r + 1) * d].to_vec(),
                other => return Err(Error::new("internal", format!("traced {name}: unexpected shape {other:?}"))),
            };
            out.push((format!("s{k}.{name}"), row));
        }
        out.push((format!("s{k}.embedding"), emb[r].clone()));
    }
    Ok(out)
}

fn compare(ours: &Named, reference: &Named, tol: f32) -> Comparison {
    let mut c = Comparison::new();
    for (name, r) in reference {
        match ours.iter().find(|(n, _)| n == name) {
            Some((_, o)) => {
                c.compare(name, o, r, tol);
            }
            None => {
                c.notes.push(format!("{name}: not produced"));
                c.layers.push(LayerReport::failed(name.as_str()));
            }
        }
    }
    c
}

/// Per tensor, the closer of two fake-quant comparisons.
fn closest(w: &Comparison, a: &Comparison) -> Comparison {
    let mut out = Comparison::new();
    for (x, y) in w.layers.iter().zip(&a.layers) {
        out.layers.push(if y.rel < x.rel || (x.rel.is_nan() && !y.rel.is_nan()) { y.clone() } else { x.clone() });
    }
    out.notes = w.notes.clone();
    out
}

/// Embeddings through the public API, grouped by prompt name, in item order.
fn api_embeddings<M: TextEmbedder>(model: &M, items: &[Item], dim: Option<usize>) -> Result<Vec<Vec<f32>>> {
    let mut out = vec![Vec::new(); items.len()];
    let mut groups: Vec<Option<String>> = items.iter().map(|i| i.prompt_name.clone()).collect();
    groups.sort();
    groups.dedup();
    for g in groups {
        let idx: Vec<usize> = (0..items.len()).filter(|&i| items[i].prompt_name == g).collect();
        let texts: Vec<&str> = idx.iter().map(|&i| items[i].text.as_str()).collect();
        let opts = EmbedOptions { prompt_name: g.clone(), dim, ..Default::default() };
        for (e, &i) in embed(model, &texts, &opts)?.into_iter().zip(&idx) {
            out[i] = e;
        }
    }
    Ok(out)
}

/// Minimum cosine of `ours` vs the first `d` values of each reference row.
fn min_cos(ours: &[Vec<f32>], reference: &[f32], full: usize, d: usize) -> f32 {
    ours.iter()
        .enumerate()
        .map(|(i, e)| reference.get(i * full..i * full + d).map_or(f32::NAN, |r| cosine_similarity(e, r)))
        .fold(f32::INFINITY, |a, c| if c.is_nan() || a.is_nan() { f32::NAN } else { a.min(c) })
}

/// `lapid verify <model>`; check `"ok"`.
pub fn verify<M: EmbedderDef>(reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    let desc = &M::DESCRIPTOR;
    let model = args.model.as_deref().unwrap_or(desc.default_model);
    let dir = reference.unwrap_or(Path::new(desc.reference_dir));
    let r = Reference::open(dir, desc.reference_script)?;
    let (items, layered) = items(&r)?;
    let dev = lapid_verify::open_device(args.device.as_deref())?;
    let mut stages = Stages::new();
    let res = resolve(model, Some(args.dtype.as_deref().unwrap_or(desc.default_dtype)))?;
    if let Some(t) = res.convert_ms {
        stages.record("convert", t);
    }
    let path = res.path;
    let dtype = file_dtype(&path)?;
    let quantized = matches!(dtype.as_str(), "int8" | "int4");
    let quality_tol = tolerance(&dtype)?;
    let m = stages.time("load", || load::<M>(&path, &dev, Precision::Stored))?;

    let mut token_mismatch = Vec::new();
    for (k, it) in items.iter().enumerate() {
        if tokenize(&m, &it.text, &it.prompt, Default::default())? != it.ids {
            token_mismatch.push(k);
        }
    }

    let ours = stages.time("traced", || traced_items(&m, &items, &layered))?;
    let pytorch: Named = ours.iter().map(|(n, _)| Ok((n.clone(), r.acts.get(n)?))).collect::<Result<_>>()?;
    let quality = compare(&ours, &pytorch, quality_tol);
    let mut kernel_json = Value::Null;
    let correctness = if quantized {
        let cpu = Device::cpu();
        let (fw, fa) = stages.time("fake_quant", || -> Result<_> {
            let w = traced_items(&load::<M>(&path, &cpu, Precision::FakeQuantWeights)?, &items, &layered)?;
            let a = traced_items(&load::<M>(&path, &cpu, Precision::FakeQuantActivations)?, &items, &layered)?;
            Ok((w, a))
        })?;
        let k = closest(&compare(&ours, &fw, KERNEL_TOLERANCE), &compare(&ours, &fa, KERNEL_TOLERANCE));
        kernel_json = json!({
            "reference": "fake-quant: same weights dequantized to f32 on cpu; per tensor the closer of f32 and int8-per-32-block activations",
            "tolerance": KERNEL_TOLERANCE,
            "comparison": k.to_json(),
            "weight_quantization_only_max_rel": compare(&fw, &pytorch, quality_tol).max_rel(),
        });
        k
    } else {
        quality.clone()
    };

    let ref_emb = r.acts.get("embeddings")?;
    let full = m.dim();
    let api = stages.time("embed", || api_embeddings(&m, &items, None))?;
    let cos: Vec<f32> = api.iter().enumerate().map(|(i, e)| ref_emb.get(i * full..(i + 1) * full).map_or(f32::NAN, |x| cosine_similarity(e, x))).collect();
    let min = min_cos(&api, &ref_emb, full, full);
    let mut matryoshka = serde_json::Map::new();
    for d in m.matryoshka_dims().into_iter().filter(|&d| d < full) {
        matryoshka.insert(d.to_string(), json!(min_cos(&api_embeddings(&m, &items, Some(d))?, &ref_emb, full, d)));
    }
    let threshold = min_cosine(&dtype);
    let ok = token_mismatch.is_empty() && correctness.ok() && threshold.is_none_or(|t| min >= t);
    let mut notes = correctness.notes.clone();
    if !token_mismatch.is_empty() {
        notes.push(format!("token ids differ for items {token_mismatch:?}"));
    }
    Ok(json!({
        "ok": ok,
        "device": dev.description(),
        "model": model,
        "dtype": dtype,
        "tokens_match": token_mismatch.is_empty(),
        "correctness_reference": if quantized { "fake-quant" } else { "pytorch-fp32" },
        "tolerance": if quantized { KERNEL_TOLERANCE } else { quality_tol },
        "first_failing_layer": correctness.first_failing(),
        "max_rel": correctness.max_rel(),
        "layers": correctness.layers,
        "kernel": kernel_json,
        "quality": {
            "reference": desc.reference_label,
            "informational": quantized,
            "tolerance": quality_tol,
            "within_tolerance": quality.ok(),
            "max_rel": quality.max_rel(),
            "layers": if quantized { serde_json::to_value(&quality.layers)? } else { Value::Null },
        },
        "embeddings": { "cosine": cos, "min_cosine": min, "threshold": threshold, "matryoshka_min_cosine": matryoshka },
        "reference_dir": dir.display().to_string(),
        "model_file": path.display().to_string(),
        "timings_ms": stages.to_json(),
        "notes": notes,
    }))
}

const CORPUS: [&str; 16] = [
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Sourdough bread needs a lively starter, patience and a very hot oven.",
    "Rust guarantees memory safety without a garbage collector.",
    "The stock market closed higher today after a volatile trading session.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "A good night's sleep improves memory, mood and concentration.",
    "The museum's new exhibition features impressionist paintings from France.",
    "Electric cars are becoming cheaper as battery prices continue to fall.",
    "The recipe calls for two cups of flour, one egg and a pinch of salt.",
    "Mount Everest is the highest mountain above sea level on Earth.",
    "Machine translation quality has improved dramatically in the last decade.",
    "The library will be closed on Monday for the public holiday.",
];
/// Sentences per benchmark pass.
const BENCH_SENTENCES: usize = 64;

fn bench_mode<M: TextEmbedder>(model: &M, texts: &[&str], opts: &EmbedOptions, batch: usize, runs: usize) -> Result<Value> {
    let opts = opts.clone().with_batch_size(batch);
    let pass = || -> Result<f64> {
        let t = Instant::now();
        if batch == 1 {
            for s in texts {
                embed(model, &[s], &opts)?;
            }
        } else {
            embed(model, texts, &opts)?;
        }
        Ok(ms(t))
    };
    let warmup_ms = pass()?;
    let totals: Vec<f64> = (0..runs).map(|_| pass()).collect::<Result<_>>()?;
    let med = median(totals.clone());
    let n = texts.len() as f64;
    Ok(json!({
        "batch_size": batch,
        "sentences": texts.len(),
        "warmup_ms": warmup_ms,
        "total_ms": stats(&totals),
        "ms_per_sentence": med / n,
        "sentences_per_s": if med > 0.0 { n / (med / 1e3) } else { 0.0 },
    }))
}

/// `lapid bench <model> [file]`: sentences/s at batch 1 and 32 (`file`: one sentence per
/// line; default: a built-in corpus), with the `document` prompt when the model has one.
pub fn bench<M: EmbedderDef>(input: Option<&Path>, args: &ModelArgs, runs: usize) -> Result<Value> {
    if runs == 0 {
        return Err(Error::new("invalid_argument", "runs must be at least 1"));
    }
    let desc = &M::DESCRIPTOR;
    let model = args.model.as_deref().unwrap_or(desc.default_model);
    let lines: Vec<String> = match input {
        Some(p) => std::fs::read_to_string(p)
            .map_err(|e| Error::new("io", format!("{}: {e}", p.display())))?
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect(),
        None => CORPUS.iter().map(|s| s.to_string()).collect(),
    };
    if lines.is_empty() {
        return Err(Error::new("invalid_argument", "no sentences"));
    }
    let texts: Vec<&str> = lines.iter().map(String::as_str).cycle().take(BENCH_SENTENCES).collect();
    let dev = lapid_verify::open_device(args.device.as_deref())?;
    let res = resolve(model, Some(args.dtype.as_deref().unwrap_or(desc.default_dtype)))?;
    let t = Instant::now();
    let m = load::<M>(&res.path, &dev, Precision::Stored)?;
    let load_ms = ms(t);
    let opts = if m.prompt("document").is_some() { EmbedOptions::document() } else { EmbedOptions::default() };
    let prompt = prompt_for(&m, &opts)?;
    let tokens: usize = texts.iter().map(|s| tokenize(&m, s, &prompt, opts.truncation).map(|v| v.len())).sum::<Result<usize>>()?;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    Ok(json!({
        "engine": "lapid",
        "model": model,
        "model_file": res.path.display().to_string(),
        "dtype": file_dtype(&res.path)?,
        "device": dev.description(),
        "backend": dev.name(),
        "host": { "arch": std::env::consts::ARCH, "os": std::env::consts::OS, "threads": threads },
        "runs": runs,
        "convert_ms": res.convert_ms,
        "load_ms": load_ms,
        "mean_tokens": tokens as f64 / texts.len() as f64,
        "batch_1": bench_mode(&m, &texts, &opts, 1, runs)?,
        "batch_32": bench_mode(&m, &texts, &opts, 32, runs)?,
    }))
}

/// Calibration for int4 (`lapid convert --dtype int4 --calibrate`): each calibration text,
/// prefixed with one of the model's prompts in turn (and once without), is one unpadded
/// single-item batch through encoder, pooling and head.
struct EmbedCalibrator<M: EmbedderDef> {
    /// Same image the model was loaded from (weights re-upload to the identical leaves).
    file: LapidFile,
    device: Device,
    model: M,
    samples: Vec<Vec<u32>>,
}

const CALIB_CHUNK_CHARS: usize = 1500;
const CALIB_MAX_SAMPLES: usize = 128;

fn calibrate<M: EmbedderDef>(path: &Path, device: &Device, source: &CalibrationSource) -> Result<Box<dyn Calibrator>> {
    let bytes = Reader::open(path)?.data();
    let file = LapidFile::from_bytes(bytes.clone())?;
    let model = M::from_file(&LapidFile::from_bytes(bytes)?, device, Precision::Stored)?;
    let mut prompts = vec![String::new()];
    prompts.extend(model.prompt_names().iter().filter_map(|n| model.prompt(n)));
    let samples = source
        .texts(CALIB_CHUNK_CHARS, CALIB_MAX_SAMPLES)?
        .iter()
        .enumerate()
        .map(|(i, t)| tokenize(&model, t, &prompts[i % prompts.len()], Default::default()))
        .collect::<Result<_>>()?;
    Ok(Box::new(EmbedCalibrator { file, device: device.clone(), model, samples }))
}

impl<M: EmbedderDef> Calibrator for EmbedCalibrator<M> {
    fn weights(&self) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::new();
        for (name, t) in self.file.tensors.iter() {
            if M::CONVERT.role(name, t.shape()) == Role::Linear && t.rank() == 2 {
                out.push((name.clone(), self.device.upload(t)?));
            }
        }
        Ok(out)
    }

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn forward(&mut self, i: usize) -> Result<Vec<Tensor>> {
        let ids = &self.samples[i];
        let batch = TokenBatch::new(&[ids.as_slice()], ids.len(), self.model.pad_id())?;
        Ok(vec![embedding_graph(&self.model, &batch, &mut Taps::off())?])
    }

    fn eval(&self, outputs: &[&Tensor]) -> Result<Vec<Vec<f32>>> {
        self.device.eval(outputs)?.iter().map(|t| self.device.to_vec_f32(t)).collect()
    }
}

/// [`AnyModel`] of an embedder (capability `embedding`).
pub struct EmbedServed<M: EmbedderDef> {
    model: M,
    meta: ModelMeta,
}

impl<M: EmbedderDef> AnyModel for EmbedServed<M> {
    fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    fn meta_mut(&mut self) -> &mut ModelMeta {
        &mut self.meta
    }

    fn embed(&mut self, texts: &[String], p: &EmbedParams) -> Result<Embeddings> {
        let o = EmbedOptions { prompt_name: p.prompt_name.clone(), dim: p.dimensions, ..Default::default() };
        let prompt = prompt_for(&self.model, &o)?;
        let mut prompt_tokens = 0;
        for t in texts {
            prompt_tokens += tokenize(&self.model, t, &prompt, o.truncation)?.len();
        }
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        Ok(Embeddings { vectors: embed(&self.model, &refs, &o)?, prompt_tokens })
    }
}

/// Registry `serve` entry of an embedder.
pub fn serve_embedder<M: EmbedderDef>(path: &Path, device: &Device, mut meta: ModelMeta, _: &LoadOptions) -> Result<Box<dyn AnyModel>> {
    let model = load::<M>(path, device, Precision::Stored)?;
    meta.capabilities = vec![Capability::Embedding];
    meta.extra.insert("embedding_dim".into(), json!(model.dim()));
    meta.extra.insert("prompts".into(), json!(model.prompt_names()));
    Ok(Box::new(EmbedServed { model, meta }))
}

impl ModelEntry {
    /// The entry of a text embedder: generic verify, bench, calibrated int4 and serving.
    pub const fn embedder<M: EmbedderDef>() -> ModelEntry {
        ModelEntry {
            name: M::DESCRIPTOR.name,
            convert: M::CONVERT,
            verify: verify::<M>,
            bench: bench::<M>,
            serve: serve_embedder::<M>,
            calibrate: Some(calibrate::<M>),
        }
    }
}
