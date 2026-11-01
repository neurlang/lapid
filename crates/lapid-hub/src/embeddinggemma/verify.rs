//! `lapid verify embeddinggemma`: compare lapid's EmbeddingGemma with the
//! sentence-transformers reference (`tools/reference/embeddinggemma_ref.py`).
//!
//! * **tokens** — our tokenizer on `prompt + text` must reproduce the reference ids.
//! * **layers** — the traced items run as *one padded batch* (exercising the padding and
//!   sliding-window masks) on the reference ids; every stage (`embed`, `layers.{i}`,
//!   `norm`, `pooled`, `dense.{k}`, `embedding`) is compared per item over its real rows.
//!   For int8 / int4 the correctness check is against the fake-quant reference (same
//!   weights dequantized, f32 kernels on the CPU, closer of f32 / int8 activations);
//!   the PyTorch comparison is reported as quality.
//! * **embeddings** — the public `embed` API (grouped by prompt name, sorted / chunked
//!   batches) vs `model.encode`: min cosine >= 0.9999 (fp32 / fp16), >= 0.999 (int8);
//!   int4 is reported only. Matryoshka sizes are checked the same way.

use std::path::{Path, PathBuf};
use std::time::Instant;

use lapid::{Device, Error, Result};
use lapid_convert::{OutDType, default_model_path};
use lapid_verify::timing::ms;
use lapid_verify::{Activations, Comparison, Stages, tolerance};
use serde_json::{Value, json};

use super::{EmbedOptions, EmbeddingGemma, Precision, cosine_similarity};

/// Kernel tolerance of quantized models vs the fake-quant reference along the full chain.
/// `lapid_verify::KERNEL_TOLERANCE` (2e-2) was measured on Whisper's 12-layer encoder; here
/// the chain is 24 layers plus pooling and two Dense layers. With int8 activations
/// (lapid-cpu), last-ulp differences flip quantized values and grow from 1.3e-3 at layer 2
/// to 2.4e-2 at the head (measured); f32-activation kernels (cuda, vulkan) stay at 1.2e-3.
pub const KERNEL_TOLERANCE: f32 = 3e-2;

/// Minimum cosine of the final embeddings vs the reference, per dtype (`None` = report only).
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

struct Reference {
    items: Vec<Item>,
    layered: Vec<usize>,
}

fn read_reference(dir: &Path) -> Result<Reference> {
    let path = dir.join("result.json");
    let raw = std::fs::read_to_string(&path).map_err(|e| {
        Error::new("io", format!("{}: {e} (run tools/reference/embeddinggemma_ref.py)", path.display()))
    })?;
    let v: Value = serde_json::from_str(&raw)?;
    let bad = |m: &str| Error::new("reference", format!("result.json: {m}"));
    let mut items = Vec::new();
    for it in v.get("items").and_then(Value::as_array).ok_or_else(|| bad("missing items"))? {
        let ids = it
            .get("input_ids")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("item without input_ids"))?
            .iter()
            .map(|x| x.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| bad("non-token id")))
            .collect::<Result<Vec<u32>>>()?;
        items.push(Item {
            text: it.get("text").and_then(Value::as_str).ok_or_else(|| bad("item without text"))?.to_string(),
            prompt_name: it.get("prompt_name").and_then(Value::as_str).map(str::to_string),
            prompt: it.get("prompt").and_then(Value::as_str).unwrap_or("").to_string(),
            ids,
        });
    }
    let layered = v
        .get("layered")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
        .unwrap_or_default();
    Ok(Reference { items, layered })
}

/// The `.lapid` file for `model` (a path, or a hub repo / HF directory converted into the
/// cache on first use) and the conversion time if one ran.
pub(crate) fn resolve_model(model: &str, dtype: &str) -> Result<(PathBuf, Option<f64>)> {
    let p = Path::new(model);
    if p.is_file() || p.extension().is_some_and(|e| e == "lapid") {
        return Ok((p.to_path_buf(), None));
    }
    let out = OutDType::parse(dtype)?;
    let key = if p.is_dir() { format!("local/{}", p.file_name().and_then(|s| s.to_str()).unwrap_or("model")) } else { model.to_string() };
    let path = default_model_path(&key, out);
    if path.is_file() {
        return Ok((path, None));
    }
    let t = Instant::now();
    super::convert::convert_hf(model, out, &path)?;
    Ok((path, Some(ms(t))))
}

fn file_dtype(path: &Path) -> Result<String> {
    let r = lapid::format::Reader::open(path)?;
    r.metadata
        .get("dtype")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| Error::new("missing_metadata", format!("{}: no \"dtype\"", path.display())))
}

/// Per-item named stages of a traced batch: ("s{k}.{stage}", values over real rows).
type Named = Vec<(String, Vec<f32>)>;

fn traced_items(model: &EmbeddingGemma, reference: &Reference) -> Result<Named> {
    let ids: Vec<&[u32]> = reference.layered.iter().map(|&k| reference.items[k].ids.as_slice()).collect();
    let (traced, emb) = model.embed_traced(&ids)?;
    let mut out = Vec::new();
    for (r, &k) in reference.layered.iter().enumerate() {
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
                c.layers.push(lapid_verify::LayerReport::failed(name.as_str()));
            }
        }
    }
    c
}

/// Per row, the closer of two fake-quant comparisons.
fn closest(w: &Comparison, a: &Comparison) -> Comparison {
    let mut out = Comparison::new();
    for (x, y) in w.layers.iter().zip(&a.layers) {
        out.layers.push(if y.rel < x.rel || (x.rel.is_nan() && !y.rel.is_nan()) { y.clone() } else { x.clone() });
    }
    out.notes = w.notes.clone();
    out
}

/// Embeddings through the public API, grouped by prompt name, in item order.
fn api_embeddings(model: &EmbeddingGemma, reference: &Reference, dim: Option<usize>) -> Result<Vec<Vec<f32>>> {
    let mut out = vec![Vec::new(); reference.items.len()];
    let mut groups: Vec<Option<String>> = reference.items.iter().map(|i| i.prompt_name.clone()).collect();
    groups.sort();
    groups.dedup();
    for g in groups {
        let idx: Vec<usize> = (0..reference.items.len()).filter(|&i| reference.items[i].prompt_name == g).collect();
        let texts: Vec<&str> = idx.iter().map(|&i| reference.items[i].text.as_str()).collect();
        let opts = EmbedOptions { prompt_name: g.clone(), dim, ..Default::default() };
        for (e, &i) in model.embed(&texts, &opts)?.into_iter().zip(&idx) {
            out[i] = e;
        }
    }
    Ok(out)
}

/// Full verification report as JSON; check `"ok"`.
pub fn verify(reference_dir: &Path, model: &str, dtype: &str, device: Option<&str>) -> Result<Value> {
    let reference = read_reference(reference_dir)?;
    let acts = Activations::open(&reference_dir.join("activations.safetensors"))?;
    let dev = lapid_verify::open_device(device)?;
    let mut stages = Stages::new();
    let (path, convert_ms) = resolve_model(model, dtype)?;
    if let Some(t) = convert_ms {
        stages.record("convert", t);
    }
    let dtype = file_dtype(&path)?;
    let quantized = matches!(dtype.as_str(), "int8" | "int4");
    let quality_tol = tolerance(&dtype)?;
    let eg = stages.time("load", || EmbeddingGemma::load(&path, &dev))?;

    // ---- tokens
    let mut token_mismatch = Vec::new();
    for (k, it) in reference.items.iter().enumerate() {
        if eg.tokenize(&it.text, &it.prompt)? != it.ids {
            token_mismatch.push(k);
        }
    }

    // ---- layers (one padded batch of the traced items, reference ids)
    let ours = stages.time("traced", || traced_items(&eg, &reference))?;
    let pytorch: Named = ours.iter().map(|(n, _)| Ok((n.clone(), acts.get(n)?))).collect::<Result<_>>()?;
    let quality = compare(&ours, &pytorch, quality_tol);
    let mut kernel_json = Value::Null;
    let correctness = if quantized {
        let cpu = Device::cpu();
        let (fw, fa) = stages.time("fake_quant", || -> Result<_> {
            let w = traced_items(&EmbeddingGemma::load_with(&path, &cpu, Precision::FakeQuantWeights)?, &reference)?;
            let a = traced_items(&EmbeddingGemma::load_with(&path, &cpu, Precision::FakeQuantActivations)?, &reference)?;
            Ok((w, a))
        })?;
        let (kw, ka) = (compare(&ours, &fw, KERNEL_TOLERANCE), compare(&ours, &fa, KERNEL_TOLERANCE));
        let k = closest(&kw, &ka);
        let wq = compare(&fw, &pytorch, quality_tol);
        kernel_json = json!({
            "reference": "fake-quant: same weights dequantized to f32 on cpu; per tensor the closer of f32 and int8-per-32-block activations",
            "tolerance": KERNEL_TOLERANCE,
            "comparison": k.to_json(),
            "weight_quantization_only_max_rel": wq.max_rel(),
        });
        k
    } else {
        quality.clone()
    };

    // ---- final embeddings through the public API
    let ref_emb = acts.get("embeddings")?;
    let full = eg.dim();
    let ours_emb = stages.time("embed", || api_embeddings(&eg, &reference, None))?;
    let cos: Vec<f32> = ours_emb
        .iter()
        .enumerate()
        .map(|(i, e)| ref_emb.get(i * full..(i + 1) * full).map_or(f32::NAN, |r| cosine_similarity(e, r)))
        .collect();
    let min_cos = cos.iter().copied().fold(f32::INFINITY, |a, c| if c.is_nan() { f32::NAN } else { a.min(c) });
    let mut matryoshka = serde_json::Map::new();
    for &d in eg.config().matryoshka_dims.iter().filter(|&&d| d < full) {
        let e = api_embeddings(&eg, &reference, Some(d))?;
        let m = e
            .iter()
            .enumerate()
            .map(|(i, v)| ref_emb.get(i * full..i * full + d).map_or(f32::NAN, |r| cosine_similarity(v, r)))
            .fold(f32::INFINITY, f32::min);
        matryoshka.insert(d.to_string(), json!(m));
    }
    let threshold = min_cosine(&dtype);
    let cos_ok = threshold.is_none_or(|t| min_cos >= t);
    let ok = token_mismatch.is_empty() && correctness.ok() && cos_ok;

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
            "reference": "PyTorch fp32 (sentence-transformers)",
            "informational": quantized,
            "tolerance": quality_tol,
            "within_tolerance": quality.ok(),
            "max_rel": quality.max_rel(),
            "layers": if quantized { serde_json::to_value(&quality.layers)? } else { Value::Null },
        },
        "embeddings": {
            "cosine": cos,
            "min_cosine": min_cos,
            "threshold": threshold,
            "matryoshka_min_cosine": matryoshka,
        },
        "reference_dir": reference_dir.display().to_string(),
        "model_file": path.display().to_string(),
        "timings_ms": stages.to_json(),
        "notes": notes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds() {
        assert_eq!(min_cosine("fp16"), Some(0.9999));
        assert_eq!(min_cosine("int8"), Some(0.999));
        assert_eq!(min_cosine("int4"), None);
    }
}
