//! `lapid verify <audio model>`: the generic driver over [`AudioDef`].
//!
//! The reference directory holds `result.json` or sub-directories that do (one per sample).
//! Per sample:
//! * **quality**: every row of [`AudioDef::stages`] against the PyTorch reference, at the
//!   dtype's tolerance (front-end rows at their own);
//! * **kernels** (quantized files): every kernel row against the fake-quant model (the same
//!   file with weights dequantized to f32, on the CPU; per row the closer of f32 and
//!   int8-per-block activations) at `KERNEL_TOLERANCE`;
//! * **argmax**: per-step argmax of the named logits rows equals the correctness reference
//!   (PyTorch for float files, the fake-quant model for quantized ones);
//! * **checks** of the stages and of [`AudioDef::end_to_end`] (token / text exactness for
//!   strict dtypes, ...).
//!
//! `ok` = correctness rows (front end + kernels or quality) + argmax + required checks
//! (+ quality for [`AudioDef::QUALITY_REQUIRED`] models).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use lapid::{Device, Error, Result};
use lapid_nn::Precision;
use lapid_verify::timing::Stages as Timings;
use lapid_verify::{Activations, Comparison, KERNEL_TOLERANCE, open_device, tolerance};
use serde_json::{Map, Value, json};

use super::audio::{Argmax, AudioDef, Check, Row, Sample, find_path};
use super::resolve::{file_dtype, resolve};
use crate::cmd::ModelArgs;

/// Sample directories under `reference` (itself, or its sub-directories with a `result.json`).
pub fn sample_dirs(reference: &Path, script: &str) -> Result<Vec<PathBuf>> {
    if reference.join("result.json").is_file() {
        return Ok(vec![reference.to_path_buf()]);
    }
    let rd = std::fs::read_dir(reference).map_err(|e| Error::new("io", format!("{}: {e} (run {script})", reference.display())))?;
    let mut v: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|d| d.join("result.json").is_file()).collect();
    v.sort();
    if v.is_empty() {
        return Err(Error::new("not_found", format!("{}: no result.json (run {script})", reference.display())));
    }
    Ok(v)
}

fn open_sample(dir: &Path, script: &str) -> Result<(Value, Activations, PathBuf)> {
    let raw = std::fs::read_to_string(dir.join("result.json")).map_err(|e| Error::new("io", format!("{}/result.json: {e} (run {script})", dir.display())))?;
    let result: Value = serde_json::from_str(&raw)?;
    let rel = result.get("audio").and_then(Value::as_str).unwrap_or("data/jfk.wav");
    let audio = find_path(dir, rel).ok_or_else(|| Error::new("not_found", format!("{}: audio {rel:?} not found", dir.display())))?;
    Ok((result, Activations::open(&dir.join("activations.safetensors"))?, audio))
}

/// The three models of one file: under test, and the two fake-quant references.
struct Loaded<M> {
    path: PathBuf,
    dtype: String,
    model: M,
    fake: Option<(M, M)>,
}

fn per_step_argmax(ours: &[f32], reference: &[f32], a: &Argmax) -> Vec<bool> {
    let am = |v: &[f32]| v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b });
    let mut cuts = vec![0];
    cuts.extend(a.splits.iter().copied().filter(|&s| s > 0 && s < a.width));
    cuts.push(a.width);
    if a.width == 0 || ours.len() != reference.len() {
        return vec![false];
    }
    ours.chunks_exact(a.width)
        .zip(reference.chunks_exact(a.width))
        .map(|(x, y)| cuts.windows(2).all(|c| am(&x[c[0]..c[1]]) == am(&y[c[0]..c[1]])))
        .collect()
}

fn checks_json(checks: &[Check]) -> Value {
    Value::Object(checks.iter().map(|c| (c.name.clone(), json!({"ok": c.ok, "required": c.required}))).collect())
}

fn verify_sample<M: AudioDef>(m: &Loaded<M>, s: &Sample, timings: &mut Timings) -> Result<Value> {
    let quantized = !matches!(m.dtype.as_str(), "fp32" | "fp16");
    let tol = tolerance(&m.dtype)?;
    let audio = lapid_nn::audio::load_audio(&s.audio, m.model.sample_rate())?;
    let ours = timings.time("stages", || m.model.stages(s, &audio))?;
    let reference: Vec<Vec<f32>> = ours.rows.iter().map(|r| M::reference_row(s, &r.name)).collect::<Result<_>>()?;

    let (mut front, mut quality) = (Comparison::new(), Comparison::new());
    for (r, want) in ours.rows.iter().zip(&reference) {
        match r.tol {
            Some(t) => front.compare(&r.name, &r.values, want, t),
            None => quality.compare(&r.name, &r.values, want, tol),
        };
    }
    // Kernel rows vs the closer fake-quant reference; the reference values argmax uses.
    let mut kernel = None;
    let mut correct_ref: BTreeMap<String, Vec<f32>> =
        ours.rows.iter().zip(&reference).map(|(r, v)| (r.name.clone(), v.clone())).collect();
    if let Some((fw, fa)) = &m.fake {
        let (sw, sa) = timings.time("fake_quant", || -> Result<_> { Ok((fw.stages(s, &audio)?, fa.stages(s, &audio)?)) })?;
        let find = |st: &super::audio::Stages, n: &str| st.rows.iter().find(|r| r.name == n).map(|r| r.values.clone());
        let (mut k, mut matched) = (Comparison::new(), Map::new());
        for r in ours.rows.iter().filter(|r| r.kernel && r.tol.is_none()) {
            let (w, a) = (find(&sw, &r.name).unwrap_or_default(), find(&sa, &r.name).unwrap_or_default());
            let rel = |v: &[f32]| lapid_verify::diff(&r.values, v).map_or(f32::NAN, |d| d.rel);
            let (rw, ra) = (rel(&w), rel(&a));
            let use_a = ra < rw || (rw.is_nan() && !ra.is_nan());
            let chosen = if use_a { a } else { w };
            k.compare(&r.name, &r.values, &chosen, KERNEL_TOLERANCE);
            matched.insert(r.name.clone(), json!(if use_a { "int8_activations" } else { "f32_activations" }));
            correct_ref.insert(r.name.clone(), chosen);
        }
        kernel = Some((k, matched));
    }

    let argmax: Map<String, Value> = ours
        .argmax
        .iter()
        .map(|a| {
            let row = ours.rows.iter().find(|r| r.name == a.row).map_or(&[][..], |r| &r.values[..]);
            let want = correct_ref.get(&a.row).map_or(&[][..], |v| &v[..]);
            (a.row.clone(), json!(per_step_argmax(row, want, a)))
        })
        .collect();
    let argmax_ok = argmax.values().all(|v| v.as_array().is_some_and(|a| !a.is_empty() && a.iter().all(|b| b == true)));

    let e2e = timings.time("end_to_end", || m.model.end_to_end(s, &audio))?;
    let checks: Vec<&Check> = ours.checks.iter().chain(&e2e.checks).collect();
    let checks_ok = checks.iter().all(|c| c.ok || !c.required);
    let correctness = match &kernel {
        Some((k, _)) => k,
        None => &quality,
    };
    let quality_ok = !quantized || !M::QUALITY_REQUIRED || quality.ok();
    let ok = front.ok() && correctness.ok() && argmax_ok && checks_ok && quality_ok;
    let mut layers = front.layers.clone();
    layers.extend(correctness.layers.iter().cloned());
    let notes = [front.notes.clone(), correctness.notes.clone()].concat();
    let all_checks: Vec<Check> = checks.iter().map(|c| Check::new(c.name.clone(), c.ok, c.required)).collect();
    let mut out = json!({
        "sample": s.dir.display().to_string(),
        "ok": ok,
        "layers": layers,
        "first_failing_layer": front.first_failing().or(correctness.first_failing()),
        "argmax_match": argmax,
        "checks": checks_json(&all_checks),
        "text": e2e.text,
        "reference_text": e2e.reference_text,
        "kernel": kernel.as_ref().map_or(Value::Null, |(k, matched)| json!({
            "reference": "fake-quant: the same weights dequantized to f32 on cpu; per tensor the closer of f32 and int8-per-32-block activations",
            "tolerance": KERNEL_TOLERANCE,
            "matched_reference": matched,
            "comparison": k.to_json(),
        })),
        "quality": {
            "reference": M::DESCRIPTOR.reference_label,
            "informational": quantized && !M::QUALITY_REQUIRED,
            "tolerance": tol,
            "within_tolerance": quality.ok(),
            "max_rel": quality.max_rel(),
            "layers": quality.layers,
        },
        "notes": notes,
    });
    if let Value::Object(o) = &mut out {
        o.extend(e2e.info);
    }
    Ok(out)
}

/// Verify `M` against its reference (default: the descriptor's); check `"ok"`.
pub fn verify<M: AudioDef>(reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    let d = &M::DESCRIPTOR;
    let dev = open_device(args.device.as_deref())?;
    let dtype_arg = args.dtype.as_deref().unwrap_or(d.default_dtype);
    let reference = reference.unwrap_or(Path::new(d.reference_dir));
    let mut timings = Timings::new();
    let mut loaded: Vec<Loaded<M>> = Vec::new();
    let (mut samples, mut ran, mut all_ok) = (Vec::new(), 0, true);
    for dir in sample_dirs(reference, d.reference_script)? {
        let (result, acts, audio) = open_sample(&dir, d.reference_script)?;
        let source = match &args.model {
            Some(m) => m.clone(),
            None => {
                let probe = Sample { dir: dir.clone(), result: result.clone(), acts: Activations::open(&dir.join("activations.safetensors"))?, audio: audio.clone(), dtype: String::new(), strict: false };
                M::sample_model(&probe).unwrap_or_else(|| d.default_model.to_string())
            }
        };
        if source.starts_with('.') || source.starts_with('/') || source.starts_with("data/") {
            if !Path::new(&source).exists() {
                samples.push(json!({"sample": dir.display().to_string(), "skipped": format!("model {source} not found")}));
                continue;
            }
        }
        let r = resolve(&source, Some(dtype_arg))?;
        if let Some(t) = r.convert_ms {
            timings.record("convert", t);
        }
        if !loaded.iter().any(|l| l.path == r.path) {
            let dtype = file_dtype(&r.path)?;
            let model = timings.time("load", || M::load_with(&r.path, &dev, Precision::Stored))?;
            let fake = if matches!(dtype.as_str(), "fp32" | "fp16") {
                None
            } else {
                let cpu = Device::cpu();
                Some((M::load_with(&r.path, &cpu, Precision::FakeQuantWeights)?, M::load_with(&r.path, &cpu, Precision::FakeQuantActivations)?))
            };
            loaded.push(Loaded { path: r.path.clone(), dtype, model, fake });
        }
        let m = loaded.iter().find(|l| l.path == r.path).ok_or_else(|| Error::new("internal", "model not loaded"))?;
        let s = Sample { dir, result, acts, audio, dtype: m.dtype.clone(), strict: d.tokens_must_match(&m.dtype) };
        let mut v = verify_sample(m, &s, &mut timings)?;
        all_ok &= v["ok"].as_bool().unwrap_or(false);
        ran += 1;
        v["model_file"] = json!(r.path.display().to_string());
        samples.push(v);
    }
    let dtypes: Vec<&str> = loaded.iter().map(|l| l.dtype.as_str()).collect();
    Ok(json!({
        "ok": all_ok && ran > 0,
        "model": args.model.as_deref().unwrap_or(d.default_model),
        "dtype": if dtypes.is_empty() { dtype_arg } else { dtypes[0] },
        "device": dev.description(),
        "correctness_reference": if dtypes.iter().any(|t| !matches!(*t, "fp32" | "fp16")) { "fake-quant" } else { "pytorch-fp32" },
        "reference_dir": reference.display().to_string(),
        "samples": samples,
        "timings_ms": timings.to_json(),
    }))
}

/// Rows from a list of `(name, values)` pairs (all kernel rows).
pub fn rows(named: Vec<(String, Vec<f32>)>) -> Vec<Row> {
    named.into_iter().map(|(n, v)| Row::new(n, v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_segments() {
        let a = Argmax { row: "x".into(), width: 5, splits: vec![3] };
        assert_eq!(per_step_argmax(&[0.0, 1.0, 0.5, 2.0, 0.0], &[0.0, 0.9, 0.1, 3.0, 1.0], &a), vec![true]);
        assert_eq!(per_step_argmax(&[0.0, 1.0, 0.5, 2.0, 0.0], &[0.0, 0.9, 0.1, 0.0, 1.0], &a), vec![false]);
        assert_eq!(per_step_argmax(&[1.0], &[], &a), vec![false]);
    }
}
