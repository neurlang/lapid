//! Calibrated int4: GPTQ (`gptq.rs`) driven by activations captured from the model's own
//! forward pass on calibration inputs.
//!
//! # How a model plugs in
//! A model implements [`Calibrator`] in its converter (see `lapid-hub` `lfm3/convert/
//! calibrate.rs`, `embeddinggemma/calibrate.rs`):
//! 1. convert the checkpoint to an f32 `.lapid` file and load the model from it with the
//!    weights as stored (`Precision::Stored`);
//! 2. [`Calibrator::weights`]: `(name, tensor)` for every linear weight as the model's
//!    graphs hold it (`device.upload(&file.get(name)?)` returns the cached, identical leaf);
//! 3. [`Calibrator::len`] / [`Calibrator::forward`]: the lazy forward graph (outputs) of
//!    each calibration input (text, audio, ...); nothing needs evaluating;
//! 4. [`Calibrator::eval`]: evaluate lazy tensors on the model's device.
//!
//! [`convert_calibrated`] then finds every `x @ Wᵀ` whose `W` is one of those leaves
//! ([`matmul_inputs`]), accumulates the Gram matrix `Σ x xᵀ` of its input on the device,
//! quantizes the weight with GPTQ and rewrites the f32 file with the dtype policy, using
//! the calibrated bytes for those weights. Gram matrices are kept on the host (f32,
//! `in²·4` bytes per distinct input); when they exceed `memory_budget` the forward passes
//! are repeated for groups of weights.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use lapid_format::Reader;
use lapid_quant::Format;
use lapid_tensor::{DType, Error, Op, Result, Tensor, bail};
use half::{bf16, f16};
use serde_json::{Value, json};

use crate::OutDType;
pub use crate::gptq::GptqOptions;
use crate::policy::{Target, target_named};
use crate::writer::{ModelWriter, RoleArc};

/// The built-in calibration text (mixed prose, dialogue, code, lists, several languages).
pub const BUILTIN_TEXT: &str = include_str!("calibration.txt");

/// Where calibration inputs come from (`--calibrate builtin|<file>`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CalibrationSource {
    Builtin,
    File(PathBuf),
}

impl CalibrationSource {
    pub fn parse(s: &str) -> CalibrationSource {
        match s.trim() {
            "" | "builtin" | "default" => CalibrationSource::Builtin,
            p => CalibrationSource::File(PathBuf::from(p)),
        }
    }

    pub fn describe(&self) -> String {
        match self {
            CalibrationSource::Builtin => "builtin".into(),
            CalibrationSource::File(p) => p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned()),
        }
    }

    /// Text samples: the builtin corpus or a UTF-8 file, split at blank lines and packed
    /// into chunks of at most `max_chars` characters (a longer paragraph is its own chunk);
    /// at most `max_chunks`, evenly spread over the text.
    pub fn texts(&self, max_chars: usize, max_chunks: usize) -> Result<Vec<String>> {
        let text = match self {
            CalibrationSource::Builtin => BUILTIN_TEXT.to_string(),
            CalibrationSource::File(p) => std::fs::read_to_string(p)
                .map_err(|e| Error::new("io", format!("calibration file {}: {e}", p.display())))?,
        };
        let all = split_text(&text, max_chars);
        if all.len() <= max_chunks.max(1) {
            return Ok(all);
        }
        let n = max_chunks.max(1);
        Ok((0..n).map(|i| all[i * all.len() / n].clone()).collect())
    }
}

/// Paragraphs (blank-line separated, CRLF or LF) packed into chunks of at most
/// `max_chars` chars; a longer paragraph is cut at whitespace into pieces of that size.
pub fn split_text(text: &str, max_chars: usize) -> Vec<String> {
    let max_chars = max_chars.max(1);
    let text = text.replace("\r\n", "\n");
    let mut pieces: Vec<String> = Vec::new();
    for para in text.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
        let mut cur = String::new();
        for word in para.split_inclusive(char::is_whitespace) {
            if !cur.is_empty() && cur.chars().count() + word.chars().count() > max_chars {
                pieces.push(std::mem::take(&mut cur).trim().to_string());
            }
            cur.push_str(word);
        }
        if !cur.trim().is_empty() {
            pieces.push(cur.trim().to_string());
        }
    }
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for p in pieces {
        if !cur.is_empty() && cur.chars().count() + p.chars().count() + 2 > max_chars {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push_str("\n\n");
        }
        cur.push_str(&p);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// A model's cooperation for calibration (see the module docs).
pub trait Calibrator {
    /// `(file tensor name, leaf tensor as used in the graphs)` of every calibratable weight.
    fn weights(&self) -> Result<Vec<(String, Tensor)>>;
    /// Number of calibration inputs (forward passes).
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Lazy outputs of the forward graph of calibration input `i` (< `len()`).
    fn forward(&mut self, i: usize) -> Result<Vec<Tensor>>;
    /// Evaluate `outputs` and return them as host f32 (row-major).
    fn eval(&self, outputs: &[&Tensor]) -> Result<Vec<Vec<f32>>>;
}

/// Identity of a leaf: the device buffer (or host allocation) behind it.
fn leaf_key(t: &Tensor) -> Option<(u8, usize, usize)> {
    match t.op() {
        Op::Device(d) => Some((0, Arc::as_ptr(d) as *const () as usize, 0)),
        Op::Host(h) => {
            let (p, n) = h.key();
            Some((1, p, n))
        }
        _ => None,
    }
}

/// Inputs `x` of every `MatMul { b_transposed: true }` in the graph of `outputs` whose `b`
/// is one of `weights` (matched by leaf identity): `(weight name, x [..., in])`.
pub fn matmul_inputs(outputs: &[Tensor], weights: &[(String, Tensor)]) -> Vec<(String, Tensor)> {
    let by_key: HashMap<_, &str> = weights.iter().filter_map(|(n, t)| Some((leaf_key(t)?, n.as_str()))).collect();
    let mut seen = HashSet::new();
    let mut stack: Vec<&Tensor> = outputs.iter().collect();
    let mut found = Vec::new();
    while let Some(t) = stack.pop() {
        if !seen.insert(t.id()) {
            continue;
        }
        if let (Op::MatMul { b_transposed: true }, [a, b]) = (t.op(), t.src())
            && let Some(name) = leaf_key(b).and_then(|k| by_key.get(&k))
        {
            found.push((name.to_string(), a.clone()));
        }
        stack.extend(t.src());
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    found
}

/// Calibration settings.
#[derive(Clone, Debug)]
pub struct CalibrationOptions {
    pub gptq: GptqOptions,
    /// Host bytes of Gram matrices per pass (default 6 GiB).
    pub memory_budget: usize,
}

impl Default for CalibrationOptions {
    fn default() -> Self {
        CalibrationOptions { gptq: GptqOptions::default(), memory_budget: 6 << 30 }
    }
}

impl CalibrationOptions {
    /// Defaults overridden by the developer knobs `LAPID_GPTQ_ACT_ORDER=0|1`,
    /// `LAPID_GPTQ_DAMP=<fraction>`, `LAPID_GPTQ_SCALE_SEARCH=0|1`.
    pub fn from_env() -> CalibrationOptions {
        let mut o = CalibrationOptions::default();
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        if let Some(v) = var("LAPID_GPTQ_ACT_ORDER") {
            o.gptq.act_order = v != "0";
        }
        if let Some(v) = var("LAPID_GPTQ_SCALE_SEARCH") {
            o.gptq.scale_search = v != "0";
        }
        if let Some(d) = var("LAPID_GPTQ_DAMP").and_then(|v| v.parse().ok()) {
            o.gptq.damp = d;
        }
        o
    }
}

fn note(msg: &str) {
    let mut e = std::io::stderr().lock();
    let _ = writeln!(e, "calibrate: {msg}");
}

/// Accumulated `Σ x xᵀ` per weight (shared by weights reading the same input).
struct Grams {
    /// weight name -> index into `mats`.
    of: BTreeMap<String, usize>,
    mats: Vec<(usize, Vec<f32>)>,
    tokens: usize,
}

/// One pass: run every forward, accumulate Gram matrices of the inputs of `names`.
fn collect(cal: &mut dyn Calibrator, weights: &[(String, Tensor)], names: &HashSet<&str>) -> Result<Grams> {
    let mine: Vec<(String, Tensor)> = weights.iter().filter(|(n, _)| names.contains(n.as_str())).cloned().collect();
    let mut g = Grams { of: BTreeMap::new(), mats: Vec::new(), tokens: 0 };
    for i in 0..cal.len() {
        let outs = cal.forward(i)?;
        let found = matmul_inputs(&outs, &mine);
        // Distinct inputs (by node) and the weights reading each.
        let mut inputs: Vec<(Tensor, Vec<String>)> = Vec::new();
        for (name, x) in found {
            match inputs.iter_mut().find(|(t, _)| t.id() == x.id()) {
                Some((_, v)) => v.push(name),
                None => inputs.push((x, vec![name])),
            }
        }
        let mut grams = Vec::with_capacity(inputs.len());
        let mut rows = 0;
        for (x, _) in &inputs {
            let k = *x.shape().last().unwrap_or(&0);
            let n = x.numel() / k.max(1);
            rows = rows.max(n);
            let x2 = x.cast(DType::F32)?.reshape(&[n, k])?;
            grams.push(x2.transpose(0, 1)?.contiguous()?.matmul(&x2)?);
        }
        drop(outs);
        let refs: Vec<&Tensor> = grams.iter().collect();
        let host = cal.eval(&refs)?;
        for ((x, users), h) in inputs.iter().zip(host) {
            let k = *x.shape().last().unwrap_or(&0);
            let idx = match users.iter().find_map(|u| g.of.get(u)) {
                Some(&i) => i,
                None => {
                    g.mats.push((k, vec![0f32; k * k]));
                    g.mats.len() - 1
                }
            };
            for u in users {
                g.of.insert(u.clone(), idx);
            }
            for (a, b) in g.mats[idx].1.iter_mut().zip(&h) {
                *a += b;
            }
        }
        g.tokens += rows;
        if (i + 1) % 8 == 0 || i + 1 == cal.len() {
            note(&format!("{}/{} forward passes, {} tokens", i + 1, cal.len(), g.tokens));
        }
    }
    let missing: Vec<&str> = names.iter().copied().filter(|n| !g.of.contains_key(*n)).collect();
    if !missing.is_empty() {
        bail!("calibration", "no activations reached {} weights (e.g. {})", missing.len(), missing[0]);
    }
    Ok(g)
}

/// f32 values of a float tensor of the f32/f16/bf16 file.
fn read_f32(r: &Reader, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let t = r.tensor(name).ok_or_else(|| Error::new("missing_tensor", name.to_string()))?;
    let b = r.tensor_bytes(t);
    let v = match t.dtype {
        DType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        DType::F16 => b.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        DType::BF16 => b.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        q if q.is_quant() && t.shape.len() == 2 => {
            let f = q.quant_format().ok_or_else(|| Error::new("internal", "quant format"))?;
            lapid_quant::dequantize(f, b, t.shape[0], t.shape[1])
        }
        d => bail!("unsupported_dtype", "{name}: cannot read {} as f32", d.name()),
    };
    Ok((t.shape.clone(), v))
}

/// Encoded tensors by name: (dtype, bytes).
pub type Encoded = BTreeMap<String, (DType, Vec<u8>)>;

/// GPTQ int4 bytes for `names` (2-D weights of `float_file`), calibrated through `cal`.
pub fn calibrate_int4(
    cal: &mut dyn Calibrator,
    float_file: &Reader,
    names: &[String],
    opts: &CalibrationOptions,
) -> Result<(Encoded, Value)> {
    let start = Instant::now();
    let weights: Vec<(String, Tensor)> = cal.weights()?.into_iter().filter(|(n, _)| names.contains(n)).collect();
    if weights.len() != names.len() {
        bail!("calibration", "the model exposes {} of the {} weights to calibrate", weights.len(), names.len());
    }
    // Passes: consecutive weights while the (unshared) Gram bytes fit the budget.
    let mut passes: Vec<Vec<&str>> = vec![Vec::new()];
    let mut used = 0usize;
    for (n, t) in &weights {
        let k = *t.shape().last().unwrap_or(&0);
        let bytes = k * k * 4;
        if used + bytes > opts.memory_budget && !passes[passes.len() - 1].is_empty() {
            passes.push(Vec::new());
            used = 0;
        }
        used += bytes;
        let last = passes.len() - 1;
        passes[last].push(n);
    }
    let mut out = BTreeMap::new();
    let mut tokens = 0;
    let mut fallbacks = Vec::new();
    for (pi, pass) in passes.iter().enumerate() {
        note(&format!("pass {}/{}: capturing inputs of {} weights", pi + 1, passes.len(), pass.len()));
        let set: HashSet<&str> = pass.iter().copied().collect();
        let t0 = Instant::now();
        let grams = collect(cal, &weights, &set)?;
        tokens = grams.tokens;
        note(&format!("captured {} Gram matrices in {:.1} s; running GPTQ", grams.mats.len(), t0.elapsed().as_secs_f64()));
        for name in pass {
            let (shape, w) = read_f32(float_file, name)?;
            let [rows, cols] = shape[..] else { bail!("shape_mismatch", "{name}: calibrated weights are 2-D, got {shape:?}") };
            let (k, h) = &grams.mats[grams.of[*name]];
            if *k != cols {
                bail!("shape_mismatch", "{name}: input width {k} != weight cols {cols}");
            }
            let bytes = match crate::gptq::quantize_int4(&w, rows, cols, h, &opts.gptq) {
                Ok((q, s)) => lapid_quant::pack_int4(&q, &s, rows, cols),
                Err(e) => {
                    note(&format!("{name}: {} — round-to-nearest instead", e.message));
                    fallbacks.push(name.to_string());
                    lapid_quant::quantize(Format::Int4, &w, rows, cols)
                }
            };
            out.insert(name.to_string(), (DType::Int4, bytes));
        }
        note(&format!("pass {}/{} done ({:.1} s)", pi + 1, passes.len(), start.elapsed().as_secs_f64()));
    }
    let report = json!({
        "method": "gptq",
        "group_size": lapid_quant::GROUP,
        "damp": opts.gptq.damp,
        "scale_search": opts.gptq.scale_search,
        "act_order": opts.gptq.act_order,
        "calibration_tokens": tokens,
        "weights": out.len(),
        "rtn_fallbacks": fallbacks,
        "passes": passes.len(),
        "seconds": (start.elapsed().as_secs_f64() * 10.0).round() / 10.0,
    });
    Ok((out, report))
}

/// Rewrite the float `.lapid` file `float_file` as `out` at `dtype` through the dtype policy
/// (`role`), storing `prequantized` tensors as given; metadata and blobs are copied, with
/// `dtype` updated and `quantization` added.
pub fn requantize(
    float_file: &Reader,
    out: &Path,
    dtype: OutDType,
    role: RoleArc,
    prequantized: Encoded,
    quantization: Value,
) -> Result<()> {
    let mut meta = float_file.metadata.clone();
    meta["dtype"] = json!(dtype.name());
    meta["quantization"] = quantization;
    let mut w = ModelWriter::create_shared(out, meta, dtype, role, float_file.tensors.len())?;
    w.set_prequantized(prequantized);
    for b in float_file.blobs() {
        w.add_blob(&b.name, &b.media_type, float_file.blob_bytes(b))?;
    }
    let mut names = Vec::with_capacity(float_file.tensors.len());
    for t in &float_file.tensors {
        if w.is_prequantized(&t.name) {
            w.add_f32(&t.name, &t.shape, &[])?;
        } else if t.dtype.is_quant() || t.dtype == DType::I32 {
            w.add_raw(&t.name, t.dtype, &t.shape, float_file.tensor_bytes(t))?;
        } else {
            let (shape, v) = read_f32(float_file, &t.name)?;
            w.add_f32(&t.name, &shape, &v)?;
        }
        names.push(t.name.clone());
    }
    w.finish(&names)
}

/// Calibrated conversion: `float_file` (an f32 `.lapid` written by the model's converter,
/// which `cal` has loaded) → `out` at `dtype` (int4 only), every weight the policy stores
/// as int4 quantized with GPTQ on `cal`'s activations.
pub fn convert_calibrated(
    float_file: &Path,
    out: &Path,
    dtype: OutDType,
    role: RoleArc,
    cal: &mut dyn Calibrator,
    source: &CalibrationSource,
    opts: &CalibrationOptions,
) -> Result<()> {
    if dtype != OutDType::Int4 {
        bail!("invalid_argument", "--calibrate supports --dtype int4 only (got {})", dtype.name());
    }
    let r = Reader::open(float_file)?;
    let n_layer = None; // the int4 policy has no layer-dependent rules
    let names: Vec<String> = r
        .tensors
        .iter()
        .filter(|t| target_named(&t.name, role(&t.name, &t.shape), &t.shape, dtype, n_layer) == Target::Quant(Format::Int4))
        .map(|t| t.name.clone())
        .collect();
    let (pre, mut report) = calibrate_int4(cal, &r, &names, opts)?;
    report["calibration"] = json!(source.describe());
    requantize(&r, out, dtype, role, pre, report)
}

#[cfg(test)]
#[path = "calibrate_tests.rs"]
mod tests;
