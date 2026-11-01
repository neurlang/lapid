//! `lapid verify gemma4moe`: compare lapid's Gemma 4 (dense or MoE) with the HF transformers
//! reference (`tools/reference/gemma4moe_ref.py`, fp32 CPU; same layout as `gemma4_ref.py`).
//!
//! Checks, in order:
//! * chat template: every `template_cases` rendering byte-exact; tokenizer: every
//!   `tokenizer_cases` id list and the case prompts exact;
//! * `short` prompt, uncached forward: embedding, every layer's attention output and layer
//!   output, final norm and all logits (relative error per dtype tolerance, per-row argmax);
//! * cached decoding, teacher-forced on the reference tokens: the raw logits of the first
//!   greedy steps (`*.gen.logits`) — prefill + KV cache + conv state path;
//! * greedy continuation: token-exact for fp32 / fp16, reported for int8 / int4;
//! * `long` prompt (> sliding window): last rows of every layer, last logits, decode steps,
//!   greedy continuation;
//! * `inject` case (when recorded): random embedding rows injected over placeholder tokens
//!   (bidirectional image-block mask on sliding layers when the config asks for it): all
//!   logits, uncached and through the chunked cached prefill.
//!
//! Quantized models (int8 / int4) are checked for kernel correctness against the fake-quant
//! reference (same file, weights dequantized to f32, on the CPU; per tensor the closer of f32
//! and int8 activations); the PyTorch comparison is then informational (quality loss).

use std::path::{Path, PathBuf};

use lapid::{Device, Error, Result};
use lapid_convert::{OutDType, default_model_path};
use lapid_verify::timing::ms;
use lapid_verify::{Activations, Comparison, KERNEL_TOLERANCE, argmax_rows, tolerance};
use serde_json::{Value, json};

use super::model::Rows;
use crate::model::Model;
use super::model::Injection;
use super::{GenerateOptions, Gemma4Moe, Precision};

/// Greedy continuation must be token-exact for these dtypes.
pub fn tokens_must_match(dtype: &str) -> bool {
    matches!(dtype, "fp32" | "fp16")
}

/// The `.lapid` file for `model`: a `.lapid` path, a `.gguf` file or an HF repo / directory
/// (converted into the lapid cache on first use). Returns the path and conversion time.
pub(crate) fn resolve_model(model: &str, dtype: &str) -> Result<(PathBuf, Option<f64>)> {
    let p = Path::new(model);
    if p.extension().is_some_and(|e| e == "lapid") {
        return Ok((p.to_path_buf(), None));
    }
    let out_dtype = OutDType::parse(dtype)?;
    let is_gguf = p.is_file() && lapid_convert::gguf::is_gguf(p)?;
    let path = if is_gguf {
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        default_model_path(&format!("gguf/{stem}"), out_dtype)
    } else if p.is_dir() {
        let name = p.canonicalize().ok().and_then(|c| c.file_name().map(|n| n.to_string_lossy().into_owned()));
        default_model_path(&format!("local/{}", name.unwrap_or_else(|| "gemma4moe".into())), out_dtype)
    } else {
        default_model_path(model, out_dtype)
    };
    if path.is_file() {
        return Ok((path, None));
    }
    let t = std::time::Instant::now();
    if is_gguf {
        super::convert::convert_gguf(p, out_dtype, &path)?;
    } else {
        super::convert::convert_hf(model, out_dtype, &path)?;
    }
    Ok((path, Some(ms(t))))
}

/// PyTorch-comparison tolerance; the ggml k-quant dtypes of GGUF imports use the int4
/// (q4_k / q5_k / q4_k_m) or int8 (q6_k) budget.
fn quality_tolerance(dtype: &str) -> Result<f32> {
    match dtype {
        "q4_k" | "q4_k_m" | "q5_k" => tolerance("int4"),
        "q6_k" => tolerance("int8"),
        other => tolerance(other),
    }
}

pub(crate) fn file_dtype(path: &Path) -> Result<String> {
    let r = lapid::format::Reader::open(path)?;
    r.metadata
        .get("dtype")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| Error::new("missing_metadata", format!("{}: no \"dtype\"", path.display())))
}

fn ids(v: &Value, key: &str) -> Result<Vec<u32>> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("reference", format!("result.json: missing array {key:?}")))?
        .iter()
        .map(|x| x.as_u64().map(|n| n as u32).ok_or_else(|| Error::new("reference", format!("{key:?}: non-integer id"))))
        .collect()
}

fn str_of<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key).and_then(Value::as_str).ok_or_else(|| Error::new("reference", format!("result.json: missing string {key:?}")))
}

/// Template and tokenizer checks. Returns (ok, details).
fn check_text(llm: &Gemma4Moe, r: &Value) -> Result<(bool, Value)> {
    let mut ok = true;
    let mut templates = Vec::new();
    for (i, c) in r.get("template_cases").and_then(Value::as_array).into_iter().flatten().enumerate() {
        let tools = c.get("tools_json").and_then(Value::as_str);
        let add_gen = c.get("add_generation_prompt").and_then(Value::as_bool).unwrap_or(false);
        let thinking = c.get("enable_thinking").and_then(Value::as_bool);
        let want = str_of(c, "text")?;
        let got = llm.render_chat_json(str_of(c, "messages_json")?, tools, add_gen, thinking);
        let m = got.as_deref().is_ok_and(|g| g == want);
        ok &= m;
        let mut e = json!({"case": i, "match": m});
        if !m {
            e["got"] = match got {
                Ok(g) => json!(g),
                Err(err) => json!(format!("error: {err}")),
            };
            e["want"] = json!(want);
        }
        templates.push(e);
    }
    let mut tokenizer = Vec::new();
    for c in r.get("tokenizer_cases").and_then(Value::as_array).into_iter().flatten() {
        let want = ids(c, "ids")?;
        let got = llm.encode(str_of(c, "text")?)?;
        let m = got == want;
        ok &= m;
        tokenizer.push(if m { json!({"match": true}) } else { json!({"match": false, "got": got, "want": want}) });
    }
    let mut prompts = serde_json::Map::new();
    for (name, c) in r.get("cases").and_then(Value::as_object).into_iter().flatten() {
        let thinking = c.get("enable_thinking").and_then(Value::as_bool);
        let text = llm.render_chat_json(str_of(c, "messages_json")?, None, true, thinking)?;
        let text_ok = text == str_of(c, "prompt_text")?;
        let ids_ok = llm.encode(&text)? == ids(c, "prompt_tokens")?;
        ok &= text_ok && ids_ok;
        prompts.insert(name.clone(), json!({"template": text_ok, "tokens": ids_ok}));
    }
    Ok((ok, json!({"template_cases": templates, "tokenizer_cases": tokenizer, "prompts": prompts})))
}

/// Teacher-forced cached decoding: logits after the prompt, then after each reference token.
fn decode_logits(llm: &Gemma4Moe, prompt: &[u32], forced: &[u32], steps: usize) -> Result<Vec<f32>> {
    let mut s = llm.new_session();
    let mut out = llm.step(&mut s, prompt)?;
    for &t in forced.iter().take(steps.saturating_sub(1)) {
        out.extend(llm.step(&mut s, &[t])?);
    }
    Ok(out)
}

fn greedy(llm: &Gemma4Moe, prompt: &[u32], n: usize) -> Result<Vec<u32>> {
    let mut s = llm.new_session();
    let opts = GenerateOptions { max_new_tokens: n.max(1), ..Default::default() };
    let mut stream = llm.generate(&mut s, prompt, opts)?;
    for t in stream.by_ref() {
        t?;
    }
    Ok(stream.generated().to_vec())
}

/// Rows `[n - tail, n)` of a row-major [n, d] buffer.
fn tail_rows(v: &[f32], d: usize, tail: usize) -> Vec<f32> {
    let n = v.len() / d.max(1);
    v[(n.saturating_sub(tail)) * d..].to_vec()
}

struct Outputs {
    short: Vec<(String, Vec<f32>)>,
    long: Vec<(String, Vec<f32>)>,
}

fn run_traced(llm: &Gemma4Moe, short: &[u32], long: &[u32], long_tail: usize) -> Result<Outputs> {
    let dev = llm.device();
    let d = llm.config().text.hidden_size;
    let mut s = llm.model().forward_traced_with(dev, short, &[], Rows::All)?;
    for (n, _) in s.iter_mut() {
        *n = format!("short.{n}");
    }
    let mut l = Vec::new();
    if !long.is_empty() {
        for (n, v) in llm.model().forward_traced_with(dev, long, &[], Rows::Last)? {
            if n == "logits" {
                l.push(("long.logits.last".to_string(), v));
            } else if n.starts_with("layers.") && !n.ends_with(".attn") {
                l.push((format!("long.{n}.tail"), tail_rows(&v, d, long_tail)));
            }
        }
    }
    Ok(Outputs { short: s, long: l })
}

fn compare(ours: &Outputs, acts: &Activations, names: &[String], tol: f32) -> Result<Comparison> {
    let mut c = Comparison::new();
    let all: Vec<(String, Vec<f32>)> = ours.short.iter().chain(&ours.long).cloned().collect();
    c.compare_named(names, &all, |n| acts.get(n), tol)?;
    Ok(c)
}

/// The `inject` case: `inject.embeds` [n, D] replaces `prompt_tokens[start..start + n]`;
/// compares the final norm of every row and the last logits (uncached forward), and the last
/// logits of the chunked cached prefill.
fn check_inject(llm: &Gemma4Moe, c: &Value, acts: &Activations, tol: f32) -> Result<(Comparison, Value)> {
    let prompt = ids(c, "prompt_tokens")?;
    let start = c.get("start").and_then(Value::as_u64).ok_or_else(|| Error::new("reference", "inject: no start"))? as usize;
    let shape = acts.shape("inject.embeds")?;
    let embeds = lapid::Tensor::from_f32(&acts.get("inject.embeds")?, &shape)?;
    let inject = [Injection { start, embeds }];
    let dev = llm.device();
    let traced = llm.model().forward_traced_with(dev, &prompt, &inject, Rows::Last)?;
    let get = |k: &str| traced.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()).unwrap_or_default();
    let mut cmp = Comparison::new();
    let want_last = acts.get("inject.logits.last")?;
    cmp.compare("inject.logits.last", &get("logits"), &want_last, tol);
    let mut s = llm.new_session();
    let last = llm.step_with(&mut s, &prompt, &inject)?;
    cmp.compare("inject.prefill.last", &last, &want_last, tol);
    let vocab = llm.config().text.vocab_size;
    let argmax = argmax_rows(&last, &want_last, vocab);
    let mut all = llm.model().forward_traced_with(dev, &prompt, &inject, Rows::All)?;
    if let Some((_, norm)) = all.iter_mut().find(|(n, _)| n == "norm") {
        cmp.compare("inject.norm", norm, &acts.get("inject.norm")?, tol);
    }
    Ok((cmp, json!({"argmax_match": argmax, "bidirectional": llm.config().bidirectional_vision})))
}

/// Full verification report as JSON; check `"ok"`.
pub fn verify(reference_dir: &Path, model: &str, dtype: &str, device: Option<&str>) -> Result<Value> {
    let raw = std::fs::read_to_string(reference_dir.join("result.json")).map_err(|e| {
        Error::new("io", format!("{}/result.json: {e} (run tools/reference/gemma4moe_ref.py)", reference_dir.display()))
    })?;
    let r: Value = serde_json::from_str(&raw)?;
    let acts = Activations::open(&reference_dir.join("activations.safetensors"))?;
    let dev = lapid_verify::open_device(device)?;
    let mut stages = lapid_verify::Stages::new();
    let (path, convert_ms) = resolve_model(model, dtype)?;
    if let Some(t) = convert_ms {
        stages.record("convert", t);
    }
    let dtype = file_dtype(&path)?;
    let quantized = !matches!(dtype.as_str(), "fp32" | "fp16");
    let quality_tol = quality_tolerance(&dtype)?;
    let llm = stages.time("load", || Gemma4Moe::load(&path, &dev))?;

    let (text_ok, text) = stages.time("text", || check_text(&llm, &r))?;

    let case = |name: &str| r.pointer(&format!("/cases/{name}")).cloned().unwrap_or(Value::Null);
    let (short, long) = (case("short"), case("long"));
    let short_prompt = ids(&short, "prompt_tokens")?;
    let long_prompt = if long.is_null() { Vec::new() } else { ids(&long, "prompt_tokens")? };
    let long_tail = acts.shape("long.layers.0.tail").ok().and_then(|s| s.first().copied()).unwrap_or(0);

    // ---- layer by layer vs PyTorch
    let mut names = vec!["short.embed".to_string()];
    for i in 0..llm.config().text.n_layers {
        names.push(format!("short.layers.{i}.attn"));
        names.push(format!("short.layers.{i}"));
    }
    names.extend(["short.norm".to_string(), "short.logits".to_string()]);
    if !long_prompt.is_empty() {
        names.extend((0..llm.config().text.n_layers).map(|i| format!("long.layers.{i}.tail")));
        names.push("long.logits.last".to_string());
    }
    let ours = stages.time("forward", || run_traced(&llm, &short_prompt, &long_prompt, long_tail))?;
    let quality = compare(&ours, &acts, &names, quality_tol)?;
    let vocab = llm.config().text.vocab_size;
    let logits_of = |o: &Outputs, n: &str| o.short.iter().chain(&o.long).find(|(k, _)| k == n).map(|(_, v)| v.clone()).unwrap_or_default();
    let mut argmax = argmax_rows(&logits_of(&ours, "short.logits"), &acts.get("short.logits")?, vocab);
    if !long_prompt.is_empty() {
        argmax.extend(argmax_rows(&logits_of(&ours, "long.logits.last"), &acts.get("long.logits.last")?, vocab));
    }

    // ---- cached decode (teacher forced) + greedy continuation
    let mut decode = Comparison::new();
    let mut decode_argmax = Vec::new();
    let mut greedy_json = serde_json::Map::new();
    let mut tokens_match = true;
    for (name, c, prompt) in [("short", &short, &short_prompt), ("long", &long, &long_prompt)] {
        if prompt.is_empty() {
            continue;
        }
        let want = ids(c, "generated_tokens")?;
        let key = format!("{name}.gen.logits");
        let steps = acts.shape(&key)?.first().copied().unwrap_or(0);
        let got = stages.time("decode", || decode_logits(&llm, prompt, &want, steps))?;
        let reference = acts.get(&key)?;
        decode.compare(&key, &got, &reference, quality_tol);
        decode_argmax.extend(argmax_rows(&got, &reference, vocab));
        let t = std::time::Instant::now();
        let ours_tokens = greedy(&llm, prompt, want.len())?;
        stages.record("greedy", ms(t));
        let m = ours_tokens == want;
        tokens_match &= m;
        let first_diff = ours_tokens.iter().zip(&want).position(|(a, b)| a != b);
        greedy_json.insert(
            name.to_string(),
            json!({"match": m, "first_mismatch": first_diff, "tokens": ours_tokens, "reference_tokens": want,
                   "text": llm.decode(&ours_tokens, false), "reference_text": c.get("generated_text")}),
        );
    }

    // ---- injected embeddings (multimodal path)
    let inject_json = match r.get("inject") {
        Some(c) => {
            let (cmp, j) = stages.time("inject", || check_inject(&llm, c, &acts, quality_tol))?;
            decode.layers.extend(cmp.layers);
            j
        }
        None => Value::Null,
    };

    // ---- kernel correctness vs fake-quant (quantized only)
    let mut kernel_json = Value::Null;
    let mut kernel: Option<Comparison> = None;
    let mut kernel_unavailable = false;
    let fake = if quantized {
        let cpu = Device::cpu();
        let r = stages.time("fake_quant", || -> Result<_> {
            let w = Gemma4Moe::load_with(&path, &cpu, Precision::FakeQuantWeights)?;
            let ow = run_traced(&w, &short_prompt, &[], 0)?;
            drop(w);
            let a = Gemma4Moe::load_with(&path, &cpu, Precision::FakeQuantActivations)?;
            let oa = run_traced(&a, &short_prompt, &[], 0)?;
            Ok((ow, oa))
        });
        match r {
            Ok(x) => Some(x),
            // e.g. lapid-nn's fake-quant loader cannot dequantize k-quant weights yet.
            Err(e) => {
                kernel_unavailable = true;
                kernel_json = json!({"unavailable": format!("{}: {}", e.code, e.message)});
                None
            }
        }
    } else {
        None
    };
    if let Some((fw, fa)) = fake {
        let short_names: Vec<String> = names.iter().filter(|n| n.starts_with("short.")).cloned().collect();
        let mut merged = Comparison::new();
        let mut which = serde_json::Map::new();
        for n in &short_names {
            let o = logits_of(&ours, n);
            let (w, a) = (logits_of(&fw, n), logits_of(&fa, n));
            let rw = lapid_verify::diff(&o, &w)?;
            let ra = lapid_verify::diff(&o, &a)?;
            let use_a = ra.rel < rw.rel || (rw.rel.is_nan() && !ra.rel.is_nan());
            merged.compare(n, &o, if use_a { &a } else { &w }, KERNEL_TOLERANCE);
            which.insert(n.clone(), json!(if use_a { "int8_activations" } else { "f32_activations" }));
        }
        let ka = argmax_rows(&logits_of(&ours, "short.logits"), &logits_of(&fw, "short.logits"), vocab);
        kernel_json = json!({
            "reference": "fake-quant: same weights dequantized to f32 on cpu; per tensor the closer of f32 and int8-per-32-block activations",
            "tolerance": KERNEL_TOLERANCE,
            "matched_reference": which,
            "logits_argmax_match_f32_activations": ka,
            "comparison": merged.to_json(),
        });
        kernel = Some(merged);
    }

    let strict = tokens_must_match(&dtype);
    let all_true = |v: &[bool]| v.iter().all(|&b| b);
    let correctness_ok = match &kernel {
        Some(k) => k.ok(),
        // No fake-quant reference: the quality budget alone (argmax flips are quantization loss).
        None if kernel_unavailable => quality.ok() && decode.ok(),
        None => quality.ok() && decode.ok() && all_true(&argmax) && all_true(&decode_argmax),
    };
    let ok = text_ok && correctness_ok && (!strict || tokens_match);
    Ok(json!({
        "ok": ok,
        "device": dev.description(),
        "model": model,
        "model_file": path.display().to_string(),
        "dtype": dtype,
        "correctness_reference": if kernel_unavailable { "pytorch-fp32 (fake-quant unavailable)" } else if quantized { "fake-quant" } else { "pytorch-fp32" },
        "text_ok": text_ok,
        "tokens_match": tokens_match,
        "tokens_must_match": strict,
        "first_failing_layer": kernel.as_ref().unwrap_or(&quality).first_failing(),
        "layers": quality.layers,
        "quality": {
            "reference": "PyTorch fp32 (HF Gemma4 text decoder)",
            "informational": quantized,
            "tolerance": quality_tol,
            "within_tolerance": quality.ok(),
            "max_rel": quality.max_rel(),
            "logits_argmax_match": argmax,
            "decode": decode.to_json(),
            "decode_argmax_match": decode_argmax,
        },
        "kernel": kernel_json,
        "greedy": greedy_json,
        "inject": inject_json,
        "text": text,
        "reference_dir": reference_dir.display().to_string(),
        "timings_ms": stages.to_json(),
        "notes": quality.notes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strictness() {
        assert!(tokens_must_match("fp16"));
        assert!(!tokens_must_match("int8"));
    }

    #[test]
    fn tails() {
        assert_eq!(tail_rows(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 2), vec![3.0, 4.0, 5.0, 6.0]);
    }
}
