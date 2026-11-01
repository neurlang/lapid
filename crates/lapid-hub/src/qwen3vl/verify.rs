//! `lapid verify qwen3vl`: compare lapid's Qwen3-VL with the HF transformers reference
//! (`tools/reference/qwen3vl_ref.py`, fp32 CPU).
//!
//! Checks: chat template / tokenizer cases; prompt expansion (image placeholders) and M-RoPE
//! positions exact; preprocessing of the reference picture vs HF `pixel_values` (resampling
//! differs by at most a few 8-bit steps); the vision encoder on the reference pixels (every
//! block, merged tokens, DeepStack features); the text decoder with the image injected
//! (embeddings, every layer, final norm, logits, per-row argmax); teacher-forced cached decoding
//! (`*.gen.logits`); greedy continuation (exact for fp32 / fp16). Quantized files are checked
//! against the fake-quant reference (same file dequantized, cpu); PyTorch is then informational.

use std::path::{Path, PathBuf};

use lapid::{Device, Error, Result};
use lapid_convert::{OutDType, default_model_path};
use lapid_verify::timing::ms;
use lapid_verify::{Activations, Comparison, KERNEL_TOLERANCE, argmax_rows, tolerance};
use serde_json::{Value, json};

use super::model::{ImageSpan, Pos3, Rows};
use super::{GenerateOptions, Precision, Prepared, Qwen3Vl};
use crate::model::Model;
use crate::vision::{Image, Patches, VisionOutput};

/// The `.lapid` for `model`: a `.lapid`, a `.gguf` (+ sibling mmproj) or an HF repo / dir.
pub fn resolve_model(model: &str, dtype: &str) -> Result<(PathBuf, Option<f64>)> {
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
        // HF cache snapshots (`models--org--name/snapshots/<hash>`) are named after the repo.
        let c = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
        let name = c
            .ancestors()
            .filter_map(|a| a.file_name()?.to_str())
            .find_map(|n| n.strip_prefix("models--").map(|r| r.replace("--", "-")))
            .or_else(|| c.file_name().and_then(|n| n.to_str()).map(str::to_string))
            .unwrap_or_else(|| "qwen3vl".into());
        default_model_path(&format!("local/{name}"), out_dtype)
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

fn file_dtype(path: &Path) -> Result<String> {
    let r = lapid::format::Reader::open(path)?;
    r.metadata.get("dtype").and_then(Value::as_str).map(str::to_string).ok_or_else(|| Error::new("missing_metadata", "no \"dtype\""))
}

fn ids(v: &Value, key: &str) -> Result<Vec<u32>> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("reference", format!("result.json: missing array {key:?}")))?
        .iter()
        .map(|x| x.as_u64().map(|n| n as u32).ok_or_else(|| Error::new("reference", format!("{key:?}: non-integer"))))
        .collect()
}

fn str_of<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key).and_then(Value::as_str).ok_or_else(|| Error::new("reference", format!("result.json: missing string {key:?}")))
}

fn check_text(vlm: &Qwen3Vl, r: &Value) -> Result<(bool, Value)> {
    let mut ok = true;
    let mut templates = Vec::new();
    for (i, c) in r.get("template_cases").and_then(Value::as_array).into_iter().flatten().enumerate() {
        let add_gen = c.get("add_generation_prompt").and_then(Value::as_bool).unwrap_or(false);
        let want = str_of(c, "text")?;
        let got = vlm.render_chat_json(str_of(c, "messages_json")?, None, add_gen);
        let m = got.as_deref().is_ok_and(|g| g == want);
        ok &= m;
        let mut e = json!({"case": i, "match": m});
        if !m {
            e["got"] = json!(got.unwrap_or_else(|err| format!("error: {err}")));
            e["want"] = json!(want);
        }
        templates.push(e);
    }
    let mut tokenizer = Vec::new();
    for c in r.get("tokenizer_cases").and_then(Value::as_array).into_iter().flatten() {
        let (want, got) = (ids(c, "ids")?, vlm.encode(str_of(c, "text")?)?);
        ok &= got == want;
        tokenizer.push(if got == want { json!({"match": true}) } else { json!({"match": false, "got": got, "want": want}) });
    }
    Ok((ok, json!({"template_cases": templates, "tokenizer_cases": tokenizer})))
}

/// One reference case with its prepared prompt (from the reference pixels).
struct Case {
    name: String,
    prepared: Prepared,
    reference: Value,
}

fn ref_patches(acts: &Activations, case: &str, c: &Value) -> Result<Option<Patches>> {
    let Some(grid) = c.get("grid_thw").and_then(Value::as_array) else { return Ok(None) };
    let g = |i: usize| grid.get(i).and_then(Value::as_u64).unwrap_or(0) as usize;
    let data = acts.get(&format!("{case}.pixel_values"))?;
    let (gh, gw) = (g(1), g(2));
    let dim = data.len() / (gh * gw).max(1);
    Ok(Some(Patches { data, grid_h: gh, grid_w: gw, dim }))
}

/// Our preprocessing of the reference picture vs HF pixel values.
fn check_preprocess(vlm: &Qwen3Vl, dir: &Path, want: &Patches) -> Result<(bool, Value)> {
    let img = Image::open(&dir.join("test.png"))?;
    let got = vlm.preprocess(&img, None)?;
    if (got.grid_h, got.grid_w) != (want.grid_h, want.grid_w) || got.data.len() != want.data.len() {
        return Ok((false, json!({"grid": [got.grid_h, got.grid_w], "reference_grid": [want.grid_h, want.grid_w]})));
    }
    let d = lapid_verify::diff(&got.data, &want.data)?;
    // One 8-bit step after (x / 255 - 0.5) / 0.5 is 0.0078; allow a few, on few pixels.
    let ok = d.max_abs <= 0.04 && d.mean_abs <= 2e-3;
    Ok((ok, json!({"ok": ok, "max_abs": d.max_abs, "mean_abs": d.mean_abs, "grid": [got.grid_h, got.grid_w]})))
}

/// Vision + text traces of a case: names prefixed `{case}.`.
type Trace = Vec<(String, Vec<f32>)>;

fn traced(vlm: &Qwen3Vl, c: &Case) -> Result<(Trace, Vec<VisionOutput>)> {
    let dev = vlm.device();
    let mut out = Vec::new();
    let mut vis = Vec::new();
    for im in &c.prepared.images {
        for (n, v) in vlm.vision().forward_traced(dev, &im.patches)? {
            out.push((format!("{}.{n}", c.name), v));
        }
        vis.push(vlm.encode_patches(&im.patches)?);
    }
    let spans: Vec<ImageSpan<'_>> = c.prepared.images.iter().zip(&vis).map(|(im, o)| ImageSpan { start: im.start, out: o }).collect();
    for (n, v) in vlm.text_model().forward_traced(dev, &c.prepared.ids, &c.prepared.positions, &spans, Rows::All)? {
        out.push((format!("{}.{n}", c.name), v));
    }
    Ok((out, vis))
}

/// Teacher-forced cached decoding: logits after the prompt, then after each forced token.
fn decode_logits(vlm: &Qwen3Vl, c: &Case, vis: &[VisionOutput], forced: &[u32], steps: usize) -> Result<Vec<f32>> {
    let m = vlm.text_model();
    let dev = vlm.device();
    let mut st = m.new_state();
    let spans: Vec<ImageSpan<'_>> = c.prepared.images.iter().zip(vis).map(|(im, o)| ImageSpan { start: im.start, out: o }).collect();
    let mut out = m.step(dev, &c.prepared.ids, &c.prepared.positions, &spans, &mut st)?;
    for &t in forced.iter().take(steps.saturating_sub(1)) {
        let p: Pos3 = [st.next_pos(); 3];
        out.extend(m.step(dev, &[t], &[p], &[], &mut st)?);
    }
    Ok(out)
}

fn greedy(vlm: &Qwen3Vl, c: &Case, n: usize) -> Result<Vec<u32>> {
    let mut s = vlm.new_session();
    let p = Prepared {
        ids: c.prepared.ids.clone(),
        positions: c.prepared.positions.clone(),
        images: c.prepared.images.iter().map(|im| super::prompt::PreparedImage { start: im.start, tokens: im.tokens, patches: im.patches.clone(), hash: im.hash }).collect(),
    };
    let mut stream = vlm.generate_prepared(&mut s, p, GenerateOptions { max_new_tokens: n.max(1), ..Default::default() })?;
    for t in stream.by_ref() {
        t?;
    }
    Ok(stream.generated().to_vec())
}

fn names_of(vlm: &Qwen3Vl, case: &str, image: bool) -> Vec<String> {
    let mut v = Vec::new();
    if image {
        v.extend((0..vlm.config().vision.depth).map(|i| format!("{case}.vision.blocks.{i}")));
        v.push(format!("{case}.vision.merged"));
        v.extend((0..vlm.config().vision.deepstack_layers.len()).map(|j| format!("{case}.vision.deepstack.{j}")));
    }
    v.push(format!("{case}.embeds"));
    v.extend((0..vlm.config().text.n_layers).map(|i| format!("{case}.layers.{i}")));
    v.extend([format!("{case}.norm"), format!("{case}.logits")]);
    v
}

/// Full verification report; check `"ok"`.
pub fn verify(reference_dir: &Path, model: &str, dtype: &str, device: Option<&str>) -> Result<Value> {
    let raw = std::fs::read_to_string(reference_dir.join("result.json"))
        .map_err(|e| Error::new("io", format!("{}/result.json: {e} (run tools/reference/qwen3vl_ref.py)", reference_dir.display())))?;
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
    let quality_tol = tolerance(&dtype).or_else(|_| tolerance("int4"))?;
    let vlm = stages.time("load", || Qwen3Vl::load(&path, &dev))?;
    let (mut text_ok, text) = stages.time("text", || check_text(&vlm, &r))?;

    let mut cases = Vec::new();
    let mut prompts = serde_json::Map::new();
    let mut pre_json = Value::Null;
    let mut pre_ok = true;
    for (name, c) in r.get("cases").and_then(Value::as_object).into_iter().flatten() {
        let rendered = vlm.render_chat_json(str_of(c, "messages_json")?, None, true)?;
        let template_ok = rendered == str_of(c, "prompt_text")?;
        let patches = ref_patches(&acts, name, c)?;
        if let Some(p) = &patches {
            let (ok, j) = check_preprocess(&vlm, reference_dir, p)?;
            pre_ok &= ok;
            pre_json = j;
        }
        let imgs: Vec<(Patches, u64)> = patches.into_iter().map(|p| (p, 0)).collect();
        let prepared = Prepared::new(vlm.config(), &vlm.encode(&rendered)?, imgs)?;
        let want_ids = ids(c, "prompt_tokens")?;
        let ids_ok = prepared.ids == want_ids;
        let pos_ok = match c.get("positions").and_then(Value::as_array) {
            Some(rows) => {
                let row = |k: usize| rows.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect::<Vec<_>>());
                let (t, h, w) = (row(0).unwrap_or_default(), row(1).unwrap_or_default(), row(2).unwrap_or_default());
                prepared.positions.len() == t.len() && prepared.positions.iter().enumerate().all(|(i, p)| p == &[t[i], h[i], w[i]])
            }
            None => true,
        };
        text_ok &= template_ok && ids_ok && pos_ok;
        prompts.insert(name.clone(), json!({"template": template_ok, "tokens": ids_ok, "positions": pos_ok}));
        cases.push(Case { name: name.clone(), prepared, reference: c.clone() });
    }

    // ---- layer by layer vs PyTorch
    let mut quality = Comparison::new();
    let mut argmax = Vec::new();
    let mut ours_all = Vec::new();
    let mut vis_all = Vec::new();
    let vocab = vlm.config().text.vocab_size;
    for c in &cases {
        let (ours, vis) = stages.time("forward", || traced(&vlm, c))?;
        let names = names_of(&vlm, &c.name, !c.prepared.images.is_empty());
        quality.compare_named(&names, &ours, |n| acts.get(n), quality_tol)?;
        let lg = ours.iter().find(|(n, _)| *n == format!("{}.logits", c.name)).map(|(_, v)| v.clone()).unwrap_or_default();
        argmax.extend(argmax_rows(&lg, &acts.get(&format!("{}.logits", c.name))?, vocab));
        ours_all.push(ours);
        vis_all.push(vis);
    }

    // ---- cached decode (teacher forced) + greedy
    let mut decode = Comparison::new();
    let mut decode_argmax = Vec::new();
    let mut greedy_json = serde_json::Map::new();
    let mut tokens_match = true;
    for (c, vis) in cases.iter().zip(&vis_all) {
        let want = ids(&c.reference, "generated_tokens")?;
        let key = format!("{}.gen.logits", c.name);
        let steps = acts.shape(&key)?.first().copied().unwrap_or(0);
        let got = stages.time("decode", || decode_logits(&vlm, c, vis, &want, steps))?;
        let reference = acts.get(&key)?;
        decode.compare(&key, &got, &reference, quality_tol);
        decode_argmax.extend(argmax_rows(&got, &reference, vocab));
        let ours_tokens = stages.time("greedy", || greedy(&vlm, c, want.len()))?;
        let m = ours_tokens == want;
        tokens_match &= m;
        greedy_json.insert(
            c.name.clone(),
            json!({"match": m, "first_mismatch": ours_tokens.iter().zip(&want).position(|(a, b)| a != b),
                   "text": vlm.decode(&ours_tokens, false), "reference_text": c.reference.get("generated_text")}),
        );
    }

    // ---- kernels vs fake-quant (quantized files)
    let mut kernel_json = Value::Null;
    let mut kernel: Option<Comparison> = None;
    if quantized {
        let cpu = Device::cpu();
        let (fw, fa) = stages.time("fake_quant", || -> Result<_> {
            let run = |p: Precision| -> Result<Vec<(String, Vec<f32>)>> {
                let m = Qwen3Vl::load_with(&path, &cpu, p)?;
                let mut v = Vec::new();
                for c in &cases {
                    v.extend(traced(&m, c)?.0);
                }
                Ok(v)
            };
            Ok((run(Precision::FakeQuantWeights)?, run(Precision::FakeQuantActivations)?))
        })?;
        let mut merged = Comparison::new();
        let get = |v: &[(String, Vec<f32>)], n: &str| v.iter().find(|(k, _)| k == n).map(|(_, x)| x.clone()).unwrap_or_default();
        for (o, c) in ours_all.iter().zip(&cases) {
            for n in names_of(&vlm, &c.name, !c.prepared.images.is_empty()) {
                let ours = get(o, &n);
                let (w, a) = (get(&fw, &n), get(&fa, &n));
                let (rw, ra) = (lapid_verify::diff(&ours, &w)?, lapid_verify::diff(&ours, &a)?);
                merged.compare(&n, &ours, if ra.rel < rw.rel { &a } else { &w }, KERNEL_TOLERANCE);
            }
        }
        kernel_json = json!({"reference": "fake-quant (cpu), closer of f32 / int8 activations", "tolerance": KERNEL_TOLERANCE, "comparison": merged.to_json()});
        kernel = Some(merged);
    }

    let strict = matches!(dtype.as_str(), "fp32" | "fp16");
    let all = |v: &[bool]| v.iter().all(|&b| b);
    let correctness_ok = match &kernel {
        Some(k) => k.ok(),
        None => quality.ok() && decode.ok() && all(&argmax) && all(&decode_argmax),
    };
    let ok = text_ok && pre_ok && correctness_ok && (!strict || tokens_match);
    Ok(json!({
        "ok": ok,
        "device": dev.description(),
        "model": model,
        "model_file": path.display().to_string(),
        "dtype": dtype,
        "correctness_reference": if quantized { "fake-quant" } else { "pytorch-fp32" },
        "text_ok": text_ok,
        "preprocess": pre_json,
        "tokens_match": tokens_match,
        "first_failing_layer": kernel.as_ref().unwrap_or(&quality).first_failing(),
        "quality": {"informational": quantized, "tolerance": quality_tol, "max_rel": quality.max_rel(), "within_tolerance": quality.ok(),
                    "layers": quality.to_json(), "logits_argmax_match": all(&argmax), "decode": decode.to_json(), "decode_argmax_match": all(&decode_argmax)},
        "kernel": kernel_json,
        "greedy": greedy_json,
        "text": text,
        "prompts": prompts,
        "timings_ms": stages.to_json(),
    }))
}
