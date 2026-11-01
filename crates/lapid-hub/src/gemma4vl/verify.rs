//! `lapid verify gemma4vl`: compare lapid's Gemma 4 vision path with the HF transformers
//! reference (`tools/reference/gemma4vl_ref.py`, fp32 CPU).
//!
//! Checks, in order:
//! * prompt: the chat template (one image part + text) and the expanded token ids exact;
//! * preprocessing: our resize / patchify of `image.png` vs HF `pixel_values` (same grid,
//!   max abs difference within 1e-6: the resize is torchvision-exact);
//! * vision tower on the reference pixels: patch embedding, every encoder layer, pooled and
//!   projected soft tokens;
//! * text decoder with the reference soft tokens injected: merged embedding, every layer's attention
//!   and output, last-row logits; teacher-forced cached decode logits;
//! * greedy continuation end to end (our preprocessing + vision + decoder): token-exact for
//!   fp32 / fp16, reported for quantized files.

use std::path::{Path, PathBuf};

use lapid::{Error, Result};
use lapid_convert::{OutDType, default_model_path};
use lapid_verify::timing::ms;
use lapid_verify::{Activations, Comparison, argmax_rows, tolerance};
use serde_json::{Value, json};

use super::{Gemma4Vl, GenerateOptions, PreparedImage, Prompt};
use crate::gemma4moe::Injection;
use crate::gemma4moe::model::Rows;
use crate::model::Model;
use crate::vision::{Image, Message};

pub fn tokens_must_match(dtype: &str) -> bool {
    matches!(dtype, "fp32" | "fp16")
}

/// The `.lapid` file for `model`: a `.lapid` path, a `.gguf` (paired with the `mmproj*.gguf`
/// next to it) or an HF repo / directory, converted with the vision tower into the lapid cache.
pub(crate) fn resolve_model(model: &str, dtype: &str) -> Result<(PathBuf, Option<f64>)> {
    let p = Path::new(model);
    if p.extension().is_some_and(|e| e == "lapid") {
        return Ok((p.to_path_buf(), None));
    }
    let out_dtype = OutDType::parse(dtype)?;
    let is_gguf = p.is_file() && lapid_convert::gguf::is_gguf(p)?;
    let path = if is_gguf {
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        default_model_path(&format!("gguf/{stem}-vision"), out_dtype)
    } else if p.is_dir() {
        let name = p.canonicalize().ok().and_then(|c| c.parent().and_then(|d| d.file_name()).map(|n| n.to_string_lossy().into_owned()));
        default_model_path(&format!("local/gemma4vl-{}", name.unwrap_or_else(|| "model".into())), out_dtype)
    } else {
        default_model_path(&format!("{}-vision", model.trim_end_matches('/')), out_dtype)
    };
    if path.is_file() {
        return Ok((path, None));
    }
    let t = std::time::Instant::now();
    if is_gguf {
        crate::gemma4moe::convert::convert_gguf(p, out_dtype, &path)?;
    } else {
        crate::gemma4moe::convert::convert_hf(model, out_dtype, &path)?;
    }
    Ok((path, Some(ms(t))))
}

/// [`resolve_model`] without the timing: the `.lapid` path for a model argument.
pub fn resolve(model: &str, dtype: &str) -> Result<PathBuf> {
    Ok(resolve_model(model, dtype)?.0)
}

fn quality_tolerance(dtype: &str) -> Result<f32> {
    match dtype {
        "q4_k" | "q4_k_m" | "q5_k" => tolerance("int4"),
        "q6_k" => tolerance("int8"),
        other => tolerance(other),
    }
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

/// The reference pixels as a prepared image.
fn reference_image(acts: &Activations, vlm: &Gemma4Vl) -> Result<PreparedImage> {
    let pos = acts.get("positions")?;
    let positions: Vec<[usize; 2]> = pos.chunks(2).map(|c| [c[0] as usize, c[1] as usize]).collect();
    PreparedImage::from_patches(acts.get("pixel_values")?, positions, vlm.processor.patch_size, vlm.processor.pooling_kernel_size)
}

fn prefixed(v: Vec<(String, Vec<f32>)>, prefix: &str) -> Vec<(String, Vec<f32>)> {
    v.into_iter().map(|(n, x)| (format!("{prefix}{n}"), x)).collect()
}

/// Full verification report as JSON; check `"ok"`.
pub fn verify(reference_dir: &Path, model: &str, dtype: &str, device: Option<&str>) -> Result<Value> {
    let raw = std::fs::read_to_string(reference_dir.join("result.json")).map_err(|e| {
        Error::new("io", format!("{}/result.json: {e} (run tools/reference/gemma4vl_ref.py)", reference_dir.display()))
    })?;
    let r: Value = serde_json::from_str(&raw)?;
    let acts = Activations::open(&reference_dir.join("activations.safetensors"))?;
    let dev = lapid_verify::open_device(device)?;
    let mut stages = lapid_verify::Stages::new();
    let (path, convert_ms) = resolve_model(model, dtype)?;
    if let Some(t) = convert_ms {
        stages.record("convert", t);
    }
    let dtype = crate::gemma4::verify::file_dtype(&path)?;
    let quantized = !matches!(dtype.as_str(), "fp32" | "fp16");
    let tol = quality_tolerance(&dtype)?;
    let mut vlm = stages.time("load", || Gemma4Vl::load(&path, &dev))?;
    if let Some(n) = r.get("max_soft_tokens").and_then(Value::as_u64) {
        vlm.set_max_soft_tokens(n as usize)?;
    }

    // ---- prompt + preprocessing
    let image = Image::open(&reference_dir.join(str_of(&r, "image")?))?;
    let messages = vec![Message::user_with_images(vec![image.clone()], str_of(&r, "prompt")?)];
    let opts = super::ChatOptions { enable_thinking: Some(false), ..Default::default() };
    let rendered = vlm.text.render_chat_json(&Gemma4Vl::messages_json(&messages), None, true, Some(false))?;
    let template_ok = rendered == str_of(&r, "prompt_text")?;
    let ours_prompt: Prompt = stages.time("prompt", || vlm.prompt(&messages, &opts))?;
    let want_ids = ids(&r, "prompt_tokens")?;
    let ids_ok = ours_prompt.ids == want_ids;
    let refimg = reference_image(&acts, &vlm)?;
    let ourimg = &ours_prompt.images[0];
    let grid_ok = ourimg.grid == refimg.grid;
    let pix = if grid_ok { lapid_verify::diff(&ourimg.patches, &refimg.patches)?.max_abs } else { f32::NAN };
    let prep_ok = grid_ok && pix <= 1e-6;

    // ---- vision tower on the reference pixels
    let d = vlm.text.config().text.hidden_size;
    let vision_ours = stages.time("vision", || -> Result<Vec<(String, Vec<f32>)>> {
        let mut trace = Vec::new();
        vlm.image_graph(&refimg, Some(&mut trace))?;
        let refs: Vec<&lapid::Tensor> = trace.iter().map(|(_, t)| t).collect();
        let out = dev.eval(&refs)?;
        trace.iter().zip(&out).map(|((n, _), t)| Ok((n.clone(), dev.to_vec_f32(t)?))).collect()
    })?;
    let mut names: Vec<String> = vec!["vision.embed".into()];
    names.extend((0..vlm.vision_config().n_layers).map(|i| format!("vision.layers.{i}")));
    names.extend(["vision.pooled".to_string(), "vision.projected".to_string()]);
    let mut quality = Comparison::new();
    quality.compare_named(&names, &vision_ours, |n| acts.get(n), tol)?;

    // ---- text decoder with the soft tokens injected
    // The decoder is checked on the reference soft tokens (layer by layer: vision errors are
    // reported above and do not compound into the text layers).
    let projected = acts.get("vision.projected")?;
    let n_soft = projected.len() / d.max(1);
    let start = want_ids.iter().position(|&t| t == vlm.image_token).unwrap_or(0);
    let inject = || -> Result<Vec<Injection>> {
        Ok(vec![Injection { start, embeds: lapid::Tensor::from_f32(&projected, &[n_soft, d])? }])
    };
    let traced = stages.time("forward", || vlm.text.model().forward_traced_with(&dev, &want_ids, &inject()?, Rows::Last))?;
    let traced = prefixed(traced, "mm.");
    let mut tnames = vec!["mm.embed".to_string()];
    for i in 0..vlm.text.config().text.n_layers {
        tnames.push(format!("mm.layers.{i}.attn"));
        tnames.push(format!("mm.layers.{i}"));
    }
    let mut ours_text: Vec<(String, Vec<f32>)> = traced.into_iter().filter(|(n, _)| n != "mm.norm").collect();
    for (n, _) in ours_text.iter_mut() {
        if n == "mm.logits" {
            *n = "mm.logits.last".into();
        }
    }
    tnames.push("mm.logits.last".into());
    quality.compare_named(&tnames, &ours_text, |n| acts.get(n), tol)?;
    let vocab = vlm.text.config().text.vocab_size;
    let last = ours_text.iter().find(|(n, _)| n == "mm.logits.last").map(|(_, v)| v.clone()).unwrap_or_default();
    let argmax = argmax_rows(&last, &acts.get("mm.logits.last")?, vocab);

    // ---- teacher-forced cached decode
    let want_gen = ids(&r, "generated_tokens")?;
    let steps = acts.shape("mm.gen.logits")?.first().copied().unwrap_or(0);
    let got = stages.time("decode", || -> Result<Vec<f32>> {
        let mut s = vlm.new_session();
        let mut out = vlm.text.step_with(&mut s, &want_ids, &inject()?)?;
        for &t in want_gen.iter().take(steps.saturating_sub(1)) {
            out.extend(vlm.text.step(&mut s, &[t])?);
        }
        Ok(out)
    })?;
    let mut decode = Comparison::new();
    let gen_ref = acts.get("mm.gen.logits")?;
    decode.compare("mm.gen.logits", &got, &gen_ref, tol);
    let decode_argmax = argmax_rows(&got, &gen_ref, vocab);

    // ---- greedy end to end (our preprocessing)
    let t = std::time::Instant::now();
    let mut s = vlm.new_session();
    let gopts = GenerateOptions { max_new_tokens: want_gen.len().max(1), ..Default::default() };
    let mut stream = vlm.generate(&mut s, &ours_prompt, gopts)?;
    for tok in stream.by_ref() {
        tok?;
    }
    let tokens = stream.generated().to_vec();
    drop(stream);
    stages.record("greedy", ms(t));
    let tokens_match = tokens == want_gen;
    let first_diff = tokens.iter().zip(&want_gen).position(|(a, b)| a != b);

    let strict = tokens_must_match(&dtype);
    let all = |v: &[bool]| v.iter().all(|&b| b);
    let text_ok = template_ok && ids_ok;
    let correctness_ok = quantized || (quality.ok() && decode.ok() && all(&argmax) && all(&decode_argmax));
    let ok = text_ok && prep_ok && correctness_ok && (!strict || tokens_match);
    Ok(json!({
        "ok": ok,
        "device": dev.description(),
        "model": model,
        "model_file": path.display().to_string(),
        "dtype": dtype,
        "text_ok": text_ok,
        "template_ok": template_ok,
        "prompt_ids_ok": ids_ok,
        "preprocess": {"ok": prep_ok, "grid": [ourimg.grid.0, ourimg.grid.1], "reference_grid": [refimg.grid.0, refimg.grid.1],
                       "max_abs_pixel_diff": pix, "n_soft": ourimg.n_soft},
        "tokens_match": tokens_match,
        "tokens_must_match": strict,
        "first_failing_layer": quality.first_failing(),
        "quality": {
            "reference": "PyTorch fp32 (HF Gemma4ForConditionalGeneration)",
            "informational": quantized,
            "tolerance": tol,
            "within_tolerance": quality.ok(),
            "max_rel": quality.max_rel(),
            "logits_argmax_match": argmax,
            "decode": decode.to_json(),
            "decode_argmax_match": decode_argmax,
        },
        "layers": quality.layers,
        "greedy": {"match": tokens_match, "first_mismatch": first_diff, "tokens": tokens, "reference_tokens": want_gen,
                   "text": vlm.decode(&tokens, false), "reference_text": r.get("generated_text")},
        "reference_dir": reference_dir.display().to_string(),
        "timings_ms": stages.to_json(),
        "notes": quality.notes,
        "info": vlm.info().name,
    }))
}
