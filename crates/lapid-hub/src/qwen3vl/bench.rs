//! `lapid bench qwen3vl`: image preprocessing + vision encoder time, prefill tokens/s of the
//! image prompt, decode tokens/s and time to first token (median over runs, fresh session).
//!
//! The picture is `input` (PNG / JPEG) or the reference picture
//! `data/reference/qwen3vl/test.png`; the prompt asks for a description and generation runs
//! greedy for `gen_tokens` with stop tokens disabled.

use std::path::Path;
use std::time::Instant;

use lapid::{Error, Result};
use lapid_verify::timing::{median, ms, stats};
use serde_json::{Value, json};

use super::verify::resolve_model;
use super::{GenerateOptions, GenerationStats, Qwen3Vl};
use crate::vision::{Image, Message};

fn run(vlm: &Qwen3Vl, msgs: &[Message], n_gen: usize) -> Result<(f64, f64, GenerationStats)> {
    let t = Instant::now();
    let mut p = vlm.prepare(msgs, None)?;
    let pre_ms = ms(t);
    let t = Instant::now();
    // Encode separately to time the vision tower, then hand the prompt over text-only spans.
    let images = std::mem::take(&mut p.images);
    let mut enc = Vec::new();
    for im in &images {
        enc.push(vlm.encode_patches(&im.patches)?);
    }
    let vis_ms = ms(t);
    p.images = images;
    let mut session = vlm.new_session();
    let opts = GenerateOptions { max_new_tokens: n_gen, stop_tokens: Some(Vec::new()), ..Default::default() };
    let mut stream = vlm.generate_prepared(&mut session, p, opts)?;
    for t in stream.by_ref() {
        t?;
    }
    drop(enc);
    Ok((pre_ms, vis_ms, stream.stats()))
}

pub fn bench(model: &str, dtype: &str, device: Option<&str>, runs: usize, gen_tokens: usize, image: Option<&Path>) -> Result<Value> {
    if runs == 0 || gen_tokens < 2 {
        return Err(Error::new("invalid_argument", "need runs >= 1 and gen tokens >= 2"));
    }
    let dev = lapid_verify::open_device(device)?;
    let (path, convert_ms) = resolve_model(model, dtype)?;
    let t = Instant::now();
    let vlm = Qwen3Vl::load(&path, &dev)?;
    let load_ms = ms(t);
    let img = Image::open(image.unwrap_or(Path::new("data/reference/qwen3vl/test.png")))?;
    let msgs = [Message::user_with_images(vec![img], "Describe this image in detail.")];
    run(&vlm, &msgs, 4)?;
    let mut measured = Vec::new();
    let (mut pre, mut vis) = (Vec::new(), Vec::new());
    for _ in 0..runs {
        let (p, v, s) = run(&vlm, &msgs, gen_tokens)?;
        pre.push(p);
        vis.push(v);
        measured.push(s);
    }
    let col = |f: &dyn Fn(&GenerationStats) -> f64| measured.iter().map(f).collect::<Vec<f64>>();
    Ok(json!({
        "engine": "lapid",
        "model": model,
        "model_file": path.display().to_string(),
        "device": dev.description(),
        "runs": runs,
        "prompt_tokens": measured.first().map(|s| s.prompt_tokens),
        "gen_tokens": gen_tokens,
        "convert_ms": convert_ms,
        "load_ms": load_ms,
        "preprocess_ms": stats(&pre),
        "vision_encode_ms": stats(&vis),
        "prefill_tokens_per_s": median(col(&|s| s.prefill_tokens_per_s())),
        "decode_tokens_per_s": median(col(&|s| s.decode_tokens_per_s())),
        "ttft_ms": stats(&col(&|s| s.time_to_first_token_ms)),
    }))
}
