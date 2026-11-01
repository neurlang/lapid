//! `lapid bench gemma4vl`: image preprocessing and encoding time, time to first token of an
//! image + text prompt (encode + prefill), and decode speed.

use std::path::Path;
use std::time::Instant;

use lapid::Result;
use lapid_verify::timing::{median, stats};
use serde_json::{Value, json};

use super::{ChatOptions, Gemma4Vl, GenerateOptions};
use crate::model::Model;
use crate::vision::{Image, Message};

/// `image` defaults to the reference picture; `runs` timed runs after one warm-up.
pub fn bench(model: &str, dtype: &str, device: Option<&str>, runs: usize, image: &Path, gen_tokens: usize) -> Result<Value> {
    let dev = lapid_verify::open_device(device)?;
    let (path, convert_ms) = super::verify::resolve_model(model, dtype)?;
    let t = Instant::now();
    let vlm = Gemma4Vl::load(&path, &dev)?;
    let load_ms = t.elapsed().as_secs_f64() * 1e3;
    let img = Image::open(image)?;
    let messages = [Message::user_with_images(vec![img.clone()], "Describe this image in detail.")];
    let opts = ChatOptions { enable_thinking: Some(false), ..Default::default() };
    let (mut prep, mut enc, mut ttft, mut decode) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut n_soft = 0;
    let mut prompt_tokens = 0;
    for run in 0..=runs {
        let (p, e, n) = vlm.time_image(&img)?;
        n_soft = n;
        let t = Instant::now();
        let prompt = vlm.prompt(&messages, &opts)?;
        prompt_tokens = prompt.ids.len();
        let mut s = vlm.new_session();
        let g = GenerateOptions { max_new_tokens: gen_tokens.max(1), stop_tokens: Some(Vec::new()), ..Default::default() };
        let mut stream = vlm.generate(&mut s, &prompt, g)?;
        let mut first = None;
        let mut count = 0usize;
        for tok in stream.by_ref() {
            tok?;
            count += 1;
            if first.is_none() {
                first = Some(Instant::now());
            }
        }
        let end = Instant::now();
        if run == 0 {
            continue;
        }
        prep.push(p);
        enc.push(e);
        if let Some(f) = first {
            ttft.push((f - t).as_secs_f64() * 1e3);
            if count > 1 {
                decode.push((count - 1) as f64 / (end - f).as_secs_f64());
            }
        }
    }
    Ok(json!({
        "model": model,
        "model_file": path.display().to_string(),
        "device": dev.description(),
        "dtype": vlm.info().dtype,
        "convert_ms": convert_ms,
        "load_ms": load_ms,
        "image": image.display().to_string(),
        "n_soft_tokens": n_soft,
        "prompt_tokens": prompt_tokens,
        "preprocess_ms": median(prep.clone()),
        "image_encode_ms": median(enc.clone()),
        "ttft_ms": median(ttft.clone()),
        "decode_tok_s": median(decode.clone()),
        "runs": {"preprocess_ms": stats(&prep), "image_encode_ms": stats(&enc), "ttft_ms": stats(&ttft), "decode_tok_s": stats(&decode)},
    }))
}
