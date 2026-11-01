//! `lapid bench gemma4moe`: prefill tokens/s, decode tokens/s and time-to-first-token.
//!
//! Per measured run (fresh session each): a `prompt_tokens`-token prompt is streamed with
//! greedy decoding for `gen_tokens` tokens, stop tokens disabled. TTFT = start → first token
//! (prefill + first sample); prefill tok/s = prompt / TTFT; decode tok/s = (gen − 1) /
//! (last token − first token). A short chat prompt's TTFT is reported separately.

use std::time::Instant;

use lapid::{Error, Result};
use lapid_verify::timing::{median, ms, stats};
use serde_json::{Value, json};

use super::verify::{file_dtype, resolve_model};
use super::{GenerateOptions, GenerationStats, Gemma4Moe, Message};

const FILLER: &str = "The quick brown fox jumps over the lazy dog while the river keeps flowing past old stone \
bridges, and travellers stop to write short notes about the weather, the roads and the towns they visit. ";

/// A chat prompt of exactly `n` tokens (`text` or filler repeated, truncated).
pub fn prompt_of(llm: &Gemma4Moe, n: usize, text: Option<&str>) -> Result<Vec<u32>> {
    let unit = text.filter(|t| !t.trim().is_empty()).unwrap_or(FILLER);
    let mut text = String::new();
    loop {
        text.push_str(unit);
        let ids = llm.encode(&llm.render_chat(&[Message::user(text.clone())], true, Some(false))?)?;
        if ids.len() >= n {
            return Ok(ids[..n].to_vec());
        }
    }
}

fn run(llm: &Gemma4Moe, prompt: &[u32], n_gen: usize) -> Result<GenerationStats> {
    let mut session = llm.new_session();
    let opts = GenerateOptions { max_new_tokens: n_gen, stop_tokens: Some(Vec::new()), ..Default::default() };
    let mut stream = llm.generate(&mut session, prompt, opts)?;
    for t in stream.by_ref() {
        t?;
    }
    Ok(stream.stats())
}

pub fn bench(
    model: &str,
    dtype: &str,
    device: Option<&str>,
    runs: usize,
    prompt_tokens: usize,
    gen_tokens: usize,
    text: Option<&str>,
) -> Result<Value> {
    if runs == 0 || prompt_tokens == 0 || gen_tokens < 2 {
        return Err(Error::new("invalid_argument", "need runs >= 1, prompt tokens >= 1, gen tokens >= 2"));
    }
    let dev = lapid_verify::open_device(device)?;
    let (path, convert_ms) = resolve_model(model, dtype)?;
    let t = Instant::now();
    let llm = Gemma4Moe::load(&path, &dev)?;
    let load_ms = ms(t);
    let prompt = prompt_of(&llm, prompt_tokens, text)?;
    let chat = llm.encode(&llm.render_chat(&[Message::user("Hello! How are you?")], true, Some(false))?)?;

    let t = Instant::now();
    run(&llm, &prompt, 4)?;
    run(&llm, &chat, 2)?;
    let warmup_ms = ms(t);

    let mut measured = Vec::with_capacity(runs);
    let mut chat_ttft = Vec::with_capacity(runs);
    for _ in 0..runs {
        measured.push(run(&llm, &prompt, gen_tokens)?);
        chat_ttft.push(run(&llm, &chat, 2)?.time_to_first_token_ms);
    }
    let col = |f: &dyn Fn(&GenerationStats) -> f64| measured.iter().map(f).collect::<Vec<f64>>();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    Ok(json!({
        "engine": "lapid",
        "model": model,
        "model_file": path.display().to_string(),
        "dtype": file_dtype(&path)?,
        "device": dev.description(),
        "backend": dev.name(),
        "host": {"arch": std::env::consts::ARCH, "os": std::env::consts::OS, "threads": threads},
        "runs": runs,
        "prompt_tokens": prompt.len(),
        "gen_tokens": gen_tokens,
        "convert_ms": convert_ms,
        "load_ms": load_ms,
        "warmup_ms": warmup_ms,
        "prefill_tokens_per_s": median(col(&|s| s.prefill_tokens_per_s())),
        "decode_tokens_per_s": median(col(&|s| s.decode_tokens_per_s())),
        "ttft_ms": stats(&col(&|s| s.time_to_first_token_ms)),
        "decode_ms": stats(&col(&|s| s.decode_ms)),
        "chat_prompt_tokens": chat.len(),
        "chat_ttft_ms": stats(&chat_ttft),
        "per_run": measured,
    }))
}
