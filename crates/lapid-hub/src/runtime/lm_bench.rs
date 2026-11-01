//! `lapid bench <causal LM>` for any [`CausalLm`]: prefill tokens/s, decode tokens/s and time
//! to first token.
//!
//! Per measured run (fresh session each): a `prompt_tokens`-token prompt is streamed with
//! greedy decoding for `gen_tokens` tokens, stop tokens disabled. TTFT = start -> first token
//! (prefill + first sample); prefill tok/s = prompt / TTFT; decode tok/s = (gen - 1) /
//! (last token - first token). A short chat prompt's TTFT is reported separately.

use std::path::Path;
use std::time::Instant;

use lapid::{Error, Result};
use lapid_verify::harness::Descriptor;
use lapid_verify::timing::{median, ms, stats};
use serde_json::{Value, json};

use super::causal_lm::{CausalLm, Llm};
use super::chat::Message;
use super::generate::{GenerateOptions, GenerationStats};
use super::resolve::{file_dtype, resolve};
use crate::cmd::ModelArgs;

const FILLER: &str = "The quick brown fox jumps over the lazy dog while the river keeps flowing past old stone \
bridges, and travellers stop to write short notes about the weather, the roads and the towns they visit. ";

/// A chat prompt of exactly `n` tokens (`text` or filler repeated, truncated).
pub fn prompt_of<M: CausalLm>(llm: &Llm<M>, n: usize, text: Option<&str>) -> Result<Vec<u32>> {
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

fn run<M: CausalLm>(llm: &Llm<M>, prompt: &[u32], n_gen: usize) -> Result<GenerationStats> {
    let mut session = llm.new_session();
    let opts = GenerateOptions { max_new_tokens: n_gen, stop_tokens: Some(Vec::new()), ..Default::default() };
    let mut stream = llm.generate(&mut session, prompt, opts)?;
    for t in stream.by_ref() {
        t?;
    }
    Ok(stream.stats())
}

/// `input` (optional): a text file used as the user message instead of the filler.
pub fn bench<M: CausalLm>(desc: &Descriptor, input: Option<&Path>, args: &ModelArgs, runs: usize) -> Result<Value> {
    let model = args.model.as_deref().unwrap_or(desc.default_model);
    let dtype = args.dtype.as_deref().unwrap_or(desc.default_dtype);
    let (prompt_tokens, gen_tokens) = desc.bench_sizes()?;
    if runs == 0 || prompt_tokens == 0 || gen_tokens < 2 {
        return Err(Error::new("invalid_argument", "need runs >= 1, prompt tokens >= 1, gen tokens >= 2"));
    }
    let text = match input {
        Some(p) => Some(std::fs::read_to_string(p).map_err(|e| Error::new("io", format!("{}: {e}", p.display())))?),
        None => None,
    };
    let dev = lapid_verify::open_device(args.device.as_deref())?;
    let res = resolve(model, Some(dtype))?;
    let t = Instant::now();
    let llm = Llm::<M>::load(&res.path, &dev)?;
    let load_ms = ms(t);
    let prompt = prompt_of(&llm, prompt_tokens, text.as_deref())?;
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
        "model_file": res.path.display().to_string(),
        "dtype": file_dtype(&res.path)?,
        "device": dev.description(),
        "backend": dev.name(),
        "host": {"arch": std::env::consts::ARCH, "os": std::env::consts::OS, "threads": threads},
        "runs": runs,
        "prompt_tokens": prompt.len(),
        "gen_tokens": gen_tokens,
        "convert_ms": res.convert_ms,
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
