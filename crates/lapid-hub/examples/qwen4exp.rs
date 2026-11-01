//! Streaming chat with Qwen4-Exp (Qwen3.8-Flash-Next). The state (KV cache, indexer keys, conv
//! windows, delta-rule states, PLE history) is reused across turns when the new prompt extends
//! the previous one. `--raw "text"` instead greedily continues a plain prompt (no template; for
//! token-level comparisons with llama.cpp's `llama-simple`) and prints the token ids.
//!
//! ```text
//! cargo run --release -p lapid-hub --example qwen4exp -- <x.gguf | x.lapid | HF dir> [int4] [--device cuda] [--think] [--greedy] [--raw "text"]
//! ```
//! `/reset` clears the conversation, `/quit` exits.

use std::io::{BufRead, Write};

use lapid_hub::qwen4exp::{ChatOptions, GenerateOptions, Message, Qwen4Exp};

fn main() -> lapid::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |f: &str| args.iter().any(|a| a == f);
    let device = match args.iter().position(|a| a == "--device") {
        Some(i) => lapid::Device::from_name(args.get(i + 1).map_or("cpu", |s| s.as_str()))?,
        None => lapid::Device::best(),
    };
    let pos: Vec<&String> = args.iter().enumerate().filter(|(i, a)| !a.starts_with("--") && (*i == 0 || !["--device", "--raw"].contains(&args[i - 1].as_str()))).map(|(_, a)| a).collect();
    let (model, dtype) = (pos.first().map_or("data/reference/qwen4exp/tiny/qwen4exp-tiny", |s| s.as_str()), pos.get(1).map_or("int4", |s| s.as_str()));
    // HF repo / directory / GGUF: converted into the lapid cache on first use.
    let llm = Qwen4Exp::load(&lapid_hub::runtime::resolve::resolve(model, Some(dtype))?.path, &device)?;
    let sampling = if flag("--greedy") { Default::default() } else { llm.recommended_sampling() };
    eprintln!("{} on {} ({} layers). /reset, /quit", model, device.description(), llm.config().n_layers());
    if let Some(raw) = args.iter().position(|a| a == "--raw").and_then(|i| args.get(i + 1)) {
        let mut session = llm.new_session();
        let mut stream = llm.generate(&mut session, &llm.encode(raw)?, GenerateOptions { max_new_tokens: 24, stop_tokens: Some(Vec::new()), ..Default::default() })?;
        for token in stream.by_ref() {
            print!("{}", token?.text);
        }
        println!("\n{:?}", stream.generated());
        return Ok(());
    }
    let (mut session, mut messages) = (llm.new_session(), Vec::<Message>::new());
    for line in std::io::stdin().lock().lines() {
        let line = line.map_err(|e| lapid::Error::new("io", e.to_string()))?;
        match line.trim() {
            "" => continue,
            "/quit" => break,
            "/reset" => {
                messages.clear();
                session.reset();
                continue;
            }
            l => messages.push(Message::user(l)),
        }
        let opts = ChatOptions {
            enable_thinking: Some(flag("--think")),
            generate: GenerateOptions { max_new_tokens: 1024, sampling: sampling.clone(), ..Default::default() },
            ..Default::default()
        };
        let mut stream = llm.chat_stream(&mut session, &messages, &opts)?;
        for token in stream.by_ref() {
            print!("{}", token?.text);
            std::io::stdout().flush().ok();
        }
        let s = stream.stats();
        eprintln!("\n[{} new prompt tokens ({} cached), ttft {:.0} ms, decode {:.1} tok/s]", s.prompt_tokens, s.cached_tokens, s.time_to_first_token_ms, s.decode_tokens_per_s());
        // Kept verbatim (thinking included) so the next prompt extends the cached tokens.
        let reply = Message::assistant(stream.text());
        drop(stream);
        messages.push(reply);
    }
    Ok(())
}
