//! Streaming chat with LFM3. The KV cache and conv state are reused across turns when the new
//! prompt extends the previous one; `--timeout-s N` cancels a reply that runs too long
//! (a `CancelToken` set from a timer thread).
//!
//! ```text
//! cargo run --release -p lapid-hub --example lfm3 -- [LiquidAI/LFM3-1.7B | dir | x.gguf | x.lapid] [int8] [--think] [--timeout-s 60]
//! ```
//! `/reset` clears the conversation, `/quit` exits. `LAPID_DEVICE` picks the device.

use std::io::{BufRead, Write};
use std::time::Duration;

use lapid_hub::lfm3::{CancelToken, ChatOptions, GenerateOptions, Lfm3, Message};

fn main() -> lapid::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |f: &str| args.iter().position(|a| a == f).and_then(|i| args.get(i + 1)).cloned();
    let timeout = value("--timeout-s").and_then(|v| v.parse::<u64>().ok());
    let pos: Vec<&String> = args.iter().enumerate().filter(|(i, a)| !a.starts_with("--") && (*i == 0 || args[i - 1] != "--timeout-s")).map(|(_, a)| a).collect();
    let (model, dtype) = (pos.first().map_or("LiquidAI/LFM3-1.7B", |s| s.as_str()), pos.get(1).map_or("int8", |s| s.as_str()));
    let device = lapid::Device::best();
    // HF repo / directory / GGUF: converted into the lapid cache on first use.
    let llm = Lfm3::load(&lapid_hub::runtime::resolve::resolve(model, Some(dtype))?.path, &device)?;
    eprintln!("{model} on {} ({} layers). /reset, /quit", device.description(), llm.config().layer_types.len());
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
        let cancel = CancelToken::new();
        if let Some(s) = timeout {
            let c = cancel.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(s));
                c.cancel();
            });
        }
        let generate = GenerateOptions { max_new_tokens: 1024, sampling: llm.recommended_sampling(), cancel: Some(cancel), ..Default::default() };
        let opts = ChatOptions { enable_thinking: Some(args.iter().any(|a| a == "--think")), generate, ..Default::default() };
        let mut stream = llm.chat_stream(&mut session, &messages, &opts)?;
        for token in stream.by_ref() {
            print!("{}", token?.text);
            std::io::stdout().flush().ok();
        }
        let s = stream.stats();
        eprintln!("\n[{:?}: ttft {:.0} ms, {} cached tokens, decode {:.1} tok/s]", stream.finish_reason(), s.time_to_first_token_ms, s.cached_tokens, s.decode_tokens_per_s());
        // Kept verbatim (thinking included) so the next prompt extends the cached tokens.
        let reply = Message::assistant(stream.text());
        drop(stream);
        messages.push(reply);
    }
    Ok(())
}
