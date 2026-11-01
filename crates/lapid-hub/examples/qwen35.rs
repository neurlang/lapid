//! Streaming chat with Qwen3.5 (dense or MoE). The state (KV cache, conv windows, delta-rule
//! states) is reused across turns when the new prompt extends the previous one.
//!
//! ```text
//! cargo run --release -p lapid-hub --example qwen35 -- [Qwen/Qwen3.5-0.8B | dir | x.gguf | x.lapid] [int8] [--device cuda] [--think] [--greedy]
//! ```
//! `/reset` clears the conversation, `/quit` exits.

use std::io::{BufRead, Write};

use lapid_hub::qwen35::{ChatOptions, GenerateOptions, Message, Qwen35};

fn main() -> lapid::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |f: &str| args.iter().any(|a| a == f);
    let device = match args.iter().position(|a| a == "--device") {
        Some(i) => lapid::Device::from_name(args.get(i + 1).map_or("cpu", |s| s.as_str()))?,
        None => lapid::Device::best(),
    };
    let pos: Vec<&String> = args.iter().enumerate().filter(|(i, a)| !a.starts_with("--") && (*i == 0 || args[i - 1] != "--device")).map(|(_, a)| a).collect();
    let (model, dtype) = (pos.first().map_or("Qwen/Qwen3.5-0.8B", |s| s.as_str()), pos.get(1).map_or("int8", |s| s.as_str()));
    // HF repo / directory / GGUF: converted into the lapid cache on first use.
    let llm = Qwen35::load(&lapid_hub::runtime::resolve::resolve(model, Some(dtype))?.path, &device)?;
    let sampling = if flag("--greedy") { Default::default() } else { llm.recommended_sampling() };
    eprintln!("{} on {} ({} layers). /reset, /quit", model, device.description(), llm.config().n_layers);
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
