//! Interactive streaming chat with DeepSeek V4, library first: load (converting an upstream
//! safetensors directory or a GGUF file into the lapid cache on first use), then stream each
//! reply token by token. The session state (sliding windows, compressor and indexer states) is
//! reused across turns when the new prompt extends the previous one (chat mode does; thinking
//! mode drops earlier reasoning, so later turns re-prefill).
//!
//! ```text
//! cargo run --release -p lapid-hub --example deepseek4 -- [model (data/reference/deepseek4/tiny/model | x.lapid | x.gguf)] [dtype (fp32)]
//!     [--device cuda] [--think] [--greedy] [--max-tokens 1024] [--timeout-s 60] [--system "..."]
//! ```
//! Commands: `/reset` clears the conversation, `/quit` exits. `--timeout-s` cancels a reply
//! that runs too long (CancelToken from a timer thread).

use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Duration;

use lapid::Device;
use lapid_hub::deepseek4::{CancelToken, ChatOptions, Deepseek4, GenerateOptions, Message};

struct Args {
    model: String,
    dtype: String,
    device: Option<String>,
    think: bool,
    greedy: bool,
    max_tokens: usize,
    timeout_s: Option<u64>,
    system: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        model: "data/reference/deepseek4/tiny/model".into(),
        dtype: "fp32".into(),
        device: None,
        think: false,
        greedy: false,
        max_tokens: 1024,
        timeout_s: None,
        system: None,
    };
    let mut pos = 0;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--device" => a.device = it.next(),
            "--think" => a.think = true,
            "--greedy" => a.greedy = true,
            "--max-tokens" => a.max_tokens = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.max_tokens),
            "--timeout-s" => a.timeout_s = it.next().and_then(|v| v.parse().ok()),
            "--system" => a.system = it.next(),
            _ if pos == 0 => {
                a.model = arg;
                pos += 1;
            }
            _ => a.dtype = arg,
        }
    }
    a
}

fn load(args: &Args, device: &Device) -> lapid::Result<Deepseek4> {
    let p = Path::new(&args.model);
    if p.extension().is_some_and(|e| e == "lapid") {
        return Deepseek4::load(p, device);
    }
    if p.is_file() {
        // GGUF (or another convertible file): convert into the lapid cache once.
        let out = lapid_hub::convert::convert(&args.model, &args.dtype, None)?;
        return Deepseek4::load(&out, device);
    }
    Deepseek4::from_hub(&args.model, &args.dtype, device)
}

/// Split a thinking reply into (reasoning, answer).
fn split_thinking(text: &str) -> (Option<String>, String) {
    match text.split_once("</think>") {
        Some((r, a)) => (Some(r.trim_start_matches("<think>").trim().to_string()), a.trim().to_string()),
        // Kept verbatim so the next turn's prompt extends the cached tokens exactly.
        None => (None, text.to_string()),
    }
}

fn main() -> lapid::Result<()> {
    let args = parse_args();
    let device = match &args.device {
        Some(d) => Device::from_name(d)?,
        None => Device::best(),
    };
    eprintln!("device: {}", device.description());
    let llm = load(&args, &device)?;
    let mut sampling = llm.recommended_sampling();
    if args.greedy {
        sampling = Default::default();
    }
    let mut session = llm.new_session();
    let mut messages: Vec<Message> = args.system.iter().map(|s| Message::system(s.clone())).collect();
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    eprintln!("ready ({} layers, thinking {}). /reset, /quit", llm.config().n_layers, if args.think { "on" } else { "off" });
    loop {
        eprint!("\n> ");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).map_err(|e| lapid::Error::new("io", e.to_string()))? == 0 {
            break;
        }
        let line = line.trim();
        match line {
            "" => continue,
            "/quit" | "/exit" => break,
            "/reset" => {
                messages.retain(|m| m.role == "system");
                session.reset();
                continue;
            }
            _ => {}
        }
        messages.push(Message::user(line));
        let cancel = CancelToken::new();
        if let Some(s) = args.timeout_s {
            let c = cancel.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(s));
                c.cancel();
            });
        }
        let opts = ChatOptions {
            thinking: args.think,
            generate: GenerateOptions { max_new_tokens: args.max_tokens, sampling: sampling.clone(), stop_tokens: None, cancel: Some(cancel) },
        };
        let mut stream = llm.chat_stream(&mut session, &messages, &opts)?;
        for token in stream.by_ref() {
            let token = token?;
            print!("{}", token.text);
            let _ = out.flush();
        }
        let s = stream.stats();
        let reason = stream.finish_reason();
        let (reasoning, answer) = split_thinking(stream.text());
        drop(stream);
        eprintln!(
            "\n[{reason:?}: {} new prompt tokens ({} cached), ttft {:.0} ms, prefill {:.0} tok/s, decode {:.1} tok/s]",
            s.prompt_tokens,
            s.cached_tokens,
            s.time_to_first_token_ms,
            s.prefill_tokens_per_s(),
            s.decode_tokens_per_s()
        );
        messages.push(Message { role: "assistant".into(), content: answer, reasoning_content: reasoning });
    }
    Ok(())
}
