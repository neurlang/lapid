//! Streaming chat with Gemma 4, library first: load (converting from the Hugging Face hub or
//! a GGUF file on first use), then stream each reply token by token. The KV cache is reused
//! across turns when the new prompt extends the previous one.
//!
//! ```text
//! cargo run --release -p lapid-hub --example gemma4 -- [model (google/gemma-4-E2B-it | x.lapid | x.gguf)] [dtype (int8)]
//!     [--device cuda] [--think] [--sample] [--max-tokens 256] [--system "..."] [--interactive]
//!     [--prompt "..."]...
//! ```
//! Without `--interactive` it runs the given `--prompt` turns (default: a 2-turn smoke test)
//! and prints time-to-first-token and decode speed per turn.

use std::io::{BufRead, Write};
use std::path::Path;

use lapid::Device;
use lapid_hub::gemma4::{ChatOptions, Gemma4, GenerateOptions, Message};

struct Args {
    model: String,
    dtype: String,
    device: Option<String>,
    think: bool,
    sample: bool,
    interactive: bool,
    max_tokens: usize,
    system: Option<String>,
    prompts: Vec<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        model: "google/gemma-4-E2B-it".into(),
        dtype: "int8".into(),
        device: None,
        think: false,
        sample: false,
        interactive: false,
        max_tokens: 256,
        system: None,
        prompts: Vec::new(),
    };
    let mut pos = 0;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--device" => a.device = it.next(),
            "--think" => a.think = true,
            "--sample" => a.sample = true,
            "--interactive" => a.interactive = true,
            "--max-tokens" => a.max_tokens = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.max_tokens),
            "--system" => a.system = it.next(),
            "--prompt" => a.prompts.extend(it.next()),
            _ if pos == 0 => {
                a.model = arg;
                pos += 1;
            }
            _ => a.dtype = arg,
        }
    }
    if a.prompts.is_empty() {
        a.prompts = vec!["Name three primary colors, briefly.".into(), "Now mix the first two: what do you get?".into()];
    }
    a
}

fn load(args: &Args, device: &Device) -> lapid::Result<Gemma4> {
    let p = Path::new(&args.model);
    if p.extension().is_some_and(|e| e == "lapid") {
        return Gemma4::load(p, device);
    }
    if p.is_file() {
        let out = lapid_hub::convert::convert(&args.model, &args.dtype, None)?;
        return Gemma4::load(&out, device);
    }
    Gemma4::from_hub(&args.model, &args.dtype, device)
}

/// Split a thinking reply into (reasoning, answer) (`<|channel>thought\n…<channel|>`).
fn split_thinking(text: &str) -> (Option<String>, String) {
    match text.split_once("<channel|>") {
        Some((r, a)) => (Some(r.trim_start_matches("<|channel>thought\n").trim().to_string()), a.trim().to_string()),
        None => (None, text.to_string()),
    }
}

fn reply(llm: &Gemma4, session: &mut lapid_hub::gemma4::Session, messages: &mut Vec<Message>, args: &Args) -> lapid::Result<()> {
    let sampling = if args.sample { llm.recommended_sampling() } else { Default::default() };
    let opts = ChatOptions {
        enable_thinking: Some(args.think),
        generate: GenerateOptions { max_new_tokens: args.max_tokens, sampling, stop_tokens: None, cancel: None },
    };
    let mut out = std::io::stdout();
    let mut stream = llm.chat_stream(session, messages, &opts)?;
    for token in stream.by_ref() {
        print!("{}", token?.text);
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
    Ok(())
}

fn main() -> lapid::Result<()> {
    let args = parse_args();
    let device = match &args.device {
        Some(d) => Device::from_name(d)?,
        None => Device::best(),
    };
    eprintln!("device: {}", device.description());
    let llm = load(&args, &device)?;
    let mut session = llm.new_session();
    let mut messages: Vec<Message> = args.system.iter().map(|s| Message::system(s.clone())).collect();
    if !args.interactive {
        for p in &args.prompts {
            eprintln!("\n> {p}");
            messages.push(Message::user(p.clone()));
            reply(&llm, &mut session, &mut messages, &args)?;
        }
        return Ok(());
    }
    let stdin = std::io::stdin();
    eprintln!("ready ({} layers). /reset, /quit", llm.config().n_layers);
    loop {
        eprint!("\n> ");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).map_err(|e| lapid::Error::new("io", e.to_string()))? == 0 {
            break;
        }
        match line.trim() {
            "" => continue,
            "/quit" | "/exit" => break,
            "/reset" => {
                messages.retain(|m| m.role == "system");
                session.reset();
            }
            l => {
                messages.push(Message::user(l));
                reply(&llm, &mut session, &mut messages, &args)?;
            }
        }
    }
    Ok(())
}
