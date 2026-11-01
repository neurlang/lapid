//! Interactive streaming chat with DeepSeek V4: load (converting an upstream safetensors
//! directory or a GGUF file into the lapid cache on first use), then stream each reply. The
//! session state is reused across turns when the new prompt extends the previous one.
//!
//! ```text
//! cargo run --release -p lapid-hub --example deepseek4 -- [model (data/reference/deepseek4/tiny/model | x.lapid | x.gguf)] [dtype (fp32)] [--think]
//! ```
//! `/reset` clears the conversation, `/quit` exits.

use std::io::{BufRead, Write};

use lapid_hub::deepseek4::{ChatOptions, Deepseek4, GenerateOptions, Message};

fn main() -> lapid::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let think = args.iter().any(|a| a == "--think");
    let mut pos = args.iter().filter(|a| !a.starts_with("--"));
    let model = pos.next().map_or("data/reference/deepseek4/tiny/model", |s| s.as_str());
    let dtype = pos.next().map_or("fp32", |s| s.as_str());
    let device = lapid::Device::best();
    let path = lapid_hub::runtime::resolve::resolve(model, Some(dtype))?.path;
    let llm = Deepseek4::load(&path, &device)?;
    let opts = ChatOptions { enable_thinking: Some(think), generate: GenerateOptions { max_new_tokens: 1024, ..Default::default() }, ..Default::default() };
    let (mut session, mut messages) = (llm.new_session(), Vec::new());
    print!("> ");
    std::io::stdout().flush().ok();
    for line in std::io::stdin().lock().lines() {
        match line?.trim() {
            "/quit" => break,
            "/reset" => messages.clear(),
            "" => {}
            text => {
                messages.push(Message::user(text));
                let mut stream = llm.chat_stream(&mut session, &messages, &opts)?;
                let mut reply = String::new();
                for t in stream.by_ref() {
                    let t = t?;
                    print!("{}", t.text);
                    std::io::stdout().flush().ok();
                    reply.push_str(&t.text);
                }
                let s = stream.stats();
                println!("\n[{} prompt tokens ({} cached), {:.1} tok/s]", s.prompt_tokens, s.cached_tokens, s.decode_tokens_per_s());
                messages.push(Message::assistant(reply));
            }
        }
        print!("> ");
        std::io::stdout().flush().ok();
    }
    Ok(())
}
