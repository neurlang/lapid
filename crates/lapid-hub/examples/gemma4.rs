//! Chat with Gemma 4 (dense, MoE, with or without images), streaming the reply.
//!
//! ```text
//! cargo run --release -p lapid-hub --example gemma4 -- [model (google/gemma-4-E2B-it | x.lapid | x.gguf)]
//!     [--image photo.png] [--prompt "..."] [--dtype int8] [--device cuda] [--think]
//! ```
//! A `.gguf` takes the vision tower from the `mmproj*.gguf` next to it.

use std::io::Write;
use std::path::Path;

use lapid_hub::gemma4::{ChatOptions, Gemma4};
use lapid_hub::vision::{Image, Message};

fn main() -> lapid::Result<()> {
    let (mut model, mut dtype, mut device) = ("google/gemma-4-E2B-it".to_string(), "int8".to_string(), None);
    let (mut image, mut prompt, mut think) = (None, None, false);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--image" => image = args.next(),
            "--prompt" => prompt = args.next(),
            "--dtype" => dtype = args.next().unwrap_or(dtype),
            "--device" => device = args.next(),
            "--think" => think = true,
            _ => model = a,
        }
    }
    let dev = match device {
        Some(d) => lapid::Device::from_name(&d)?,
        None => lapid::Device::best(),
    };
    let path = lapid_hub::runtime::vlm_verify::resolve_vision(&model, &dtype)?.0;
    let llm = Gemma4::load(&path, &dev)?;
    let message = match &image {
        Some(p) => Message::user_with_images(vec![Image::open(Path::new(p))?], prompt.unwrap_or_else(|| "Describe this image.".into())),
        None => Message::user(prompt.unwrap_or_else(|| "Write a haiku about rust.".into())),
    };
    let mut session = llm.new_session();
    let opts = ChatOptions { enable_thinking: Some(think), ..Default::default() };
    let mut stream = llm.mm_chat_stream(&mut session, &[message], &opts)?;
    for t in stream.by_ref() {
        print!("{}", t?.text);
        std::io::stdout().flush().ok();
    }
    let s = stream.stats();
    println!("\n[{} prompt tokens, TTFT {:.0} ms, {:.1} tok/s]", s.prompt_tokens, s.time_to_first_token_ms, s.decode_tokens_per_s());
    Ok(())
}
