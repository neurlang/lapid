//! Describe an image with Gemma 4 (vision tower + text decoder), streaming the answer.
//!
//! cargo run --release -p lapid-hub --example gemma4vl -- <model|.lapid|.gguf> <image> [prompt] [--dtype D] [--device D]
//!
//! `model` is an HF repo / directory (converted with the vision tower), a `.lapid`, or a
//! model GGUF with its `mmproj*.gguf` in the same directory.

use std::io::Write;
use std::path::Path;

use lapid_hub::gemma4vl::{ChatOptions, Gemma4Vl};
use lapid_hub::vision::{Image, Message};

fn main() -> lapid::Result<()> {
    let mut pos = Vec::new();
    let (mut dtype, mut device) = ("int8".to_string(), None::<String>);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dtype" => dtype = args.next().unwrap_or(dtype),
            "--device" => device = args.next(),
            _ => pos.push(a),
        }
    }
    let model = pos.first().cloned().unwrap_or_else(|| "google/gemma-4-E2B-it".into());
    let image = pos.get(1).cloned().unwrap_or_else(|| "data/reference/gemma4vl/image.png".into());
    let prompt = pos.get(2).cloned().unwrap_or_else(|| "Describe this image.".into());
    let dev = match device {
        Some(d) => lapid::Device::from_name(&d)?,
        None => lapid::Device::best(),
    };
    let path = lapid_hub::gemma4vl::verify::resolve(&model, &dtype)?;
    let vlm = Gemma4Vl::load(&path, &dev)?;
    let messages = [Message::user_with_images(vec![Image::open(Path::new(&image))?], prompt)];
    let mut session = vlm.new_session();
    let opts = ChatOptions { enable_thinking: Some(false), ..Default::default() };
    let mut stream = vlm.chat_stream(&mut session, &messages, &opts)?;
    for t in stream.by_ref() {
        print!("{}", t?.text);
        std::io::stdout().flush().ok();
    }
    let s = stream.stats();
    println!("\n[{} prompt tokens, TTFT {:.0} ms, {} tokens]", s.prompt_tokens, s.time_to_first_token_ms, s.generated_tokens);
    Ok(())
}
