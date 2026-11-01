//! Describe an image with Qwen3-VL (or chat without one), streaming the answer.
//!
//! ```text
//! cargo run --release -p lapid-hub --example qwen3vl -- [model (Qwen/Qwen3-VL-2B-Instruct | dir | x.gguf | x.lapid)]
//!     [--image photo.jpg] [--prompt "What is in this picture?"] [--dtype int8] [--device cuda]
//! ```
//! A `.gguf` model needs its `mmproj*.gguf` in the same directory (or `LAPID_MMPROJ`).

use std::io::Write;
use std::path::Path;

use lapid_hub::qwen3vl::{ChatOptions, Qwen3Vl};
use lapid_hub::vision::{Image, Message};

fn main() -> lapid::Result<()> {
    let (mut model, mut dtype, mut device) = ("Qwen/Qwen3-VL-2B-Instruct".to_string(), "int8".to_string(), None);
    let (mut image, mut prompt) = (None, "Describe this image.".to_string());
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--image" => image = args.next(),
            "--prompt" => prompt = args.next().unwrap_or_default(),
            "--dtype" => dtype = args.next().unwrap_or(dtype),
            "--device" => device = args.next(),
            _ => model = a,
        }
    }
    let dev = match device {
        Some(d) => lapid::Device::from_name(&d)?,
        None => lapid::Device::best(),
    };
    let vlm = Qwen3Vl::load(&lapid_hub::runtime::vlm_verify::resolve_vision(&model, &dtype)?.0, &dev)?;
    let images = image.map(|p| Image::open(Path::new(&p))).transpose()?.into_iter().collect();
    let mut session = vlm.new_session();
    let mut stream = vlm.mm_chat_stream(&mut session, &[Message::user_with_images(images, prompt)], &ChatOptions::default())?;
    for t in stream.by_ref() {
        print!("{}", t?.text);
        std::io::stdout().flush().ok();
    }
    let s = stream.stats();
    eprintln!("\n[{} prompt tokens, TTFT {:.0} ms, {:.1} tok/s, {}]", s.prompt_tokens, s.time_to_first_token_ms, s.decode_tokens_per_s(), dev.description());
    Ok(())
}
