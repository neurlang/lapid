//! Describe an image with Qwen3-VL, streaming the answer.
//!
//! ```text
//! cargo run --release -p lapid-hub --example qwen3vl -- <repo|dir|model.gguf|file.lapid> \
//!     --image photo.jpg [--prompt "What is in this picture?"] [--dtype int8] [--device cuda]
//! ```
//! A `.gguf` model needs its `mmproj*.gguf` in the same directory (or `LAPID_MMPROJ`).

use std::io::Write;
use std::path::Path;

use lapid_hub::qwen3vl::{ChatOptions, Qwen3Vl};
use lapid_hub::vision::{Image, Message};

fn main() -> lapid::Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().unwrap_or_else(|| "Qwen/Qwen3-VL-2B-Instruct".to_string());
    let (mut image, mut prompt, mut dtype, mut device) = (None, "Describe this image.".to_string(), "int8".to_string(), None);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--image" => image = args.next(),
            "--prompt" => prompt = args.next().unwrap_or_default(),
            "--dtype" => dtype = args.next().unwrap_or(dtype),
            "--device" => device = args.next(),
            other => return Err(lapid::Error::new("invalid_argument", format!("unknown argument {other:?}"))),
        }
    }
    let device = match device {
        Some(d) => lapid::Device::from_name(&d)?,
        None => lapid::Device::best(),
    };
    let (path, _) = lapid_hub::qwen3vl::verify::resolve_model(&model, &dtype)?;
    let vlm = Qwen3Vl::load(&path, &device)?;
    let images = match image {
        Some(p) => vec![Image::open(Path::new(&p))?],
        None => Vec::new(),
    };
    let msg = Message::user_with_images(images, prompt);
    let mut session = vlm.new_session();
    let mut stream = vlm.chat_stream(&mut session, &[msg], &ChatOptions::default())?;
    for t in stream.by_ref() {
        print!("{}", t?.text);
        std::io::stdout().flush().ok();
    }
    let s = stream.stats();
    eprintln!(
        "\n[{} prompt tokens, TTFT {:.0} ms, {:.1} tok/s decode, {}]",
        s.prompt_tokens,
        s.time_to_first_token_ms,
        s.decode_tokens_per_s(),
        device.description()
    );
    Ok(())
}
