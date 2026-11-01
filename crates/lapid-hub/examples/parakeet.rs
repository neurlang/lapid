//! Transcribe an audio file with Parakeet TDT, library first: load from the hub cache
//! (converting from Hugging Face on first use), a `.lapid` file, or a Parakeet GGUF;
//! print sentence segments and word timestamps.
//!
//! ```text
//! cargo run --release -p lapid-hub --example parakeet -- data/jfk.wav \
//!     [nvidia/parakeet-tdt-0.6b-v3 | model.lapid | model.gguf] [int8]
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use lapid::Device;
use lapid_hub::Parakeet;
use lapid_hub::parakeet::TranscribeOptions;

fn main() -> lapid::Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(audio) = args.next().map(PathBuf::from) else {
        eprintln!("usage: transcribe_parakeet <audio.wav> [repo | .lapid | .gguf (nvidia/parakeet-tdt-0.6b-v3)] [dtype (int8)]");
        std::process::exit(2);
    };
    let model = args.next().unwrap_or_else(|| "nvidia/parakeet-tdt-0.6b-v3".to_string());
    let dtype = args.next().unwrap_or_else(|| "int8".to_string());

    let device = Device::best();
    eprintln!("device: {}", device.description());
    let t = Instant::now();
    let p = Path::new(&model);
    let parakeet = if p.extension().is_some_and(|e| e == "lapid") {
        Parakeet::load(p, &device)?
    } else if p.is_file() {
        let out = lapid_hub::convert::convert(&model, &dtype, None)?;
        Parakeet::load(&out, &device)?
    } else {
        Parakeet::from_hub(&model, &dtype, &device)?
    };
    eprintln!("loaded in {:.0} ms", t.elapsed().as_secs_f64() * 1e3);

    let t = Instant::now();
    let out = parakeet.transcribe_file(&audio, &TranscribeOptions::default())?;
    eprintln!("transcribed in {:.0} ms ({} encoder windows)", t.elapsed().as_secs_f64() * 1e3, out.windows);

    for seg in &out.segments {
        println!("[{:7.2} --> {:7.2}] {}", seg.start, seg.end, seg.text);
    }
    println!();
    for w in &out.words {
        println!("{:7.2} {:7.2}  {}", w.start, w.end, w.word);
    }
    Ok(())
}
