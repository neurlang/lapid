//! Transcribe a WAV file with Whisper: load from the hub cache (converting from Hugging Face
//! on first use) or a `.lapid` file, transcribe, print timestamped segments.
//!
//! ```text
//! cargo run --release -p lapid-hub --example whisper -- data/jfk.wav [openai/whisper-small | model.lapid] [int8]
//!     --bytes   load a .lapid file through `Whisper::load_bytes` (in-memory image)
//!     --json    print the transcript as JSON
//! ```
//! `LAPID_DEVICE=cpu|cuda|vulkan|metal` picks the device (default: best available).

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use lapid::Device;
use lapid_hub::{TranscribeOptions, Whisper};

fn main() -> lapid::Result<()> {
    let (flags, args): (Vec<String>, Vec<String>) = std::env::args().skip(1).partition(|a| a.starts_with("--"));
    let [audio, rest @ ..] = &args[..] else {
        eprintln!("usage: whisper <audio.wav> [repo (openai/whisper-small) | model.lapid] [dtype (int8)] [--bytes] [--json]");
        std::process::exit(2);
    };
    let model = rest.first().map_or("openai/whisper-small", String::as_str);
    let dtype = rest.get(1).map_or("int8", String::as_str);
    let device = Device::best();
    let t = Instant::now();
    let whisper = match Path::new(model) {
        p if p.is_file() && flags.iter().any(|f| f == "--bytes") => {
            Whisper::load_bytes(Arc::new(std::fs::read(p).map_err(|e| lapid::Error::new("io", format!("{model}: {e}")))?), &device)?
        }
        p if p.is_file() => Whisper::load(p, &device)?,
        _ => Whisper::from_hub(model, dtype, &device)?,
    };
    let load_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let transcript = whisper.transcribe_file(Path::new(audio), &TranscribeOptions::default())?;
    let run_ms = t.elapsed().as_secs_f64() * 1e3;
    if flags.iter().any(|f| f == "--json") {
        println!("{}", serde_json::to_string_pretty(&transcript)?);
        return Ok(());
    }
    eprintln!("device: {}, load {load_ms:.0} ms, transcribe {run_ms:.0} ms, language {}", device.description(), transcript.language);
    for seg in &transcript.segments {
        println!("[{:7.2} --> {:7.2}] {}", seg.start, seg.end, seg.text);
    }
    Ok(())
}
