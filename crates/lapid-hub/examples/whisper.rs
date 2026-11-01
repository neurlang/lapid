//! Transcribe a WAV file with Whisper, library first: load from the hub cache (converting
//! from Hugging Face on first use) or a `.lapid` file, transcribe, print timestamped segments.
//!
//! ```text
//! cargo run --release -p lapid-hub --example whisper -- data/jfk.wav [openai/whisper-small | model.lapid] [int8]
//!     --bytes   load a .lapid file through `Whisper::load_bytes` (in-memory image)
//!     --json    print the transcript (segments, language, timings) as JSON
//! ```
//! `LAPID_DEVICE=cpu|cuda|vulkan|metal` picks the device (default: best available).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use lapid::Device;
use lapid_hub::{Model, TranscribeOptions, Whisper};

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn main() -> lapid::Result<()> {
    let (flags, args): (Vec<String>, Vec<String>) = std::env::args().skip(1).partition(|a| a.starts_with("--"));
    let mut args = args.into_iter();
    let Some(audio) = args.next().map(PathBuf::from) else {
        eprintln!("usage: transcribe <audio.wav> [repo (openai/whisper-small) | model.lapid] [dtype (int8)] [--bytes] [--json]");
        std::process::exit(2);
    };
    let model = args.next().unwrap_or_else(|| "openai/whisper-small".to_string());
    let dtype = args.next().unwrap_or_else(|| "int8".to_string());
    let json = flags.iter().any(|f| f == "--json");
    let bytes = flags.iter().any(|f| f == "--bytes");

    let device = Device::best();
    let t = Instant::now();
    let whisper = if Path::new(&model).is_file() {
        if bytes {
            let image = std::fs::read(&model).map_err(|e| lapid::Error::new("io", format!("{model}: {e}")))?;
            Whisper::load_bytes(Arc::new(image), &device)?
        } else {
            Whisper::load(Path::new(&model), &device)?
        }
    } else {
        Whisper::from_hub(&model, &dtype, &device)?
    };
    let load_ms = ms(t);

    let t = Instant::now();
    let transcript = whisper.transcribe_file(&audio, &TranscribeOptions::default())?;
    let run_ms = ms(t);

    if json {
        let v = serde_json::json!({
            "text": transcript.text,
            "language": transcript.language,
            "segments": transcript.segments,
            "model": whisper.info(),
            "device": device.description(),
            "timing_ms": { "load": load_ms, "transcribe": run_ms },
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    eprintln!("device: {}, load {load_ms:.0} ms, transcribe {run_ms:.0} ms", device.description());
    eprintln!("language: {}", transcript.language);
    for seg in &transcript.segments {
        println!("[{:7.2} --> {:7.2}] {}", seg.start, seg.end, seg.text);
    }
    Ok(())
}
