//! Native check of the lapid-web code path (same `load_whisper` / `transcribe_json` the wasm
//! exports call), by default on the WebGPU backend:
//!
//! ```text
//! cargo run --release -p lapid-web --example transcribe -- model.lapid audio.wav [webgpu|cpu|webgpu:N] [runs]
//! ```

use std::path::Path;
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> lapid::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err(lapid::Error::new("invalid_argument", "usage: transcribe <model.lapid> <audio.wav> [device] [runs]"));
    }
    let device = lapid::Device::from_name(args.get(2).map(String::as_str).unwrap_or("webgpu"))?;
    let runs: usize = args.get(3).and_then(|r| r.parse().ok()).unwrap_or(1).max(1);
    eprintln!("device: {}", device.description());
    let bytes = std::fs::read(&args[0]).map_err(|e| lapid::Error::new("io", format!("{}: {e}", args[0])))?;
    let t = Instant::now();
    let model = lapid_web::load_whisper(bytes, &device)?;
    eprintln!("load: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let samples = lapid_hub::audio::load_audio_16k_mono(Path::new(&args[1]))?;
    let opts = lapid_web::options(None, None);
    for i in 0..runs {
        let t = Instant::now();
        let json = lapid_web::transcribe_json(&model, &samples, &opts)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if i == 0 {
            println!("{json}");
        }
        eprintln!("run {i}: transcribe {ms:.0} ms ({:.1} s audio)", samples.len() as f64 / 16000.0);
    }
    Ok(())
}
