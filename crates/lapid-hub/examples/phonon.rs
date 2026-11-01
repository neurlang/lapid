//! Text to speech with Phonon, streaming: PCM chunks are written as each 80 ms codec frame
//! is decoded; prints the time to first audio and the real-time factor.
//!
//! ```text
//! cargo run --release -p lapid-hub --example phonon -- "Hello from Phonon." [out.wav]
//!     --model=gradium/phonon-40m-en | --model=model.lapid   (default gradium/phonon-40m-en)
//!     --dtype=fp16          weight dtype when converting from the hub (fp32, fp16, int8, int4)
//!     --voice=NAME          voice (default "default"); --voices=extras.lapid adds voices,
//!     --voice-file=x.safetensors  adds (and selects) a voice file (gradium/phonon-voices format)
//!     --seed=N              deterministic noise seed
//!     --temperature=T       flow sampler noise variance (default 0.3)
//! ```
//! `LAPID_DEVICE=cpu|cuda|vulkan|metal` picks the device (default: best available).

use std::path::{Path, PathBuf};
use std::time::Instant;

use lapid::Device;
use lapid_hub::{Model, Phonon, SpeakOptions};

fn main() -> lapid::Result<()> {
    let (flags, args): (Vec<String>, Vec<String>) = std::env::args().skip(1).partition(|a| a.starts_with("--"));
    let flag = |name: &str| flags.iter().find_map(|f| f.strip_prefix(&format!("--{name}="))).map(str::to_string);
    let mut args = args.into_iter();
    let Some(text) = args.next() else {
        eprintln!(
            "usage: phonon <text> [out.wav] [--model=..] [--dtype=..] [--voice=..] [--voices=..] [--seed=N] [--temperature=T]"
        );
        std::process::exit(2);
    };
    let out = PathBuf::from(args.next().unwrap_or_else(|| "speech.wav".to_string()));
    let model = flag("model").unwrap_or_else(|| "gradium/phonon-40m-en".to_string());
    let dtype = flag("dtype").unwrap_or_else(|| "fp16".to_string());
    let parse = |name: &str| -> lapid::Result<Option<f64>> {
        flag(name)
            .map(|v| v.parse::<f64>().map_err(|e| lapid::Error::new("invalid_argument", format!("--{name}: {e}"))))
            .transpose()
    };

    let device = Device::best();
    let t = Instant::now();
    let mut tts = if Path::new(&model).is_file() {
        Phonon::load(Path::new(&model), &device)?
    } else {
        Phonon::from_hub(&model, &dtype, &device)?
    };
    if let Some(extras) = flag("voices") {
        tts.add_voices(Path::new(&extras))?;
    }
    let mut voice = flag("voice");
    if let Some(file) = flag("voice-file") {
        let name = Path::new(&file).file_stem().and_then(|s| s.to_str()).unwrap_or("custom").to_string();
        tts.add_voice_safetensors(&name, Path::new(&file))?;
        voice = Some(name);
    }
    eprintln!(
        "loaded {} ({}) on {} in {:.0} ms; voices: {}",
        tts.info().name,
        tts.info().dtype,
        device.description(),
        t.elapsed().as_secs_f64() * 1e3,
        tts.voices().join(", ")
    );

    let opts = SpeakOptions {
        voice,
        seed: parse("seed")?.map(|s| s as u64),
        temperature: parse("temperature")?.map(|v| v as f32),
        ..Default::default()
    };
    let t = Instant::now();
    let mut pcm = Vec::new();
    let mut first = None;
    for chunk in tts.speak_stream(&text, &opts)? {
        let chunk = chunk?;
        if first.is_none() {
            first = Some(t.elapsed().as_secs_f64() * 1e3);
        }
        pcm.extend(chunk); // a player would start playing here
    }
    let total = t.elapsed().as_secs_f64();
    let secs = pcm.len() as f64 / tts.sample_rate() as f64;
    lapid_hub::phonon::wav::write_wav(&out, &pcm, tts.sample_rate())?;
    eprintln!(
        "{:.2} s of audio -> {} | first audio {:.0} ms | total {:.0} ms | real-time factor {:.3}",
        secs,
        out.display(),
        first.unwrap_or(0.0),
        total * 1e3,
        if secs > 0.0 { total / secs } else { 0.0 }
    );
    Ok(())
}
