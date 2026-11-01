//! Streaming speech recognition with a Zipformer2 transducer: the WAV is fed in 100 ms pieces
//! as if from a microphone; partials update in place, finals print at endpoints. `--vad`
//! cuts the audio with Silero VAD first and decodes each segment with a fresh stream.
//!
//! ```text
//! cargo run --release -p lapid-hub --example zipformer -- data/jfk.wav \
//!     [data/hf/zipformer2-en-streaming-2023-05-17 | model.lapid] [fp16] [--vad]
//! ```

use std::path::Path;
use std::time::Instant;

use lapid::Device;
use lapid_hub::zipformer::{StreamOptions, Zipformer};
use lapid_hub::{StreamResult, Vad, VadOptions};

/// Stream `samples` through a fresh stream; print partials (in place) and finals.
fn run(z: &Zipformer, samples: &[f32], opts: &StreamOptions, offset_s: f64) -> lapid::Result<()> {
    let mut s = z.stream(opts)?;
    let show = |r: &StreamResult| match r.is_final {
        true => println!("\r\x1b[2K[{:7.2}] {}", offset_s + f64::from(r.start), r.text),
        false => eprint!("\r\x1b[2K{}", r.text),
    };
    for piece in samples.chunks(1600) {
        s.accept_waveform(piece)?.iter().for_each(show);
    }
    s.input_finished()?.iter().for_each(show);
    Ok(())
}

fn main() -> lapid::Result<()> {
    let (flags, args): (Vec<String>, Vec<String>) = std::env::args().skip(1).partition(|a| a.starts_with("--"));
    let [audio, rest @ ..] = &args[..] else {
        eprintln!("usage: zipformer <audio.wav> [model dir | .lapid] [dtype (fp16)] [--vad]");
        std::process::exit(2);
    };
    let model = rest.first().map_or("data/hf/zipformer2-en-streaming-2023-05-17", String::as_str);
    let device = Device::best();
    let z = match Path::new(model) {
        p if p.extension().is_some_and(|e| e == "lapid") => Zipformer::load(p, &device)?,
        _ => Zipformer::from_hub(model, rest.get(1).map_or("fp16", String::as_str), &device)?,
    };
    let samples = lapid_hub::audio::load_audio_16k_mono(Path::new(audio))?;
    let t = Instant::now();
    if flags.iter().any(|f| f == "--vad") {
        let vad = Vad::from_hub("data/reference/vad/model", "fp32", &device)?;
        let opts = StreamOptions { endpoint: None, ..StreamOptions::default() };
        for seg in vad.process(&samples, &VadOptions::default())? {
            run(&z, &samples[seg.start..seg.end], &opts, seg.start_seconds(16_000))?;
        }
    } else {
        run(&z, &samples, &StreamOptions::default(), 0.0)?;
    }
    let secs = samples.len() as f64 / 16_000.0;
    eprintln!("{} : {secs:.2} s of audio in {:.2} s (RTF {:.3})", device.description(), t.elapsed().as_secs_f64(), t.elapsed().as_secs_f64() / secs);
    Ok(())
}
