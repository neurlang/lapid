//! Streaming speech recognition with a Zipformer2 transducer: the WAV is fed in pieces as if
//! it came from a microphone, partial results update in place, finals print at endpoints.
//! With `--vad`, Silero VAD cuts the audio into speech segments first and each segment is
//! decoded by a fresh stream.
//!
//! ```text
//! cargo run --release -p lapid-hub --example zipformer -- data/jfk.wav \
//!     [--model data/hf/zipformer2-en-streaming-2023-05-17 | model.lapid] [--dtype fp16] \
//!     [--chunk 16] [--piece-ms 100] [--vad [data/reference/vad/model]]
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use lapid::Device;
use lapid_hub::zipformer::{StreamOptions, Zipformer};
use lapid_hub::{Vad, VadOptions};

struct Args {
    audio: PathBuf,
    model: String,
    dtype: String,
    chunk: Option<usize>,
    piece_ms: usize,
    vad: Option<String>,
}

fn parse() -> Option<Args> {
    let mut it = std::env::args().skip(1);
    let mut a = Args {
        audio: PathBuf::new(),
        model: "data/hf/zipformer2-en-streaming-2023-05-17".into(),
        dtype: "fp16".into(),
        chunk: None,
        piece_ms: 100,
        vad: None,
    };
    let mut rest: Vec<String> = it.by_ref().collect();
    rest.reverse();
    while let Some(x) = rest.pop() {
        match x.as_str() {
            "--model" => a.model = rest.pop()?,
            "--dtype" => a.dtype = rest.pop()?,
            "--chunk" => a.chunk = rest.pop()?.parse().ok(),
            "--piece-ms" => a.piece_ms = rest.pop()?.parse().ok()?,
            "--vad" => {
                let next = rest.last().filter(|n| !n.starts_with("--")).cloned();
                if next.is_some() {
                    rest.pop();
                }
                a.vad = Some(next.unwrap_or_else(|| "data/reference/vad/model".into()));
            }
            _ if a.audio.as_os_str().is_empty() => a.audio = PathBuf::from(x),
            _ => return None,
        }
    }
    (!a.audio.as_os_str().is_empty()).then_some(a)
}

fn load(model: &str, dtype: &str, device: &Device) -> lapid::Result<Zipformer> {
    let p = Path::new(model);
    if p.extension().is_some_and(|e| e == "lapid") { Zipformer::load(p, device) } else { Zipformer::from_hub(model, dtype, device) }
}

/// Stream `samples` through a fresh stream; print partials (in place) and finals.
fn run(z: &Zipformer, samples: &[f32], opts: &StreamOptions, piece: usize, offset_s: f64) -> lapid::Result<()> {
    let mut s = z.stream(opts)?;
    let show = |r: &lapid_hub::StreamResult| {
        if r.is_final {
            eprint!("\r\x1b[2K");
            println!("[{:7.2}] {}", offset_s + f64::from(r.start), r.text);
        } else {
            eprint!("\r\x1b[2K{}", r.text);
        }
    };
    for p in samples.chunks(piece.max(1)) {
        for r in s.accept_waveform(p)? {
            show(&r);
        }
    }
    for r in s.input_finished()? {
        show(&r);
    }
    Ok(())
}

fn main() -> lapid::Result<()> {
    let Some(args) = parse() else {
        eprintln!("usage: zipformer <audio.wav> [--model DIR|.lapid] [--dtype fp16] [--chunk 16] [--piece-ms 100] [--vad [DIR|.lapid]]");
        std::process::exit(2);
    };
    let device = Device::best();
    eprintln!("device: {}", device.description());
    let t = Instant::now();
    let z = load(&args.model, &args.dtype, &device)?;
    eprintln!("loaded in {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let samples = lapid_hub::audio::load_audio_16k_mono(&args.audio)?;
    let sr = z.config().sample_rate as usize;
    let piece = sr * args.piece_ms / 1000;
    let t = Instant::now();
    match &args.vad {
        None => {
            let opts = StreamOptions { chunk_size: args.chunk, ..StreamOptions::default() };
            run(&z, &samples, &opts, piece, 0.0)?;
        }
        Some(vad_model) => {
            let vad = if vad_model.ends_with(".lapid") { Vad::load(Path::new(vad_model), &device)? } else { Vad::from_hub(vad_model, "fp32", &device)? };
            let segments = vad.process(&samples, &VadOptions::default())?;
            eprintln!("vad: {} speech segments", segments.len());
            let opts = StreamOptions { chunk_size: args.chunk, left_context_frames: None, endpoint: None };
            for seg in segments {
                run(&z, &samples[seg.start..seg.end], &opts, piece, seg.start as f64 / sr as f64)?;
            }
        }
    }
    let secs = samples.len() as f64 / sr as f64;
    let el = t.elapsed().as_secs_f64();
    eprintln!("{:.2} s of audio in {:.2} s (RTF {:.3})", secs, el, el / secs);
    Ok(())
}
