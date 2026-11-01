//! Streaming text to speech with Phonon: PCM chunks arrive as each 80 ms codec frame is
//! decoded (a player would start playing at the first one); writes a WAV file and prints the
//! time to first audio and the real-time factor.
//!
//! ```text
//! cargo run --release -p lapid-hub --example phonon -- "Hello from Phonon." [out.wav] [voice.safetensors]
//! ```
//! The optional voice file is a `speaker_wavs` safetensors (`gradium/phonon-voices` format).
//! `LAPID_DEVICE=cpu|cuda|vulkan|metal` picks the device (default: best available).

use std::path::Path;
use std::time::Instant;

use lapid::Device;
use lapid_hub::{Phonon, SpeakOptions};

fn main() -> lapid::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let text = args.first().map_or("Hello from Phonon, streaming speech on lapid.", String::as_str);
    let out = Path::new(args.get(1).map_or("speech.wav", String::as_str));

    let device = Device::best();
    let mut tts = Phonon::from_hub("gradium/phonon-40m-en", "fp16", &device)?;
    let mut opts = SpeakOptions { seed: Some(0), ..Default::default() };
    if let Some(file) = args.get(2) {
        tts.add_voice_safetensors("custom", Path::new(file))?;
        opts.voice = Some("custom".into());
    }

    let t = Instant::now();
    let (mut pcm, mut first_ms) = (Vec::new(), None);
    for chunk in tts.speak_stream(text, &opts)? {
        first_ms.get_or_insert(t.elapsed().as_secs_f64() * 1e3);
        pcm.extend(chunk?);
    }
    let secs = pcm.len() as f64 / f64::from(tts.sample_rate());
    let total = t.elapsed().as_secs_f64();
    lapid_hub::audio::write_wav(out, &pcm, tts.sample_rate())?;
    eprintln!(
        "{secs:.2} s of audio -> {} | first audio {:.0} ms | real-time factor {:.3}",
        out.display(),
        first_ms.unwrap_or(0.0),
        total / secs.max(1e-9)
    );
    Ok(())
}
