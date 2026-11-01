//! Any model through one API: `lapid_hub::load` resolves the source (HF repo, directory,
//! `.gguf`, `.lapid`; converted into the lapid cache on first use) and returns an
//! `AnyModel`, whose capabilities say what it can do.
//!
//! ```text
//! cargo run --release -p lapid-hub --example run -- LiquidAI/LFM3-1.7B "Write a haiku about rust."
//! cargo run --release -p lapid-hub --example run -- openai/whisper-small data/jfk.wav
//! cargo run --release -p lapid-hub --example run -- gradium/phonon-40m-en "Hello there." out.wav
//! cargo run --release -p lapid-hub --example run -- google/embeddinggemma-300m "a cat" "a kitten"
//! ```

use std::io::Write;

use lapid::Device;
use lapid_hub::runtime::serve::{AsrParams, Capability, ChatParams, Delta, EmbedParams, TtsParams};
use lapid_hub::LoadOptions;

fn main() -> lapid::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (source, inputs) = args.split_first().expect("usage: run <model> [inputs ...]");
    let mut model = lapid_hub::load(source, &Device::best(), &LoadOptions::default())?;
    let meta = model.meta().clone();
    eprintln!("{} ({}, {}): {:?}", meta.id, meta.architecture, meta.dtype, meta.capabilities);

    if meta.has(Capability::Chat) {
        let req = ChatParams { messages: serde_json::json!([{"role": "user", "content": inputs.join(" ")}]), ..Default::default() };
        let s = model.chat(&req, &mut |d| {
            if let Delta::Content(t) = d {
                print!("{t}");
                let _ = std::io::stdout().flush();
            }
            true
        })?;
        println!("\n[{} tokens, {:.1} tok/s]", s.completion_tokens, s.decode_tps());
    } else if meta.has(Capability::Transcription) {
        for f in inputs {
            let samples = lapid_hub::audio::load_audio_16k_mono(f.as_ref())?;
            println!("{f}: {}", model.transcribe(&samples, &AsrParams::default())?.text);
        }
    } else if meta.has(Capability::Speech) {
        let mut pcm = Vec::new();
        model.speak(&inputs[0], &TtsParams::default(), &mut |chunk| {
            pcm.extend_from_slice(chunk);
            true
        })?;
        let out = inputs.get(1).map_or("speech.wav", String::as_str);
        lapid_hub::audio::write_wav(out.as_ref(), &pcm, model.sample_rate())?;
        println!("{out}: {:.2} s", pcm.len() as f64 / f64::from(model.sample_rate()));
    } else if meta.has(Capability::Embedding) {
        let e = model.embed(inputs, &EmbedParams::default())?;
        for (t, v) in inputs.iter().zip(&e.vectors) {
            println!("{t}: [{:.4}, {:.4}, ...] ({} dims)", v[0], v[1], v.len());
        }
    }
    Ok(())
}
