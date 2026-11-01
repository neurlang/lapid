//! Long-form transcription: the whole signal is featurized once (global per-feature
//! normalization, as for a single pass), the encoder runs on windows of `chunk_seconds`
//! with `context_seconds` of extra audio on each side (only the window's core frames are
//! kept, so every kept frame saw at least that much context), and one TDT decode runs over
//! the stitched encoder frames. Audio up to `chunk + 2 * context` seconds is one pass —
//! exactly the reference computation.

use lapid::{Error, Result, Tensor};
use serde::Serialize;

use super::decode::{self, Path, Segment, Token, Word};
use super::mel::Mel;
use super::model::subsampling::encoder_frames;
use super::Parakeet;

#[derive(Clone, Debug)]
pub struct TranscribeOptions {
    /// Encoder window core length for long audio (seconds).
    pub chunk_seconds: f32,
    /// Extra audio encoded on each side of a window core (seconds).
    pub context_seconds: f32,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        TranscribeOptions { chunk_seconds: 120.0, context_seconds: 15.0 }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Transcript {
    pub text: String,
    /// Sentence-level segments.
    pub segments: Vec<Segment>,
    pub words: Vec<Word>,
    pub tokens: Vec<Token>,
    /// Encoder windows used (1 = single pass).
    pub windows: usize,
}

/// Encoder windows over `frames` encoder frames: (window start, window end, core start,
/// core end) in encoder frames.
pub(crate) fn windows(frames: usize, core: usize, ctx: usize) -> Vec<(usize, usize, usize, usize)> {
    if frames <= core + 2 * ctx || core == 0 {
        return vec![(0, frames, 0, frames)];
    }
    let mut out = Vec::new();
    let mut a = 0;
    while a < frames {
        let b = (a + core).min(frames);
        out.push((a.saturating_sub(ctx), (b + ctx).min(frames), a, b));
        a = b;
    }
    out
}

/// Projected encoder frames of the whole signal, [1, T, H] realized on the model's device.
pub(crate) fn encode_long(p: &Parakeet, mel: &Mel, opts: &TranscribeOptions) -> Result<(Tensor, usize)> {
    let cfg = &p.cfg;
    let stages = cfg.subsampling_stages();
    let sub = cfg.subsampling_factor;
    let total = encoder_frames(mel.n_frames, stages);
    let per_s = 1.0 / cfg.frame_seconds() as f32;
    let core = (opts.chunk_seconds * per_s).round().max(1.0) as usize;
    let ctx = (opts.context_seconds * per_s).round().max(0.0) as usize;
    let wins = windows(total, core, ctx);
    if wins.len() == 1 {
        return Ok((p.encode_mel(mel.frames(0, mel.n_frames), mel.n_frames)?, 1));
    }
    let hdim = cfg.decoder_hidden;
    let mut host = Vec::with_capacity(total * hdim);
    for &(ws, we, cs, ce) in &wins {
        let (m0, m1) = (ws * sub, (we * sub).min(mel.n_frames));
        let enc = p.encode_mel(mel.frames(m0, m1), m1 - m0)?;
        let got = enc.shape().get(1).copied().unwrap_or(0);
        if got < ce - ws {
            return Err(Error::new("internal", format!("window [{ws}, {we}) gave {got} frames, need {}", ce - ws)));
        }
        let v = p.device.to_vec_f32(&enc)?;
        host.extend_from_slice(&v[(cs - ws) * hdim..(ce - ws) * hdim]);
    }
    let enc = p.device.upload(&Tensor::from_f32(&host, &[1, total, hdim])?)?;
    Ok((enc, wins.len()))
}

pub(crate) fn transcribe(p: &Parakeet, samples: &[f32], opts: &TranscribeOptions) -> Result<Transcript> {
    let mel = p.features.compute(samples);
    if mel.n_frames == 0 {
        return Ok(Transcript { text: String::new(), segments: Vec::new(), words: Vec::new(), tokens: Vec::new(), windows: 0 });
    }
    let (enc, windows) = encode_long(p, &mel, opts)?;
    let frames = enc.shape().get(1).copied().unwrap_or(0);
    let decoded = p.decode_frames(&enc, frames, Path::Greedy, false)?;
    let emitted = decoded.emitted(p.cfg.blank_id);
    let tokens = decode::tokens(&p.tokenizer, &emitted, p.cfg.frame_seconds(), p.cfg.blank_id);
    Ok(Transcript {
        text: decode::text(&tokens),
        segments: decode::segments(&tokens),
        words: decode::words(&tokens),
        tokens,
        windows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parakeet::Parakeet;

    #[test]
    fn window_plan() {
        assert_eq!(windows(100, 750, 125), vec![(0, 100, 0, 100)]);
        assert_eq!(windows(1000, 750, 125), vec![(0, 1000, 0, 1000)]);
        let w = windows(2000, 750, 125);
        assert_eq!(w, vec![(0, 875, 0, 750), (625, 1625, 750, 1500), (1375, 2000, 1500, 2000)]);
    }

    /// Manual check of the windowed encoder against one full-attention pass:
    /// `PARAKEET_LONG_AUDIO=<wav> [PARAKEET_MODEL=<.lapid>] [PARAKEET_DEVICE=cuda] cargo test
    /// -p lapid-hub --release chunked_vs_single_pass -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn chunked_vs_single_pass() {
        let Ok(audio) = std::env::var("PARAKEET_LONG_AUDIO") else { return };
        let dev = lapid_verify::open_device(std::env::var("PARAKEET_DEVICE").ok().as_deref()).expect("device");
        let model = std::env::var("PARAKEET_MODEL").map(std::path::PathBuf::from).unwrap_or_else(|_| {
            lapid_convert::default_model_path("nvidia/parakeet-tdt-0.6b-v3", lapid_convert::OutDType::Int8)
        });
        let p = Parakeet::load(&model, &dev).expect("load");
        let samples = lapid_nn::audio::load_audio(std::path::Path::new(&audio), 16_000).expect("audio");
        let t = std::time::Instant::now();
        let chunked = p.transcribe(&samples, &TranscribeOptions::default()).expect("chunked");
        let tc = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let single = p.transcribe(&samples, &TranscribeOptions { chunk_seconds: 1e6, context_seconds: 0.0 }).expect("single");
        let ts = t.elapsed().as_secs_f64();
        if let Ok(dir) = std::env::var("PARAKEET_DUMP") {
            std::fs::write(format!("{dir}/chunked.txt"), &chunked.text).expect("dump");
            std::fs::write(format!("{dir}/single.txt"), &single.text).expect("dump");
        }
        let (a, b): (Vec<&str>, Vec<&str>) = (chunked.text.split_whitespace().collect(), single.text.split_whitespace().collect());
        // word-level edit distance
        let mut prev: Vec<usize> = (0..=b.len()).collect();
        for (i, wa) in a.iter().enumerate() {
            let mut cur = vec![i + 1; b.len() + 1];
            for (j, wb) in b.iter().enumerate() {
                cur[j + 1] = (prev[j] + usize::from(wa != wb)).min(prev[j + 1] + 1).min(cur[j] + 1);
            }
            prev = cur;
        }
        println!(
            "audio {:.1} s: chunked {} windows {tc:.2} s, single pass {ts:.2} s; {} vs {} words, word edit distance {} ({:.2}%)",
            samples.len() as f64 / 16_000.0,
            chunked.windows,
            a.len(),
            b.len(),
            prev[b.len()],
            100.0 * prev[b.len()] as f64 / b.len().max(1) as f64
        );
    }
}
