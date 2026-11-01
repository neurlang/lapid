//! NVIDIA Parakeet TDT speech recognition (FastConformer encoder + token-and-duration
//! transducer), e.g. `nvidia/parakeet-tdt-0.6b-v3` (25 European languages, punctuation and
//! capitalization, word timestamps), on the shared transducer pipeline.
//!
//! ```no_run
//! let device = lapid::Device::best();
//! let model = lapid_hub::Parakeet::from_hub("nvidia/parakeet-tdt-0.6b-v3", "int8", &device)?;
//! let out = model.transcribe_file("data/jfk.wav".as_ref(), &Default::default())?;
//! for w in &out.words {
//!     println!("{:6.2} {:6.2} {}", w.start, w.end, w.word);
//! }
//! # Ok::<(), lapid::Error>(())
//! ```
//!
//! [`mel`] (NeMo log-mel) -> [`model`] (conv subsampling + FastConformer, LSTM prediction
//! network, joint) -> `runtime::transducer` greedy TDT decoding -> tokens, words, segments.
//! Long audio is encoded in overlapping windows (`transducer::context_windows`).

pub mod config;
pub mod convert;
pub mod mel;
pub(crate) mod model;

use std::path::Path;
use std::time::Instant;

use lapid::{Device, Error, LapidFile, Result, Tensor};
use lapid_convert::engine::ConvertSpec;
use lapid_nn::{Loader, Tokenizer};
use lapid_verify::Descriptor;
use serde::Serialize;
use serde_json::json;

pub use config::ParakeetConfig;
pub use lapid_nn::Precision;
pub use crate::runtime::transducer::Step;
use crate::model::{Bytes, Model, ModelInfo, hub_model_path};
use crate::runtime::audio::{Argmax, AsrModel, AudioDef, BenchRun, Check, EndToEnd, Row, Sample, Stages, asr_entry};
use crate::runtime::serve::{AsrParams, AsrSegment, AsrWord, TranscriptOut};
use crate::runtime::transducer::{GreedyOptions, Transducer, context_windows, forced_decode, greedy_decode};
use mel::{Features, Mel};
use model::decoder::{Joint, LstmState, Prediction};
use model::encoder::Encoder;

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
pub struct Token {
    pub id: u32,
    /// The SentencePiece piece (`▁` marks a word start).
    pub piece: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Word {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub tokens: Vec<u32>,
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

/// A loaded Parakeet TDT model bound to a device.
pub struct Parakeet {
    device: Device,
    info: ModelInfo,
    cfg: ParakeetConfig,
    tokenizer: Tokenizer,
    features: Features,
    encoder: Encoder,
    prediction: Prediction,
    joint: Joint,
}

impl Parakeet {
    /// Load `repo` (e.g. "nvidia/parakeet-tdt-0.6b-v3") from the lapid cache, converting it
    /// from the Hugging Face hub first if it is missing.
    pub fn from_hub(repo: &str, dtype: &str, device: &Device) -> Result<Parakeet> {
        Parakeet::load(&hub_model_path(repo, dtype, |r, d, o| convert::CONVERT.convert(r, d, o))?, device)
    }

    pub fn load(path: &Path, device: &Device) -> Result<Parakeet> {
        Parakeet::load_with(path, device, Precision::Stored)
    }

    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Parakeet> {
        Parakeet::from_file(&LapidFile::from_bytes(bytes)?, device, Precision::Stored)
    }

    /// Load with a weight precision (the fake-quant modes build verification references).
    pub fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Parakeet> {
        Parakeet::from_file(&LapidFile::open(path)?, device, precision)
    }

    fn from_file(file: &LapidFile, device: &Device, precision: Precision) -> Result<Parakeet> {
        let cfg = ParakeetConfig::from_metadata(&file.metadata)?;
        let name = file.metadata.pointer("/parakeet/tokenizer_blob").and_then(serde_json::Value::as_str).unwrap_or(config::TOKENIZER_BLOB);
        let blob = file.blob(name).ok_or_else(|| Error::new("metadata", format!("parakeet: tokenizer blob {name:?} missing")))?;
        let json = std::str::from_utf8(blob).map_err(|e| Error::new("tokenizer", format!("tokenizer.json: {e}")))?;
        let loader = Loader::new(file, device, precision);
        Ok(Parakeet {
            device: device.clone(),
            info: ModelInfo::from_file(file, "automatic-speech-recognition"),
            tokenizer: Tokenizer::from_tokenizer_json(json)?,
            features: Features::new(&cfg)?,
            encoder: Encoder::load(&loader, &cfg)?,
            prediction: Prediction::load(&loader, &cfg)?,
            joint: Joint::load(&loader, &cfg)?,
            cfg,
        })
    }

    pub fn transcribe_file(&self, path: &Path, opts: &TranscribeOptions) -> Result<Transcript> {
        self.transcribe(&lapid_nn::audio::load_audio(path, self.cfg.sample_rate)?, opts)
    }

    /// Transcribe mono samples at `config().sample_rate` (16 kHz), any length.
    pub fn transcribe(&self, samples: &[f32], opts: &TranscribeOptions) -> Result<Transcript> {
        let mel = self.features(samples);
        if mel.n_frames == 0 {
            return Ok(Transcript { text: String::new(), segments: Vec::new(), words: Vec::new(), tokens: Vec::new(), windows: 0 });
        }
        let (enc, windows) = self.encode_long(&mel, opts)?;
        let tokens = self.tokens(&greedy_decode(self, &enc, &self.greedy())?.steps);
        Ok(Transcript { text: text(&tokens), segments: segments(&tokens), words: words(&tokens), tokens, windows })
    }

    /// TDT steps advance a few frames and emit often: the 8198-wide joint one frame per
    /// evaluation with the argmax on the host does the least work.
    fn greedy(&self) -> GreedyOptions {
        GreedyOptions { max_symbols_per_frame: self.cfg.max_symbols_per_step, joint_window: 1, host_argmax: true, ..GreedyOptions::default() }
    }

    /// Normalized log-mel features of a whole signal (`[T, n_mels]`, valid frames only).
    pub fn features(&self, samples: &[f32]) -> Mel {
        self.features.compute(samples)
    }

    /// Encoder on mel `[frames, n_mels]`: projected frames `[T', joint_hidden]`, realized.
    pub fn encode_mel(&self, mel: &[f32], frames: usize) -> Result<Tensor> {
        let g = self.encoder.graph(mel, frames, false)?;
        let t = g.projected.shape()[1];
        Ok(self.device.eval(&[&g.projected.reshape(&[t, self.cfg.decoder_hidden])?])?.remove(0))
    }

    /// Encoder intermediates: "encoder.subsampling", "encoder.layers.{i}", "encoder.projected".
    pub fn encode_traced(&self, mel: &[f32], frames: usize) -> Result<Vec<(String, Vec<f32>)>> {
        let g = self.encoder.graph(mel, frames, true)?;
        let realized = self.device.eval(&g.traced.iter().collect::<Vec<_>>())?;
        g.traced.iter().zip(&realized).map(|(l, t)| Ok((l.name().unwrap_or("encoder").to_string(), self.device.to_vec_f32(t)?))).collect()
    }

    /// Projected encoder frames of a whole signal `[T, H]`: one pass for audio up to
    /// `chunk + 2 context` seconds, else windows whose core frames are stitched.
    fn encode_long(&self, mel: &Mel, opts: &TranscribeOptions) -> Result<(Tensor, usize)> {
        let cfg = &self.cfg;
        let total = model::subsampling::encoder_frames(mel.n_frames, cfg.subsampling_stages());
        let per_s = 1.0 / cfg.frame_seconds() as f32;
        let core = (opts.chunk_seconds * per_s).round().max(1.0) as usize;
        let wins = context_windows(total, core, (opts.context_seconds * per_s).round().max(0.0) as usize);
        if wins.len() == 1 {
            return Ok((self.encode_mel(mel.frames(0, mel.n_frames), mel.n_frames)?, 1));
        }
        let (h, sub) = (cfg.decoder_hidden, cfg.subsampling_factor);
        let mut host = Vec::with_capacity(total * h);
        for &(ws, we, cs, ce) in &wins {
            let (m0, m1) = (ws * sub, (we * sub).min(mel.n_frames));
            let enc = self.encode_mel(mel.frames(m0, m1), m1 - m0)?;
            if enc.shape()[0] < ce - ws {
                return Err(Error::new("internal", format!("window [{ws}, {we}) gave {} frames, need {}", enc.shape()[0], ce - ws)));
            }
            host.extend_from_slice(&self.device.to_vec_f32(&enc)?[(cs - ws) * h..(ce - ws) * h]);
        }
        Ok((self.device.upload(&Tensor::from_f32(&host, &[total, h])?)?, wins.len()))
    }

    /// Token offsets in seconds (`start = frame`, `end = frame + duration`); punctuation takes
    /// the previous token's end (NeMo `_refine_timestamps_tdt`); special tokens skipped.
    fn tokens(&self, emitted: &[Step]) -> Vec<Token> {
        const PUNCTUATION: [&str; 11] = ["?", "'", "¡", "¿", "-", ":", ",", "%", "/", ".", "!"];
        let fs = self.cfg.frame_seconds();
        let mut out: Vec<Token> = Vec::new();
        for s in emitted.iter().filter(|s| s.token != self.cfg.blank_id && !self.tokenizer.is_special(s.token)) {
            let piece = self.tokenizer.id_to_token(s.token).unwrap_or("").to_string();
            let (mut start, mut end) = (s.start(fs), s.end(fs));
            if let Some(prev) = out.last().filter(|_| PUNCTUATION.contains(&piece.as_str())) {
                (start, end) = (prev.end, prev.end);
            }
            out.push(Token { id: s.token, piece, start, end });
        }
        out
    }

    pub fn config(&self) -> &ParakeetConfig {
        &self.cfg
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

/// Pieces -> text (Metaspace: `▁` is a space, the leading one dropped).
fn text(tokens: &[Token]) -> String {
    tokens.iter().map(|t| t.piece.as_str()).collect::<String>().replace('▁', " ").trim().to_string()
}

/// Words: a new word starts at every piece beginning with `▁`.
fn words(tokens: &[Token]) -> Vec<Word> {
    let mut out: Vec<Word> = Vec::new();
    for t in tokens {
        let body = t.piece.trim_start_matches('▁');
        match out.last_mut() {
            Some(w) if !t.piece.starts_with('▁') => {
                w.word.push_str(body);
                w.end = t.end;
            }
            _ => out.push(Word { word: body.to_string(), start: t.start, end: t.end }),
        }
    }
    out.retain(|w| !w.word.is_empty());
    out
}

/// Segments ending at words whose text ends with `.`, `?` or `!` (NeMo `segment_seperators`).
fn segments(tokens: &[Token]) -> Vec<Segment> {
    let (mut out, mut cur) = (Vec::new(), Vec::<Token>::new());
    for (i, t) in tokens.iter().enumerate() {
        cur.push(t.clone());
        let next_starts_word = tokens.get(i + 1).is_none_or(|n| n.piece.starts_with('▁'));
        if (t.piece.trim_end().ends_with(['.', '?', '!']) && next_starts_word) || i + 1 == tokens.len() {
            let (start, end) = (cur[0].start, cur[cur.len() - 1].end);
            out.push(Segment { start, end, text: text(&cur), tokens: cur.iter().map(|t| t.id).collect() });
            cur.clear();
        }
    }
    out
}

impl Model for Parakeet {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

impl Transducer for Parakeet {
    fn device(&self) -> &Device {
        &self.device
    }

    fn blank_id(&self) -> u32 {
        self.cfg.blank_id
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }

    fn durations(&self) -> &[usize] {
        &self.cfg.durations
    }

    fn frame_seconds(&self) -> f64 {
        self.cfg.frame_seconds()
    }

    fn predictor_init(&self) -> Result<Vec<Tensor>> {
        Ok(self.prediction.zero_state().tensors().into_iter().cloned().collect())
    }

    fn predict(&self, _history: &[u32], token: u32, state: &[Tensor]) -> Result<(Tensor, Vec<Tensor>)> {
        let (out, next) = self.prediction.step(token, &LstmState::from_tensors(state.to_vec(), self.prediction.n_layers()))?;
        Ok((out, next.tensors().into_iter().cloned().collect()))
    }

    fn joint(&self, enc: &Tensor, pred: &Tensor) -> Result<Tensor> {
        self.joint.logits(enc, pred)
    }
}

impl AsrModel for Parakeet {
    fn recognize(&self, samples: &[f32], _: &AsrParams) -> Result<TranscriptOut> {
        let t = self.transcribe(samples, &TranscribeOptions::default())?;
        Ok(TranscriptOut {
            text: t.text.trim().to_string(),
            language: None,
            segments: t.segments.into_iter().map(|s| AsrSegment { start: s.start, end: s.end, text: s.text, tokens: s.tokens }).collect(),
            words: t.words.into_iter().map(|w| AsrWord { word: w.word, start: w.start, end: w.end }).collect(),
        })
    }
}

impl AudioDef for Parakeet {
    const DESCRIPTOR: Descriptor = Descriptor {
        name: "parakeet",
        default_model: "nvidia/parakeet-tdt-0.6b-v3",
        default_dtype: "int8",
        reference_dir: "data/reference/parakeet",
        reference_script: "tools/reference/parakeet_ref.py",
        reference_label: "PyTorch fp32 (HF ParakeetForTDT)",
        strict_dtypes: &["fp32", "fp16", "int8"],
        bench_prompt_tokens: 0,
        bench_gen_tokens: 0,
    };
    const CONVERT: &'static ConvertSpec = &convert::CONVERT;

    fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Self> {
        Parakeet::load_with(path, device, precision)
    }

    fn sample_rate(&self) -> u32 {
        self.cfg.sample_rate
    }

    /// Mel from the audio; the encoder on the reference mel; the joint logits of every step
    /// of the reference path (frames, tokens, durations) on the reference encoding.
    fn stages(&self, s: &Sample, audio: &[f32]) -> Result<Stages> {
        let mut rows = vec![Row::front("mel", self.features(audio).data, 1e-3)];
        let frames = s.acts.shape("mel")?.first().copied().unwrap_or(0);
        rows.extend(self.encode_traced(&s.get("mel")?, frames)?.into_iter().map(|(n, v)| Row::new(n, v)));
        let (f, t, d) = (s.u32s("/steps/frames")?, s.u32s("/steps/tokens")?, s.u32s("/steps/durations")?);
        let path: Vec<Step> = (0..f.len().min(t.len()).min(d.len())).map(|i| Step { frame: f[i] as usize, token: t[i], duration: d[i] as usize }).collect();
        let enc = s.get("encoder.projected")?;
        let h = self.cfg.decoder_hidden;
        let enc = self.device.upload(&Tensor::from_f32(&enc, &[enc.len() / h, h])?)?;
        rows.push(Row::new("decoder.step_logits", forced_decode(self, &enc, &path, true)?.logits.concat()));
        let argmax = vec![Argmax { row: "decoder.step_logits".into(), width: self.cfg.joint_dim(), splits: vec![self.cfg.vocab_size] }];
        Ok(Stages { rows, argmax, checks: vec![] })
    }

    fn end_to_end(&self, s: &Sample, audio: &[f32]) -> Result<EndToEnd> {
        let t = self.transcribe(audio, &TranscribeOptions::default())?;
        let ours: Vec<u32> = t.tokens.iter().map(|t| t.id).collect();
        let want: Vec<u32> = s.u32s("/tokens")?.into_iter().filter(|&t| t != self.cfg.blank_id && !self.tokenizer.is_special(t)).collect();
        let reference_text = s.text("text");
        let words_match = s.result["words"].as_array().is_some_and(|r| {
            r.len() == t.words.len()
                && r.iter().zip(&t.words).all(|(r, w)| {
                    let f = |k: &str| r[k].as_f64().unwrap_or(f64::NAN);
                    r["word"].as_str() == Some(w.word.as_str()) && (f("start") - f64::from(w.start)).abs() < 1e-3 && (f("end") - f64::from(w.end)).abs() < 1e-3
                })
        });
        let checks = vec![
            Check::new("tokens_match", ours == want, s.strict),
            Check::new("text_match", t.text == reference_text, s.strict),
            Check::new("words_match", words_match, false),
        ];
        Ok(EndToEnd { text: t.text, reference_text, checks, info: json!({"windows": t.windows}).as_object().cloned().unwrap_or_default() })
    }

    /// `features` (log-mel), `encode` (all encoder windows), `decode` (greedy TDT), `total`.
    fn bench_run(&self, audio: &[f32]) -> Result<BenchRun> {
        let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let mel = self.features(audio);
        let features = ms(t);
        let t = Instant::now();
        let (enc, windows) = self.encode_long(&mel, &TranscribeOptions::default())?;
        let encode = ms(t);
        let t = Instant::now();
        let tokens = self.tokens(&greedy_decode(self, &enc, &self.greedy())?.steps);
        let decode = ms(t);
        Ok(BenchRun {
            times: vec![("features", features), ("encode", encode), ("decode", decode), ("total", features + encode + decode)],
            values: vec![("tokens", tokens.len() as f64), ("encoder_windows", windows as f64)],
            text: text(&tokens),
        })
    }
}

pub static ENTRY: crate::runtime::ModelEntry = asr_entry::<Parakeet>();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_and_segments() {
        let tk = |piece: &str, start: f32, end: f32| Token { id: 0, piece: piece.into(), start, end };
        let t = vec![tk("▁And", 0.24, 0.4), tk("▁so", 0.56, 0.8), tk(",", 0.8, 0.8), tk("▁ask", 1.0, 1.2), tk("▁no", 1.2, 1.3), tk("t", 1.3, 1.5), tk(".", 1.5, 1.5), tk("▁Yes", 2.0, 2.3)];
        assert_eq!(text(&t), "And so, ask not. Yes");
        let w = words(&t);
        assert_eq!(w.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(), ["And", "so,", "ask", "not.", "Yes"]);
        assert_eq!((w[3].start, w[3].end), (1.2, 1.5));
        let s = segments(&t);
        assert_eq!((s.len(), s[0].text.as_str(), s[1].start, s[1].end), (2, "And so, ask not.", 2.0, 2.3));
    }
}
