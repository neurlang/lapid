//! OpenAI Whisper speech recognition: model graphs, tokenizer, OpenAI decoding semantics
//! (greedy, timestamp rules, long-form seek loop), conversion from Hugging Face and
//! whisper.cpp checkpoints, and the verification / benchmark wiring.

pub mod bench;
pub mod cmd;
pub mod convert;
mod decode;
pub(crate) mod mel;
mod model;
mod tokenizer;
mod transcribe;
pub mod verify;

use std::path::Path;

use lapid::{Device, LapidFile, Result, Tensor};
use serde::Serialize;

use crate::audio;
use crate::model::{Bytes, Model, ModelInfo, hub_model_path};
use model::config::WhisperConfig;
use model::decoder::Decoder;
use model::encoder::Encoder;
use model::first;
pub use lapid_nn::Precision;
use lapid_nn::audio::MelSpectrogram;
use lapid_nn::{Loader, Tokenizer};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    Transcribe,
    Translate,
}

#[derive(Clone, Debug)]
pub struct TranscribeOptions {
    /// Language code ("en", "de", ...); `None` = detect from the first 30 s.
    pub language: Option<String>,
    pub task: Task,
    /// Timestamp tokens / segment-level timing (OpenAI default: on).
    pub timestamps: bool,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        TranscribeOptions { language: None, task: Task::Transcribe, timestamps: true }
    }
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
    pub language: String,
    pub segments: Vec<Segment>,
}

/// A loaded Whisper model bound to a device.
pub struct Whisper {
    device: Device,
    info: ModelInfo,
    cfg: WhisperConfig,
    tokenizer: Tokenizer,
    encoder: Encoder,
    decoder: Decoder,
    mel: MelSpectrogram,
    /// SuppressTokens list (non-speech set + special tokens), sorted.
    suppress: Vec<u32>,
    /// SuppressBlank list (`" "` + eot).
    suppress_blank: Vec<u32>,
}

impl Whisper {
    /// Load `repo` (e.g. "openai/whisper-small") from the lapid cache, converting it from the
    /// Hugging Face hub into `lapid_convert::default_model_path` first if it is missing.
    pub fn from_hub(repo: &str, dtype: &str, device: &Device) -> Result<Whisper> {
        let path = hub_model_path(repo, dtype, convert::convert_hf)?;
        Whisper::load(&path, device)
    }

    /// Load a converted `.lapid` Whisper model and upload its weights to `device`.
    pub fn load(path: &Path, device: &Device) -> Result<Whisper> {
        Whisper::load_with(path, device, Precision::Stored)
    }

    /// Load a `.lapid` Whisper image from memory (zero copy until upload), e.g.
    /// `Arc::new(std::fs::read(path)?)` or `Arc::new(include_bytes!("whisper.lapid").as_slice())`.
    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Whisper> {
        Whisper::from_file(LapidFile::from_bytes(bytes)?, device, Precision::Stored)
    }

    /// Load with a given weight precision. The fake-quant modes build the reference models
    /// quantized kernels are checked against (same weights, f32 arithmetic).
    pub fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Whisper> {
        Whisper::from_file(LapidFile::open(path)?, device, precision)
    }

    fn from_file(file: LapidFile, device: &Device, precision: Precision) -> Result<Whisper> {
        let cfg = WhisperConfig::from_metadata(&file.metadata)?;
        let tokenizer = tokenizer::from_strings(&convert::load_token_strings(&file)?, cfg.special.eot)?;
        let loader = Loader::new(&file, device, precision);
        let encoder = Encoder::load(&loader, &cfg)?;
        let decoder = Decoder::load(&loader, &cfg)?;
        let (suppress, suppress_blank) = suppress_lists(&cfg, &tokenizer);
        Ok(Whisper {
            device: device.clone(),
            info: ModelInfo::from_file(&file, "automatic-speech-recognition"),
            mel: mel::whisper_mel(cfg.n_mels)?,
            cfg,
            tokenizer,
            encoder,
            decoder,
            suppress,
            suppress_blank,
        })
    }

    pub fn transcribe_file(&self, path: &Path, opts: &TranscribeOptions) -> Result<Transcript> {
        let samples = audio::load_audio_16k_mono(path)?;
        self.transcribe(&samples, opts)
    }

    /// Transcribe 16 kHz mono samples (any length).
    pub fn transcribe(&self, samples_16k_mono: &[f32], opts: &TranscribeOptions) -> Result<Transcript> {
        transcribe::transcribe(self, samples_16k_mono, opts)
    }

    /// Log-mel of the first 30 s (zero padded): [n_mels * 3000], HF WhisperFeatureExtractor.
    pub fn log_mel(&self, samples_16k_mono: &[f32]) -> Vec<f32> {
        mel::log_mel_window(samples_16k_mono, &self.mel)
    }

    /// Run the encoder on a [n_mels * 3000] log-mel window; realized [1, n_audio_ctx, d].
    pub fn encode(&self, mel: &[f32]) -> Result<Tensor> {
        let (out, _) = self.encoder.graph(mel)?;
        first(self.device.eval(&[&out])?)
    }

    /// Encoder with intermediates: "encoder.embed", "encoder.layers.{i}", "encoder.out".
    pub fn encode_traced(&self, mel: &[f32]) -> Result<Vec<(String, Vec<f32>)>> {
        let (_, traced) = self.encoder.graph(mel)?;
        let refs: Vec<&Tensor> = traced.iter().collect();
        let realized = self.device.eval(&refs)?;
        traced
            .iter()
            .zip(&realized)
            .map(|(lazy, t)| {
                let name = lazy.name().unwrap_or("encoder.unnamed").to_string();
                Ok((name, self.device.to_vec_f32(t)?))
            })
            .collect()
    }

    /// Full (uncached) decoder pass: logits for every position, [tokens.len() * n_vocab].
    pub fn decoder_logits(&self, encoded: &Tensor, tokens: &[u32]) -> Result<Vec<f32>> {
        let cross = self.decoder.cross_kv_graph(encoded)?;
        let mut cache = self.decoder.new_cache();
        let logits = self.decoder.forward(tokens, &mut cache, &cross, None)?;
        let realized = first(self.device.eval(&[&logits])?)?;
        self.device.to_vec_f32(&realized)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl Model for Whisper {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

/// OpenAI `_get_suppress_tokens` + SuppressBlank token lists.
fn suppress_lists(cfg: &WhisperConfig, tok: &Tokenizer) -> (Vec<u32>, Vec<u32>) {
    let sp = &cfg.special;
    let to_u32 = |v: &i64| u32::try_from(*v).ok();
    let mut suppress: Vec<u32> = cfg.suppress_tokens.iter().filter_map(to_u32).collect();
    if cfg.suppress_tokens.is_empty() || cfg.suppress_tokens.contains(&-1) {
        suppress.extend(tokenizer::non_speech_tokens(tok));
    }
    suppress.extend([sp.transcribe, sp.translate, sp.sot, sp.prev]);
    suppress.extend(sp.sot_lm);
    suppress.extend(sp.no_speech);
    suppress.retain(|&t| (t as usize) < cfg.n_vocab);
    suppress.sort_unstable();
    suppress.dedup();

    let mut blank = cfg.begin_suppress_tokens.clone();
    if blank.is_empty() {
        blank.extend(tok.token_for_bytes(b" "));
        blank.push(sp.eot);
    }
    (suppress, blank)
}
