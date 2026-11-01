//! OpenAI Whisper speech recognition on the shared seq2seq pipeline: the network
//! ([`model`]), log-mel front end ([`mel`]), tokenizer, OpenAI decoding rules and the
//! long-form seek loop ([`transcribe`]), conversion ([`convert`]) and the verification /
//! benchmark descriptor (generic harness, `runtime::audio`).

pub mod config;
pub mod convert;
pub(crate) mod mel;
mod model;
mod tokenizer;
mod transcribe;

use std::path::Path;
use std::time::Instant;

use lapid::{Device, LapidFile, Result, Tensor};
use lapid_convert::engine::ConvertSpec;
use lapid_nn::audio::MelSpectrogram;
use lapid_nn::{Loader, Tokenizer};
use lapid_verify::Descriptor;
use serde::Serialize;
use serde_json::json;

use crate::model::{Bytes, Model, ModelInfo, hub_model_path};
use crate::runtime::audio::{Argmax, AsrModel, AudioDef, BenchRun, Check, EndToEnd, Row, Sample, Stages, asr_entry};
use crate::runtime::seq2seq::Seq2Seq;
use crate::runtime::serve::{AsrParams, TranscriptOut};
/// A transcript segment (seconds, text, tokens incl. timestamps).
pub use crate::runtime::serve::AsrSegment as Segment;
use config::WhisperConfig;
pub use lapid_nn::Precision;
use lapid_nn::KvCache;
pub use model::CrossKv;
use model::{Decoder, Encoder};

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
    /// Hugging Face hub first if it is missing.
    pub fn from_hub(repo: &str, dtype: &str, device: &Device) -> Result<Whisper> {
        Whisper::load(&hub_model_path(repo, dtype, |r, d, o| convert::CONVERT.convert(r, d, o))?, device)
    }

    /// Load a converted `.lapid` Whisper model and upload its weights to `device`.
    pub fn load(path: &Path, device: &Device) -> Result<Whisper> {
        Whisper::load_with(path, device, Precision::Stored)
    }

    /// Load a `.lapid` image from memory (zero copy until upload).
    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Whisper> {
        Whisper::from_file(LapidFile::from_bytes(bytes)?, device, Precision::Stored)
    }

    /// Load with a weight precision (the fake-quant modes build verification references).
    pub fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Whisper> {
        Whisper::from_file(LapidFile::open(path)?, device, precision)
    }

    fn from_file(file: LapidFile, device: &Device, precision: Precision) -> Result<Whisper> {
        let cfg = WhisperConfig::from_metadata(&file.metadata)?;
        let tokenizer = tokenizer::from_strings(&convert::load_token_strings(&file)?, cfg.special.eot)?;
        let loader = Loader::new(&file, device, precision);
        let (suppress, suppress_blank) = tokenizer::suppress_lists(&cfg, &tokenizer);
        Ok(Whisper {
            device: device.clone(),
            info: ModelInfo::from_file(&file, "automatic-speech-recognition"),
            mel: mel::whisper_mel(cfg.n_mels)?,
            encoder: Encoder::load(&loader, &cfg)?,
            decoder: Decoder::load(&loader, &cfg)?,
            cfg,
            tokenizer,
            suppress,
            suppress_blank,
        })
    }

    pub fn transcribe_file(&self, path: &Path, opts: &TranscribeOptions) -> Result<Transcript> {
        self.transcribe(&crate::audio::load_audio_16k_mono(path)?, opts)
    }

    /// Transcribe 16 kHz mono samples (any length).
    pub fn transcribe(&self, samples_16k_mono: &[f32], opts: &TranscribeOptions) -> Result<Transcript> {
        transcribe::transcribe(self, samples_16k_mono, opts)
    }

    /// Log-mel of the first 30 s (zero padded): `[n_mels * 3000]`, HF WhisperFeatureExtractor.
    pub fn log_mel(&self, samples_16k_mono: &[f32]) -> Vec<f32> {
        mel::log_mel_window(samples_16k_mono, &self.mel)
    }

    /// The encoder on a `[n_mels * 3000]` log-mel window: realized `[1, n_audio_ctx, d]`.
    pub fn encode(&self, mel: &[f32]) -> Result<Tensor> {
        let (out, _) = self.encoder.graph(mel)?;
        Ok(self.device.eval(&[&out])?.remove(0))
    }

    /// Encoder intermediates: "encoder.embed", "encoder.layers.{i}", "encoder.out".
    pub fn encode_traced(&self, mel: &[f32]) -> Result<Vec<(String, Vec<f32>)>> {
        let (_, traced) = self.encoder.graph(mel)?;
        let realized = self.device.eval(&traced.iter().collect::<Vec<_>>())?;
        traced.iter().zip(&realized).map(|(l, t)| Ok((l.name().unwrap_or("encoder").to_string(), self.device.to_vec_f32(t)?))).collect()
    }

    /// Uncached decoder pass over `tokens`: logits of every position, `[tokens.len() * n_vocab]`.
    pub fn decoder_logits(&self, encoded: &Tensor, tokens: &[u32]) -> Result<Vec<f32>> {
        let cross = self.decoder.cross_kv(encoded)?;
        let logits = self.decoder.forward(tokens, &mut self.decoder.new_cache(), &cross, None)?;
        self.device.to_vec_f32(&self.device.eval(&[&logits])?.remove(0))
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

impl Seq2Seq for Whisper {
    /// A `[n_mels * 3000]` log-mel window.
    type Input = [f32];
    /// Realized cross-attention K/V of the window.
    type Memory = CrossKv;
    type Cache = KvCache;

    fn encode(&self, mel: &[f32]) -> Result<CrossKv> {
        let (enc, _) = self.encoder.graph(mel)?;
        let lazy = self.decoder.cross_kv(&enc)?;
        let mut kv = self.device.eval(&lazy.k.iter().chain(&lazy.v).collect::<Vec<_>>())?;
        let v = kv.split_off(lazy.k.len());
        Ok(CrossKv { k: kv, v })
    }

    fn new_cache(&self, _: &CrossKv) -> Result<KvCache> {
        Ok(self.decoder.new_cache())
    }

    fn cache_len(&self, cache: &KvCache) -> usize {
        cache.offset()
    }

    fn max_positions(&self) -> usize {
        self.decoder.n_ctx()
    }

    fn decode(&self, ids: &[u32], cache: &mut KvCache, memory: &CrossKv, rows: &[usize]) -> Result<Vec<f32>> {
        self.decoder.step(&self.device, ids, cache, memory, rows)
    }
}

impl AsrModel for Whisper {
    fn recognize(&self, samples: &[f32], opts: &AsrParams) -> Result<TranscriptOut> {
        let language = opts.language.clone().filter(|l| !l.is_empty() && l != "auto");
        let task = if opts.translate { Task::Translate } else { Task::Transcribe };
        let t = self.transcribe(samples, &TranscribeOptions { language, task, timestamps: true })?;
        Ok(TranscriptOut { text: t.text.trim().to_string(), language: Some(t.language), segments: t.segments, words: Vec::new() })
    }
}

/// Tolerance of our log-mel spectrogram (independent of the model dtype).
const MEL_TOLERANCE: f32 = 1e-3;

impl AudioDef for Whisper {
    const DESCRIPTOR: Descriptor = Descriptor {
        name: "whisper",
        default_model: "openai/whisper-small",
        default_dtype: "int8",
        reference_dir: "data/reference/whisper-small",
        reference_script: "tools/reference/whisper_ref.py",
        reference_label: "PyTorch fp32 (HF WhisperForConditionalGeneration)",
        strict_dtypes: &["fp32", "fp16", "int8"],
        bench_prompt_tokens: 0,
        bench_gen_tokens: 0,
    };
    const CONVERT: &'static ConvertSpec = &convert::CONVERT;

    fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Self> {
        Whisper::load_with(path, device, precision)
    }

    fn sample_rate(&self) -> u32 {
        mel::SAMPLE_RATE as u32
    }

    /// Mel from the audio; the encoder on the reference mel; the prompt logits of the
    /// decoder on the reference encoding.
    fn stages(&self, s: &Sample, audio: &[f32]) -> Result<Stages> {
        let mut rows = vec![Row::front("mel", self.log_mel(audio), MEL_TOLERANCE)];
        rows.extend(self.encode_traced(&s.get("mel")?)?.into_iter().map(|(n, v)| Row::new(n, v)));
        let enc = s.get("encoder.out")?;
        let d = self.cfg.n_audio_state;
        let enc = self.device.upload(&Tensor::from_f32(&enc, &[1, enc.len() / d, d])?)?;
        rows.push(Row::new("decoder.logits.prompt", self.decoder_logits(&enc, &s.u32s("/prompt_tokens")?)?));
        Ok(Stages { rows, argmax: vec![Argmax { row: "decoder.logits.prompt".into(), width: self.cfg.n_vocab, splits: vec![] }], checks: vec![] })
    }

    fn end_to_end(&self, s: &Sample, audio: &[f32]) -> Result<EndToEnd> {
        let opts = TranscribeOptions { language: Some("en".into()), ..TranscribeOptions::default() };
        let t = self.transcribe(audio, &opts)?;
        let eot = self.cfg.special.eot;
        let ours: Vec<u32> = t.segments.iter().flat_map(|s| s.tokens.iter().copied()).filter(|&t| t < eot).collect();
        let want: Vec<u32> = s.u32s("/tokens")?.into_iter().filter(|&t| t < eot).collect();
        let (text, reference_text) = (t.text.trim().to_string(), s.text("text"));
        let checks = vec![Check::new("tokens_match", ours == want, s.strict), Check::new("text_match", text == reference_text, s.strict)];
        let info = json!({"language": t.language, "tokens": ours, "reference_tokens": want});
        Ok(EndToEnd { text, reference_text, checks, info: info.as_object().cloned().unwrap_or_default() })
    }

    /// `mel` (log-mel), `encode` (encoder + cross K/V), `total` (a full `transcribe`; English,
    /// as whisper-cli); `decode` = total - mel - encode (exact for audio <= 30 s).
    fn bench_run(&self, audio: &[f32]) -> Result<BenchRun> {
        let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let mel = self.log_mel(audio);
        let mel_ms = ms(t);
        let t = Instant::now();
        drop(Seq2Seq::encode(self, &mel[..])?);
        let encode_ms = ms(t);
        let t = Instant::now();
        let tr = self.transcribe(audio, &TranscribeOptions { language: Some("en".into()), ..TranscribeOptions::default() })?;
        let total = ms(t);
        let tokens = tr.segments.iter().map(|s| s.tokens.len()).sum::<usize>() as f64;
        let decode = (total - mel_ms - encode_ms).max(0.0);
        Ok(BenchRun {
            times: vec![("mel", mel_ms), ("encode", encode_ms), ("decode", decode), ("total", total)],
            values: vec![("tokens", tokens), ("tokens_per_s", if decode > 0.0 { tokens / (decode / 1e3) } else { 0.0 })],
            text: tr.text.trim().to_string(),
        })
    }
}

pub static ENTRY: crate::runtime::ModelEntry = asr_entry::<Whisper>();
