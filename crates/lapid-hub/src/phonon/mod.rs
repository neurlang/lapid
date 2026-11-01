//! Phonon text-to-speech (Gradium, built on Kyutai's Pocket TTS and the Mimi codec):
//! a flow-matching transformer predicts one 32-d audio latent per 80 ms frame from the text
//! and a voice prompt, and the streaming Mimi decoder turns each latent into 1920 samples
//! of 24 kHz audio as soon as it is sampled.
//!
//! Only the model's math, config, conversion tables and verification stages live here; text
//! chunking, end of speech, fade-in, streaming, verification, benchmarks and serving are the
//! shared speech pipeline (`runtime::tts`); the streaming transformer and transposed
//! convolution are `lapid_nn::speech` layers.
//!
//! ```no_run
//! let device = lapid::Device::best();
//! let tts = lapid_hub::Phonon::from_hub("gradium/phonon-40m-en", "fp16", &device)?;
//! for pcm in tts.speak_stream("Hello from Phonon.", &Default::default())? {
//!     let pcm = pcm?; // f32 mono chunk at tts.sample_rate()
//!     # let _ = pcm;
//! }
//! tts.speak_to_file("Hello again.", "hello.wav".as_ref())?;
//! # Ok::<(), lapid::Error>(())
//! ```

pub mod config;
pub mod convert;
mod flow;
mod lm;
mod mimi;
mod stream;
pub mod text;
mod transformer;
mod verify;
/// WAV writing (moved to [`crate::audio`]).
pub use crate::audio as wav;

use std::collections::BTreeMap;
use std::path::Path;

use lapid::{DType, Device, Error, LapidFile, Result, Tensor, bail};
use lapid_convert::engine::ConvertSpec;
use lapid_nn::{Loader, Tokenizer};
use lapid_verify::Descriptor;
use lapid_verify::harness::Reference;

use crate::model::{Bytes, Model, ModelInfo, hub_model_path};
use crate::runtime::ModelEntry;
use crate::runtime::tts::{self, ReferenceSpeech, SpeechDef, SpeechRequest};
use config::PhononConfig;
pub use lapid_nn::Precision;
use lm::FlowLm;
use mimi::MimiDecoder;
pub use crate::runtime::tts::StreamStats;

/// Iterator of PCM chunks (mono f32 at [`Phonon::sample_rate`]); see [`Phonon::speak_stream`].
pub type SpeechStream<'a> = tts::SpeechStream<Phonon, &'a Phonon>;

/// Options of one synthesis call; `None` fields take the model's defaults.
#[derive(Clone, Debug, Default)]
pub struct SpeakOptions {
    /// Voice name (`"default"`, or one added with [`Phonon::add_voices`]).
    pub voice: Option<String>,
    /// Variance of the flow sampler's noise (generation_config: 0.3). 0 = deterministic.
    pub temperature: Option<f32>,
    /// Noise seed; `None` draws one from the clock. Same seed + options = same audio.
    pub seed: Option<u64>,
    /// Euler steps of the LSD sampler (default 1).
    pub num_flow_steps: Option<usize>,
    /// End-of-speech logit threshold (default 4.0).
    pub eos_threshold: Option<f32>,
    /// Frame budget per text chunk (default `ceil((tokens / 3 + 2) * frame_rate)`).
    pub max_frames: Option<usize>,
    /// Speaking-duration condition (default 0.0, the model's training value).
    pub duration_delta: Option<f32>,
    /// Codec frames (80 ms each) per yielded PCM chunk (default 1: lowest latency).
    pub frames_per_chunk: Option<usize>,
    /// Maximum tokens per text chunk; longer text is split at sentence ends (default 50).
    pub max_chunk_tokens: Option<usize>,
}

/// A loaded Phonon model bound to a device.
pub struct Phonon {
    device: Device,
    info: ModelInfo,
    cfg: PhononConfig,
    tokenizer: Tokenizer,
    lm: FlowLm,
    mimi: MimiDecoder,
    /// Voice-prompt latents [frames, prompt_dim] on the device, by name.
    voices: BTreeMap<String, Tensor>,
}

impl Phonon {
    /// Load `repo` (e.g. "gradium/phonon-40m-en") from the lapid cache, converting it from the
    /// Hugging Face hub first if missing (`dtype`: "fp32", "fp16", "int8", "int4").
    pub fn from_hub(repo: &str, dtype: &str, device: &Device) -> Result<Phonon> {
        Phonon::load(&hub_model_path(repo, dtype, |r, d, out| convert::CONVERT.convert(r, d, out))?, device)
    }

    /// Load a converted `.lapid` Phonon model.
    pub fn load(path: &Path, device: &Device) -> Result<Phonon> {
        Phonon::load_with(path, device, Precision::Stored)
    }

    /// Load from an in-memory `.lapid` image (no filesystem; zero copy).
    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Phonon> {
        Phonon::from_file(&LapidFile::from_bytes(bytes)?, device, Precision::Stored)
    }

    /// Load with a weight precision (the fake-quant modes build verification references).
    pub fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Phonon> {
        Phonon::from_file(&LapidFile::open(path)?, device, precision)
    }

    fn from_file(file: &LapidFile, device: &Device, precision: Precision) -> Result<Phonon> {
        let cfg = PhononConfig::from_metadata(&file.metadata)?;
        let tokenizer = load_tokenizer(file, &cfg)?;
        let loader = Loader::new(file, device, precision);
        let lm = FlowLm::load(&loader, &cfg)?;
        let mimi = MimiDecoder::load(&loader, &cfg.mimi, device)?;
        let info = ModelInfo::from_file(file, "text-to-speech");
        let mut tts = Phonon { device: device.clone(), info, cfg, tokenizer, lm, mimi, voices: BTreeMap::new() };
        tts.voices_from(file)?;
        if !tts.voices.contains_key(&tts.cfg.assets.default_voice) {
            bail!("missing_tensor", "model file has no default voice {:?}", tts.cfg.assets.default_voice);
        }
        Ok(tts)
    }

    fn voices_from(&mut self, file: &LapidFile) -> Result<usize> {
        let prefix = self.cfg.assets.voice_prefix.clone();
        let dim = self.cfg.lm.audio_prompt_dim;
        let mut n = 0;
        for (name, t) in file.tensors.range(prefix.clone()..) {
            let Some(voice) = name.strip_prefix(&prefix) else {
                break;
            };
            if t.rank() != 2 || t.shape()[1] != dim {
                bail!("shape_mismatch", "voice {name:?} must be [frames, {dim}], got {:?}", t.shape());
            }
            let v = self.device.upload(&t.cast(DType::F32)?)?;
            let v = self.device.eval(&[&v])?.pop().ok_or_else(|| Error::new("internal", "eval returned nothing"))?;
            self.voices.insert(voice.to_string(), v);
            n += 1;
        }
        Ok(n)
    }

    /// Add the voices of a `.lapid` voices extras file (see [`convert::convert_voices`]).
    /// Returns how many were added.
    pub fn add_voices(&mut self, path: &Path) -> Result<usize> {
        let file = LapidFile::open(path)?;
        match file.extends() {
            Some(e) if e.architecture == "phonon" => {}
            _ => bail!("unsupported", "{}: not a phonon voices extras file", path.display()),
        }
        self.voices_from(&file)
    }

    /// Add a voice from its prompt latents (row-major [frames, prompt_dim]).
    pub fn add_voice(&mut self, name: &str, latents: &[f32]) -> Result<()> {
        let dim = self.cfg.lm.audio_prompt_dim;
        if latents.is_empty() || !latents.len().is_multiple_of(dim) {
            bail!("shape_mismatch", "voice latents must be [frames, {dim}], got {} values", latents.len());
        }
        let t = self.device.upload(&Tensor::from_f32(latents, &[latents.len() / dim, dim])?)?;
        self.voices.insert(name.to_string(), t);
        Ok(())
    }

    /// Add a voice from a voice safetensors file (`speaker_wavs` [1, prompt_dim, frames], the
    /// format of `gradium/phonon-voices` and the package's `default-voice.safetensors`).
    pub fn add_voice_safetensors(&mut self, name: &str, path: &Path) -> Result<()> {
        let (data, _) = convert::read_voice_safetensors(path)?;
        self.add_voice(name, &data)
    }

    pub fn voices(&self) -> Vec<String> {
        self.voices.keys().cloned().collect()
    }

    pub fn sample_rate(&self) -> u32 {
        self.cfg.mimi.sample_rate
    }

    pub fn config(&self) -> &PhononConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Prepared text and its token ids (as the model sees them).
    pub fn tokenize(&self, text: &str) -> Result<(String, Vec<u32>)> {
        let prepared = text::prepare(text);
        let ids = text::encode(&self.tokenizer, &prepared)?;
        Ok((prepared, ids))
    }

    /// Stream speech for `text`: an iterator of PCM chunks (mono f32 at
    /// [`sample_rate`](Self::sample_rate)), one per `frames_per_chunk` codec frames.
    pub fn speak_stream(&self, text: &str, opts: &SpeakOptions) -> Result<SpeechStream<'_>> {
        tts::speak(self, text, opts)
    }

    /// The whole utterance as one buffer.
    pub fn speak(&self, text: &str, opts: &SpeakOptions) -> Result<Vec<f32>> {
        tts::synthesize(self, text, opts)
    }

    /// Synthesize `text` with default options into a 16-bit PCM WAV file.
    pub fn speak_to_file(&self, text: &str, path: &Path) -> Result<()> {
        self.speak_to_file_with(text, path, &SpeakOptions::default())
    }

    pub fn speak_to_file_with(&self, text: &str, path: &Path, opts: &SpeakOptions) -> Result<()> {
        let pcm = self.speak(text, opts)?;
        wav::write_wav(path, &pcm, self.sample_rate())
    }
}

impl SpeechDef for Phonon {
    const DESCRIPTOR: Descriptor = Descriptor {
        name: "phonon",
        default_model: "gradium/phonon-40m-en",
        default_dtype: "fp16",
        reference_dir: "data/reference/phonon-40m-en",
        reference_script: "tools/reference/phonon_ref.py",
        reference_label: "PyTorch fp32 (HF GradiumPhononTTSForConditionalGeneration + PocketTTSAudioTokenizer)",
        strict_dtypes: &["fp32", "fp16"],
        bench_prompt_tokens: 0,
        bench_gen_tokens: 0,
    };
    const CONVERT: &'static ConvertSpec = &convert::CONVERT;

    fn from_file(file: &LapidFile, device: &Device, precision: Precision) -> Result<Self> {
        Phonon::from_file(file, device, precision)
    }

    fn voices(&self) -> Vec<String> {
        Phonon::voices(self)
    }

    fn default_voice(&self) -> Option<String> {
        Some(self.cfg.assets.default_voice.clone())
    }

    fn request_options(&self, req: &SpeechRequest) -> SpeakOptions {
        SpeakOptions {
            // OpenAI voice names ("alloy", ...) fall back to the model's default voice.
            voice: req.voice.clone().filter(|v| self.voices.contains_key(v)),
            seed: req.seed,
            temperature: req.temperature,
            frames_per_chunk: req.frames_per_chunk,
            ..Default::default()
        }
    }

    fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tokenize(text)?.1)
    }

    fn reference_stages(&self, r: &Reference) -> Result<Vec<(String, Vec<f32>)>> {
        verify::run_model(self, &verify::Inputs::from_reference(r)?)
    }

    fn reference_speech(&self, r: &Reference) -> Result<ReferenceSpeech> {
        verify::reference_speech(self, r)
    }
}

pub static ENTRY: ModelEntry = ModelEntry::speech::<Phonon>();

impl Model for Phonon {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

/// The tokenizer blob (tokenizer.json; the optional SentencePiece model is not used).
fn load_tokenizer(file: &LapidFile, cfg: &PhononConfig) -> Result<Tokenizer> {
    if let Some(bytes) = file.blob(&cfg.assets.tokenizer) {
        return Tokenizer::from_bytes(bytes);
    }
    bail!("missing_metadata", "phonon: model file has no tokenizer blob {:?}", cfg.assets.tokenizer)
}
