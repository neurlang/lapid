//! NVIDIA Parakeet TDT speech recognition (FastConformer encoder + token-and-duration
//! transducer), e.g. `nvidia/parakeet-tdt-0.6b-v3` (25 European languages, punctuation and
//! capitalization, word timestamps).
//!
//! ```no_run
//! let device = lapid::Device::best();
//! let model = lapid_hub::Parakeet::from_hub("nvidia/parakeet-tdt-0.6b-v3", "int8", &device)?;
//! let out = model.transcribe_file("data/jfk.wav".as_ref(), &Default::default())?;
//! println!("{}", out.text);
//! for w in &out.words {
//!     println!("{:6.2} {:6.2} {}", w.start, w.end, w.word);
//! }
//! # Ok::<(), lapid::Error>(())
//! ```
//!
//! Pipeline: [`mel`] (NeMo log-mel) → [`model::encoder`] (conv subsampling + FastConformer
//! with relative-position attention, on the device) → [`decode`] (greedy TDT, host loop over
//! small device graphs: LSTM prediction network + joint) → tokens, words, segments.
//! Long audio is encoded in overlapping windows ([`transcribe`]).

pub mod bench;
pub mod cmd;
pub mod config;
pub mod convert;
pub mod decode;
pub mod mel;
pub(crate) mod model;
mod transcribe;
pub mod verify;

use std::path::Path;

use lapid::{Device, Error, LapidFile, Result, Tensor};
use lapid_nn::{Loader, Tokenizer};

pub use config::ParakeetConfig;
pub use decode::{Segment, Step, Token, Word};
pub use lapid_nn::Precision;
pub use transcribe::{TranscribeOptions, Transcript};

use crate::model::{Bytes, Model, ModelInfo, hub_model_path};
use decode::{Decoded, Path as DecodePath};
use mel::{Features, Mel};
use model::decoder::{Joint, Prediction};
use model::encoder::Encoder;
use model::first;

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
    /// from the Hugging Face hub into `lapid_convert::default_model_path` first if missing.
    pub fn from_hub(repo: &str, dtype: &str, device: &Device) -> Result<Parakeet> {
        let path = hub_model_path(repo, dtype, convert::convert_hf)?;
        Parakeet::load(&path, device)
    }

    /// Load a converted `.lapid` Parakeet model and upload its weights to `device`.
    pub fn load(path: &Path, device: &Device) -> Result<Parakeet> {
        Parakeet::load_with(path, device, Precision::Stored)
    }

    /// Load from an in-memory `.lapid` image (zero copy).
    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Parakeet> {
        Parakeet::from_file(&LapidFile::from_bytes(bytes)?, device, Precision::Stored)
    }

    /// Load with a given weight precision (the fake-quant modes build the references that
    /// quantized kernels are verified against).
    pub fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Parakeet> {
        Parakeet::from_file(&LapidFile::open(path)?, device, precision)
    }

    fn from_file(file: &LapidFile, device: &Device, precision: Precision) -> Result<Parakeet> {
        let cfg = ParakeetConfig::from_metadata(&file.metadata)?;
        let blob_name = file
            .metadata
            .pointer("/parakeet/tokenizer_blob")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(config::TOKENIZER_BLOB);
        let blob = file
            .blob(blob_name)
            .ok_or_else(|| Error::new("metadata", format!("parakeet: tokenizer blob {blob_name:?} missing")))?;
        let json = std::str::from_utf8(blob).map_err(|e| Error::new("tokenizer", format!("tokenizer.json: {e}")))?;
        let tokenizer = Tokenizer::from_tokenizer_json(json)?;
        let loader = Loader::new(file, device, precision);
        Ok(Parakeet {
            device: device.clone(),
            info: ModelInfo::from_file(file, "automatic-speech-recognition"),
            features: Features::new(&cfg)?,
            encoder: Encoder::load(&loader, &cfg)?,
            prediction: Prediction::load(&loader, &cfg)?,
            joint: Joint::load(&loader, &cfg)?,
            tokenizer,
            cfg,
        })
    }

    pub fn transcribe_file(&self, path: &Path, opts: &TranscribeOptions) -> Result<Transcript> {
        let samples = lapid_nn::audio::load_audio(path, self.cfg.sample_rate)?;
        self.transcribe(&samples, opts)
    }

    /// Transcribe mono samples at `config().sample_rate` (16 kHz), any length.
    pub fn transcribe(&self, samples: &[f32], opts: &TranscribeOptions) -> Result<Transcript> {
        transcribe::transcribe(self, samples, opts)
    }

    /// Normalized log-mel features of a whole signal ([T, n_mels], valid frames only).
    pub fn features(&self, samples: &[f32]) -> Mel {
        self.features.compute(samples)
    }

    /// Encoder on mel [frames, n_mels]: projected frames [1, T', joint_hidden], realized.
    pub fn encode_mel(&self, mel: &[f32], frames: usize) -> Result<Tensor> {
        let g = self.encoder.graph(mel, frames, false)?;
        first(self.device.eval(&[&g.projected])?)
    }

    /// Encoder with intermediates: "encoder.subsampling", "encoder.layers.{i}",
    /// "encoder.projected" (host copies).
    pub fn encode_traced(&self, mel: &[f32], frames: usize) -> Result<Vec<(String, Vec<f32>)>> {
        let g = self.encoder.graph(mel, frames, true)?;
        let refs: Vec<&Tensor> = g.traced.iter().collect();
        let realized = self.device.eval(&refs)?;
        g.traced
            .iter()
            .zip(&realized)
            .map(|(lazy, t)| Ok((lazy.name().unwrap_or("encoder.unnamed").to_string(), self.device.to_vec_f32(t)?)))
            .collect()
    }

    /// TDT decode over projected frames `enc` [1, T, H] (on this device).
    pub(crate) fn decode_frames(&self, enc: &Tensor, frames: usize, path: DecodePath<'_>, keep_logits: bool) -> Result<Decoded> {
        decode::decode(self, enc, frames, path, keep_logits)
    }

    /// Greedy decode steps (blank steps included) over projected frames.
    pub fn decode_steps(&self, enc: &Tensor) -> Result<Vec<Step>> {
        let frames = enc.shape().get(1).copied().unwrap_or(0);
        Ok(self.decode_frames(enc, frames, DecodePath::Greedy, false)?.steps)
    }

    pub fn config(&self) -> &ParakeetConfig {
        &self.cfg
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

}

impl Model for Parakeet {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
