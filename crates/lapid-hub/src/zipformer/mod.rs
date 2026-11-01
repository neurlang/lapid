//! Streaming speech recognition with icefall's Zipformer2 transducer (the model family
//! sherpa-onnx runs), e.g. the 66M LibriSpeech streaming model
//! `Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17` or its Hebrew fine-tune
//! `thewh1teagle/zipformer2-66m-streaming-he`.
//!
//! ```no_run
//! use lapid_hub::zipformer::{StreamOptions, Zipformer};
//! let device = lapid::Device::best();
//! let model = Zipformer::load("zipformer2-en.lapid".as_ref(), &device)?;
//! let mut stream = model.stream(&StreamOptions::default())?;
//! let audio = lapid_hub::audio::load_audio_16k_mono("data/jfk.wav".as_ref())?;
//! for piece in audio.chunks(1600) {
//!     for r in stream.accept_waveform(piece)? {
//!         println!("{} {}", if r.is_final { "final" } else { "partial" }, r.text);
//!     }
//! }
//! for r in stream.input_finished()? {
//!     println!("final {}", r.text);
//! }
//! # Ok::<(), lapid::Error>(())
//! ```
//!
//! Pipeline: [`fbank`] (lhotse / kaldi fbank, computed incrementally) → [`model::embed`]
//! (conv subsampling with a ConvNeXt cache) → [`model::encoder`] (6 multi-rate stacks of
//! [`model::layer`], cached keys / values / nonlinear-attention inputs / conv context per
//! layer) → greedy transducer search ([`stream`]: stateless decoder + joiner) → tokens
//! ([`tokens`]), with sherpa-onnx [`endpoint`] rules. Checkpoints come from icefall through
//! `tools/reference/zipformer_ref.py export` and [`convert`].

pub mod bench;
pub mod cmd;
pub mod config;
pub mod convert;
pub mod endpoint;
pub mod fbank;
pub(crate) mod model;
pub mod stream;
pub mod tokens;
pub mod verify;

use std::path::Path;
use std::sync::Arc;

use lapid::{Device, Error, LapidFile, Result};
use lapid_nn::Loader;

pub use config::ZipformerConfig;
pub use endpoint::{EndpointConfig, EndpointRule};
pub use lapid_nn::Precision;
pub use stream::{StreamOptions, StreamResult, StreamStats, ZipformerStream};
pub use tokens::Tokens;

use crate::model::{Bytes, Model, ModelInfo, hub_model_path};
use model::decoder::{Decoder, Joiner};
use model::encoder::Encoder;

/// Shared weights of a loaded model (streams hold an `Arc` to it).
pub(crate) struct Inner {
    pub device: Device,
    pub info: ModelInfo,
    pub cfg: ZipformerConfig,
    pub tokens: Tokens,
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub joiner: Joiner,
}

/// A loaded Zipformer2 transducer bound to a device. Cheap to clone.
#[derive(Clone)]
pub struct Zipformer {
    inner: Arc<Inner>,
}

impl Zipformer {
    /// Load `src` (a directory exported by `tools/reference/zipformer_ref.py export`, or an HF
    /// repo with the same files: `config.json`, `model.safetensors`, `tokens.txt`) from the
    /// lapid cache, converting it first if missing.
    pub fn from_hub(src: &str, dtype: &str, device: &Device) -> Result<Zipformer> {
        let path = hub_model_path(src, dtype, convert::convert_hf)?;
        Zipformer::load(&path, device)
    }

    /// Load a converted `.lapid` model and upload its weights to `device`.
    pub fn load(path: &Path, device: &Device) -> Result<Zipformer> {
        Zipformer::load_with(path, device, Precision::Stored)
    }

    /// Load from an in-memory `.lapid` image (zero copy).
    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Zipformer> {
        Zipformer::from_file(&LapidFile::from_bytes(bytes)?, device, Precision::Stored)
    }

    /// Load with a given weight precision (fake-quant references for verification).
    pub fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Zipformer> {
        Zipformer::from_file(&LapidFile::open(path)?, device, precision)
    }

    fn from_file(file: &LapidFile, device: &Device, precision: Precision) -> Result<Zipformer> {
        let cfg = ZipformerConfig::from_metadata(&file.metadata)?;
        let blob = file
            .blob(config::TOKENS_BLOB)
            .ok_or_else(|| Error::new("metadata", format!("zipformer: blob {:?} missing", config::TOKENS_BLOB)))?;
        let text = std::str::from_utf8(blob).map_err(|e| Error::new("tokenizer", format!("tokens.txt: {e}")))?;
        let tokens = Tokens::parse(text)?;
        if tokens.len() != cfg.vocab_size {
            return Err(Error::new("tokenizer", format!("tokens.txt has {} pieces, model vocabulary is {}", tokens.len(), cfg.vocab_size)));
        }
        let loader = Loader::new(file, device, precision);
        let inner = Inner {
            device: device.clone(),
            info: ModelInfo::from_file(file, "automatic-speech-recognition"),
            encoder: Encoder::load(&loader, &cfg)?,
            decoder: Decoder::load(&loader, &cfg)?,
            joiner: Joiner::load(&loader, &cfg)?,
            tokens,
            cfg,
        };
        Ok(Zipformer { inner: Arc::new(inner) })
    }

    /// A new stream (fresh encoder state and hypothesis).
    pub fn stream(&self, opts: &StreamOptions) -> Result<ZipformerStream> {
        ZipformerStream::new(Arc::clone(&self.inner), opts)
    }

    /// Whole-signal transcription through the streaming path (no endpointing): the same
    /// tokens as icefall's streaming decode of the utterance.
    pub fn transcribe(&self, samples: &[f32], chunk_size: Option<usize>) -> Result<StreamResult> {
        let opts = StreamOptions { chunk_size, left_context_frames: None, endpoint: None };
        let mut s = self.stream(&opts)?;
        s.accept_waveform(samples)?;
        s.input_finished()?;
        Ok(s.result())
    }

    pub fn transcribe_file(&self, path: &Path, chunk_size: Option<usize>) -> Result<StreamResult> {
        let samples = lapid_nn::audio::load_audio(path, self.inner.cfg.sample_rate)?;
        self.transcribe(&samples, chunk_size)
    }

    pub fn config(&self) -> &ZipformerConfig {
        &self.inner.cfg
    }

    pub fn tokens(&self) -> &Tokens {
        &self.inner.tokens
    }
}

impl Model for Zipformer {
    fn info(&self) -> ModelInfo {
        self.inner.info.clone()
    }

    fn device(&self) -> &Device {
        &self.inner.device
    }
}
