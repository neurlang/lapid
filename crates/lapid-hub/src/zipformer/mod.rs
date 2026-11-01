//! Streaming speech recognition with icefall's Zipformer2 transducer (the model family
//! sherpa-onnx runs), e.g. `Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17`
//! or its Hebrew fine-tune `thewh1teagle/zipformer2-66m-streaming-he`, on the shared
//! streaming transducer pipeline.
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
//! `lapid_nn::audio::OnlineFbank` (lhotse / kaldi fbank, incremental) -> [`model`] (conv subsampling with a
//! ConvNeXt cache, 6 multi-rate stacks with cached attention / conv context, stateless
//! decoder, joiner) -> `runtime::transducer` (icefall chunk schedule, greedy search with one
//! symbol per frame) -> [`stream`] (segments, sherpa-onnx endpoints) -> [`tokens`].
//! Checkpoints come from icefall through `tools/reference/zipformer_ref.py export`.

pub mod config;
pub mod convert;
pub(crate) mod model;
pub mod stream;
pub mod tokens;
mod verify;

use std::path::Path;
use std::sync::Arc;

use lapid::{Device, Error, LapidFile, Result, Tensor};
use lapid_nn::Loader;

pub use crate::runtime::endpoint::{EndpointConfig, EndpointRule};
pub use crate::runtime::transducer::StreamStats;
pub use config::ZipformerConfig;
pub use lapid_nn::Precision;
pub use stream::{StreamOptions, StreamResult, ZipformerStream};
pub use tokens::Tokens;

use crate::model::{Bytes, Model, ModelInfo, hub_model_path};
use crate::runtime::transducer::{StreamingTransducer, Transducer};
use config::{LOG_EPS, OUTPUT_FRAME_SECONDS, PAD_FRAMES, TAIL_FRAMES};
use lapid_nn::audio::{Fbank, OnlineFbank};
use model::decoder::{Decoder, Joiner};
use model::encoder::{Encoder, EncoderState, Plan};
use model::layer::STATE_NAMES;

/// The network and its token table (streams share it through an `Arc`).
pub struct ZipformerModel {
    device: Device,
    info: ModelInfo,
    cfg: ZipformerConfig,
    tokens: Tokens,
    encoder: Encoder,
    decoder: Decoder,
    joiner: Joiner,
}

/// Encoder state of a stream: the chunk geometry constants and the caches.
pub struct ChunkState {
    plan: Plan,
    caches: EncoderState,
    /// The last chunk's graph, dropped when the next chunk starts: releasing a ~13k-node
    /// graph is then not charged to the greedy search between chunks (CPU).
    last: Option<model::encoder::ChunkGraph>,
}

/// Host copies of one traced chunk (verification).
pub(crate) struct ChunkTrace {
    pub traced: Vec<(String, Vec<f32>)>,
    pub state: Vec<(String, Vec<f32>)>,
}

impl ZipformerModel {
    fn from_file(file: &LapidFile, device: &Device, precision: Precision) -> Result<ZipformerModel> {
        let cfg = ZipformerConfig::from_metadata(&file.metadata)?;
        let blob = file.blob(config::TOKENS_BLOB).ok_or_else(|| Error::new("metadata", format!("zipformer: blob {:?} missing", config::TOKENS_BLOB)))?;
        let tokens = Tokens::parse(std::str::from_utf8(blob).map_err(|e| Error::new("tokenizer", format!("tokens.txt: {e}")))?)?;
        if tokens.len() != cfg.vocab_size {
            return Err(Error::new("tokenizer", format!("tokens.txt has {} pieces, model vocabulary is {}", tokens.len(), cfg.vocab_size)));
        }
        let l = Loader::new(file, device, precision);
        Ok(ZipformerModel {
            device: device.clone(),
            info: ModelInfo::from_file(file, "automatic-speech-recognition"),
            encoder: Encoder::load(&l, &cfg)?,
            decoder: Decoder::load(&l, &cfg)?,
            joiner: Joiner::load(&l, &cfg)?,
            tokens,
            cfg,
        })
    }

    pub fn config(&self) -> &ZipformerConfig {
        &self.cfg
    }

    pub fn tokens(&self) -> &Tokens {
        &self.tokens
    }

    /// Fresh encoder state for `chunk` / `left` (encoder frames at 50 Hz).
    pub fn chunk_state(&self, chunk: usize, left: usize) -> Result<ChunkState> {
        let plan = self.encoder.plan(chunk, left, &self.device)?;
        Ok(ChunkState { caches: self.encoder.init_state(&plan), plan, last: None })
    }

    fn frontend(&self) -> OnlineFbank {
        OnlineFbank::new(Fbank::new(self.cfg.feature_dim, self.cfg.sample_rate))
    }

    /// One chunk `[2 chunk + 13, F]`: realized encoder frames `[chunk / 2, J]`, the caches
    /// updated; with `trace`, host copies of the stack outputs and the new caches.
    pub(crate) fn run_chunk(&self, feats: &[f32], st: &mut ChunkState, trace: bool) -> Result<(Tensor, Option<ChunkTrace>)> {
        st.last = None;
        let x = Tensor::from_f32(feats, &[2 * st.plan.chunk + PAD_FRAMES, self.cfg.feature_dim])?;
        let g = self.encoder.chunk(&st.plan, &x, &st.caches, trace)?;
        let mut outs: Vec<&Tensor> = vec![&g.out];
        outs.extend(g.traced.iter());
        outs.extend(g.state.tensors());
        let realized = self.device.eval(&outs)?;
        let n = 1 + g.traced.len();
        let caches = EncoderState::from_tensors(&realized[n..], g.state.processed)?;
        let tr = if trace {
            let host = |t: &Tensor| self.device.to_vec_f32(t);
            let traced = g.traced.iter().zip(&realized[1..n]).map(|(l, t)| Ok((l.name().unwrap_or("unnamed").to_string(), host(t)?))).collect::<Result<_>>()?;
            let mut state = vec![("state.embed".to_string(), host(&caches.embed)?)];
            let mut layers = caches.layers.iter();
            for (i, &count) in self.cfg.num_encoder_layers.iter().enumerate() {
                for j in 0..count {
                    let layer = layers.next().ok_or_else(|| Error::new("internal", "zipformer: layer count"))?;
                    for (name, t) in STATE_NAMES.iter().zip(layer.tensors()) {
                        state.push((format!("state.{i}.{j}.{name}"), host(t)?));
                    }
                }
            }
            Some(ChunkTrace { traced, state })
        } else {
            None
        };
        st.caches = caches;
        st.last = Some(g);
        Ok((realized[0].clone(), tr))
    }
}

impl Transducer for ZipformerModel {
    fn device(&self) -> &Device {
        &self.device
    }

    fn blank_id(&self) -> u32 {
        self.cfg.blank_id
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }

    fn frame_seconds(&self) -> f64 {
        OUTPUT_FRAME_SECONDS
    }

    fn predictor_init(&self) -> Result<Vec<Tensor>> {
        Ok(Vec::new())
    }

    /// Stateless: the last `context_size` tokens of `[-1 .., blank] + history + token`.
    fn predict(&self, history: &[u32], token: u32, _: &[Tensor]) -> Result<(Tensor, Vec<Tensor>)> {
        let ctx = self.cfg.context_size;
        let mut hyp: Vec<i64> = vec![-1; ctx - 1];
        hyp.push(i64::from(self.cfg.blank_id));
        hyp.extend(history.iter().map(|&t| i64::from(t)));
        if token != self.cfg.blank_id {
            hyp.push(i64::from(token));
        }
        Ok((self.decoder.forward(&hyp[hyp.len() - ctx..])?, Vec::new()))
    }

    fn joint(&self, enc: &Tensor, pred: &Tensor) -> Result<Tensor> {
        self.joiner.logits(enc, pred)
    }
}

impl crate::runtime::transducer::FeatureFrontend for OnlineFbank {
    fn accept(&mut self, samples: &[f32]) -> Vec<f32> {
        OnlineFbank::accept(self, samples);
        self.take_all()
    }

    fn finish(&mut self) -> Vec<f32> {
        OnlineFbank::finish(self);
        self.take_all()
    }
}

impl StreamingTransducer for ZipformerModel {
    type EncoderState = ChunkState;
    type Frontend = OnlineFbank;

    fn feature_dim(&self) -> usize {
        self.cfg.feature_dim
    }

    /// icefall: a chunk of `c` encoder frames reads `2c + 13` fbank frames, advancing `2c`.
    fn chunk_frames(&self, st: &ChunkState) -> (usize, usize) {
        (2 * st.plan.chunk + PAD_FRAMES, 2 * st.plan.chunk)
    }

    /// icefall's flush: 13 + 30 frames of `LOG_EPS`.
    fn tail_padding(&self) -> (usize, f32) {
        (PAD_FRAMES + TAIL_FRAMES, LOG_EPS)
    }

    fn encode_chunk(&self, feats: &[f32], st: &mut ChunkState) -> Result<Tensor> {
        Ok(self.run_chunk(feats, st, false)?.0)
    }
}

/// A loaded Zipformer2 transducer bound to a device. Cheap to clone.
#[derive(Clone)]
pub struct Zipformer {
    inner: Arc<ZipformerModel>,
}

impl Zipformer {
    /// Load `src` (a directory exported by `tools/reference/zipformer_ref.py export`, or an HF
    /// repo with the same files) from the lapid cache, converting it first if missing.
    pub fn from_hub(src: &str, dtype: &str, device: &Device) -> Result<Zipformer> {
        let path = if Path::new(src).is_dir() {
            crate::runtime::resolve::resolve(src, Some(dtype))?.path
        } else {
            hub_model_path(src, dtype, |r, d, o| convert::CONVERT.convert(r, d, o))?
        };
        Zipformer::load(&path, device)
    }

    pub fn load(path: &Path, device: &Device) -> Result<Zipformer> {
        Zipformer::load_with(path, device, Precision::Stored)
    }

    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Zipformer> {
        Ok(Zipformer { inner: Arc::new(ZipformerModel::from_file(&LapidFile::from_bytes(bytes)?, device, Precision::Stored)?) })
    }

    /// Load with a weight precision (fake-quant references for verification).
    pub fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Zipformer> {
        Ok(Zipformer { inner: Arc::new(ZipformerModel::from_file(&LapidFile::open(path)?, device, precision)?) })
    }

    /// A new stream (fresh encoder state and hypothesis).
    pub fn stream(&self, opts: &StreamOptions) -> Result<ZipformerStream> {
        ZipformerStream::new(Arc::clone(&self.inner), opts)
    }

    /// Whole-signal transcription through the streaming path (no endpointing): the same
    /// tokens as icefall's streaming decode of the utterance.
    pub fn transcribe(&self, samples: &[f32], chunk_size: Option<usize>) -> Result<StreamResult> {
        let mut s = self.stream(&StreamOptions { chunk_size, left_context_frames: None, endpoint: None })?;
        s.accept_waveform(samples)?;
        s.input_finished()?;
        Ok(s.result())
    }

    pub fn transcribe_file(&self, path: &Path, chunk_size: Option<usize>) -> Result<StreamResult> {
        self.transcribe(&lapid_nn::audio::load_audio(path, self.inner.cfg.sample_rate)?, chunk_size)
    }

    pub fn config(&self) -> &ZipformerConfig {
        &self.inner.cfg
    }

    pub fn tokens(&self) -> &Tokens {
        &self.inner.tokens
    }

    pub fn model(&self) -> &ZipformerModel {
        &self.inner
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

pub static ENTRY: crate::runtime::ModelEntry = crate::runtime::audio::asr_entry::<Zipformer>();
