//! Silero VAD v5 voice activity detection (16 kHz), e.g. `snakers4/silero-vad` exported by
//! `tools/reference/silero_vad_ref.py`.
//!
//! ```no_run
//! use lapid_hub::{Vad, VadEvent, VadOptions};
//! let device = lapid::Device::cpu();
//! let vad = Vad::load("silero_vad.lapid".as_ref(), &device)?;
//! let samples = lapid_hub::audio::load_audio_16k_mono("data/jfk.wav".as_ref())?;
//! for s in vad.process(&samples, &VadOptions::default())? {
//!     println!("{:.2}-{:.2}", s.start_seconds(16_000), s.end_seconds(16_000));
//! }
//! let mut stream = vad.stream(&VadOptions::default());
//! for chunk in samples.chunks(1600) {
//!     for ev in stream.accept_waveform(chunk)? {
//!         println!("{ev:?}");
//!     }
//! }
//! stream.flush()?;
//! # Ok::<(), lapid::Error>(())
//! ```
//!
//! Semantics follow silero-vad 5.1.2: [`Vad::probabilities`] = `model.audio_forward`
//! (512-sample windows, the previous 64 samples prepended, state reset per call, the last
//! window zero-padded), [`Vad::process`] = `get_speech_timestamps`, [`VadStream`] =
//! `VADIterator` fed with 512-sample chunks. See [`model`] for the device / host split.

pub mod config;
pub mod convert;
pub(crate) mod model;
pub mod segments;
mod stream;
pub mod verify;

use std::path::Path;
use std::sync::Arc;

use lapid::{Device, LapidFile, Result};
use lapid_nn::{Loader, Precision};

pub use config::VadConfig;
pub use model::LstmState;
pub use segments::{SpeechSegment, VadEvent, VadOptions};
pub use stream::VadStream;

use crate::model::{Bytes, Model, ModelInfo, hub_model_path};
use model::Net;

/// Windows per device evaluation in [`Vad::probabilities`] (bounds the graph's memory:
/// 2048 × 640 input floats).
pub const BATCH_WINDOWS: usize = 2048;
/// Smallest device batch; fewer windows are padded with silence (see `Vad::windows`).
/// Backends run matmuls with few rows through a GEMV path (CPU up to 8 rows, CUDA / WGSL /
/// Metal up to 4) whose summation order differs from their GEMM; 9 keeps every evaluation on
/// the GEMM path, which makes streaming bit-identical to `probabilities`.
pub const MIN_BATCH_WINDOWS: usize = 9;

struct Inner {
    device: Device,
    info: ModelInfo,
    net: Net,
}

/// A loaded Silero VAD bound to a device. Cheap to clone (shared weights).
#[derive(Clone)]
pub struct Vad {
    inner: Arc<Inner>,
}

/// Per-window intermediates of [`Vad::trace`] (host copies).
#[derive(Clone, Debug, Default)]
pub struct VadTrace {
    /// STFT magnitude [n, 129, 4].
    pub stft: Vec<f32>,
    /// Encoder output [n, 128].
    pub encoder: Vec<f32>,
    /// LSTM h / c after each window [n, 128].
    pub h: Vec<f32>,
    pub c: Vec<f32>,
    /// Speech probability per window [n].
    pub probs: Vec<f32>,
}

impl Vad {
    /// Load `repo_or_dir` (an exported silero_vad directory with `config.json` +
    /// `model.safetensors`, or an HF repo with those files) from the lapid cache,
    /// converting it on first use.
    pub fn from_hub(repo_or_dir: &str, dtype: &str, device: &Device) -> Result<Vad> {
        let path = if Path::new(repo_or_dir).is_dir() {
            crate::runtime::resolve::resolve(repo_or_dir, Some(dtype))?.path
        } else {
            hub_model_path(repo_or_dir, dtype, |r, d, o| convert::CONVERT.convert(r, d, o))?
        };
        Vad::load(&path, device)
    }

    /// Load a converted `.lapid` file.
    pub fn load(path: &Path, device: &Device) -> Result<Vad> {
        Vad::from_file(&LapidFile::open(path)?, device)
    }

    /// Load from an in-memory `.lapid` image (zero copy).
    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Vad> {
        Vad::from_file(&LapidFile::from_bytes(bytes)?, device)
    }

    fn from_file(file: &LapidFile, device: &Device) -> Result<Vad> {
        let cfg = VadConfig::from_metadata(&file.metadata)?;
        let loader = Loader::new(file, device, Precision::Stored);
        let net = Net::load(&loader, &cfg)?;
        Ok(Vad {
            inner: Arc::new(Inner { device: device.clone(), info: ModelInfo::from_file(file, "voice-activity-detection"), net }),
        })
    }

    pub fn config(&self) -> &VadConfig {
        self.inner.net.config()
    }

    pub fn sample_rate(&self) -> u32 {
        self.config().sample_rate
    }

    /// Samples per probability (512).
    pub fn window_size(&self) -> usize {
        self.config().window_size
    }

    /// Speech probability of every 512-sample window of `samples` (16 kHz mono), the last
    /// one zero-padded: silero `audio_forward`.
    pub fn probabilities(&self, samples: &[f32]) -> Result<Vec<f32>> {
        Ok(self.run(samples, false)?.probs)
    }

    /// [`probabilities`](Self::probabilities) with the intermediates of the first
    /// `max_windows` windows (all outputs cover only those windows).
    pub fn trace(&self, samples: &[f32], max_windows: usize) -> Result<VadTrace> {
        let w = self.window_size();
        let n = samples.len().div_ceil(w).min(max_windows);
        self.run(&samples[..samples.len().min(n * w)], true)
    }

    /// Speech segments of a whole buffer: silero `get_speech_timestamps`.
    pub fn process(&self, samples: &[f32], opts: &VadOptions) -> Result<Vec<SpeechSegment>> {
        let probs = self.probabilities(samples)?;
        Ok(segments::speech_timestamps(&probs, samples.len(), self.sample_rate(), self.window_size(), opts))
    }

    /// A streaming detector (silero `VADIterator`) sharing this model.
    pub fn stream(&self, opts: &VadOptions) -> VadStream {
        VadStream::new(self, opts)
    }

    fn run(&self, samples: &[f32], traced: bool) -> Result<VadTrace> {
        let net = &self.inner.net;
        let (w, c) = (self.window_size(), self.config().context_size);
        let n = samples.len().div_ceil(w);
        let mut state = LstmState::zeros(self.config().hidden_size);
        let mut out = VadTrace { probs: Vec::with_capacity(n), ..VadTrace::default() };
        let mut inputs = Vec::new();
        let zeros = vec![0f32; c];
        for first in (0..n).step_by(BATCH_WINDOWS) {
            let count = BATCH_WINDOWS.min(n - first);
            inputs.clear();
            for i in first..first + count {
                let start = i * w;
                let context = if i == 0 { &zeros[..] } else { &samples[start - c..start] };
                net.push_input(context, &samples[start..samples.len().min(start + w)], &mut inputs);
            }
            self.windows(&inputs, count, &mut state, traced, &mut out)?;
        }
        Ok(out)
    }

    /// Run `count` prepared windows from `state`, appending to `out`.
    pub(crate) fn windows(&self, inputs: &[f32], count: usize, state: &mut LstmState, traced: bool, out: &mut VadTrace) -> Result<()> {
        let net = &self.inner.net;
        let cfg = self.config();
        let g = 4 * cfg.hidden_size;
        // Small batches are padded with silent windows so every evaluation takes the same
        // kernel paths (backends switch matmul strategy for tiny row counts): the stream,
        // whatever its chunking, then reproduces `probabilities` bit for bit.
        let padded;
        let (inputs, n) = if count < MIN_BATCH_WINDOWS {
            let mut v = inputs.to_vec();
            v.resize(MIN_BATCH_WINDOWS * net.input_len(), 0.0);
            padded = v;
            (&padded[..], MIN_BATCH_WINDOWS)
        } else {
            (inputs, count)
        };
        let mut front = net.front(&self.inner.device, inputs, n, traced)?;
        front.gates.truncate(count * g);
        if let (Some(s), Some(e)) = (front.stft.as_mut(), front.encoder.as_mut()) {
            s.truncate(count * cfg.n_bins() * cfg.stft_frames());
            e.truncate(count * cfg.hidden_size);
        }
        for gx in front.gates.chunks_exact(g) {
            out.probs.push(net.step(state, gx));
            if traced {
                out.h.extend_from_slice(&state.h);
                out.c.extend_from_slice(&state.c);
            }
        }
        if let (Some(s), Some(e)) = (front.stft, front.encoder) {
            out.stft.extend(s);
            out.encoder.extend(e);
        }
        Ok(())
    }

    pub(crate) fn net(&self) -> &Net {
        &self.inner.net
    }
}

impl Model for Vad {
    fn info(&self) -> ModelInfo {
        self.inner.info.clone()
    }

    fn device(&self) -> &Device {
        &self.inner.device
    }
}

/// Registry entry: generic verify and bench (not served).
pub static ENTRY: crate::runtime::ModelEntry = crate::runtime::audio::audio_entry::<Vad>();
