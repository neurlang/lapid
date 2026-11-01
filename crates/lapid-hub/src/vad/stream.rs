//! Streaming VAD: silero `VADIterator` over chunks of any size.

use lapid::Result;

use super::model::LstmState;
use super::segments::{Detector, VadEvent, VadOptions};
use super::{Vad, VadTrace};

/// Streaming detector bound to a [`Vad`] (shares its weights). Samples are buffered into
/// 512-sample windows; the LSTM state and the 64-sample context carry across calls, so the
/// probabilities equal [`Vad::probabilities`] of the concatenated input.
pub struct VadStream {
    vad: Vad,
    opts: VadOptions,
    detector: Detector,
    state: LstmState,
    /// Last `context_size` samples of the previous window (zeros at the start).
    context: Vec<f32>,
    /// Samples not yet forming a full window.
    pending: Vec<f32>,
    /// Real samples accepted since the last reset.
    samples: usize,
    last_probs: Vec<f32>,
}

impl VadStream {
    pub fn new(vad: &Vad, opts: &VadOptions) -> VadStream {
        let cfg = vad.config();
        VadStream {
            vad: vad.clone(),
            opts: opts.clone(),
            detector: Detector::new(opts, cfg.sample_rate),
            state: LstmState::zeros(cfg.hidden_size),
            context: vec![0.0; cfg.context_size],
            pending: Vec::with_capacity(cfg.window_size),
            samples: 0,
            last_probs: Vec::new(),
        }
    }

    pub fn options(&self) -> &VadOptions {
        &self.opts
    }

    /// Whether the detector is inside a speech segment.
    pub fn in_speech(&self) -> bool {
        self.detector.triggered()
    }

    /// Samples accepted since the start (or the last reset / flush).
    pub fn samples_accepted(&self) -> usize {
        self.samples
    }

    /// Probabilities of the windows completed by the last `accept_waveform` / `flush` call.
    pub fn last_probabilities(&self) -> &[f32] {
        &self.last_probs
    }

    /// Feed 16 kHz mono samples (any length); returns the events of the windows completed.
    /// All complete windows of one call run as one device evaluation.
    pub fn accept_waveform(&mut self, samples: &[f32]) -> Result<Vec<VadEvent>> {
        self.samples += samples.len();
        self.pending.extend_from_slice(samples);
        let w = self.vad.window_size();
        let count = self.pending.len() / w;
        self.last_probs.clear();
        if count == 0 {
            return Ok(Vec::new());
        }
        let events = self.run(count)?;
        self.pending.drain(..count * w);
        Ok(events)
    }

    /// End of stream: run the buffered remainder as a zero-padded window and close an open
    /// segment at the last real sample (like `get_speech_timestamps` does at the end of the
    /// audio; `VADIterator` itself has no end). The stream is reset afterwards.
    pub fn flush(&mut self) -> Result<Vec<VadEvent>> {
        let w = self.vad.window_size();
        let mut events = Vec::new();
        self.last_probs.clear();
        if !self.pending.is_empty() {
            self.pending.resize(w, 0.0);
            events = self.run(1)?;
        }
        let total = self.samples;
        for ev in &mut events {
            match ev {
                VadEvent::SpeechStart { sample } | VadEvent::SpeechEnd { sample } => *sample = (*sample).min(total),
            }
        }
        if self.detector.triggered() {
            events.push(VadEvent::SpeechEnd { sample: total });
        }
        let probs = std::mem::take(&mut self.last_probs);
        self.reset();
        self.last_probs = probs;
        Ok(events)
    }

    /// Forget all state (LSTM, context, buffered samples, detector).
    pub fn reset(&mut self) {
        let cfg = self.vad.config();
        self.detector.reset();
        self.state = LstmState::zeros(cfg.hidden_size);
        self.context = vec![0.0; cfg.context_size];
        self.pending.clear();
        self.samples = 0;
        self.last_probs.clear();
    }

    /// Run the first `count` buffered windows.
    fn run(&mut self, count: usize) -> Result<Vec<VadEvent>> {
        let (w, c) = (self.vad.window_size(), self.context.len());
        let net = self.vad.net();
        let mut inputs = Vec::with_capacity(count * net.input_len());
        for i in 0..count {
            let chunk = &self.pending[i * w..(i + 1) * w];
            net.push_input(&self.context, chunk, &mut inputs);
            self.context.copy_from_slice(&chunk[w - c..]);
        }
        let mut out = VadTrace::default();
        self.vad.windows(&inputs, count, &mut self.state, false, &mut out)?;
        let events = out.probs.iter().filter_map(|&p| self.detector.push(p, w)).collect();
        self.last_probs = out.probs;
        Ok(events)
    }
}
