//! Streaming recognition: the shared `TransducerStream` (online fbank, icefall chunk schedule
//! and tail padding, greedy search with all caches carried) plus segments and endpoints.
//! A stream fed in any pieces gives exactly icefall's tokens.

use std::sync::Arc;

use lapid::Result;
use serde::Serialize;

use super::config::OUTPUT_FRAME_SECONDS;
use super::{Zipformer, ZipformerModel};
use crate::runtime::endpoint::EndpointConfig;
use crate::runtime::serve::{AsrStream, Delay, StreamParams, StreamUpdate};
use crate::runtime::transducer::{ChunkOutput, GreedyOptions, StreamStats, Transducer, TransducerStream};

/// Streaming configuration.
#[derive(Clone, Debug)]
pub struct StreamOptions {
    /// Encoder frames (50 Hz) per chunk; `None` = the model's default (16 = 320 ms for the
    /// LibriSpeech model). Latency grows with the chunk, accuracy too.
    pub chunk_size: Option<usize>,
    /// Cached left context in encoder frames (50 Hz); `None` = the model's default.
    pub left_context_frames: Option<usize>,
    /// Endpoint rules; `None` = one segment for the whole stream.
    pub endpoint: Option<EndpointConfig>,
}

impl Default for StreamOptions {
    fn default() -> StreamOptions {
        StreamOptions { chunk_size: None, left_context_frames: None, endpoint: Some(EndpointConfig::default()) }
    }
}

/// Recognition result of one segment (between endpoints).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct StreamResult {
    /// Segment index (incremented at every endpoint that closed a non-empty segment).
    pub segment: usize,
    pub text: String,
    pub tokens: Vec<u32>,
    /// Emission time of each token (seconds from the stream start).
    pub timestamps: Vec<f32>,
    /// Segment start (seconds).
    pub start: f32,
    /// Closed by an endpoint or the end of input.
    pub is_final: bool,
}

/// One greedy search (icefall `greedy_search`: one symbol per frame; the joiner covers a
/// whole chunk per evaluation).
pub(crate) fn greedy() -> GreedyOptions {
    GreedyOptions { max_symbols_per_frame: 1, joint_window: 1 << 16, ..GreedyOptions::default() }
}

/// A streaming recognizer over one audio stream (see [`super::Zipformer::stream`]).
pub struct ZipformerStream {
    inner: TransducerStream<ZipformerModel, Arc<ZipformerModel>>,
    endpoint: Option<EndpointConfig>,
    seg_tokens: Vec<u32>,
    seg_times: Vec<f32>,
    /// Encoder frame (25 Hz) where the open segment starts.
    seg_start: usize,
    trailing: usize,
    segment: usize,
}

impl ZipformerStream {
    pub(crate) fn new(model: Arc<ZipformerModel>, opts: &StreamOptions) -> Result<ZipformerStream> {
        let cfg = &model.cfg;
        let state = model.chunk_state(opts.chunk_size.unwrap_or(cfg.chunk_size), opts.left_context_frames.unwrap_or(cfg.left_context_frames))?;
        let frontend = model.frontend();
        Ok(ZipformerStream {
            inner: TransducerStream::new(model, state, frontend, greedy())?,
            endpoint: opts.endpoint,
            seg_tokens: Vec::new(),
            seg_times: Vec::new(),
            seg_start: 0,
            trailing: 0,
            segment: 0,
        })
    }

    /// Chunk size in encoder frames (50 Hz).
    pub fn chunk_size(&self) -> usize {
        self.inner.encoder_state().plan.chunk
    }

    pub fn left_context_frames(&self) -> usize {
        self.inner.encoder_state().plan.left
    }

    /// Audio (seconds) a chunk consumes once the stream is running.
    pub fn chunk_seconds(&self) -> f64 {
        self.chunk_size() as f64 * 0.02
    }

    pub fn stats(&self) -> StreamStats {
        self.inner.stats()
    }

    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    /// Feed 16 kHz mono samples; a partial result for every chunk that emitted tokens and a
    /// final result at every endpoint.
    pub fn accept_waveform(&mut self, samples: &[f32]) -> Result<Vec<StreamResult>> {
        let chunks = self.inner.accept_waveform(samples)?;
        Ok(self.results(&chunks))
    }

    /// End of input: flush the audio (icefall tail padding) and close the segment.
    pub fn input_finished(&mut self) -> Result<Vec<StreamResult>> {
        if self.inner.is_finished() {
            return Ok(Vec::new());
        }
        let chunks = self.inner.input_finished()?;
        let mut out = self.results(&chunks);
        let r = self.result();
        if !r.tokens.is_empty() || out.is_empty() {
            out.push(r);
        }
        Ok(out)
    }

    /// Current (partial or last final) result of the open segment.
    pub fn result(&self) -> StreamResult {
        StreamResult {
            segment: self.segment,
            text: self.inner.model().tokens.text(&self.seg_tokens),
            tokens: self.seg_tokens.clone(),
            timestamps: self.seg_times.clone(),
            start: (self.seg_start as f64 * OUTPUT_FRAME_SECONDS) as f32,
            is_final: self.inner.is_finished(),
        }
    }

    /// Segment bookkeeping of decoded chunks.
    pub(crate) fn results(&mut self, chunks: &[ChunkOutput]) -> Vec<StreamResult> {
        let blank = self.inner.model().blank_id();
        let mut out = Vec::new();
        for c in chunks {
            let mut emitted = false;
            for s in c.steps.iter().filter(|s| s.token != blank) {
                self.seg_tokens.push(s.token);
                self.seg_times.push(s.start(OUTPUT_FRAME_SECONDS));
                emitted = true;
            }
            self.trailing = c.trailing(self.trailing, blank);
            if emitted {
                out.push(StreamResult { is_final: false, ..self.result() });
            }
            self.check_endpoint(c.start + c.frames, &mut out);
        }
        out
    }

    fn check_endpoint(&mut self, frames: usize, out: &mut Vec<StreamResult>) {
        let Some(ep) = self.endpoint else { return };
        let seconds = |n: usize| (n as f64 * OUTPUT_FRAME_SECONDS) as f32;
        if !ep.is_endpoint(seconds(frames - self.seg_start), seconds(self.trailing)) {
            return;
        }
        if !self.seg_tokens.is_empty() {
            let mut r = self.result();
            r.is_final = true;
            out.push(r);
            self.segment += 1;
        }
        // sherpa-onnx Reset: encoder state and decoder context are kept
        self.seg_tokens.clear();
        self.seg_times.clear();
        (self.seg_start, self.trailing) = (frames, 0);
    }
}

/// Chunk and left context (50 Hz encoder frames) a [`Delay`] level streams with: the chunk sizes
/// icefall trains the model with (16 = 320 ms, 32 = 640 ms, 64 = 1.28 s), paired with its
/// decoding defaults. Longer chunks give the encoder more context per step; the model has no
/// level beyond its largest trained chunk, so `High` and `XHigh` clamp to `Medium`.
pub(crate) fn delay_plan(delay: Delay) -> (usize, usize) {
    match delay {
        Delay::Minimal => (16, 128),
        Delay::Low => (32, 128),
        Delay::Medium | Delay::High | Delay::XHigh => (64, 256),
    }
}

/// [`AsrStream`] of a [`ZipformerStream`]: one segment, or one per endpoint.
struct ZipformerAsr(ZipformerStream);

fn update(r: StreamResult) -> StreamUpdate {
    StreamUpdate { segment: r.segment, text: r.text, is_final: r.is_final, start: r.start }
}

impl AsrStream for ZipformerAsr {
    fn accept(&mut self, samples_16k: &[f32]) -> Result<Vec<StreamUpdate>> {
        Ok(self.0.accept_waveform(samples_16k)?.into_iter().map(update).collect())
    }

    fn finish(&mut self) -> Result<Vec<StreamUpdate>> {
        Ok(self.0.input_finished()?.into_iter().map(update).collect())
    }
}

impl Zipformer {
    /// A stream for serving at `params.delay`; a model whose geometry cannot run that plan
    /// (see `ZipformerConfig::check_stream`) streams at its own defaults.
    pub(crate) fn open_asr_stream(&self, params: &StreamParams) -> Result<Box<dyn AsrStream>> {
        let (chunk, left) = delay_plan(params.delay);
        let plan = self.config().check_stream(chunk, left).is_ok().then_some((chunk, left));
        let opts = StreamOptions { chunk_size: plan.map(|p| p.0), left_context_frames: plan.map(|p| p.1), endpoint: params.endpoint };
        Ok(Box::new(ZipformerAsr(self.stream(&opts)?)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zipformer::config::tests::medium;

    #[test]
    fn every_delay_level_has_a_valid_plan() {
        let cfg = medium();
        for d in Delay::ALL {
            let (chunk, left) = delay_plan(d);
            cfg.check_stream(chunk, left).unwrap_or_else(|e| panic!("{}: {}", d.name(), e.message));
        }
        // latency never decreases with the level
        let chunks: Vec<usize> = Delay::ALL.iter().map(|d| delay_plan(*d).0).collect();
        assert!(chunks.windows(2).all(|w| w[0] <= w[1]), "{chunks:?}");
    }

    #[test]
    fn results_become_updates() {
        let r = StreamResult { segment: 2, text: "ahoj".into(), tokens: vec![5], timestamps: vec![0.4], start: 1.5, is_final: true };
        assert_eq!(update(r), StreamUpdate { segment: 2, text: "ahoj".into(), is_final: true, start: 1.5 });
    }
}
