//! Streaming recognition: audio → online fbank → encoder chunks (all caches carried) →
//! greedy transducer search → partial / final results, with endpointing.
//!
//! Chunk schedule (icefall `streaming_decode.py`, sherpa-onnx): a chunk of `c` encoder frames
//! reads `2c + 13` fbank frames and advances by `2c`. At end of input, 13 + 30 frames of
//! `LOG_EPS` are appended and the remaining frames are processed in padded chunks, so a
//! stream fed in any pieces gives exactly icefall's result.
//!
//! Greedy search emits at most one token per encoder frame (icefall `greedy_search`,
//! sherpa-onnx): with the decoder output fixed, the joiner runs over all remaining frames of
//! the chunk in one evaluation; at the first non-blank frame the token is emitted and the new
//! decoder output and the joiner over the frames after it are evaluated together.

use std::sync::Arc;
use std::time::Instant;

use lapid::{Result, Tensor, bail};
use serde::Serialize;

use super::Inner;
use super::config::{LOG_EPS, OUTPUT_FRAME_SECONDS, PAD_FRAMES, TAIL_FRAMES};
use super::endpoint::EndpointConfig;
use super::fbank::{Fbank, OnlineFbank};
use super::model::encoder::{EncoderState, Plan};
use super::model::first;
use super::model::layer::STATE_NAMES;

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

/// Timing counters of a stream.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct StreamStats {
    pub chunks: usize,
    pub fbank_ms: f64,
    pub encoder_ms: f64,
    pub decode_ms: f64,
}

/// Host copies of one chunk's encoder output, traced activations and new state (verify).
pub(crate) struct ChunkTrace {
    pub encoder_out: Vec<f32>,
    pub traced: Vec<(String, Vec<f32>)>,
    pub state: Vec<(String, Vec<f32>)>,
}

/// A streaming recognizer over one audio stream (see [`super::Zipformer::stream`]).
pub struct ZipformerStream {
    model: Arc<Inner>,
    plan: Plan,
    endpoint: Option<EndpointConfig>,
    fbank: OnlineFbank,
    /// Pending fbank frames, row-major [n, F].
    feats: Vec<f32>,
    state: EncoderState,
    hyp: Vec<i64>,
    /// Realized projected decoder output [1, J] for the last `context` tokens of `hyp`.
    dec: Tensor,
    seg_tokens: Vec<u32>,
    seg_times: Vec<f32>,
    seg_start: usize,
    /// Encoder output frames (25 Hz) decoded so far.
    frames: usize,
    trailing: usize,
    segment: usize,
    finished: bool,
    stats: StreamStats,
}

impl ZipformerStream {
    pub(crate) fn new(model: Arc<Inner>, opts: &StreamOptions) -> Result<ZipformerStream> {
        let cfg = &model.cfg;
        let chunk = opts.chunk_size.unwrap_or(cfg.chunk_size);
        let left = opts.left_context_frames.unwrap_or(cfg.left_context_frames);
        let plan = model.encoder.plan(chunk, left, &model.device)?;
        let state = model.encoder.init_state(&plan);
        let ctx = cfg.context_size;
        let mut hyp = vec![-1i64; ctx - 1];
        hyp.push(i64::from(cfg.blank_id));
        let dec = first(model.device.eval(&[&model.decoder.forward(&hyp)?])?)?;
        Ok(ZipformerStream {
            fbank: OnlineFbank::new(Fbank::new(cfg.feature_dim, cfg.sample_rate)),
            feats: Vec::new(),
            plan,
            endpoint: opts.endpoint,
            state,
            hyp,
            dec,
            seg_tokens: Vec::new(),
            seg_times: Vec::new(),
            seg_start: 0,
            frames: 0,
            trailing: 0,
            segment: 0,
            finished: false,
            stats: StreamStats::default(),
            model,
        })
    }

    /// Chunk size in encoder frames (50 Hz).
    pub fn chunk_size(&self) -> usize {
        self.plan.chunk
    }

    pub fn left_context_frames(&self) -> usize {
        self.plan.left
    }

    /// Audio (seconds) a chunk consumes once the stream is running: `2 chunk` fbank frames.
    pub fn chunk_seconds(&self) -> f64 {
        self.plan.chunk as f64 * 0.02
    }

    pub fn stats(&self) -> StreamStats {
        self.stats
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Feed 16 kHz mono samples; returns a partial result for every chunk that emitted
    /// tokens and a final result at every endpoint.
    pub fn accept_waveform(&mut self, samples: &[f32]) -> Result<Vec<StreamResult>> {
        if self.finished {
            bail!("invalid_argument", "zipformer stream: input already finished");
        }
        let t = Instant::now();
        self.fbank.accept(samples);
        let frames = self.fbank.take_all();
        self.stats.fbank_ms += t.elapsed().as_secs_f64() * 1e3;
        self.accept_features(&frames)
    }

    /// End of input: flush the remaining audio (icefall tail padding) and close the segment.
    pub fn input_finished(&mut self) -> Result<Vec<StreamResult>> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.fbank.finish();
        let frames = self.fbank.take_all();
        let mut out = self.accept_features(&frames)?;
        out.extend(self.finish_features()?);
        Ok(out)
    }

    /// Current (partial or last final) result of the open segment.
    pub fn result(&self) -> StreamResult {
        StreamResult {
            segment: self.segment,
            text: self.model.tokens.text(&self.seg_tokens),
            tokens: self.seg_tokens.clone(),
            timestamps: self.seg_times.clone(),
            start: (self.seg_start as f64 * OUTPUT_FRAME_SECONDS) as f32,
            is_final: self.finished,
        }
    }

    /// Fbank frames [n, F] (row-major) as if computed from audio.
    pub(crate) fn accept_features(&mut self, frames: &[f32]) -> Result<Vec<StreamResult>> {
        self.feats.extend_from_slice(frames);
        let f = self.model.cfg.feature_dim;
        let (window, shift) = (2 * self.plan.chunk + PAD_FRAMES, 2 * self.plan.chunk);
        let mut out = Vec::new();
        while self.feats.len() >= window * f {
            let x: Vec<f32> = self.feats[..window * f].to_vec();
            self.feats.drain(..shift * f);
            self.step(&x, false, &mut out)?;
        }
        Ok(out)
    }

    /// Tail padding (13 + 30 frames of `LOG_EPS`) and the remaining chunks.
    pub(crate) fn finish_features(&mut self) -> Result<Vec<StreamResult>> {
        let f = self.model.cfg.feature_dim;
        self.feats.extend(std::iter::repeat_n(LOG_EPS, (PAD_FRAMES + TAIL_FRAMES) * f));
        let (window, shift) = (2 * self.plan.chunk + PAD_FRAMES, 2 * self.plan.chunk);
        let mut out = Vec::new();
        while !self.feats.is_empty() {
            let n = (self.feats.len() / f).min(window);
            let mut x: Vec<f32> = self.feats[..n * f].to_vec();
            x.resize(window * f, LOG_EPS);
            let drain = (shift * f).min(self.feats.len());
            self.feats.drain(..drain);
            self.step(&x, false, &mut out)?;
        }
        self.finished = true;
        let r = self.result();
        if !r.tokens.is_empty() || out.is_empty() {
            out.push(r);
        }
        Ok(out)
    }

    /// Run one chunk on fbank frames `x` [2c + 13, F]; with `trace`, return host copies.
    pub(crate) fn step(&mut self, x: &[f32], trace: bool, out: &mut Vec<StreamResult>) -> Result<Option<ChunkTrace>> {
        let m = Arc::clone(&self.model);
        let t0 = Instant::now();
        let window = 2 * self.plan.chunk + PAD_FRAMES;
        let feats = Tensor::from_f32(x, &[window, m.cfg.feature_dim])?;
        let g = m.encoder.chunk(&self.plan, &feats, &self.state, trace)?;
        let mut outputs: Vec<&Tensor> = vec![&g.out];
        outputs.extend(g.traced.iter());
        let states = g.state.tensors();
        let n_traced = g.traced.len();
        outputs.extend(states.iter().copied());
        let realized = m.device.eval(&outputs)?;
        let enc = realized[0].clone();
        let new_state = EncoderState::from_tensors(&realized[1 + n_traced..], g.state.processed)?;
        let traced = if trace {
            let mut tr = ChunkTrace { encoder_out: m.device.to_vec_f32(&enc)?, traced: Vec::new(), state: Vec::new() };
            for (lazy, t) in g.traced.iter().zip(&realized[1..1 + n_traced]) {
                tr.traced.push((lazy.name().unwrap_or("unnamed").to_string(), m.device.to_vec_f32(t)?));
            }
            tr.state.push(("state.embed".into(), m.device.to_vec_f32(&new_state.embed)?));
            let mut li = 0;
            for (i, &layers) in m.cfg.num_encoder_layers.iter().enumerate() {
                for j in 0..layers {
                    for (name, t) in STATE_NAMES.iter().zip(new_state.layers[li].tensors()) {
                        tr.state.push((format!("state.{i}.{j}.{name}"), m.device.to_vec_f32(t)?));
                    }
                    li += 1;
                }
            }
            Some(tr)
        } else {
            None
        };
        self.state = new_state;
        self.stats.encoder_ms += t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        let emitted = self.greedy(&enc)?;
        self.stats.decode_ms += t1.elapsed().as_secs_f64() * 1e3;
        self.stats.chunks += 1;
        if emitted {
            out.push(self.result());
        }
        self.check_endpoint(out);
        Ok(traced)
    }

    /// Greedy search over this chunk's frames; true when a token was emitted.
    fn greedy(&mut self, enc: &Tensor) -> Result<bool> {
        let m = Arc::clone(&self.model);
        let (t, blank, ctx) = (enc.shape()[0], m.cfg.blank_id as i32, m.cfg.context_size);
        let argmax = |from: usize, dec: &Tensor| -> Result<Tensor> { m.joiner.logits(&enc.slice(0, from, t)?, dec)?.argmax_last() };
        let mut start = 0;
        let mut am = m.device.to_vec_i32(&first(m.device.eval(&[&argmax(0, &self.dec)?])?)?)?;
        let mut emitted = false;
        loop {
            let Some(j) = am.iter().position(|&y| y != blank) else {
                self.trailing += t - start;
                break;
            };
            let frame = start + j;
            let token = am[j] as u32;
            self.hyp.push(i64::from(token));
            self.seg_tokens.push(token);
            self.seg_times.push(((self.frames + frame) as f64 * OUTPUT_FRAME_SECONDS) as f32);
            self.trailing = 0;
            emitted = true;
            start = frame + 1;
            let dec = m.decoder.forward(&self.hyp[self.hyp.len() - ctx..])?;
            if start < t {
                let r = m.device.eval(&[&dec, &argmax(start, &dec)?])?;
                self.dec = r[0].clone();
                am = m.device.to_vec_i32(&r[1])?;
            } else {
                self.dec = first(m.device.eval(&[&dec])?)?;
                break;
            }
        }
        self.frames += t;
        Ok(emitted)
    }

    fn check_endpoint(&mut self, out: &mut Vec<StreamResult>) {
        let Some(ep) = self.endpoint else { return };
        let utterance = ((self.frames - self.seg_start) as f64 * OUTPUT_FRAME_SECONDS) as f32;
        let trailing = (self.trailing as f64 * OUTPUT_FRAME_SECONDS) as f32;
        if !ep.is_endpoint(utterance, trailing) {
            return;
        }
        if !self.seg_tokens.is_empty() {
            let mut r = self.result();
            r.is_final = true;
            out.push(r);
            self.segment += 1;
        }
        // sherpa-onnx Reset: encoder state and decoder context are kept.
        self.seg_tokens.clear();
        self.seg_times.clear();
        self.seg_start = self.frames;
        self.trailing = 0;
    }
}
