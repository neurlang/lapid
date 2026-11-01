//! Transducer (RNN-T / TDT) pipeline: greedy decoding over encoder frames, and a streaming
//! session that feeds audio or features chunk by chunk with the encoder cache carried.
//!
//! How a model plugs in:
//! * Implement [`Transducer`]: the predictor step and the joint as lazy graphs, blank id,
//!   vocabulary size and (TDT) durations. Nothing else.
//! * Offline models (e.g. Parakeet TDT) encode the signal themselves (see
//!   [`context_windows`] for long audio) and call [`greedy_decode`] on the encoder frames.
//! * Streaming models (e.g. Zipformer RNN-T) also implement [`StreamingTransducer`] (feature
//!   front end, chunk geometry, one encoder chunk with its own cache state) and wrap a
//!   [`TransducerStream`].
//!
//! Greedy semantics (NeMo `GreedyTDTInfer`, icefall `greedy_search`): at frame `t` the joint
//! gives the token argmax and, for TDT, the duration argmax. A non-blank token runs the
//! predictor; a blank with duration 0 advances one frame; after `max_symbols_per_frame`
//! symbols on one frame the decoder advances one frame. With the predictor output fixed the
//! joint runs over up to `joint_window` frames in one evaluation (argmax on the device), and
//! the predictor step for a new token is evaluated together with the next joint window.

use std::ops::Deref;

use lapid::{Device, Error, Result, Tensor, bail};
use lapid_nn::sampling::argmax;
use serde::Serialize;

/// The model's math: predictor and joint. Encoder frames are `[T, H]` realized tensors.
pub trait Transducer {
    fn device(&self) -> &Device;
    fn blank_id(&self) -> u32;
    /// Token logits per frame (vocabulary including blank). TDT duration logits follow them.
    fn vocab_size(&self) -> usize;
    /// TDT duration values (frames) per duration logit; empty for plain RNN-T.
    fn durations(&self) -> &[usize] {
        &[]
    }
    /// Seconds per encoder frame (for timestamps).
    fn frame_seconds(&self) -> f64;
    /// Initial predictor state as realized tensors (empty for a stateless predictor).
    fn predictor_init(&self) -> Result<Vec<Tensor>>;
    /// Lazy predictor step on `token`: (output `[1, H]`, new state with as many tensors as
    /// `state`). `history` holds the tokens emitted before `token` (the start symbol, a blank
    /// with an empty history, is not part of it), so a stateless predictor can build its
    /// token context from it.
    fn predict(&self, history: &[u32], token: u32, state: &[Tensor]) -> Result<(Tensor, Vec<Tensor>)>;
    /// Lazy joint: encoder frames `[n, H]`, predictor output `[1, H]` -> logits
    /// `[n, vocab_size + durations.len()]`.
    fn joint(&self, enc: &Tensor, pred: &Tensor) -> Result<Tensor>;
}

/// One decoder step (a blank step or an emitted token).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Step {
    /// Encoder frame index since the start of decoding.
    pub frame: usize,
    pub token: u32,
    /// Predicted duration in frames (before the blank / max-symbols adjustments; 0 for RNN-T).
    pub duration: usize,
}

impl Step {
    /// Start time in seconds.
    pub fn start(&self, frame_seconds: f64) -> f32 {
        (self.frame as f64 * frame_seconds) as f32
    }

    /// End time in seconds (`frame + duration`).
    pub fn end(&self, frame_seconds: f64) -> f32 {
        ((self.frame + self.duration) as f64 * frame_seconds) as f32
    }
}

/// Greedy decoding options.
#[derive(Clone, Debug)]
pub struct GreedyOptions {
    /// Symbols emitted on one frame before the decoder is forced to the next frame
    /// (NeMo default 10; icefall streaming uses 1).
    pub max_symbols_per_frame: usize,
    /// Frames the joint covers in one evaluation while the predictor output is unchanged.
    pub joint_window: usize,
    /// Keep the host joint logits of every step (forces a window of 1; for verification).
    pub keep_logits: bool,
    /// Also return blank steps (the full path).
    pub record_blanks: bool,
}

impl Default for GreedyOptions {
    fn default() -> Self {
        GreedyOptions { max_symbols_per_frame: 10, joint_window: 16, keep_logits: false, record_blanks: false }
    }
}

/// Result of [`greedy_decode`].
#[derive(Clone, Debug, Default)]
pub struct Decoded {
    pub steps: Vec<Step>,
    /// Joint logits per step when `keep_logits` was set.
    pub logits: Vec<Vec<f32>>,
}

impl Decoded {
    /// Emitted (non-blank) steps.
    pub fn emitted(&self, blank: u32) -> Vec<Step> {
        self.steps.iter().copied().filter(|s| s.token != blank).collect()
    }
}

/// Decode all of `enc` (`[T, H]`, realized) from a fresh state.
pub fn greedy_decode<M: Transducer + ?Sized>(model: &M, enc: &Tensor, opts: &GreedyOptions) -> Result<Decoded> {
    let mut d = GreedyDecoder::new(model, opts.clone())?;
    let steps = d.decode_chunk(model, enc)?;
    Ok(Decoded { steps, logits: d.take_logits() })
}

/// Follow a given path (verification: every stage sees the reference path). Returns the
/// steps (equal to `path` on success) and, with `keep_logits`, the joint logits.
pub fn forced_decode<M: Transducer + ?Sized>(model: &M, enc: &Tensor, path: &[Step], keep_logits: bool) -> Result<Decoded> {
    let opts = GreedyOptions { keep_logits, record_blanks: true, ..GreedyOptions::default() };
    let mut d = GreedyDecoder::new(model, opts)?;
    let steps = d.run(model, enc, Some(path))?;
    Ok(Decoded { steps, logits: d.take_logits() })
}

/// Joint results over frames `[start, start + n)` for one predictor output.
struct Window {
    start: usize,
    tokens: Vec<u32>,
    durations: Vec<usize>,
    logits: Vec<Vec<f32>>,
}

/// Greedy decoder state, carried across encoder chunks.
pub struct GreedyDecoder {
    opts: GreedyOptions,
    /// Realized predictor output for the last token.
    pred_out: Option<Tensor>,
    pred_state: Vec<Tensor>,
    /// Token whose predictor step has not run yet (blank = start symbol).
    pending: Option<u32>,
    history: Vec<u32>,
    symbols: usize,
    /// Frames to skip at the start of the next chunk (a TDT duration past the chunk end).
    skip: usize,
    frames: usize,
    steps: usize,
    last_emission: Option<usize>,
    logits: Vec<Vec<f32>>,
}

impl GreedyDecoder {
    pub fn new<M: Transducer + ?Sized>(model: &M, opts: GreedyOptions) -> Result<GreedyDecoder> {
        let (pred_state, pending) = (model.predictor_init()?, Some(model.blank_id()));
        let (history, logits) = (Vec::new(), Vec::new());
        Ok(GreedyDecoder { opts, pred_out: None, pred_state, pending, history, symbols: 0, skip: 0, frames: 0, steps: 0, last_emission: None, logits })
    }

    /// Emitted tokens so far.
    pub fn tokens(&self) -> &[u32] {
        &self.history
    }

    /// Encoder frames decoded so far.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Frame of the last emitted token.
    pub fn last_emission(&self) -> Option<usize> {
        self.last_emission
    }

    /// Realized predictor state (after the last evaluated step).
    pub fn predictor_state(&self) -> &[Tensor] {
        &self.pred_state
    }

    /// Joint logits kept so far (`keep_logits`), drained.
    pub fn take_logits(&mut self) -> Vec<Vec<f32>> {
        std::mem::take(&mut self.logits)
    }

    /// Decode the next encoder chunk `enc` `[T, H]` (realized); frames are numbered from the
    /// start of the stream. Returns emitted steps (and blanks with `record_blanks`).
    pub fn decode_chunk<M: Transducer + ?Sized>(&mut self, model: &M, enc: &Tensor) -> Result<Vec<Step>> {
        self.run(model, enc, None)
    }

    fn run<M: Transducer + ?Sized>(&mut self, m: &M, enc: &Tensor, forced: Option<&[Step]>) -> Result<Vec<Step>> {
        if enc.shape().len() != 2 {
            bail!("shape_mismatch", "transducer: encoder frames must be [T, H], got {:?}", enc.shape());
        }
        let n = enc.shape()[0];
        let blank = m.blank_id();
        let full = self.opts.keep_logits || forced.is_some();
        let width = if full { 1 } else { self.opts.joint_window.max(1) };
        let max_symbols = self.opts.max_symbols_per_frame.max(1);
        let limit = n * (max_symbols + 1) + 1;
        let mut out = Vec::new();
        let mut t = self.skip.min(n);
        self.skip -= t;
        let mut win: Option<Window> = None;
        let mut taken = 0usize;
        while t < n {
            taken += 1;
            if taken > limit {
                bail!("internal", "transducer decode: step limit exceeded");
            }
            let covered = self.pending.is_none() && win.as_ref().is_some_and(|w| t >= w.start && t < w.start + w.tokens.len());
            if !covered {
                let pending = self.pending.take();
                match self.eval_window(m, enc, t, (t + width).min(n), full, pending) {
                    Ok(w) => win = Some(w),
                    Err(e) => {
                        self.pending = pending;
                        return Err(e);
                    }
                }
            }
            let w = win.as_ref().ok_or_else(|| Error::new("internal", "transducer decode: no joint window"))?;
            let i = t - w.start;
            let frame = self.frames + t;
            let (token, duration) = match forced {
                None => (w.tokens[i], w.durations[i]),
                Some(path) => {
                    let s = path.get(self.steps).ok_or_else(|| Error::new("reference", "replay: reference path ended early"))?;
                    if s.frame != frame {
                        bail!("reference", "replay: step {} at frame {frame}, reference at {}", self.steps, s.frame);
                    }
                    (s.token, s.duration)
                }
            };
            if self.opts.keep_logits {
                self.logits.push(w.logits[i].clone());
            }
            self.steps += 1;
            if token != blank || self.opts.record_blanks {
                out.push(Step { frame, token, duration });
            }
            let mut adv = duration;
            if token != blank {
                self.pending = Some(token);
                self.symbols += 1;
                self.last_emission = Some(frame);
            }
            if adv == 0 && (token == blank || self.symbols >= max_symbols) {
                adv = 1;
            }
            if adv > 0 {
                self.symbols = 0;
            }
            t += adv;
        }
        self.skip += t.saturating_sub(n);
        self.frames += n;
        Ok(out)
    }

    /// Joint over frames `[a, b)`, preceded by the pending predictor step (one evaluation).
    fn eval_window<M: Transducer + ?Sized>(&mut self, m: &M, enc: &Tensor, a: usize, b: usize, full: bool, pending: Option<u32>) -> Result<Window> {
        let (v, durs) = (m.vocab_size(), m.durations());
        let (pred, state) = match pending {
            Some(tok) => m.predict(&self.history, tok, &self.pred_state).map(|(o, s)| (o, Some(s)))?,
            None => (self.pred_out.clone().ok_or_else(|| Error::new("internal", "transducer decode: no predictor output"))?, None),
        };
        let logits = m.joint(&enc.slice(0, a, b)?, &pred)?;
        let mut outs: Vec<Tensor> = if full {
            vec![logits]
        } else if durs.is_empty() {
            vec![logits.argmax_last()?]
        } else {
            vec![logits.slice(1, 0, v)?.argmax_last()?, logits.slice(1, v, v + durs.len())?.argmax_last()?]
        };
        let n_main = outs.len();
        if let Some(s) = &state {
            outs.push(pred);
            outs.extend(s.iter().cloned());
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        let mut realized = m.device().eval(&refs)?;
        if realized.len() != outs.len() {
            bail!("backend", "device.eval returned {} outputs, expected {}", realized.len(), outs.len());
        }
        let mut new = realized.split_off(n_main).into_iter();
        let dev = m.device();
        let n = b - a;
        let duration =
            |i: usize| durs.get(i).copied().ok_or_else(|| Error::new("internal", format!("duration index {i} out of range")));
        let mut w = Window { start: a, tokens: Vec::new(), durations: Vec::new(), logits: Vec::new() };
        if full {
            let host = dev.to_vec_f32(&realized[0])?;
            let width = v + durs.len();
            if host.len() != n * width {
                bail!("shape_mismatch", "joint gave {} logits, expected {}", host.len(), n * width);
            }
            for row in host.chunks_exact(width) {
                w.tokens.push(argmax(&row[..v]) as u32);
                w.durations.push(if durs.is_empty() { 0 } else { duration(argmax(&row[v..]))? });
                w.logits.push(row.to_vec());
            }
        } else {
            let toks = dev.to_vec_i32(&realized[0])?;
            if toks.len() != n {
                bail!("shape_mismatch", "joint argmax gave {} frames, expected {n}", toks.len());
            }
            w.tokens = toks.iter().map(|&x| x as u32).collect();
            w.durations = match realized.get(1) {
                Some(d) => dev.to_vec_i32(d)?.iter().map(|&i| duration(i.max(0) as usize)).collect::<Result<_>>()?,
                None => vec![0; n],
            };
        }
        if state.is_some() {
            self.pred_out = new.next();
            self.pred_state = new.collect();
            if let Some(tok) = pending.filter(|&t| t != m.blank_id()) {
                self.history.push(tok);
            }
        }
        Ok(w)
    }
}

/// Encoder windows for long offline audio over `frames` encoder frames: a core of `core`
/// frames with `ctx` frames of context on each side, as (window start, window end, core
/// start, core end). Short inputs (`<= core + 2 ctx`) are one window.
pub fn context_windows(frames: usize, core: usize, ctx: usize) -> Vec<(usize, usize, usize, usize)> {
    if frames <= core + 2 * ctx || core == 0 {
        return vec![(0, frames, 0, frames)];
    }
    let mut out = Vec::new();
    let mut a = 0;
    while a < frames {
        let b = (a + core).min(frames);
        out.push((a.saturating_sub(ctx), (b + ctx).min(frames), a, b));
        a = b;
    }
    out
}

/// Audio to feature frames, incrementally (row-major `[n, feature_dim]`).
pub trait FeatureFrontend {
    fn accept(&mut self, samples: &[f32]) -> Vec<f32>;
    /// End of input: the remaining frames.
    fn finish(&mut self) -> Vec<f32>;
}

/// A transducer with a chunked streaming encoder.
pub trait StreamingTransducer: Transducer {
    /// Encoder caches carried between chunks (plus anything chunk geometry depends on).
    type EncoderState;
    type Frontend: FeatureFrontend;
    fn feature_dim(&self) -> usize;
    /// (window, shift) in feature frames: a chunk reads `window` frames and advances `shift`.
    fn chunk_frames(&self, state: &Self::EncoderState) -> (usize, usize);
    /// Feature frames appended at the end of input, and the value padding them and the
    /// last partial window.
    fn tail_padding(&self) -> (usize, f32) {
        (0, 0.0)
    }
    /// Encode one window `[window, feature_dim]`; returns realized frames `[T, H]` and
    /// updates `state`.
    fn encode_chunk(&self, feats: &[f32], state: &mut Self::EncoderState) -> Result<Tensor>;
}

/// Timing clock: `std::time::Instant` natively; reads zero on wasm32.
#[derive(Clone, Copy)]
struct Clock(#[cfg(not(target_arch = "wasm32"))] std::time::Instant);

impl Clock {
    fn now() -> Clock {
        Clock(#[cfg(not(target_arch = "wasm32"))] std::time::Instant::now())
    }

    fn ms(&self) -> f64 {
        #[cfg(not(target_arch = "wasm32"))]
        return self.0.elapsed().as_secs_f64() * 1e3;
        #[cfg(target_arch = "wasm32")]
        0.0
    }
}

/// Timing counters of a stream.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct StreamStats {
    pub chunks: usize,
    pub frontend_ms: f64,
    pub encoder_ms: f64,
    pub decode_ms: f64,
}

/// Output of one encoder chunk.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChunkOutput {
    /// First encoder frame of the chunk.
    pub start: usize,
    /// Encoder frames in the chunk.
    pub frames: usize,
    /// Emitted steps (and blanks with `record_blanks`).
    pub steps: Vec<Step>,
}

impl ChunkOutput {
    /// Blank frames since the last emission, given the count before this chunk.
    pub fn trailing(&self, before: usize, blank: u32) -> usize {
        match self.steps.iter().rev().find(|s| s.token != blank) {
            Some(s) => (self.start + self.frames).saturating_sub(s.frame + 1),
            None => before + self.frames,
        }
    }
}

/// Streaming recognizer: audio or features in, per-chunk emissions out.
pub struct TransducerStream<M: StreamingTransducer + ?Sized, P: Deref<Target = M>> {
    model: P,
    frontend: M::Frontend,
    encoder: M::EncoderState,
    decoder: GreedyDecoder,
    /// Pending feature frames, row-major.
    feats: Vec<f32>,
    finished: bool,
    stats: StreamStats,
}

impl<M: StreamingTransducer + ?Sized, P: Deref<Target = M>> TransducerStream<M, P> {
    pub fn new(model: P, encoder: M::EncoderState, frontend: M::Frontend, opts: GreedyOptions) -> Result<Self> {
        let decoder = GreedyDecoder::new(&*model, opts)?;
        Ok(TransducerStream { model, frontend, encoder, decoder, feats: Vec::new(), finished: false, stats: StreamStats::default() })
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn decoder(&self) -> &GreedyDecoder {
        &self.decoder
    }

    pub fn encoder_state(&self) -> &M::EncoderState {
        &self.encoder
    }

    pub fn encoder_state_mut(&mut self) -> &mut M::EncoderState {
        &mut self.encoder
    }

    pub fn stats(&self) -> StreamStats {
        self.stats
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Feed mono samples; one output per encoder chunk that ran.
    pub fn accept_waveform(&mut self, samples: &[f32]) -> Result<Vec<ChunkOutput>> {
        self.check_open()?;
        let (t, frames) = (Clock::now(), self.frontend.accept(samples));
        self.stats.frontend_ms += t.ms();
        self.accept_features(&frames)
    }

    /// End of input: flush the front end, pad the tail and run the remaining chunks.
    pub fn input_finished(&mut self) -> Result<Vec<ChunkOutput>> {
        if self.finished {
            return Ok(Vec::new());
        }
        let (t, frames) = (Clock::now(), self.frontend.finish());
        self.stats.frontend_ms += t.ms();
        let mut out = self.accept_features(&frames)?;
        out.extend(self.finish_features()?);
        Ok(out)
    }

    /// Feed feature frames `[n, feature_dim]` (row-major).
    pub fn accept_features(&mut self, frames: &[f32]) -> Result<Vec<ChunkOutput>> {
        self.check_open()?;
        let f = self.model.feature_dim();
        if f == 0 || !frames.len().is_multiple_of(f) {
            bail!("shape_mismatch", "transducer stream: {} feature values for dim {f}", frames.len());
        }
        self.feats.extend_from_slice(frames);
        self.run_chunks(None)
    }

    /// Tail padding and the remaining (padded) chunks; closes the stream.
    pub fn finish_features(&mut self) -> Result<Vec<ChunkOutput>> {
        self.check_open()?;
        let (tail, value) = self.model.tail_padding();
        self.feats.extend(std::iter::repeat_n(value, tail * self.model.feature_dim()));
        let out = self.run_chunks(Some(value))?;
        self.finished = true;
        Ok(out)
    }

    /// Run every full window; with `flush`, also the last partial ones padded with it.
    fn run_chunks(&mut self, flush: Option<f32>) -> Result<Vec<ChunkOutput>> {
        let f = self.model.feature_dim();
        let mut out = Vec::new();
        loop {
            let (window, shift) = self.model.chunk_frames(&self.encoder);
            if window == 0 || shift == 0 || shift > window {
                bail!("invalid_argument", "transducer stream: bad chunk geometry window {window}, shift {shift}");
            }
            let n = self.feats.len() / f;
            let x = match flush {
                _ if n >= window => self.feats[..window * f].to_vec(),
                Some(v) if n > 0 => {
                    let mut x = self.feats[..n * f].to_vec();
                    x.resize(window * f, v);
                    x
                }
                _ => break,
            };
            let drain = (shift * f).min(self.feats.len());
            self.feats.drain(..drain);
            out.push(self.encode_and_decode(&x)?);
        }
        Ok(out)
    }

    /// Decode encoder frames produced outside the stream (e.g. a traced encoder run).
    pub fn decode_encoded(&mut self, enc: &Tensor) -> Result<ChunkOutput> {
        let t = Clock::now();
        let start = self.decoder.frames();
        let steps = self.decoder.decode_chunk(&*self.model, enc)?;
        self.stats.decode_ms += t.ms();
        self.stats.chunks += 1;
        Ok(ChunkOutput { start, frames: self.decoder.frames() - start, steps })
    }

    fn encode_and_decode(&mut self, x: &[f32]) -> Result<ChunkOutput> {
        let t = Clock::now();
        let enc = self.model.encode_chunk(x, &mut self.encoder)?;
        self.stats.encoder_ms += t.ms();
        self.decode_encoded(&enc)
    }

    fn check_open(&self) -> Result<()> {
        if self.finished {
            bail!("invalid_argument", "transducer stream: input already finished");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// Toy: H = V + D. Encoder rows carry the logits; the predictor output is -20 on the last
    /// token (so it is not repeated on the same frame) and counts its steps in its state.
    struct Toy {
        device: Device,
        v: usize,
        durs: Vec<usize>,
        predicts: Cell<usize>,
    }

    impl Toy {
        fn new(v: usize, durs: Vec<usize>) -> Toy {
            Toy { device: Device::cpu(), v, durs, predicts: Cell::new(0) }
        }

        fn h(&self) -> usize {
            self.v + self.durs.len()
        }

        /// Rows: (token wanted or None for blank, duration index); uploaded as `[T, H]`.
        fn frames(&self, rows: &[(Option<usize>, usize)]) -> (Vec<f32>, Tensor) {
            let h = self.h();
            let mut x = vec![0.0f32; rows.len() * h];
            for (t, &(tok, d)) in rows.iter().enumerate() {
                x[t * h] = 1.0; // blank
                tok.inspect(|&k| x[t * h + k] = 10.0);
                if !self.durs.is_empty() {
                    x[t * h + self.v + d] = 5.0;
                }
            }
            let enc = self.device.upload(&Tensor::from_f32(&x, &[rows.len(), h]).expect("tensor")).expect("upload");
            (x, enc)
        }
    }

    #[rustfmt::skip]
    impl Transducer for Toy {
        fn device(&self) -> &Device { &self.device }
        fn blank_id(&self) -> u32 { 0 }
        fn vocab_size(&self) -> usize { self.v }
        fn durations(&self) -> &[usize] { &self.durs }
        fn frame_seconds(&self) -> f64 { 0.04 }
        fn predictor_init(&self) -> Result<Vec<Tensor>> { Ok(vec![Tensor::from_f32(&[0.0], &[1, 1])?]) }
        fn predict(&self, history: &[u32], token: u32, state: &[Tensor]) -> Result<(Tensor, Vec<Tensor>)> {
            self.predicts.set(self.predicts.get() + 1);
            assert!(!history.contains(&0));
            let mut p = vec![0.0f32; self.h()];
            if token != 0 {
                p[token as usize] = -20.0;
            }
            let next = state[0].add(&Tensor::from_f32(&[1.0], &[1, 1])?)?;
            Ok((Tensor::from_f32(&p, &[1, self.h()])?, vec![next]))
        }
        fn joint(&self, enc: &Tensor, pred: &Tensor) -> Result<Tensor> { enc.add(pred) }
    }

    fn tokens(steps: &[Step]) -> Vec<(usize, u32)> {
        steps.iter().filter(|s| s.token != 0).map(|s| (s.frame, s.token)).collect()
    }

    const ROWS: [(Option<usize>, usize); 7] =
        [(Some(1), 0), (None, 0), (Some(2), 0), (Some(3), 0), (None, 0), (None, 0), (Some(1), 0)];
    const WANT: [(usize, u32); 4] = [(0, 1), (2, 2), (3, 3), (6, 1)];

    #[test]
    fn rnnt_greedy() {
        let m = Toy::new(4, vec![]);
        let (_, enc) = m.frames(&ROWS);
        for window in [1, 3, 16] {
            let opts = GreedyOptions { joint_window: window, max_symbols_per_frame: 1, ..Default::default() };
            assert_eq!(tokens(&greedy_decode(&m, &enc, &opts).expect("decode").steps), WANT, "window {window}");
        }
        let opts = GreedyOptions { keep_logits: true, record_blanks: true, ..Default::default() };
        let d = greedy_decode(&m, &enc, &opts).expect("decode");
        assert_eq!((tokens(&d.steps), d.logits.len()), (WANT.to_vec(), d.steps.len()));
        assert_eq!(forced_decode(&m, &enc, &d.steps, false).expect("replay").steps, d.steps);
        // predictor state: start symbol + one step per emission (each re-checks its frame)
        m.predicts.set(0);
        let mut d = GreedyDecoder::new(&m, GreedyOptions::default()).expect("new");
        assert_eq!(d.decode_chunk(&m, &enc).expect("decode").len(), 4);
        assert_eq!(m.device.to_vec_f32(&d.predictor_state()[0]).expect("state"), vec![5.0]);
        assert_eq!((m.predicts.get(), d.tokens(), d.last_emission()), (5, &[1, 2, 3, 1][..], Some(6)));
    }

    #[test]
    fn tdt_durations_and_max_symbols() {
        // durations [0, 1, 2]: token 1 at frame 0 with duration 2 jumps to frame 2
        let m = Toy::new(4, vec![0, 1, 2]);
        let (_, enc) = m.frames(&[(Some(1), 2), (Some(3), 1), (Some(2), 1), (None, 2), (None, 0), (Some(3), 1)]);
        for window in [1, 4] {
            let opts = GreedyOptions { joint_window: window, record_blanks: true, ..Default::default() };
            let d = greedy_decode(&m, &enc, &opts).expect("decode");
            let path: Vec<(usize, u32, usize)> = d.steps.iter().map(|s| (s.frame, s.token, s.duration)).collect();
            assert_eq!(path, vec![(0, 1, 2), (2, 2, 1), (3, 0, 2), (5, 3, 1)], "window {window}");
        }
        // duration 0 keeps the decoder on frame 0, where tokens 1 > 2 > 3 all score
        let m = Toy::new(4, vec![0, 1]);
        let (mut x, _) = m.frames(&[(Some(1), 0), (None, 1)]);
        (x[2], x[3]) = (9.0, 8.0);
        let enc = m.device.upload(&Tensor::from_f32(&x, &[2, m.h()]).expect("tensor")).expect("upload");
        let d = greedy_decode(&m, &enc, &GreedyOptions { max_symbols_per_frame: 2, ..Default::default() }).expect("decode");
        assert_eq!(tokens(&d.steps), vec![(0, 1), (0, 2)]);
        let w = context_windows(2000, 750, 125);
        assert_eq!(w, vec![(0, 875, 0, 750), (625, 1625, 750, 1500), (1375, 2000, 1500, 2000)]);
        assert_eq!(context_windows(100, 750, 125), vec![(0, 100, 0, 100)]);
    }

    struct Identity;

    #[rustfmt::skip]
    impl FeatureFrontend for Identity {
        fn accept(&mut self, samples: &[f32]) -> Vec<f32> { samples.to_vec() }
        fn finish(&mut self) -> Vec<f32> { Vec::new() }
    }

    /// Streaming toy: a window of 3 feature frames, shift 2, keeps the first 2 rows.
    #[rustfmt::skip]
    impl StreamingTransducer for Toy {
        type EncoderState = usize;
        type Frontend = Identity;
        fn feature_dim(&self) -> usize { self.h() }
        fn chunk_frames(&self, _: &usize) -> (usize, usize) { (3, 2) }
        fn tail_padding(&self) -> (usize, f32) { (1, 0.0) }
        fn encode_chunk(&self, feats: &[f32], chunks: &mut usize) -> Result<Tensor> {
            *chunks += 1;
            self.device.upload(&Tensor::from_f32(&feats[..2 * self.h()], &[2, self.h()])?)
        }
    }

    #[test]
    fn streaming_matches_offline() {
        let m = Toy::new(4, vec![]);
        let ((x, _), h) = (m.frames(&ROWS), m.h());
        let opts = GreedyOptions { max_symbols_per_frame: 1, ..Default::default() };
        let mut s = TransducerStream::new(&m, 0usize, Identity, opts).expect("stream");
        let mut outs = Vec::new();
        for piece in [&x[..h], &x[h..4 * h], &x[4 * h..]] {
            outs.extend(s.accept_waveform(piece).expect("accept"));
        }
        outs.extend(s.input_finished().expect("finish"));
        assert!(s.is_finished() && s.accept_features(&x[..h]).is_err());
        let steps: Vec<Step> = outs.iter().flat_map(|o| o.steps.clone()).collect();
        assert_eq!(tokens(&steps), WANT);
        // 7 frames + 1 tail frame = 8 frames in 4 chunks of 2
        assert_eq!((*s.encoder_state(), s.stats().chunks), (4, 4));
        assert_eq!(outs.iter().map(|o| o.frames).sum::<usize>(), 8);
        assert_eq!(outs.iter().fold(0, |tr, o| o.trailing(tr, 0)), 1);
        assert!((steps[3].start(m.frame_seconds()) - 0.24).abs() < 1e-6);
    }
}
