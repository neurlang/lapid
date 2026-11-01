//! TTS / audio-stream pipeline: text front end -> segments, an autoregressive frame
//! generator per segment, codec / vocoder decoding, and a chunked PCM iterator with
//! end-of-speech rules, fade-in, frame budgets and cancellation.
//!
//! How a model plugs in: implement [`SpeechModel`]. `segments` turns text into token or
//! phoneme segments (with frame budgets), `begin_segment` makes the per-segment generation
//! state (e.g. a fresh KV cache), and `step` produces one frame: the AR step plus the codec
//! decode of that frame, ideally in one device evaluation, returning PCM and an optional
//! end-of-speech logit. Utterance-wide state (codec caches, voice, noise RNG, conditioning)
//! is the model's `Stream` type, built by the model from its own options. [`SpeechStream`]
//! does the rest.

use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use lapid::{Result, bail};
use serde::Serialize;

/// One segment of text to speak.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment<I> {
    /// Tokens, phonemes, or whatever the model's `step` consumes.
    pub input: I,
    /// Frame budget of the segment.
    pub max_frames: usize,
    /// Frames still generated after end of speech was detected.
    pub frames_after_eos: usize,
}

/// One generated frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Frame {
    /// Decoded mono PCM of this frame.
    pub pcm: Vec<f32>,
    /// End-of-speech logit, compared with [`StreamOptions::eos_threshold`].
    pub eos_logit: Option<f32>,
    /// The model ends the segment after this frame.
    pub end: bool,
}

/// The model's math.
pub trait SpeechModel {
    /// One segment's input (e.g. token ids).
    type Input;
    /// Utterance-wide state: codec / vocoder caches, voice, RNG, conditioning.
    type Stream;
    /// Per-segment generation state (e.g. the backbone KV cache and previous latent).
    type SegmentState;
    fn sample_rate(&self) -> u32;
    /// Text front end: normalization, tokens / phonemes, splitting with frame budgets.
    fn segments(&self, text: &str) -> Result<Vec<Segment<Self::Input>>>;
    fn begin_segment(&self, stream: &mut Self::Stream, input: &Self::Input) -> Result<Self::SegmentState>;
    /// Frame `frame` (0-based within the segment): generate and decode.
    fn step(&self, stream: &mut Self::Stream, input: &Self::Input, seg: &mut Self::SegmentState, frame: usize) -> Result<Frame>;
    /// Codec tail after the last frame.
    fn flush(&self, _stream: &mut Self::Stream) -> Result<Vec<f32>> {
        Ok(Vec::new())
    }
}

/// Streaming options.
#[derive(Clone, Debug)]
pub struct StreamOptions {
    /// Frames per yielded PCM chunk.
    pub frames_per_chunk: usize,
    /// End of speech once `eos_logit > threshold`; `None` = only `Frame::end` and budgets.
    pub eos_threshold: Option<f32>,
    /// Frames before end of speech may be detected.
    pub min_frames_before_eos: usize,
    /// Linear fade-in over the first samples of the stream.
    pub fade_in_samples: usize,
    /// Checked before every frame; when set the stream ends (see [`SpeechStream::cancelled`]).
    pub cancel: Option<Arc<AtomicBool>>,
}

impl Default for StreamOptions {
    fn default() -> Self {
        StreamOptions { frames_per_chunk: 1, eos_threshold: None, min_frames_before_eos: 0, fade_in_samples: 0, cancel: None }
    }
}

/// Timing counters (milliseconds since the stream was created).
#[derive(Clone, Debug, Default, Serialize)]
pub struct StreamStats {
    pub first_audio_ms: Option<f64>,
    pub frames: usize,
    pub samples: usize,
    pub elapsed_ms: f64,
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

struct Current<I, S> {
    seg: Segment<I>,
    state: S,
    frames: usize,
    /// Frames left after end of speech was detected.
    remaining: Option<usize>,
}

/// Iterator of PCM chunks (mono f32 at the model's sample rate).
pub struct SpeechStream<M: SpeechModel + ?Sized, P: Deref<Target = M>> {
    model: P,
    segments: VecDeque<Segment<M::Input>>,
    cur: Option<Current<M::Input, M::SegmentState>>,
    stream: M::Stream,
    opts: StreamOptions,
    emitted: usize,
    clock: Clock,
    stats: StreamStats,
    done: bool,
    cancelled: bool,
}

/// Segment `text` with the model's front end and stream it.
pub fn speak<M: SpeechModel + ?Sized, P: Deref<Target = M>>(
    model: P,
    text: &str,
    stream: M::Stream,
    opts: StreamOptions,
) -> Result<SpeechStream<M, P>> {
    let segments = model.segments(text)?;
    if segments.is_empty() {
        bail!("invalid_argument", "nothing to say: the text is empty");
    }
    Ok(SpeechStream::new(model, segments, stream, opts))
}

/// Speak `text` to completion: all PCM.
pub fn synthesize<M: SpeechModel + ?Sized, P: Deref<Target = M>>(
    model: P,
    text: &str,
    stream: M::Stream,
    opts: StreamOptions,
) -> Result<Vec<f32>> {
    let mut out = Vec::new();
    for chunk in speak(model, text, stream, opts)? {
        out.extend(chunk?);
    }
    Ok(out)
}

impl<M: SpeechModel + ?Sized, P: Deref<Target = M>> SpeechStream<M, P> {
    pub fn new(model: P, segments: Vec<Segment<M::Input>>, stream: M::Stream, opts: StreamOptions) -> Self {
        SpeechStream {
            model,
            segments: segments.into(),
            cur: None,
            stream,
            opts,
            emitted: 0,
            clock: Clock::now(),
            stats: StreamStats::default(),
            done: false,
            cancelled: false,
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.model.sample_rate()
    }

    pub fn stats(&self) -> StreamStats {
        let mut s = self.stats.clone();
        s.elapsed_ms = self.clock.ms();
        s
    }

    /// The utterance state (e.g. recorded latents for verification).
    pub fn state(&self) -> &M::Stream {
        &self.stream
    }

    pub fn state_mut(&mut self) -> &mut M::Stream {
        &mut self.stream
    }

    /// Ended by the cancel flag.
    pub fn cancelled(&self) -> bool {
        self.cancelled
    }

    fn is_cancelled(&self) -> bool {
        self.opts.cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed))
    }

    /// One frame of PCM; `Ok(None)` when every segment is finished.
    pub fn next_frame(&mut self) -> Result<Option<Vec<f32>>> {
        let model = &*self.model;
        let mut cur = match self.cur.take() {
            Some(c) => c,
            None => match self.segments.pop_front() {
                Some(seg) => {
                    let state = model.begin_segment(&mut self.stream, &seg.input)?;
                    Current { seg, state, frames: 0, remaining: None }
                }
                None => return Ok(None),
            },
        };
        let frame = model.step(&mut self.stream, &cur.seg.input, &mut cur.state, cur.frames)?;

        // HF PocketTTSEosThresholdCriteria: the crossing frame and `frames_after_eos` more.
        let can_end = cur.frames >= self.opts.min_frames_before_eos;
        cur.frames += 1;
        let crossed = match (self.opts.eos_threshold, frame.eos_logit) {
            (Some(th), Some(l)) => l > th,
            _ => false,
        };
        if cur.remaining.is_none() && can_end && crossed {
            cur.remaining = Some(cur.seg.frames_after_eos);
        }
        let finished = match cur.remaining {
            Some(0) => true,
            Some(n) => {
                cur.remaining = Some(n - 1);
                false
            }
            None => false,
        } || frame.end
            || cur.frames >= cur.seg.max_frames;
        if !finished {
            self.cur = Some(cur);
        }
        let mut pcm = frame.pcm;
        self.fade(&mut pcm);
        self.stats.frames += 1;
        Ok(Some(pcm))
    }

    fn fade(&mut self, pcm: &mut [f32]) {
        let fade = self.opts.fade_in_samples;
        for (i, s) in pcm.iter_mut().enumerate() {
            let pos = self.emitted + i;
            if pos >= fade {
                break;
            }
            *s *= pos as f32 / (fade.max(2) - 1) as f32;
        }
        self.emitted += pcm.len();
        self.stats.samples += pcm.len();
        if self.stats.first_audio_ms.is_none() && !pcm.is_empty() {
            self.stats.first_audio_ms = Some(self.clock.ms());
        }
    }
}

impl<M: SpeechModel + ?Sized, P: Deref<Target = M>> Iterator for SpeechStream<M, P> {
    type Item = Result<Vec<f32>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let mut out = Vec::new();
        for _ in 0..self.opts.frames_per_chunk.max(1) {
            if self.is_cancelled() {
                self.cancelled = true;
                self.done = true;
                break;
            }
            match self.next_frame() {
                Ok(Some(pcm)) => out.extend(pcm),
                Ok(None) => {
                    self.done = true;
                    match self.model.flush(&mut self.stream) {
                        Ok(mut tail) => {
                            self.fade(&mut tail);
                            out.extend(tail);
                        }
                        Err(e) => return Some(Err(e)),
                    }
                    break;
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            }
        }
        if out.is_empty() { None } else { Some(Ok(out)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Toy: one segment per character; frame `f` of segment `s` is 2 samples of
    /// `100 s + f`, its eos logit is `f`; the codec counts decoded frames and flushes one
    /// sample of -1.
    struct Toy {
        max_frames: usize,
        end_at: Option<usize>,
    }

    impl SpeechModel for Toy {
        type Input = usize;
        type Stream = usize;
        type SegmentState = Vec<usize>;
        fn sample_rate(&self) -> u32 {
            24_000
        }
        fn segments(&self, text: &str) -> Result<Vec<Segment<usize>>> {
            Ok((0..text.chars().count()).map(|i| Segment { input: i, max_frames: self.max_frames, frames_after_eos: 1 }).collect())
        }
        fn begin_segment(&self, _: &mut usize, _: &usize) -> Result<Vec<usize>> {
            Ok(Vec::new())
        }
        fn step(&self, decoded: &mut usize, input: &usize, seg: &mut Vec<usize>, frame: usize) -> Result<Frame> {
            assert_eq!(seg.len(), frame);
            seg.push(frame);
            *decoded += 1;
            let v = (100 * input + frame) as f32;
            Ok(Frame { pcm: vec![v, v], eos_logit: Some(frame as f32), end: self.end_at == Some(frame) })
        }
        fn flush(&self, _: &mut usize) -> Result<Vec<f32>> {
            Ok(vec![-1.0])
        }
    }

    fn opts(frames_per_chunk: usize, min: usize) -> StreamOptions {
        StreamOptions { frames_per_chunk, eos_threshold: Some(2.5), min_frames_before_eos: min, ..Default::default() }
    }

    #[test]
    fn eos_rules_and_chunks() {
        let m = Toy { max_frames: 10, end_at: None };
        // eos crosses at frame 3, plus 1 frame: 5 frames per segment
        let s = speak(&m, "ab", 0usize, opts(3, 0)).expect("speak");
        let chunks: Vec<Vec<f32>> = s.map(|c| c.expect("chunk")).collect();
        assert_eq!(chunks.iter().map(Vec::len).collect::<Vec<_>>(), vec![6, 6, 6, 3]);
        let all: Vec<f32> = chunks.concat();
        assert_eq!(&all[..10], &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
        assert_eq!(&all[10..12], &[100.0, 100.0]);
        assert_eq!(all.last(), Some(&-1.0));

        // eos only counts from frame 4: crossing at 4 + 1 = 6 frames
        let all = synthesize(&m, "a", 0usize, opts(2, 4)).expect("synth");
        assert_eq!(all.len(), 6 * 2 + 1);
    }

    #[test]
    fn budgets_end_flag_and_stats() {
        let m = Toy { max_frames: 2, end_at: None };
        assert_eq!(synthesize(&m, "abc", 0usize, opts(1, 0)).expect("synth").len(), 3 * 2 * 2 + 1);
        let m = Toy { max_frames: 10, end_at: Some(0) };
        let mut s = speak(&m, "ab", 0usize, StreamOptions::default()).expect("speak");
        assert_eq!(s.next().expect("chunk").expect("ok"), vec![0.0, 0.0]);
        assert_eq!(s.next().expect("chunk").expect("ok"), vec![100.0, 100.0]);
        assert_eq!(s.next().expect("chunk").expect("ok"), vec![-1.0]);
        assert!(s.next().is_none());
        assert_eq!(*s.state(), 2);
        let st = s.stats();
        assert_eq!((st.frames, st.samples), (2, 5));
        assert!(st.first_audio_ms.is_some());
        assert!(speak(&m, "", 0usize, StreamOptions::default()).is_err());
    }

    #[test]
    fn fade_and_cancel() {
        let m = Toy { max_frames: 10, end_at: None };
        let o = StreamOptions { fade_in_samples: 4, ..opts(1, 0) };
        let all = synthesize(&m, "b", 0usize, o).expect("synth");
        // segment 0 frames: 0 0 1 1 2 2 ... ; fade weights 0, 1/3, 2/3, 1
        assert_eq!(&all[..6], &[0.0, 0.0, 1.0 * 2.0 / 3.0, 1.0, 2.0, 2.0]);

        let flag = Arc::new(AtomicBool::new(false));
        let o = StreamOptions { cancel: Some(flag.clone()), ..opts(1, 0) };
        let mut s = speak(&m, "ab", 0usize, o).expect("speak");
        assert!(s.next().is_some());
        flag.store(true, Ordering::Relaxed);
        assert!(s.next().is_none());
        assert!(s.cancelled());
        assert_eq!(*s.state(), 1);
    }
}
