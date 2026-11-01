//! Speech segments from per-window probabilities, reproducing silero-vad 5.1.2's
//! `get_speech_timestamps` (whole buffer) and `VADIterator` (streaming events) exactly,
//! including the max-duration splitting, the 98 ms "possible split" silence, padding and the
//! merge of padded neighbours. Arithmetic follows the Python: sample positions are integers,
//! derived limits (`sr * ms / 1000`, ...) are floats (f64), and option values are taken at their
//! shortest decimal spelling so `0.6f32` behaves like Python's `0.6`.

use serde::{Deserialize, Serialize};

/// Options of [`super::Vad::process`] / [`super::VadStream`] (silero `get_speech_timestamps` /
/// `VADIterator` arguments; the defaults are silero's).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VadOptions {
    /// Probabilities at or above this are speech.
    pub threshold: f32,
    /// While in speech, probabilities below this are silence; `None` = `threshold - 0.15`.
    pub neg_threshold: Option<f32>,
    /// Segments shorter than this are dropped (whole-buffer only).
    pub min_speech_duration_ms: u32,
    /// Silence needed to close a segment.
    pub min_silence_duration_ms: u32,
    /// Padding added on each side of a segment.
    pub speech_pad_ms: u32,
    /// Longer segments are split (at the last silence over 98 ms if any, else hard);
    /// `f32::INFINITY` = never (whole-buffer only).
    pub max_speech_duration_s: f32,
}

impl Default for VadOptions {
    fn default() -> VadOptions {
        VadOptions {
            threshold: 0.5,
            neg_threshold: None,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
            max_speech_duration_s: f32::INFINITY,
        }
    }
}

/// A speech segment in samples: `start..end` (end exclusive, like silero).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct SpeechSegment {
    pub start: usize,
    pub end: usize,
}

impl SpeechSegment {
    pub fn start_seconds(&self, sample_rate: u32) -> f64 {
        self.start as f64 / sample_rate as f64
    }

    pub fn end_seconds(&self, sample_rate: u32) -> f64 {
        self.end as f64 / sample_rate as f64
    }

    pub fn duration_seconds(&self, sample_rate: u32) -> f64 {
        self.end.saturating_sub(self.start) as f64 / sample_rate as f64
    }
}

/// A streaming event (silero `VADIterator` output) at an absolute sample position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VadEvent {
    SpeechStart { sample: usize },
    SpeechEnd { sample: usize },
}

/// f32 option → the f64 Python would hold for the same literal (`0.6f32` → `0.6`).
pub(crate) fn decimal(x: f32) -> f64 {
    if !x.is_finite() {
        return x as f64;
    }
    format!("{x}").parse::<f64>().unwrap_or(x as f64)
}

impl VadOptions {
    pub(crate) fn threshold64(&self) -> f64 {
        decimal(self.threshold)
    }

    pub(crate) fn neg_threshold64(&self) -> f64 {
        match self.neg_threshold {
            Some(n) => decimal(n),
            None => self.threshold64() - 0.15,
        }
    }
}

/// `get_speech_timestamps` over precomputed `probs` (one per `window` samples) of a signal
/// of `num_samples` samples at `sample_rate`.
pub fn speech_timestamps(probs: &[f32], num_samples: usize, sample_rate: u32, window: usize, opts: &VadOptions) -> Vec<SpeechSegment> {
    let sr = sample_rate as f64;
    let w = window as i64;
    let threshold = opts.threshold64();
    let neg_threshold = opts.neg_threshold64();
    let min_speech_samples = sr * opts.min_speech_duration_ms as f64 / 1000.0;
    let speech_pad_samples = sr * opts.speech_pad_ms as f64 / 1000.0;
    let max_speech_samples = sr * decimal(opts.max_speech_duration_s) - window as f64 - 2.0 * speech_pad_samples;
    let min_silence_samples = sr * opts.min_silence_duration_ms as f64 / 1000.0;
    let min_silence_at_max_speech = sr * 98.0 / 1000.0;
    let len = num_samples as i64;

    // Python dicts: `start`/`end` present or not; 0 means "unset" for the int trackers.
    let mut speeches: Vec<(i64, i64)> = Vec::new();
    let mut current: Option<i64> = None; // current_speech['start']
    let mut triggered = false;
    let (mut temp_end, mut prev_end, mut next_start) = (0i64, 0i64, 0i64);
    for (i, &p) in probs.iter().enumerate() {
        let p = p as f64;
        let pos = w * i as i64;
        if p >= threshold && temp_end != 0 {
            temp_end = 0;
            if next_start < prev_end {
                next_start = pos;
            }
        }
        if p >= threshold && !triggered {
            triggered = true;
            current = Some(pos);
            continue;
        }
        if triggered && (pos - current.unwrap_or(0)) as f64 > max_speech_samples {
            if prev_end != 0 {
                speeches.push((current.unwrap_or(0), prev_end));
                current = None;
                if next_start < prev_end {
                    triggered = false;
                } else {
                    current = Some(next_start);
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
            } else {
                speeches.push((current.unwrap_or(0), pos));
                current = None;
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
                continue;
            }
        }
        if p < neg_threshold && triggered {
            if temp_end == 0 {
                temp_end = pos;
            }
            if (pos - temp_end) as f64 > min_silence_at_max_speech {
                prev_end = temp_end;
            }
            if ((pos - temp_end) as f64) < min_silence_samples {
                continue;
            }
            let start = current.unwrap_or(0);
            if ((temp_end - start) as f64) > min_speech_samples {
                speeches.push((start, temp_end));
            }
            current = None;
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            triggered = false;
        }
    }
    if let Some(start) = current.filter(|&s| ((len - s) as f64) > min_speech_samples) {
        speeches.push((start, len));
    }

    let n = speeches.len();
    for i in 0..n {
        if i == 0 {
            speeches[0].0 = (speeches[0].0 as f64 - speech_pad_samples).max(0.0) as i64;
        }
        if i + 1 != n {
            let silence = speeches[i + 1].0 - speeches[i].1;
            if (silence as f64) < 2.0 * speech_pad_samples {
                let half = silence.div_euclid(2);
                speeches[i].1 += half;
                speeches[i + 1].0 = (speeches[i + 1].0 - half).max(0);
            } else {
                speeches[i].1 = (len as f64).min(speeches[i].1 as f64 + speech_pad_samples) as i64;
                speeches[i + 1].0 = (speeches[i + 1].0 as f64 - speech_pad_samples).max(0.0) as i64;
            }
        } else {
            speeches[i].1 = (len as f64).min(speeches[i].1 as f64 + speech_pad_samples) as i64;
        }
    }
    speeches.into_iter().map(|(s, e)| SpeechSegment { start: s.max(0) as usize, end: e.max(0) as usize }).collect()
}

/// `VADIterator` state machine over a stream of window probabilities.
#[derive(Clone, Debug)]
pub(crate) struct Detector {
    threshold: f64,
    neg_threshold: f64,
    min_silence_samples: f64,
    speech_pad_samples: f64,
    triggered: bool,
    temp_end: i64,
    current_sample: i64,
}

impl Detector {
    pub fn new(opts: &VadOptions, sample_rate: u32) -> Detector {
        let sr = sample_rate as f64;
        Detector {
            threshold: opts.threshold64(),
            neg_threshold: opts.neg_threshold64(),
            min_silence_samples: sr * opts.min_silence_duration_ms as f64 / 1000.0,
            speech_pad_samples: sr * opts.speech_pad_ms as f64 / 1000.0,
            triggered: false,
            temp_end: 0,
            current_sample: 0,
        }
    }

    pub fn reset(&mut self) {
        self.triggered = false;
        self.temp_end = 0;
        self.current_sample = 0;
    }

    pub fn triggered(&self) -> bool {
        self.triggered
    }

    /// Feed the probability of the next `window` samples.
    pub fn push(&mut self, prob: f32, window: usize) -> Option<VadEvent> {
        let p = prob as f64;
        let w = window as i64;
        self.current_sample += w;
        if p >= self.threshold && self.temp_end != 0 {
            self.temp_end = 0;
        }
        if p >= self.threshold && !self.triggered {
            self.triggered = true;
            let start = (self.current_sample as f64 - self.speech_pad_samples - w as f64).max(0.0);
            return Some(VadEvent::SpeechStart { sample: start as usize });
        }
        if p < self.neg_threshold && self.triggered {
            if self.temp_end == 0 {
                self.temp_end = self.current_sample;
            }
            if ((self.current_sample - self.temp_end) as f64) < self.min_silence_samples {
                return None;
            }
            let end = self.temp_end as f64 + self.speech_pad_samples - w as f64;
            self.temp_end = 0;
            self.triggered = false;
            return Some(VadEvent::SpeechEnd { sample: end.max(0.0) as usize });
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 512;

    fn probs(pattern: &str) -> Vec<f32> {
        pattern.chars().map(|c| if c == '#' { 0.9 } else if c == '~' { 0.4 } else { 0.0 }).collect()
    }

    #[test]
    fn defaults_and_decimal() {
        let o = VadOptions::default();
        assert_eq!(o.threshold64(), 0.5);
        assert_eq!(o.neg_threshold64(), 0.5 - 0.15);
        assert_eq!(decimal(0.6), 0.6);
        assert!(decimal(f32::INFINITY).is_infinite());
    }

    #[test]
    fn padding_merge_and_min_speech() {
        // 10 speech windows, 5 silent windows (the 5th is 2048 >= 1600 samples past temp_end).
        let p = probs("..##########.....##########.....");
        let n = p.len() * W;
        let s = speech_timestamps(&p, n, 16_000, W, &VadOptions::default());
        // temp_end = 12*512; padding 480 each side.
        assert_eq!(s, vec![SpeechSegment { start: 2 * W - 480, end: 12 * W + 480 }, SpeechSegment { start: 17 * W - 480, end: 27 * W + 480 }]);
        // Four silent windows are not enough: one segment.
        let p = probs("..##########....##########");
        assert_eq!(speech_timestamps(&p, p.len() * W, 16_000, W, &VadOptions::default()).len(), 1);
        // Short burst below 250 ms (4000 samples) is dropped.
        let p = probs("..###.......");
        assert!(speech_timestamps(&p, p.len() * W, 16_000, W, &VadOptions::default()).is_empty());
        // Gap shorter than 2 * pad: the padding splits the gap in half.
        let o = VadOptions { min_silence_duration_ms: 0, speech_pad_ms: 100, ..VadOptions::default() };
        let p = probs("##########.##########");
        let s = speech_timestamps(&p, p.len() * W, 16_000, W, &o);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].end, 10 * W + 256);
        assert_eq!(s[1].start, 11 * W - 256);
    }

    #[test]
    fn open_segment_runs_to_end() {
        let p = probs("....##########");
        let n = p.len() * W - 100;
        let s = speech_timestamps(&p, n, 16_000, W, &VadOptions::default());
        assert_eq!(s, vec![SpeechSegment { start: 4 * W - 480, end: n }]);
    }

    #[test]
    fn max_duration_hard_split() {
        let o = VadOptions { max_speech_duration_s: 0.5, speech_pad_ms: 0, ..VadOptions::default() };
        // max_speech_samples = 8000 - 512 = 7488 -> split when pos - start > 7488.
        let p = probs(&"#".repeat(40));
        let s = speech_timestamps(&p, 40 * W, 16_000, W, &o);
        assert_eq!(s[0], SpeechSegment { start: 0, end: 15 * W });
        assert!(s.len() >= 2);
    }

    #[test]
    fn iterator_events() {
        let mut d = Detector::new(&VadOptions::default(), 16_000);
        let p = probs("..####.....");
        let ev: Vec<_> = p.iter().filter_map(|&x| d.push(x, W)).collect();
        assert_eq!(ev, vec![VadEvent::SpeechStart { sample: 2 * W - 480 }, VadEvent::SpeechEnd { sample: 6 * W + 480 }]);
        assert!(!d.triggered());
    }
}
