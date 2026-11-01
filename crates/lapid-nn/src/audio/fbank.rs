//! Kaldi-style log-mel filterbank exactly as icefall trains with it: lhotse `Fbank` /
//! `Wav2LogFilterBank` defaults (16 kHz, 25 ms Povey window, 10 ms shift, 512-point FFT,
//! remove DC, pre-emphasis 0.97, dither 0, 80 mel bins from 20 Hz to Nyquist - 400 Hz,
//! power spectrum, `log(max(x, f32::EPSILON))`, `snip_edges = false`).
//!
//! `snip_edges = false` centers frame `f` on sample `160 f + 80`: the signal is reflected by
//! 120 samples at the start (`x[119], ..., x[0]`) and at the end, and a signal of `n` samples
//! has `(n + 80) / 160` frames. [`OnlineFbank`] produces the same frames incrementally: frame
//! `f` is ready once samples up to `160 f + 280` arrived, the last ones (which read the end
//! reflection) when the input is finished, so streaming and whole-signal features agree.
//!
//! Host f64 math (lhotse runs f32 torch; differences are ~1e-6 relative).

use std::f64::consts::PI;

pub const FRAME_LENGTH: usize = 400;
pub const FRAME_SHIFT: usize = 160;
const N_FFT: usize = 512;
const PREEMPH: f64 = 0.97;
const LOW_FREQ: f64 = 20.0;
const HIGH_FREQ: f64 = -400.0;
/// Samples reflected at the start: (400 - 160) / 2.
const LEFT_PAD: usize = (FRAME_LENGTH - FRAME_SHIFT) / 2;

fn lin2mel(x: f64) -> f64 {
    1127.0 * (1.0 + x / 700.0).ln()
}

/// One triangular mel filter: first FFT bin and weights.
#[derive(Clone)]
struct Filter {
    start: usize,
    weights: Vec<f64>,
}

/// Fbank extractor (stateless; see [`OnlineFbank`] for streaming).
#[derive(Clone)]
pub struct Fbank {
    window: Vec<f64>,
    filters: Vec<Filter>,
    fft: Fft,
    n_mels: usize,
}

impl Fbank {
    pub fn new(n_mels: usize, sample_rate: u32) -> Fbank {
        // Povey window: symmetric Hann ^ 0.85.
        let window = (0..FRAME_LENGTH)
            .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / (FRAME_LENGTH - 1) as f64).cos()).powf(0.85))
            .collect();
        // torchaudio / lhotse `get_mel_banks` (bins 0 .. n_fft / 2; the Nyquist bin gets 0).
        let sr = sample_rate as f64;
        let nyquist = sr / 2.0;
        let high = if HIGH_FREQ <= 0.0 { HIGH_FREQ + nyquist } else { HIGH_FREQ };
        let (mlo, mhi) = (lin2mel(LOW_FREQ), lin2mel(high));
        let delta = (mhi - mlo) / (n_mels + 1) as f64;
        let bin_width = sr / N_FFT as f64;
        let filters = (0..n_mels)
            .map(|m| {
                let (left, center, right) = (mlo + m as f64 * delta, mlo + (m + 1) as f64 * delta, mlo + (m + 2) as f64 * delta);
                let w: Vec<f64> = (0..N_FFT / 2)
                    .map(|k| {
                        let mel = lin2mel(bin_width * k as f64);
                        let up = (mel - left) / (center - left);
                        let down = (right - mel) / (right - center);
                        up.min(down).max(0.0)
                    })
                    .collect();
                let start = w.iter().position(|&v| v > 0.0).unwrap_or(0);
                let end = w.iter().rposition(|&v| v > 0.0).map_or(start, |e| e + 1);
                Filter { start, weights: w[start..end].to_vec() }
            })
            .collect();
        Fbank { window, filters, fft: Fft::new(N_FFT), n_mels }
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Log-mel energies of one 400-sample frame into `out` [n_mels].
    pub fn frame(&self, samples: &[f64], out: &mut [f32]) {
        let n = samples.len().min(FRAME_LENGTH);
        let mean = samples[..n].iter().sum::<f64>() / n.max(1) as f64;
        let mut re = vec![0.0f64; N_FFT];
        let mut im = vec![0.0f64; N_FFT];
        // Remove DC, pre-emphasis (replicate padding: x[-1] = x[0]), window.
        for i in 0..n {
            let x = samples[i] - mean;
            let prev = if i == 0 { x } else { samples[i - 1] - mean };
            re[i] = (x - PREEMPH * prev) * self.window[i];
        }
        self.fft.forward(&mut re, &mut im);
        let power: Vec<f64> = (0..N_FFT / 2).map(|k| re[k] * re[k] + im[k] * im[k]).collect();
        for (o, f) in out.iter_mut().zip(&self.filters) {
            let e: f64 = f.weights.iter().zip(&power[f.start..]).map(|(w, p)| w * p).sum();
            *o = e.max(f32::EPSILON as f64).ln() as f32;
        }
    }

    /// Frames of a whole signal of `n` samples (`snip_edges = false`).
    pub fn num_frames(n: usize) -> usize {
        (n + FRAME_SHIFT / 2) / FRAME_SHIFT
    }

    /// Features of a whole signal: [frames, n_mels] row-major.
    pub fn compute(&self, samples: &[f32]) -> Vec<f32> {
        let mut online = OnlineFbank::new(self.clone());
        online.accept(samples);
        online.finish();
        online.take_all()
    }
}

/// Incremental fbank over a growing signal; emits frames as soon as they are complete.
pub struct OnlineFbank {
    fbank: Fbank,
    /// Samples not yet needed only for past frames, starting at absolute sample `offset`.
    samples: Vec<f32>,
    offset: usize,
    /// First 120 samples of the signal (for the start reflection).
    head: Vec<f32>,
    total: usize,
    frames_done: usize,
    finished: bool,
    ready: Vec<f32>,
}

impl OnlineFbank {
    pub fn new(fbank: Fbank) -> OnlineFbank {
        OnlineFbank { fbank, samples: Vec::new(), offset: 0, head: Vec::new(), total: 0, frames_done: 0, finished: false, ready: Vec::new() }
    }

    pub fn n_mels(&self) -> usize {
        self.fbank.n_mels
    }

    /// Frames produced so far (including the ones taken).
    pub fn frames_done(&self) -> usize {
        self.frames_done
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Append samples and compute every frame that became complete.
    pub fn accept(&mut self, samples: &[f32]) {
        if self.finished {
            return;
        }
        if self.head.len() < LEFT_PAD {
            let need = (LEFT_PAD - self.head.len()).min(samples.len());
            self.head.extend_from_slice(&samples[..need]);
        }
        self.samples.extend_from_slice(samples);
        self.total += samples.len();
        // Frame f reads padded [160 f, 160 f + 400) = signal [160 f - 120, 160 f + 280).
        while self.frames_done * FRAME_SHIFT + FRAME_LENGTH - LEFT_PAD <= self.total {
            self.emit_frame(false);
        }
        self.trim();
    }

    /// End of input: the remaining frames (reading the end reflection).
    pub fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        let n = Fbank::num_frames(self.total);
        while self.frames_done < n {
            self.emit_frame(true);
        }
        self.trim();
    }

    /// Take the computed frames ([k, n_mels] row-major).
    pub fn take_all(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.ready)
    }

    /// Frames waiting in `take_all`.
    pub fn pending(&self) -> usize {
        self.ready.len() / self.fbank.n_mels.max(1)
    }

    fn sample(&self, i: isize, at_end: bool) -> f64 {
        let n = self.total as isize;
        let j = if i < 0 {
            // lhotse: flip(x[:120]) -> x[-i - 1]
            let k = (-i - 1) as usize;
            return self.head.get(k).copied().unwrap_or(0.0) as f64;
        } else if i >= n {
            if !at_end || n == 0 {
                return 0.0;
            }
            // lhotse: flip(x[-npad_right:]) -> x[2n - 1 - i]
            2 * n - 1 - i
        } else {
            i
        };
        let j = j.max(0) as usize;
        if j < self.offset {
            return 0.0;
        }
        self.samples.get(j - self.offset).copied().unwrap_or(0.0) as f64
    }

    fn emit_frame(&mut self, at_end: bool) {
        let start = (self.frames_done * FRAME_SHIFT) as isize - LEFT_PAD as isize;
        let frame: Vec<f64> = (0..FRAME_LENGTH as isize).map(|k| self.sample(start + k, at_end)).collect();
        let mut out = vec![0.0f32; self.fbank.n_mels];
        self.fbank.frame(&frame, &mut out);
        self.ready.extend_from_slice(&out);
        self.frames_done += 1;
    }

    /// Drop samples no future frame reads (keep enough for the end reflection).
    fn trim(&mut self) {
        let next_start = (self.frames_done * FRAME_SHIFT).saturating_sub(LEFT_PAD);
        let keep_from = next_start.min(self.total.saturating_sub(FRAME_LENGTH));
        if keep_from > self.offset + 4 * FRAME_LENGTH {
            self.samples.drain(..keep_from - self.offset);
            self.offset = keep_from;
        }
    }
}

/// Iterative radix-2 complex FFT (power-of-two sizes).
#[derive(Clone)]
struct Fft {
    n: usize,
    cos: Vec<f64>,
    sin: Vec<f64>,
}

impl Fft {
    fn new(n: usize) -> Fft {
        let cos = (0..n / 2).map(|k| (2.0 * PI * k as f64 / n as f64).cos()).collect();
        let sin = (0..n / 2).map(|k| -(2.0 * PI * k as f64 / n as f64).sin()).collect();
        Fft { n, cos, sin }
    }

    fn forward(&self, re: &mut [f64], im: &mut [f64]) {
        let n = self.n;
        let bits = n.trailing_zeros();
        for i in 0..n {
            let j = i.reverse_bits() >> (usize::BITS - bits);
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let step = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..len / 2 {
                    let (wr, wi) = (self.cos[k * step], self.sin[k * step]);
                    let (a, b) = (start + k, start + k + len / 2);
                    let (xr, xi) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                }
            }
            len *= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_dft() {
        let n = 16;
        let fft = Fft::new(n);
        let x: Vec<f64> = (0..n).map(|i| ((i * 7 % 5) as f64 - 2.0) * 0.3).collect();
        let (mut re, mut im) = (x.clone(), vec![0.0; n]);
        fft.forward(&mut re, &mut im);
        for k in 0..n {
            let (mut r, mut i) = (0.0, 0.0);
            for (t, v) in x.iter().enumerate() {
                let a = -2.0 * PI * (k * t) as f64 / n as f64;
                r += v * a.cos();
                i += v * a.sin();
            }
            assert!((re[k] - r).abs() < 1e-9 && (im[k] - i).abs() < 1e-9, "bin {k}");
        }
    }

    #[test]
    fn online_equals_offline_for_any_split() {
        let fb = Fbank::new(80, 16_000);
        let x: Vec<f32> = (0..5_123).map(|i| (i as f32 * 0.013).sin() * 0.3 + ((i * 31 % 17) as f32 - 8.0) * 0.001).collect();
        let whole = fb.compute(&x);
        assert_eq!(whole.len(), Fbank::num_frames(x.len()) * 80);
        for piece in [1usize, 97, 160, 1000] {
            let mut on = OnlineFbank::new(Fbank::new(80, 16_000));
            let mut got = Vec::new();
            for c in x.chunks(piece) {
                on.accept(c);
                got.extend(on.take_all());
            }
            on.finish();
            got.extend(on.take_all());
            assert_eq!(got, whole, "piece {piece}");
        }
    }
}
