//! Short-time Fourier transform (torch.stft / librosa semantics), computed in f64.

use std::sync::Arc;

use lapid::{Result, bail};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// Padding of the signal edges when frames are centered.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PadMode {
    /// Mirror without repeating the edge sample (torch / numpy "reflect").
    Reflect,
    /// Zeros ("constant").
    Zeros,
}

#[derive(Clone, Debug)]
pub struct StftConfig {
    pub n_fft: usize,
    pub hop_length: usize,
    /// Window length (<= n_fft); the window is zero-padded to n_fft, centered (torch).
    pub win_length: usize,
    /// Periodic Hann (`torch.hann_window(periodic=True)`, Whisper) or symmetric (NeMo).
    pub periodic_window: bool,
    /// Frames centered on `t * hop` (signal padded by n_fft / 2 on both sides).
    pub center: bool,
    pub pad_mode: PadMode,
}

impl StftConfig {
    /// win_length = n_fft, periodic Hann, centered with reflect padding.
    pub fn new(n_fft: usize, hop_length: usize) -> StftConfig {
        StftConfig { n_fft, hop_length, win_length: n_fft, periodic_window: true, center: true, pad_mode: PadMode::Reflect }
    }
}

/// Hann window of length `n`: periodic = `np.hanning(n + 1)[:-1]`, else `np.hanning(n)`.
pub fn hann_window(n: usize, periodic: bool) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    let denom = if periodic { n } else { n - 1 } as f64;
    (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / denom).cos()).collect()
}

pub struct Stft {
    pub config: StftConfig,
    fft: Arc<dyn Fft<f64>>,
    /// n_fft long (zero-padded window).
    window: Vec<f64>,
}

impl Stft {
    pub fn new(config: StftConfig) -> Result<Stft> {
        let c = &config;
        if c.n_fft == 0 || c.hop_length == 0 || c.win_length == 0 || c.win_length > c.n_fft {
            bail!(
                "invalid_argument",
                "stft: need n_fft, hop > 0 and 0 < win_length <= n_fft (n_fft {}, hop {}, win {})",
                c.n_fft,
                c.hop_length,
                c.win_length
            );
        }
        let fft = FftPlanner::<f64>::new().plan_fft_forward(c.n_fft);
        let mut window = vec![0.0; c.n_fft];
        let left = (c.n_fft - c.win_length) / 2;
        window[left..left + c.win_length].copy_from_slice(&hann_window(c.win_length, c.periodic_window));
        Ok(Stft { config, fft, window })
    }

    /// Frequency bins per frame: n_fft / 2 + 1.
    pub fn n_bins(&self) -> usize {
        self.config.n_fft / 2 + 1
    }

    /// Frames of a full STFT of `len` samples.
    pub fn n_frames(&self, len: usize) -> usize {
        let c = &self.config;
        if c.center {
            1 + len / c.hop_length
        } else if len < c.n_fft {
            0
        } else {
            1 + (len - c.n_fft) / c.hop_length
        }
    }

    /// For frames `0..n_frames`, call `f(frame, spectrum)` with `|X|^power` per bin
    /// (power 2 = power spectrum, 1 = magnitude). Frames past the signal read padding.
    pub fn for_each_frame(&self, signal: &[f64], n_frames: usize, power: f64, mut f: impl FnMut(usize, &[f64])) {
        let c = &self.config;
        let n = signal.len() as isize;
        let pad = if c.center { (c.n_fft / 2) as isize } else { 0 };
        let reflect = c.center && c.pad_mode == PadMode::Reflect;
        let sample = |i: isize| -> f64 {
            if i >= 0 && i < n {
                return signal[i as usize];
            }
            if !reflect || n == 0 {
                return 0.0;
            }
            if n == 1 {
                return signal[0];
            }
            let period = 2 * (n - 1);
            let mut j = i.rem_euclid(period);
            if j >= n {
                j = period - j;
            }
            signal[j as usize]
        };
        let n_bins = self.n_bins();
        let mut buf = vec![Complex::new(0.0, 0.0); c.n_fft];
        let mut scratch = vec![Complex::new(0.0, 0.0); self.fft.get_inplace_scratch_len()];
        let mut spec = vec![0.0f64; n_bins];
        for frame in 0..n_frames {
            let start = (frame * c.hop_length) as isize - pad;
            for (k, b) in buf.iter_mut().enumerate() {
                *b = Complex::new(sample(start + k as isize) * self.window[k], 0.0);
            }
            self.fft.process_with_scratch(&mut buf, &mut scratch);
            for (s, z) in spec.iter_mut().zip(&buf[..n_bins]) {
                let p = z.re * z.re + z.im * z.im;
                *s = if power == 2.0 { p } else { p.powf(power / 2.0) };
            }
            f(frame, &spec);
        }
    }

    /// `|X|^power` for every frame: [n_frames, n_bins] row-major.
    pub fn spectrogram(&self, signal: &[f64], power: f64) -> Vec<f64> {
        let n_frames = self.n_frames(signal.len());
        let n_bins = self.n_bins();
        let mut out = vec![0.0; n_frames * n_bins];
        self.for_each_frame(signal, n_frames, power, |f, s| out[f * n_bins..(f + 1) * n_bins].copy_from_slice(s));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct DFT power of one frame.
    fn dft_power(frame: &[f64]) -> Vec<f64> {
        let n = frame.len();
        (0..n / 2 + 1)
            .map(|k| {
                let (mut re, mut im) = (0.0, 0.0);
                for (t, &x) in frame.iter().enumerate() {
                    let a = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                    re += x * a.cos();
                    im += x * a.sin();
                }
                re * re + im * im
            })
            .collect()
    }

    fn signal(n: usize) -> Vec<f64> {
        (0..n).map(|i| ((i * 37 % 101) as f64 - 50.0) / 50.0).collect()
    }

    #[test]
    fn windows() {
        let p = hann_window(4, true);
        assert!((p[0] - 0.0).abs() < 1e-12 && (p[2] - 1.0).abs() < 1e-12);
        let s = hann_window(5, false);
        assert!((s[4] - 0.0).abs() < 1e-12 && (s[2] - 1.0).abs() < 1e-12);
        assert_eq!(hann_window(1, true), vec![1.0]);
    }

    #[test]
    fn matches_direct_dft_reflect_and_zeros() {
        let x = signal(50);
        for (mode, win) in [(PadMode::Reflect, 16), (PadMode::Zeros, 12)] {
            let cfg = StftConfig { win_length: win, pad_mode: mode, periodic_window: win == 16, ..StftConfig::new(16, 4) };
            let stft = Stft::new(cfg.clone()).unwrap();
            assert_eq!(stft.n_frames(50), 13);
            let spec = stft.spectrogram(&x, 2.0);
            let mut window = [0.0; 16];
            let left = (16 - win) / 2;
            window[left..left + win].copy_from_slice(&hann_window(win, cfg.periodic_window));
            for f in [0usize, 5, 12] {
                let frame: Vec<f64> = (0..16)
                    .map(|k| {
                        let i = (f * 4 + k) as isize - 8;
                        let v = if (0..50).contains(&i) {
                            x[i as usize]
                        } else if mode == PadMode::Zeros {
                            0.0
                        } else if i < 0 {
                            x[(-i) as usize]
                        } else {
                            x[(2 * 49 - i) as usize]
                        };
                        v * window[k]
                    })
                    .collect();
                let want = dft_power(&frame);
                for (b, w) in want.iter().enumerate() {
                    assert!((spec[f * 9 + b] - w).abs() < 1e-9 * w.max(1.0), "frame {f} bin {b}");
                }
            }
        }
    }

    #[test]
    fn uncentered_frames_and_magnitude() {
        let x = signal(40);
        let cfg = StftConfig { center: false, ..StftConfig::new(16, 8) };
        let stft = Stft::new(cfg).unwrap();
        assert_eq!(stft.n_frames(40), 4);
        assert_eq!(stft.n_frames(10), 0);
        let pow = stft.spectrogram(&x, 2.0);
        let mag = stft.spectrogram(&x, 1.0);
        for (p, m) in pow.iter().zip(&mag) {
            assert!((p.sqrt() - m).abs() < 1e-9);
        }
        let frame: Vec<f64> = (0..16).map(|k| x[8 + k] * hann_window(16, true)[k]).collect();
        let want = dft_power(&frame);
        for (b, w) in want.iter().enumerate() {
            assert!((pow[9 + b] - w).abs() < 1e-9 * w.max(1.0));
        }
    }

    #[test]
    fn rejects_bad_config() {
        assert!(Stft::new(StftConfig { win_length: 20, ..StftConfig::new(16, 4) }).is_err());
        assert!(Stft::new(StftConfig::new(16, 0)).is_err());
    }
}
