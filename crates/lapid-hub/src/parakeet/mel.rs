//! Parakeet / NeMo log-mel features (HF `ParakeetFeatureExtractor` semantics) on top of
//! `lapid_nn::audio`: pre-emphasis, STFT (n_fft 512, symmetric Hann of 400 zero-padded and
//! centered, center zero padding), power spectrum, slaney mel filterbank, `ln(x + 2^-24)`,
//! per-feature standardization over the valid frames (unbiased variance, `std + 1e-5`).
//! Dither is off (inference).
//!
//! The extractor computes `1 + L / hop` frames but only the first `L / hop` are valid (HF
//! masks the last one to zero and the encoder never sees it); only valid frames are
//! returned. Computed in f64, returned as f32, frames-major `[T, n_mels]`.

use lapid::Result;
use lapid_nn::audio::{MelFilterbank, MelNorm, MelScale, MelSpectrogram, PadMode, Stft, StftConfig};

use super::config::ParakeetConfig;

const LOG_GUARD: f64 = 1.0 / 16_777_216.0; // 2^-24
const STD_EPS: f64 = 1e-5;

pub(crate) struct Features {
    mel: MelSpectrogram,
    preemphasis: f64,
    hop: usize,
}

/// Normalized log-mel of a whole signal.
pub struct Mel {
    pub n_mels: usize,
    pub n_frames: usize,
    /// [n_frames, n_mels] row-major.
    pub data: Vec<f32>,
}

impl Mel {
    /// Frames `[start, end)` as a [end - start, n_mels] slice.
    pub fn frames(&self, start: usize, end: usize) -> &[f32] {
        &self.data[start * self.n_mels..end * self.n_mels]
    }
}

impl Features {
    pub fn new(cfg: &ParakeetConfig) -> Result<Features> {
        let stft = Stft::new(StftConfig {
            n_fft: cfg.n_fft,
            hop_length: cfg.hop_length,
            win_length: cfg.win_length,
            periodic_window: false,
            center: true,
            pad_mode: PadMode::Zeros,
        })?;
        let filters = MelFilterbank::new(
            cfg.sample_rate,
            cfg.n_fft,
            cfg.n_mels,
            0.0,
            cfg.sample_rate as f64 / 2.0,
            MelScale::Slaney,
            MelNorm::Slaney,
        )?;
        Ok(Features { mel: MelSpectrogram::new(stft, filters, 2.0)?, preemphasis: cfg.preemphasis, hop: cfg.hop_length })
    }

    /// Valid frames of `n_samples` of audio.
    pub fn n_frames(&self, n_samples: usize) -> usize {
        n_samples / self.hop
    }

    pub fn compute(&self, samples: &[f32]) -> Mel {
        let n_mels = self.mel.n_mels();
        let n_frames = self.n_frames(samples.len());
        if n_frames == 0 {
            return Mel { n_mels, n_frames: 0, data: Vec::new() };
        }
        let mut signal = Vec::with_capacity(samples.len());
        let mut prev = 0.0f64;
        for (i, &s) in samples.iter().enumerate() {
            let s = s as f64;
            signal.push(if i == 0 { s } else { s - self.preemphasis * prev });
            prev = s;
        }
        // [n_mels, n_frames] linear energies.
        let mut energies = self.mel.compute(&signal, n_frames);
        for v in energies.iter_mut() {
            *v = (*v + LOG_GUARD).ln();
        }
        let mut data = vec![0.0f32; n_frames * n_mels];
        for m in 0..n_mels {
            let row = &energies[m * n_frames..(m + 1) * n_frames];
            let mean = row.iter().sum::<f64>() / n_frames as f64;
            let var = if n_frames > 1 {
                row.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n_frames - 1) as f64
            } else {
                0.0
            };
            let inv = 1.0 / (var.sqrt() + STD_EPS);
            for (t, &x) in row.iter().enumerate() {
                data[t * n_mels + m] = ((x - mean) * inv) as f32;
            }
        }
        Mel { n_mels, n_frames, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_and_standardized() {
        let f = Features::new(&ParakeetConfig::v3()).expect("features");
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin() * 0.3 + (i % 7) as f32 * 1e-3)
            .collect();
        let mel = f.compute(&samples);
        assert_eq!((mel.n_frames, mel.n_mels), (100, 128));
        // per-feature mean ~0
        let m0: f64 = (0..mel.n_frames).map(|t| mel.data[t * 128 + 10] as f64).sum::<f64>() / 100.0;
        assert!(m0.abs() < 1e-4, "{m0}");
        assert_eq!(f.compute(&[0.0; 100]).n_frames, 0);
    }
}
