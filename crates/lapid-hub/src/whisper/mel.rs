//! Whisper log-mel spectrogram on top of `lapid_nn::audio` (STFT n_fft 400, hop 160,
//! periodic Hann, center reflect padding, slaney mel filterbank, power spectrum).
//!
//! * [`log_mel_window`] — HF `WhisperFeatureExtractor` semantics for one 30 s window: pad /
//!   trim to 480000 samples, log10(max(x, 1e-10)), floor at max - 8, (x + 4) / 4, drop the
//!   last frame → exactly 3000 frames.
//! * [`LongMel`] — OpenAI `transcribe()` semantics for long-form audio: the whole signal plus
//!   30 s of zeros is analysed at once (global max), then 30 s windows are sliced out and
//!   zero-padded (`pad_or_trim`).
//!
//! Computed in f64 (like the numpy reference) and returned as f32.

use lapid::Result;
use lapid_nn::audio::{MelFilterbank, MelNorm, MelScale, MelSpectrogram, Stft, StftConfig};

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const CHUNK_SAMPLES: usize = 480_000;
pub const N_FRAMES: usize = 3000;

/// Whisper's mel front-end for `n_mels` bands (80 or 128).
pub(crate) fn whisper_mel(n_mels: usize) -> Result<MelSpectrogram> {
    let stft = Stft::new(StftConfig::new(N_FFT, HOP_LENGTH))?;
    let filters = MelFilterbank::new(
        SAMPLE_RATE as u32,
        N_FFT,
        n_mels,
        0.0,
        SAMPLE_RATE as f64 / 2.0,
        MelScale::Slaney,
        MelNorm::Slaney,
    )?;
    MelSpectrogram::new(stft, filters, 2.0)
}

/// log10(clamp 1e-10), floor at global max - 8, (x + 4) / 4.
fn finalize(mel: &[f64]) -> Vec<f32> {
    let logs: Vec<f64> = mel.iter().map(|&x| x.max(1e-10).log10()).collect();
    let max = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let floor = max - 8.0;
    logs.iter().map(|&x| ((x.max(floor) + 4.0) / 4.0) as f32).collect()
}

/// HF WhisperFeatureExtractor for one window: [n_mels * 3000].
pub(crate) fn log_mel_window(samples: &[f32], mel: &MelSpectrogram) -> Vec<f32> {
    let mut signal = vec![0.0f64; CHUNK_SAMPLES];
    for (d, &s) in signal.iter_mut().zip(samples) {
        *d = s as f64;
    }
    // 1 + 480000/160 = 3001 frames; the last one is dropped.
    finalize(&mel.compute(&signal, N_FRAMES))
}

/// OpenAI long-form log-mel: audio + 30 s zeros, global normalisation.
pub(crate) struct LongMel {
    n_mels: usize,
    /// [n_mels, total_frames]
    data: Vec<f32>,
    total_frames: usize,
}

impl LongMel {
    pub fn new(samples: &[f32], mel: &MelSpectrogram) -> LongMel {
        let mut signal: Vec<f64> = Vec::with_capacity(samples.len() + CHUNK_SAMPLES);
        signal.extend(samples.iter().map(|&s| s as f64));
        signal.resize(samples.len() + CHUNK_SAMPLES, 0.0);
        // torch.stft(center=True) gives 1 + L/hop frames; [..., :-1] drops the last.
        let total_frames = signal.len() / HOP_LENGTH;
        LongMel {
            n_mels: mel.n_mels(),
            data: finalize(&mel.compute(&signal, total_frames)),
            total_frames,
        }
    }

    /// `pad_or_trim(mel[:, seek:seek+size], 3000)` → [n_mels * 3000] (zero padded).
    pub fn window(&self, seek: usize, size: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_mels * N_FRAMES];
        let size = size.min(N_FRAMES);
        let end = (seek + size).min(self.total_frames);
        if seek >= end {
            return out;
        }
        for m in 0..self.n_mels {
            let row = &self.data[m * self.total_frames..(m + 1) * self.total_frames];
            out[m * N_FRAMES..m * N_FRAMES + (end - seek)].copy_from_slice(&row[seek..end]);
        }
        out
    }
}

/// Number of mel frames covering `n_samples` of real audio (`content_frames` in transcribe.py).
pub(crate) fn content_frames(n_samples: usize) -> usize {
    n_samples / HOP_LENGTH
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_shape_and_range() {
        let mel = whisper_mel(80).unwrap();
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let out = log_mel_window(&samples, &mel);
        assert_eq!(out.len(), 80 * N_FRAMES);
        let max = out.iter().copied().fold(f32::MIN, f32::max);
        let min = out.iter().copied().fold(f32::MAX, f32::min);
        assert!(max - min <= 2.0 + 1e-5); // 8 decades / 4
    }

    #[test]
    fn long_mel_windows() {
        let mel = whisper_mel(80).unwrap();
        let samples = vec![0.1f32; 16000 * 40];
        let long = LongMel::new(&samples, &mel);
        assert_eq!(content_frames(samples.len()), 4000);
        let w = long.window(3000, 1000);
        assert_eq!(w.len(), 80 * N_FRAMES);
        assert_eq!(w[1500], 0.0); // padded tail
    }
}
