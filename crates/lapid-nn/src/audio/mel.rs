//! Mel filterbanks (librosa / HF `mel_filter_bank` semantics) and mel spectrograms.
//!
//! Log compression and normalization differ per model (Whisper: log10 + dynamic-range floor,
//! NeMo: ln + per-feature standardization), so [`MelSpectrogram::compute`] returns linear
//! mel energies and the model applies its own scaling.

use lapid::{Result, bail};

use super::stft::Stft;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MelScale {
    /// Linear below 1 kHz, logarithmic above (librosa default, Whisper, NeMo).
    Slaney,
    /// 2595 log10(1 + hz / 700).
    Htk,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MelNorm {
    /// Peak 1 triangles.
    None,
    /// Unit-area triangles: each scaled by 2 / (f_hi - f_lo).
    Slaney,
}

pub fn hz_to_mel(hz: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + hz / 700.0).log10(),
        MelScale::Slaney => {
            const MIN_LOG_HZ: f64 = 1000.0;
            const MIN_LOG_MEL: f64 = 15.0;
            let logstep = 27.0 / 6.4f64.ln();
            if hz >= MIN_LOG_HZ { MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() * logstep } else { 3.0 * hz / 200.0 }
        }
    }
}

pub fn mel_to_hz(mel: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 700.0 * (10f64.powf(mel / 2595.0) - 1.0),
        MelScale::Slaney => {
            const MIN_LOG_HZ: f64 = 1000.0;
            const MIN_LOG_MEL: f64 = 15.0;
            let logstep = 6.4f64.ln() / 27.0;
            if mel >= MIN_LOG_MEL { MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp() } else { 200.0 * mel / 3.0 }
        }
    }
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f64;
    (0..n).map(|i| a + step * i as f64).collect()
}

/// Triangular mel filters over `n_fft / 2 + 1` FFT bins, stored sparsely as
/// (first bin, contiguous weights).
#[derive(Clone, Debug)]
pub struct MelFilterbank {
    n_mels: usize,
    n_bins: usize,
    filters: Vec<(usize, Vec<f64>)>,
}

impl MelFilterbank {
    /// HF `mel_filter_bank(n_fft/2 + 1, n_mels, f_min, f_max, sample_rate, norm, mel_scale)`.
    pub fn new(
        sample_rate: u32,
        n_fft: usize,
        n_mels: usize,
        f_min: f64,
        f_max: f64,
        scale: MelScale,
        norm: MelNorm,
    ) -> Result<MelFilterbank> {
        if n_fft == 0 || n_mels == 0 || sample_rate == 0 || !(0.0 <= f_min && f_min < f_max) {
            bail!(
                "invalid_argument",
                "mel filterbank: need n_fft, n_mels, sample_rate > 0 and 0 <= f_min < f_max (got {n_fft}, {n_mels}, {sample_rate}, {f_min}..{f_max})"
            );
        }
        let n_bins = n_fft / 2 + 1;
        let mel_freqs = linspace(hz_to_mel(f_min, scale), hz_to_mel(f_max, scale), n_mels + 2);
        let filter_freqs: Vec<f64> = mel_freqs.iter().map(|&m| mel_to_hz(m, scale)).collect();
        let fft_freqs = linspace(0.0, sample_rate as f64 / 2.0, n_bins);

        let mut filters = Vec::with_capacity(n_mels);
        for m in 0..n_mels {
            let (lo, center, hi) = (filter_freqs[m], filter_freqs[m + 1], filter_freqs[m + 2]);
            let enorm = match norm {
                MelNorm::Slaney => 2.0 / (hi - lo),
                MelNorm::None => 1.0,
            };
            let mut first = None;
            let mut weights = Vec::new();
            for (k, &f) in fft_freqs.iter().enumerate() {
                let down = (f - lo) / (center - lo);
                let up = (hi - f) / (hi - center);
                let w = down.min(up).max(0.0) * enorm;
                if w > 0.0 {
                    let start = *first.get_or_insert(k);
                    // keep the contiguous run (triangles are convex)
                    weights.resize(k - start, 0.0);
                    weights.push(w);
                }
            }
            filters.push((first.unwrap_or(0), weights));
        }
        Ok(MelFilterbank { n_mels, n_bins, filters })
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Dense [n_mels, n_bins] matrix.
    pub fn dense(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.n_mels * self.n_bins];
        for (m, (start, w)) in self.filters.iter().enumerate() {
            out[m * self.n_bins + start..m * self.n_bins + start + w.len()].copy_from_slice(w);
        }
        out
    }

    /// out[m] = sum_k filter[m][k] * spectrum[k]; `spectrum` has n_bins values.
    pub fn apply(&self, spectrum: &[f64], out: &mut [f64]) {
        for (o, (start, w)) in out.iter_mut().zip(&self.filters) {
            *o = w.iter().zip(&spectrum[*start..]).map(|(a, b)| a * b).sum();
        }
    }
}

/// STFT + mel filterbank. `power` 2.0 = power spectrum (Whisper, NeMo), 1.0 = magnitude.
pub struct MelSpectrogram {
    pub stft: Stft,
    pub filters: MelFilterbank,
    pub power: f64,
}

impl MelSpectrogram {
    pub fn new(stft: Stft, filters: MelFilterbank, power: f64) -> Result<MelSpectrogram> {
        if stft.n_bins() != filters.n_bins() {
            bail!("invalid_argument", "mel: STFT has {} bins, filterbank {}", stft.n_bins(), filters.n_bins());
        }
        Ok(MelSpectrogram { stft, filters, power })
    }

    pub fn n_mels(&self) -> usize {
        self.filters.n_mels()
    }

    /// Frames of a full STFT of `len` samples.
    pub fn n_frames(&self, len: usize) -> usize {
        self.stft.n_frames(len)
    }

    /// Linear mel energies of frames `0..n_frames`: [n_mels, n_frames] (mel-major, the
    /// layout of `[n_mels, T]` model inputs).
    pub fn compute(&self, signal: &[f64], n_frames: usize) -> Vec<f64> {
        let n_mels = self.n_mels();
        let mut col = vec![0.0f64; n_mels];
        let mut out = vec![0.0f64; n_mels * n_frames];
        self.stft.for_each_frame(signal, n_frames, self.power, |f, spec| {
            self.filters.apply(spec, &mut col);
            for (m, &v) in col.iter().enumerate() {
                out[m * n_frames + f] = v;
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::stft::StftConfig;

    fn whisper_bank(n_mels: usize) -> MelFilterbank {
        MelFilterbank::new(16_000, 400, n_mels, 0.0, 8000.0, MelScale::Slaney, MelNorm::Slaney).unwrap()
    }

    #[test]
    fn mel_scale_roundtrip() {
        for scale in [MelScale::Slaney, MelScale::Htk] {
            for hz in [0.0, 100.0, 999.0, 1000.0, 4000.0, 8000.0] {
                assert!((mel_to_hz(hz_to_mel(hz, scale), scale) - hz).abs() < 1e-6);
            }
        }
        assert!((hz_to_mel(1000.0, MelScale::Slaney) - 15.0).abs() < 1e-12);
        assert!((hz_to_mel(700.0, MelScale::Htk) - 2595.0 * 2f64.log10()).abs() < 1e-9);
    }

    #[test]
    fn filterbank_shape_and_area() {
        for n_mels in [80, 128] {
            let f = whisper_bank(n_mels);
            let dense = f.dense();
            let n_bins = 201;
            assert_eq!(dense.len(), n_mels * n_bins);
            let bin_hz = 16_000.0 / 400.0;
            for m in 0..n_mels {
                let row = &dense[m * n_bins..(m + 1) * n_bins];
                assert!(row.iter().all(|&w| w >= 0.0));
                let sum: f64 = row.iter().sum();
                assert!(sum > 0.0, "filter {m} is empty");
                // slaney norm => unit area in Hz; wide filters are well sampled by 40 Hz bins.
                let top = hz_to_mel(8000.0, MelScale::Slaney);
                let lo = mel_to_hz(top * m as f64 / (n_mels + 1) as f64, MelScale::Slaney);
                let hi = mel_to_hz(top * (m + 2) as f64 / (n_mels + 1) as f64, MelScale::Slaney);
                if hi - lo > 10.0 * bin_hz {
                    assert!((sum * bin_hz - 1.0).abs() < 0.05, "filter {m}: {}", sum * bin_hz);
                }
            }
        }
    }

    #[test]
    fn filterbank_first_filter_matches_librosa() {
        // librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)[0, 1] == 0.02486259
        let d = whisper_bank(80).dense();
        assert_eq!(d[0], 0.0);
        assert!((d[1] - 0.024_862_59).abs() < 1e-6, "{}", d[1]);
    }

    #[test]
    fn htk_unnormalized_peaks_at_one() {
        let f = MelFilterbank::new(16_000, 512, 40, 0.0, 8000.0, MelScale::Htk, MelNorm::None).unwrap();
        let d = f.dense();
        for m in 0..40 {
            let peak = d[m * 257..(m + 1) * 257].iter().copied().fold(0.0, f64::max);
            assert!(peak <= 1.0 + 1e-12 && peak > 0.3, "filter {m} peak {peak}");
        }
        assert!(MelFilterbank::new(16_000, 512, 40, 100.0, 50.0, MelScale::Htk, MelNorm::None).is_err());
    }

    #[test]
    fn spectrogram_equals_filterbank_times_stft() {
        let x: Vec<f64> = (0..3000).map(|i| (i as f64 * 0.3).sin() * 0.5).collect();
        let stft = Stft::new(StftConfig::new(400, 160)).unwrap();
        let spec = stft.spectrogram(&x, 2.0);
        let n_frames = stft.n_frames(x.len());
        let mel = MelSpectrogram::new(Stft::new(StftConfig::new(400, 160)).unwrap(), whisper_bank(80), 2.0).unwrap();
        assert_eq!(mel.n_frames(3000), n_frames);
        let out = mel.compute(&x, n_frames);
        let dense = mel.filters.dense();
        for f in [0, 7, n_frames - 1] {
            for m in [0, 40, 79] {
                let want: f64 = (0..201).map(|k| dense[m * 201 + k] * spec[f * 201 + k]).sum();
                assert!((out[m * n_frames + f] - want).abs() <= 1e-9 * want.abs().max(1.0));
            }
        }
        let bad = MelSpectrogram::new(Stft::new(StftConfig::new(512, 160)).unwrap(), whisper_bank(80), 2.0);
        assert!(bad.is_err());
    }
}
