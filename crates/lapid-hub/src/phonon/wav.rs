//! Minimal WAV writer (mono, 16-bit PCM). Candidate for lapid-nn `audio` (which only reads).

use std::io::Write;
use std::path::Path;

use lapid::{Error, Result};

/// RIFF/WAVE bytes of mono `samples` (clamped to [-1, 1], 16-bit PCM).
pub fn wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut b = Vec::with_capacity(44 + samples.len() * 2);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&1u16.to_le_bytes()); // mono
    b.extend_from_slice(&sample_rate.to_le_bytes());
    b.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

/// Write mono `samples` to `path` as 16-bit PCM WAV.
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let mut f = std::fs::File::create(path).map_err(|e| Error::new("io", format!("{}: {e}", path.display())))?;
    f.write_all(&wav_bytes(samples, sample_rate)).map_err(|e| Error::new("io", format!("{}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_through_parser() {
        let s: Vec<f32> = (0..100).map(|i| (i as f32 / 50.0) - 1.0).collect();
        let w = lapid_nn::audio::parse_wav(&wav_bytes(&s, 24000)).expect("parse");
        assert_eq!(w.sample_rate, 24000);
        assert_eq!(w.samples.len(), 100);
        assert!((w.samples[10] - s[10]).abs() < 1e-3);
    }
}
