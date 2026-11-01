//! WAV (RIFF/WAVE) parsing and loading.
//!
//! Supported: PCM 8 (unsigned) / 16 / 24 / 32-bit int, IEEE float 32 / 64,
//! WAVE_FORMAT_EXTENSIBLE wrapping either, any channel count, any sample rate.

use std::path::Path;

use lapid::{Error, Result};

use super::resample::resample;

/// Decoded WAV: interleaved f32 samples in [-1, 1].
#[derive(Clone, Debug)]
pub struct Wav {
    pub sample_rate: u32,
    pub channels: u16,
    /// Interleaved, `frames * channels` values.
    pub samples: Vec<f32>,
}

const FORMAT_PCM: u16 = 1;
const FORMAT_FLOAT: u16 = 3;
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

fn wav_err(msg: impl Into<String>) -> Error {
    Error::new("unsupported_audio", msg)
}

fn u16_le(b: &[u8], at: usize) -> Result<u16> {
    b.get(at..at + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| wav_err("truncated WAV header"))
}

fn u32_le(b: &[u8], at: usize) -> Result<u32> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| wav_err("truncated WAV header"))
}

struct Fmt {
    format: u16,
    channels: u16,
    sample_rate: u32,
    block_align: u16,
    bits: u16,
}

fn parse_fmt(c: &[u8]) -> Result<Fmt> {
    if c.len() < 16 {
        return Err(wav_err("fmt chunk too short"));
    }
    let mut format = u16_le(c, 0)?;
    let channels = u16_le(c, 2)?;
    let sample_rate = u32_le(c, 4)?;
    let block_align = u16_le(c, 12)?;
    let bits = u16_le(c, 14)?;
    if format == FORMAT_EXTENSIBLE {
        // cbSize(2) validBits(2) channelMask(4) SubFormat GUID(16) — first 2 bytes = format
        if c.len() < 40 {
            return Err(wav_err("WAVE_FORMAT_EXTENSIBLE fmt chunk too short"));
        }
        format = u16_le(c, 24)?;
    }
    Ok(Fmt { format, channels, sample_rate, block_align, bits })
}

/// Parse a RIFF/WAVE file from memory.
pub fn parse_wav(bytes: &[u8]) -> Result<Wav> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(wav_err("not a RIFF/WAVE file (only WAV input is supported)"));
    }
    let mut fmt: Option<Fmt> = None;
    let mut data: Option<&[u8]> = None;
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32_le(bytes, pos + 4)? as usize;
        let body_start = pos + 8;
        // Streaming writers may leave 0 / 0xFFFFFFFF sizes: clamp to the file.
        let body_end = body_start.saturating_add(size).min(bytes.len());
        let body = &bytes[body_start..body_end];
        match id {
            b"fmt " => fmt = Some(parse_fmt(body)?),
            b"data" => {
                let body = if size == u32::MAX as usize {
                    &bytes[body_start..]
                } else {
                    body
                };
                data = Some(body);
                if fmt.is_some() {
                    break;
                }
            }
            _ => {}
        }
        // chunks are padded to even sizes
        pos = body_start.saturating_add(size).saturating_add(size & 1);
    }
    let fmt = fmt.ok_or_else(|| wav_err("WAV has no fmt chunk"))?;
    let data = data.ok_or_else(|| wav_err("WAV has no data chunk"))?;
    if fmt.channels == 0 {
        return Err(wav_err("WAV has 0 channels"));
    }
    if fmt.sample_rate == 0 {
        return Err(wav_err("WAV has sample rate 0"));
    }

    let channels = fmt.channels as usize;
    let mut bytes_per_sample = (fmt.bits as usize).div_ceil(8);
    if fmt.block_align as usize >= channels && (fmt.block_align as usize).is_multiple_of(channels) {
        let from_align = fmt.block_align as usize / channels;
        if from_align != 0 {
            bytes_per_sample = from_align;
        }
    }
    let decode: fn(&[u8]) -> f32 = match (fmt.format, bytes_per_sample) {
        (FORMAT_PCM, 1) => |s: &[u8]| (s[0] as f32 - 128.0) / 128.0,
        (FORMAT_PCM, 2) => |s: &[u8]| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
        (FORMAT_PCM, 3) => |s: &[u8]| {
            let v = i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8;
            v as f32 / 8_388_608.0
        },
        (FORMAT_PCM, 4) => |s: &[u8]| {
            (i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64 / 2_147_483_648.0) as f32
        },
        (FORMAT_FLOAT, 4) => |s: &[u8]| f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
        (FORMAT_FLOAT, 8) => |s: &[u8]| {
            f64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]) as f32
        },
        (f, b) => {
            return Err(wav_err(format!(
                "unsupported WAV encoding: format tag {f}, {b} bytes per sample \
                 (supported: PCM 8/16/24/32-bit, float 32/64-bit)"
            )));
        }
    };
    let frame_bytes = bytes_per_sample * channels;
    let n_frames = data.len() / frame_bytes;
    let mut samples = Vec::with_capacity(n_frames * channels);
    for chunk in data[..n_frames * frame_bytes].chunks_exact(bytes_per_sample) {
        let v = decode(chunk);
        samples.push(if v.is_finite() { v } else { 0.0 });
    }
    Ok(Wav { sample_rate: fmt.sample_rate, channels: fmt.channels, samples })
}

/// Average all channels of interleaved samples.
pub fn mixdown(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let inv = 1.0 / channels as f32;
    samples.chunks_exact(channels).map(|f| f.iter().sum::<f32>() * inv).collect()
}

/// Read and parse a WAV file.
pub fn load_wav(path: &Path) -> Result<Wav> {
    let bytes = std::fs::read(path)
        .map_err(|e| Error::new("io", format!("cannot read {}: {e}", path.display())))?;
    parse_wav(&bytes).map_err(|e| Error::new(e.code, format!("{}: {}", path.display(), e.message)))
}

/// Load a WAV file as mono f32 at `sample_rate` Hz (channels averaged, band-limited resampling).
pub fn load_audio(path: &Path, sample_rate: u32) -> Result<Vec<f32>> {
    let wav = load_wav(path)?;
    let mono = mixdown(&wav.samples, wav.channels as usize);
    Ok(resample(&mono, wav.sample_rate, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav_bytes(format: u16, channels: u16, rate: u32, bits: u16, data: &[u8], ext: bool) -> Vec<u8> {
        let bps = bits.div_ceil(8);
        let block_align = channels * bps;
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&(if ext { FORMAT_EXTENSIBLE } else { format }).to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&(rate * block_align as u32).to_le_bytes());
        fmt.extend_from_slice(&block_align.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        if ext {
            fmt.extend_from_slice(&22u16.to_le_bytes());
            fmt.extend_from_slice(&bits.to_le_bytes());
            fmt.extend_from_slice(&0u32.to_le_bytes());
            fmt.extend_from_slice(&format.to_le_bytes());
            fmt.extend_from_slice(&[0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71]);
            fmt.extend_from_slice(&[0, 0]);
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(b"WAVE");
        // an unknown odd-sized chunk to exercise padding
        out.extend_from_slice(b"LIST");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&[1, 2, 3, 0]);
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        out.extend_from_slice(&fmt);
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        let riff = (out.len() - 8) as u32;
        out[4..8].copy_from_slice(&riff.to_le_bytes());
        out
    }

    #[test]
    fn pcm16_stereo() {
        let mut data = Vec::new();
        for v in [16384i16, -16384, 32767, -32768] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let wav = parse_wav(&wav_bytes(1, 2, 16000, 16, &data, false)).unwrap();
        assert_eq!(wav.channels, 2);
        assert_eq!(wav.samples, vec![0.5, -0.5, 32767.0 / 32768.0, -1.0]);
        let mono = mixdown(&wav.samples, 2);
        assert_eq!(mono.len(), 2);
        assert!(mono[0].abs() < 1e-7);
    }

    #[test]
    fn pcm8_and_24_and_32() {
        let w8 = parse_wav(&wav_bytes(1, 1, 8000, 8, &[128, 255, 0], false)).unwrap();
        assert_eq!(w8.samples, vec![0.0, 127.0 / 128.0, -1.0]);

        let w24 = parse_wav(&wav_bytes(1, 1, 8000, 24, &[0, 0, 0x40, 0, 0, 0xC0], false)).unwrap();
        assert_eq!(w24.samples, vec![0.5, -0.5]);

        let mut d32 = Vec::new();
        d32.extend_from_slice(&(1i32 << 30).to_le_bytes());
        let w32 = parse_wav(&wav_bytes(1, 1, 8000, 32, &d32, false)).unwrap();
        assert_eq!(w32.samples, vec![0.5]);
    }

    #[test]
    fn float_and_extensible() {
        let mut d = Vec::new();
        d.extend_from_slice(&0.25f32.to_le_bytes());
        d.extend_from_slice(&(-0.75f32).to_le_bytes());
        let w = parse_wav(&wav_bytes(3, 1, 44100, 32, &d, false)).unwrap();
        assert_eq!(w.samples, vec![0.25, -0.75]);
        assert_eq!(w.sample_rate, 44100);

        let we = parse_wav(&wav_bytes(3, 1, 48000, 32, &d, true)).unwrap();
        assert_eq!(we.samples, vec![0.25, -0.75]);

        let mut d64 = Vec::new();
        d64.extend_from_slice(&0.125f64.to_le_bytes());
        let w64 = parse_wav(&wav_bytes(3, 1, 16000, 64, &d64, false)).unwrap();
        assert_eq!(w64.samples, vec![0.125]);

        let mut d16 = Vec::new();
        d16.extend_from_slice(&8192i16.to_le_bytes());
        let w16e = parse_wav(&wav_bytes(1, 1, 16000, 16, &d16, true)).unwrap();
        assert_eq!(w16e.samples, vec![0.25]);
    }

    #[test]
    fn load_from_disk_resamples() {
        let mut data = Vec::new();
        for i in 0..8000i32 {
            let v = ((i as f32 * 0.05).sin() * 16000.0) as i16;
            data.extend_from_slice(&v.to_le_bytes());
            data.extend_from_slice(&v.to_le_bytes());
        }
        let dir = std::env::temp_dir().join(format!("lapid-nn-wav-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tone.wav");
        std::fs::write(&path, wav_bytes(1, 2, 8000, 16, &data, false)).unwrap();
        let wav = load_wav(&path).unwrap();
        assert_eq!((wav.channels, wav.sample_rate, wav.samples.len()), (2, 8000, 16000));
        let mono16k = load_audio(&path, 16000).unwrap();
        assert_eq!(mono16k.len(), 16000);
        assert!(load_audio(&dir.join("missing.wav"), 16000).unwrap_err().message.contains("missing.wav"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(parse_wav(b"not a wav file at all").unwrap_err().code, "unsupported_audio");
        let bad = wav_bytes(2, 1, 16000, 4, &[0, 0], false); // ADPCM
        assert!(parse_wav(&bad).is_err());
    }
}
