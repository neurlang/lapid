//! Output formats shared by the server and `lapid run`: subtitles (SRT / WebVTT) and
//! 16-bit PCM WAV.

use lapid_hub::runtime::serve::AsrSegment;

fn timestamp(t: f32, sep: char) -> String {
    let ms = (t.max(0.0) * 1000.0).round() as u64;
    let (h, m, s, ms) = (ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000);
    format!("{h:02}:{m:02}:{s:02}{sep}{ms:03}")
}

pub fn srt(segments: &[AsrSegment]) -> String {
    let mut out = String::new();
    for (i, s) in segments.iter().enumerate() {
        out.push_str(&format!("{}\n{} --> {}\n{}\n\n", i + 1, timestamp(s.start, ','), timestamp(s.end, ','), s.text.trim()));
    }
    out
}

pub fn vtt(segments: &[AsrSegment]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for s in segments {
        out.push_str(&format!("{} --> {}\n{}\n\n", timestamp(s.start, '.'), timestamp(s.end, '.'), s.text.trim()));
    }
    out
}

/// f32 samples as 16-bit little-endian PCM.
pub fn pcm16(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// 44-byte mono 16-bit PCM WAV header; `data_bytes = None` writes the "unknown length"
/// sizes (0xFFFFFFFF) used for streamed WAV.
pub fn wav_header(sample_rate: u32, data_bytes: Option<u32>) -> Vec<u8> {
    let data = data_bytes.unwrap_or(u32::MAX);
    let riff = data_bytes.map_or(u32::MAX, |d| d.saturating_add(36));
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&riff.to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes()); // PCM
    h.extend_from_slice(&1u16.to_le_bytes()); // mono
    h.extend_from_slice(&sample_rate.to_le_bytes());
    h.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    h.extend_from_slice(&2u16.to_le_bytes());
    h.extend_from_slice(&16u16.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data.to_le_bytes());
    h
}

/// A complete WAV file.
pub fn wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let pcm = pcm16(samples);
    let mut out = wav_header(sample_rate, Some(pcm.len() as u32));
    out.extend_from_slice(&pcm);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subtitles() {
        let s = vec![AsrSegment { start: 0.0, end: 3661.5, text: " hi ".into(), tokens: vec![] }];
        assert_eq!(srt(&s), "1\n00:00:00,000 --> 01:01:01,500\nhi\n\n");
        assert_eq!(vtt(&s), "WEBVTT\n\n00:00:00.000 --> 01:01:01.500\nhi\n\n");
    }

    #[test]
    fn wav_roundtrip() {
        let w = wav(&[0.0, 0.5, -1.0], 24000);
        let parsed = lapid_hub::audio::parse_wav(&w).expect("parse");
        assert_eq!((parsed.sample_rate, parsed.channels, parsed.samples.len()), (24000, 1, 3));
    }
}
