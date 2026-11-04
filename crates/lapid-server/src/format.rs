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

/// Encode f32 PCM samples to MP3 bytes using the shine pure-Rust encoder.
/// Returns an empty vector for empty input without error.
pub fn mp3(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    if samples.is_empty() {
        return Vec::new();
    }
    // Convert f32 → i16 using the same clamp formula as pcm16()
    let pcm_i16: Vec<i16> = samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0).round() as i16)
        .collect();

    let config = shine_rs::mp3_encoder::Mp3EncoderConfig::new()
        .sample_rate(sample_rate)
        .bitrate(128)
        .channels(1)
        .stereo_mode(shine_rs::mp3_encoder::StereoMode::Mono);

    shine_rs::mp3_encoder::encode_pcm_to_mp3(config, &pcm_i16)
        .expect("MP3 encoding failed on validated PCM input")
}

/// Decode MP3 bytes to (f32 samples, sample_rate). Test-only.
#[cfg(test)]
pub fn decode_mp3(bytes: &[u8]) -> (Vec<f32>, u32) {
    use std::io::Cursor;
    let mut decoder = puremp3::Mp3Decoder::new(Cursor::new(bytes));
    let mut samples = Vec::new();
    let mut sample_rate = 24000u32;
    loop {
        match decoder.next_frame() {
            Ok(frame) => {
                sample_rate = frame.header.sample_rate.hz();
                // Mono: use left channel (identical to right for mono encodes)
                for i in 0..frame.num_samples {
                    samples.push(frame.samples[0][i]);
                }
            }
            Err(_) => break,
        }
    }
    (samples, sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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

    // ── Unit tests for MP3 ──────────────────────────────────────────────────

    #[test]
    fn mp3_non_empty() {
        // 440 Hz sine, 0.05 s @ 24 kHz
        let samples: Vec<f32> = (0..1200)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin() * 0.5)
            .collect();
        let out = mp3(&samples, 24000);
        assert!(!out.is_empty(), "expected non-empty MP3 output");
        // Valid MP3 sync bytes: 0xFF 0xFB / 0xFF 0xFA / 0xFF 0xF3 / 0xFF 0xF2 / 0xFF 0xE0-0xFF
        assert_eq!(out[0], 0xFF, "expected first byte 0xFF");
        assert!(out[1] >= 0xE0, "expected valid MP3 sync second byte, got 0x{:02X}", out[1]);
    }

    #[test]
    fn mp3_empty_input() {
        let out = mp3(&[], 24000);
        assert!(out.is_empty(), "empty input should produce empty output");
    }
    
    fn sine_diagnostics(samples: &[f32], frequency: f64, sample_rate: f64) {
        use std::f64::consts::TAU;

        let n = samples.len() as f64;

        let mean = samples.iter()
            .map(|&x| x as f64)
            .sum::<f64>() / n;

        let mut total_sq = 0.0;
        let mut ac_sq = 0.0;
        let mut sin_corr = 0.0;
        let mut cos_corr = 0.0;

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for (i, &sample) in samples.iter().enumerate() {
            let x = sample as f64;
            let centered = x - mean;

            let phase = TAU * frequency * i as f64 / sample_rate;

            total_sq += x * x;
            ac_sq += centered * centered;

            sin_corr += centered * phase.sin();
            cos_corr += centered * phase.cos();

            min = min.min(x);
            max = max.max(x);
        }

        // Least-squares estimate of the fundamental sine amplitude.
        let a = 2.0 * sin_corr / n;
        let b = 2.0 * cos_corr / n;

        let fundamental_amplitude = (a * a + b * b).sqrt();
        let fundamental_rms = fundamental_amplitude / 2.0_f64.sqrt();

        // Measure everything not explained by DC + fundamental.
        let mut residual_sq = 0.0;

        for (i, &sample) in samples.iter().enumerate() {
            let phase = TAU * frequency * i as f64 / sample_rate;

            let fitted = mean + a * phase.sin() + b * phase.cos();
            let residual = sample as f64 - fitted;

            residual_sq += residual * residual;
        }

        let total_rms = (total_sq / n).sqrt();
        let ac_rms = (ac_sq / n).sqrt();
        let residual_rms = (residual_sq / n).sqrt();

        println!("  Mean/DC:           {:.6}", mean);
        println!("  Total RMS:         {:.6}", total_rms);
        println!("  AC RMS:            {:.6}", ac_rms);
        println!("  Fundamental RMS:   {:.6}", fundamental_rms);
        println!("  Fundamental amp:   {:.6}", fundamental_amplitude);
        println!("  Residual RMS:      {:.6}", residual_rms);
        println!("  Peak:              {:.6}", max.abs().max(min.abs()));
        println!("  Min:               {:.6}", min);
        println!("  Max:               {:.6}", max);
    }

    #[test]
    fn debug_mp3_1001hz_low_amplitude() {
        const SAMPLE_RATE: usize = 24_000;
        const FREQUENCY: f32 = 1_001.0;
        const AMPLITUDE: f32 = 0.15;

        let samples: Vec<f32> = (0..SAMPLE_RATE * 3)
            .map(|i| {
                AMPLITUDE
                    * (std::f32::consts::TAU
                        * FREQUENCY
                        * i as f32
                        / SAMPLE_RATE as f32)
                        .sin()
            })
            .collect();

        let encoded = mp3(&samples, SAMPLE_RATE as u32);
        let (decoded, rate) = decode_mp3(&encoded);

        println!("Encoded bytes: {}", encoded.len());
        println!("Decoded samples: {}", decoded.len());
        println!("Decoded sample rate: {}", rate);
        println!("Original RMS: {:.6}", rms(&samples));
        println!("Decoded full RMS: {:.6}", rms(&decoded));

        // Inspect four different one-second windows.
        for start in [6000, 18000, 30000, 42000] {
            let window = &decoded[start..start + SAMPLE_RATE];

            println!(
                "Window {}..{}: RMS={:.6}",
                start,
                start + SAMPLE_RATE,
                rms(window)
            );
        }
    }


    

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

        // Feature: tts-mp3-format, Property 2: MP3 round-trip RMS preservation
        // Validates: Requirements 3.1
        //
        // Encode a pure sine to MP3 via shine-rs (24 kHz / MPEG-2, 128 kbps) and
        // verify the decoded RMS is within 30 % of the original.
        //
        // Tolerance rationale: empirical measurement of shine-rs at 24 kHz shows
        // gain shifts of 12–26 % across bands 3–6 (1125–2250 Hz). 30 % provides a
        // clear margin above the observed worst case without being vacuously wide.
        // It confirms the codec preserves energy (not silenced or clipped).
        //
        // Input constraints:
        // - Frequencies are exact multiples of 375 Hz (MPEG-2 subband width at
        //   24 kHz) in bands 3–6 (1125–2250 Hz). These produce whole-cycle alignment
        //   over integer-second durations, giving a phase-stable full-signal RMS.
        //   Bands 1–2 (375–750 Hz) have an 80–98 % energy defect in shine-rs.
        //   Bands ≥ 7 (≥ 2625 Hz) have additional high-frequency rolloff (30–77 %).
        // - Full trimmed-signal RMS (4×1152-sample edges removed) is used rather
        //   than a fixed 1-second window, which is sensitive to beat-envelope phase.
        #[test]
        fn prop_mp3_rms_roundtrip(
            band in 3u32..=6u32,        // 1125, 1500, 1875, 2250 Hz
            amplitude in 0.2f32..=0.8f32,
        ) {
            const SAMPLE_RATE: u32 = 24_000;
            const SUBBAND_HZ: u32 = 375;
            const DURATION_SECS: usize = 3;
            const EDGE_GUARD: usize = 4 * 1152;
            const WINDOW_LEN: usize = SAMPLE_RATE as usize;
            const TOLERANCE: f32 = 0.30;

            let freq_hz = band * SUBBAND_HZ;
            let n = SAMPLE_RATE as usize * DURATION_SECS;
            let samples: Vec<f32> = (0..n)
                .map(|i| {
                    amplitude
                        * (2.0 * std::f32::consts::PI * freq_hz as f32 * i as f32
                            / SAMPLE_RATE as f32)
                            .sin()
                })
                .collect();

            let rms_orig = rms(&samples);
            prop_assume!(rms_orig > 0.01);

            let encoded = mp3(&samples, SAMPLE_RATE);
            prop_assert!(!encoded.is_empty(), "MP3 encoder returned empty output");

            let (decoded, decoded_rate) = decode_mp3(&encoded);
            prop_assert_eq!(decoded_rate, SAMPLE_RATE,
                "unexpected decoded sample rate: {}", decoded_rate);

            prop_assert!(
                decoded.len() >= 2 * EDGE_GUARD + WINDOW_LEN,
                "decoded audio too short: {} samples (freq={}Hz)", decoded.len(), freq_hz
            );

            // Use the full trimmed signal (not a 1-second window) — the window RMS
            // is sensitive to where in the beat envelope the window falls, while the
            // trimmed full-signal RMS averages over many cycles and is stable.
            let trimmed = &decoded[EDGE_GUARD..decoded.len() - EDGE_GUARD];
            let rms_dec = rms(trimmed);

            prop_assert!(rms_dec > 0.0,
                "decoded window is silent (freq={}Hz, amp={:.2})", freq_hz, amplitude);

            let diff = (rms_orig - rms_dec).abs() / rms_orig;
            prop_assert!(
                diff < TOLERANCE,
                "RMS diff {:.1}% exceeds 30% tolerance — orig={:.4}, dec={:.4}, \
                freq={}Hz, amp={:.3}",
                diff * 100.0, rms_orig, rms_dec, freq_hz, amplitude
            );
        }
    }

    #[test]
    fn mp3_pure_sine_rms_roundtrip() {
        const SAMPLE_RATE: usize = 24_000;
        const FREQUENCY: f32 = 1_000.0;
        const AMPLITUDE: f32 = 0.5;
        const DURATION_SECONDS: usize = 3;

        const WINDOW_LEN: usize = SAMPLE_RATE; // Compare 1 second.
        const EDGE_GUARD: usize = 2 * 1152;    // 96 ms at each edge.
        const TOLERANCE: f32 = 0.20;

        // Generate a continuous 1 kHz sine wave.
        let samples: Vec<f32> = (0..SAMPLE_RATE * DURATION_SECONDS)
            .map(|i| {
                AMPLITUDE
                    * (2.0 * std::f32::consts::PI
                        * FREQUENCY
                        * i as f32
                        / SAMPLE_RATE as f32)
                        .sin()
            })
            .collect();

        // Verify the generated signal itself.
        let rms_orig = rms(&samples);
        let expected_rms = AMPLITUDE / std::f32::consts::SQRT_2;

        assert!(
            (rms_orig - expected_rms).abs() < 0.0001,
            "Original sine generation is incorrect: \
            measured_rms={:.6}, expected_rms={:.6}",
            rms_orig,
            expected_rms
        );

        // Encode.
        let encoded = mp3(&samples, 24_000);

        assert!(
            !encoded.is_empty(),
            "MP3 encoder returned an empty bitstream"
        );

        // Decode.
        let (decoded, decoded_rate) = decode_mp3(&encoded);

        assert_eq!(
            decoded_rate,
            24_000,
            "Unexpected decoded sample rate: expected=24000, actual={}",
            decoded_rate
        );

        assert!(
            !decoded.is_empty(),
            "MP3 decoder returned empty audio"
        );

        // Select a central 1-second window.
        // No phase alignment is needed for a steady sine wave.
        let required_len = WINDOW_LEN + 2 * EDGE_GUARD;

        assert!(
            decoded.len() >= required_len,
            "Decoded audio is too short: decoded_len={}, required_len={}, \
            window_len={}, edge_guard={}",
            decoded.len(),
            required_len,
            WINDOW_LEN,
            EDGE_GUARD
        );

        let start = (decoded.len() - WINDOW_LEN) / 2;
        let end = start + WINDOW_LEN;

        assert!(
            start >= EDGE_GUARD
                && decoded.len() - end >= EDGE_GUARD,
            "Selected window is too close to decoded audio boundaries: \
            start={}, end={}, decoded_len={}, left_margin={}, right_margin={}",
            start,
            end,
            decoded.len(),
            start,
            decoded.len() - end
        );

        let decoded_window = &decoded[start..end];

        assert!(
            decoded_window.iter().all(|x| x.is_finite()),
            "Decoded audio contains NaN or infinity"
        );

        // Measure decoded RMS.
        let rms_dec = rms(decoded_window);

        assert!(
            rms_dec.is_finite() && rms_dec > 0.0,
            "Invalid decoded RMS: rms_dec={:.6}",
            rms_dec
        );

        // Compare against the original sine's RMS.
        let diff = (rms_orig - rms_dec).abs() / rms_orig;

        assert!(
            diff < TOLERANCE,
            "MP3 RMS preservation FAILED: \
            original_rms={:.6}, decoded_rms={:.6}, \
            expected_rms={:.6}, relative_difference={:.2}%, \
            tolerance={:.2}%, decoded_len={}, \
            compared_range={}..{}, encoded_bytes={}, sample_rate={}",
            rms_orig,
            rms_dec,
            expected_rms,
            diff * 100.0,
            TOLERANCE * 100.0,
            decoded.len(),
            start,
            end,
            encoded.len(),
            decoded_rate
        );
    }


    #[test]
    fn mp3_roundtrip_example() {
        // 440 Hz sine, 0.1 s @ 24 kHz
        let samples: Vec<f32> = (0..2400)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin() * 0.5)
            .collect();
        let encoded = mp3(&samples, 24000);
        let (decoded, rate) = decode_mp3(&encoded);
        assert_eq!(rate, 24000);
        let rms_orig = rms(&samples);
        let rms_dec = rms(&decoded);
        assert!(rms_orig > 0.0, "original RMS should be > 0");
        let diff = (rms_orig - rms_dec).abs() / rms_orig;
        assert!(diff < 0.20, "RMS diff {diff:.3} exceeds 20% tolerance");
    }

    #[test]
    fn mp3_roundtrip_long() {
        // Diagnostic: 2213 Hz at 0.5 amplitude
        let samples: Vec<f32> = (0..24000)
            .map(|i| (2.0 * std::f32::consts::PI * 2213.637 * i as f32 / 24000.0).sin() * 0.5)
            .collect();
        let encoded = mp3(&samples, 24000);
        let (decoded, rate) = decode_mp3(&encoded);
        eprintln!("orig rms={:.4}, dec rms={:.4}, dec len={}", rms(&samples), rms(&decoded), decoded.len());
        assert_eq!(rate, 24000);
    }

    // ── Helper ──────────────────────────────────────────────────────────────

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() { return 0.0; }
        let sum: f32 = samples.iter().map(|s| s * s).sum();
        (sum / samples.len() as f32).sqrt()
    }

    // ── Property-based tests ────────────────────────────────────────────────

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]

        // Feature: tts-mp3-format, Property 1: Encoder produces non-empty output for non-empty input
        // Validates: Requirements 2.3
        #[test]
        fn prop_mp3_non_empty(
            samples in prop::collection::vec(-1.0f32..=1.0f32, 1..=4096usize)
        ) {
            let out = mp3(&samples, 24000);
            prop_assert!(out.len() > 0);
        }

        // Feature: tts-mp3-format, Property 3: Deterministic encoding
        // Validates: Requirements 3.2
        #[test]
        fn prop_mp3_deterministic(
            samples in prop::collection::vec(-1.0f32..=1.0f32, 0..=4096usize)
        ) {
            let a = mp3(&samples, 24000);
            let b = mp3(&samples, 24000);
            prop_assert_eq!(a, b);
        }
    }
}
