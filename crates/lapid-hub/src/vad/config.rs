//! Silero VAD hyper-parameters, read from the `.lapid` metadata (`metadata["silero_vad"]`,
//! written by [`super::convert`] from the exported `config.json`).
//!
//! The network is fixed by the upstream model (v5, 16 kHz branch); the config records the
//! framing (window, context, STFT) and the encoder channel plan so a file of a different
//! shape fails at load instead of producing garbage.

use lapid::{Error, Result, bail};
use serde_json::{Value, json};

/// One encoder block: Conv1d(in, out, k=3, stride, pad=1) + ReLU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderBlock {
    pub in_channels: usize,
    pub out_channels: usize,
    pub stride: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VadConfig {
    /// Upstream model version ("v5").
    pub version: String,
    pub sample_rate: u32,
    /// New samples per model call (512 at 16 kHz).
    pub window_size: usize,
    /// Samples of the previous window prepended to each window (64 at 16 kHz).
    pub context_size: usize,
    /// STFT basis length (256) and hop (128); the input is reflect-padded on the right by
    /// `filter_length / 4`.
    pub filter_length: usize,
    pub hop_length: usize,
    pub encoder: Vec<EncoderBlock>,
    /// LSTM cell width (128).
    pub hidden_size: usize,
}

impl VadConfig {
    /// Silero VAD v5, 16 kHz.
    pub fn v5_16k() -> VadConfig {
        let b = |i, o, s| EncoderBlock { in_channels: i, out_channels: o, stride: s };
        VadConfig {
            version: "v5".into(),
            sample_rate: 16_000,
            window_size: 512,
            context_size: 64,
            filter_length: 256,
            hop_length: 128,
            encoder: vec![b(129, 128, 1), b(128, 64, 2), b(64, 64, 2), b(64, 128, 1)],
            hidden_size: 128,
        }
    }

    /// Frequency bins of the STFT magnitude (`filter_length / 2 + 1`).
    pub fn n_bins(&self) -> usize {
        self.filter_length / 2 + 1
    }

    /// Right reflect padding of the STFT input.
    pub fn stft_pad(&self) -> usize {
        self.filter_length / 4
    }

    /// Samples the model sees per call: context + window.
    pub fn input_size(&self) -> usize {
        self.context_size + self.window_size
    }

    /// STFT frames per call.
    pub fn stft_frames(&self) -> usize {
        (self.input_size() + self.stft_pad() - self.filter_length) / self.hop_length + 1
    }

    /// Time steps after the encoder (must be 1: one LSTM step per window).
    pub fn encoder_frames(&self) -> usize {
        self.encoder.iter().fold(self.stft_frames(), |l, b| (l + 2 - 3) / b.stride + 1)
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(Error::new("unsupported_model", format!("silero_vad config: {m}")));
        if self.sample_rate == 0 || self.window_size == 0 || self.hop_length == 0 || self.filter_length < 4 {
            return bad("zero sizes".into());
        }
        if self.input_size() + self.stft_pad() < self.filter_length || self.stft_pad() >= self.input_size() {
            return bad(format!("window {} + context {} too short for filter {}", self.window_size, self.context_size, self.filter_length));
        }
        let mut c = self.n_bins();
        for (i, b) in self.encoder.iter().enumerate() {
            if b.in_channels != c || b.stride == 0 {
                return bad(format!("encoder block {i} expects {} channels, previous gives {c}", b.in_channels));
            }
            c = b.out_channels;
        }
        if c != self.hidden_size {
            return bad(format!("encoder output {c} != hidden_size {}", self.hidden_size));
        }
        if self.encoder_frames() != 1 {
            return bad(format!("encoder leaves {} frames per window (expected 1)", self.encoder_frames()));
        }
        Ok(())
    }

    pub fn to_json(&self) -> Value {
        json!({
            "version": self.version,
            "sample_rate": self.sample_rate,
            "window_size_samples": self.window_size,
            "context_size_samples": self.context_size,
            "filter_length": self.filter_length,
            "hop_length": self.hop_length,
            "encoder": self.encoder.iter().map(|b| json!([b.in_channels, b.out_channels, b.stride])).collect::<Vec<_>>(),
            "hidden_size": self.hidden_size,
        })
    }

    /// From a JSON object with the [`to_json`](Self::to_json) keys (also the exported HF-style
    /// `config.json`); missing keys keep the v5 16 kHz values.
    pub fn from_json(v: &Value) -> Result<VadConfig> {
        let mut c = VadConfig::v5_16k();
        let u = |k: &str, d: usize| v.get(k).and_then(Value::as_u64).map_or(d, |x| x as usize);
        if let Some(ver) = v.get("version").and_then(Value::as_str) {
            if ver != "v5" {
                bail!("unsupported_model", "silero_vad version {ver:?} (only \"v5\" is implemented)");
            }
            c.version = ver.to_string();
        }
        c.sample_rate = u("sample_rate", c.sample_rate as usize) as u32;
        if c.sample_rate != 16_000 {
            bail!("unsupported_model", "silero_vad sample_rate {} (only the 16 kHz branch is implemented)", c.sample_rate);
        }
        c.window_size = u("window_size_samples", c.window_size);
        c.context_size = u("context_size_samples", c.context_size);
        c.filter_length = u("filter_length", c.filter_length);
        c.hop_length = u("hop_length", c.hop_length);
        c.hidden_size = u("hidden_size", c.hidden_size);
        if let Some(enc) = v.get("encoder").and_then(Value::as_array) {
            c.encoder = enc
                .iter()
                .map(|b| {
                    let n = |i: usize| b.get(i).and_then(Value::as_u64).map(|x| x as usize);
                    match (n(0), n(1), n(2)) {
                        (Some(i), Some(o), Some(s)) => Ok(EncoderBlock { in_channels: i, out_channels: o, stride: s }),
                        _ => Err(Error::new("metadata", "silero_vad encoder entries must be [in, out, stride]")),
                    }
                })
                .collect::<Result<_>>()?;
        }
        c.validate()?;
        Ok(c)
    }

    /// From `.lapid` metadata (`metadata["silero_vad"]`).
    pub fn from_metadata(meta: &Value) -> Result<VadConfig> {
        let v = meta.get("silero_vad").ok_or_else(|| Error::new("missing_metadata", "no \"silero_vad\" config in metadata"))?;
        VadConfig::from_json(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v5_shapes_and_roundtrip() {
        let c = VadConfig::v5_16k();
        c.validate().expect("valid");
        assert_eq!((c.n_bins(), c.stft_pad(), c.input_size(), c.stft_frames(), c.encoder_frames()), (129, 64, 576, 4, 1));
        assert_eq!(VadConfig::from_json(&c.to_json()).expect("parse"), c);
        assert_eq!(VadConfig::from_json(&json!({"sample_rate": 8000})).unwrap_err().code, "unsupported_model");
        assert_eq!(VadConfig::from_json(&json!({"version": "v6"})).unwrap_err().code, "unsupported_model");
    }
}
