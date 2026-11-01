//! Parakeet TDT hyper-parameters, read from the `.lapid` metadata (`metadata["parakeet"]`,
//! written by [`super::convert`]).

use lapid::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(test)]
use serde_json::json;

/// Blob holding the Hugging Face `tokenizer.json`.
pub(crate) const TOKENIZER_BLOB: &str = "tokenizer/tokenizer.json";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParakeetConfig {
    // ---- feature extractor
    pub sample_rate: u32,
    pub n_mels: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    #[serde(default = "default_preemphasis")]
    pub preemphasis: f64,
    // ---- encoder (FastConformer)
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ff_dim: usize,
    pub conv_kernel: usize,
    pub subsampling_factor: usize,
    pub subsampling_channels: usize,
    // ---- TDT decoder + joint
    pub decoder_hidden: usize,
    pub decoder_layers: usize,
    /// Tokens incl. blank (blank is the last id).
    pub vocab_size: usize,
    pub blank_id: u32,
    pub durations: Vec<usize>,
    pub max_symbols_per_step: usize,
}

impl ParakeetConfig {
    /// parakeet-tdt-0.6b-v3 values (defaults for anything a source does not state).
    pub fn v3() -> ParakeetConfig {
        ParakeetConfig {
            sample_rate: 16_000,
            n_mels: 128,
            n_fft: 512,
            hop_length: 160,
            win_length: 400,
            preemphasis: 0.97,
            d_model: 1024,
            n_layers: 24,
            n_heads: 8,
            ff_dim: 4096,
            conv_kernel: 9,
            subsampling_factor: 8,
            subsampling_channels: 256,
            decoder_hidden: 640,
            decoder_layers: 2,
            vocab_size: 8193,
            blank_id: 8192,
            durations: vec![0, 1, 2, 3, 4],
            max_symbols_per_step: 10,
        }
    }

    /// Seconds per encoder frame (hop × subsampling / rate): 0.08 s for v3.
    pub fn frame_seconds(&self) -> f64 {
        (self.hop_length * self.subsampling_factor) as f64 / self.sample_rate as f64
    }

    /// Number of stride-2 subsampling stages (log2 of the factor).
    pub fn subsampling_stages(&self) -> usize {
        self.subsampling_factor.trailing_zeros() as usize
    }

    /// Mel bins left after the subsampling stages (128 / 8 = 16).
    pub fn subsampled_mels(&self) -> usize {
        let mut f = self.n_mels;
        for _ in 0..self.subsampling_stages() {
            f = f.div_ceil(2);
        }
        f
    }

    /// Joint output width: tokens (incl. blank) + durations.
    pub fn joint_dim(&self) -> usize {
        self.vocab_size + self.durations.len()
    }

    /// The metadata object (`metadata["parakeet"]`): the fields plus the tokenizer blob name.
    pub fn to_json(&self) -> Value {
        let mut v = serde_json::to_value(self).unwrap_or_default();
        v["tokenizer_blob"] = Value::from(TOKENIZER_BLOB);
        v
    }

    pub fn from_metadata(metadata: &Value) -> Result<ParakeetConfig> {
        let p = metadata
            .get("parakeet")
            .ok_or_else(|| meta_err("not a parakeet model: metadata has no 'parakeet' object"))?;
        let mut cfg = ParakeetConfig::deserialize(p).map_err(|e| meta_err(format!("parakeet metadata: {e}")))?;
        cfg.max_symbols_per_step = cfg.max_symbols_per_step.max(1);
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(meta_err(format!("parakeet config: {m}")));
        if self.n_heads == 0 || !self.d_model.is_multiple_of(self.n_heads) {
            return bad(format!("d_model {} not divisible by {} heads", self.d_model, self.n_heads));
        }
        if !self.subsampling_factor.is_power_of_two() || self.subsampling_factor < 2 {
            return bad(format!("subsampling factor {} must be a power of two >= 2", self.subsampling_factor));
        }
        if self.conv_kernel.is_multiple_of(2) {
            return bad(format!("conv kernel {} must be odd", self.conv_kernel));
        }
        if self.blank_id as usize >= self.vocab_size || self.durations.is_empty() {
            return bad(format!("blank id {} / vocab {} / {} durations", self.blank_id, self.vocab_size, self.durations.len()));
        }
        if self.win_length > self.n_fft || self.hop_length == 0 || self.sample_rate == 0 {
            return bad(format!("stft n_fft {} win {} hop {}", self.n_fft, self.win_length, self.hop_length));
        }
        Ok(())
    }
}

fn default_preemphasis() -> f64 {
    0.97
}

fn meta_err(msg: impl Into<String>) -> Error {
    Error::new("metadata", msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_derived() {
        let c = ParakeetConfig::v3();
        let back = ParakeetConfig::from_metadata(&json!({ "parakeet": c.to_json() })).expect("parse");
        assert_eq!(back, c);
        assert!((c.frame_seconds() - 0.08).abs() < 1e-12);
        assert_eq!(c.subsampling_stages(), 3);
        assert_eq!(c.subsampled_mels(), 16);
        assert_eq!(c.joint_dim(), 8198);
        assert_eq!(ParakeetConfig::from_metadata(&json!({})).unwrap_err().code, "metadata");
    }
}
