//! Audio input for the hub's speech models: WAV loading and resampling come from
//! [`lapid_nn::audio`]; this module fixes the 16 kHz mono format the models expect.

use std::path::Path;

use lapid::Result;

pub use lapid_nn::audio::{Wav, mixdown, parse_wav, resample};

/// Sample rate of `load_audio_16k_mono` output.
pub const SAMPLE_RATE: u32 = 16_000;

/// Load a WAV file as 16 kHz mono f32.
pub fn load_audio_16k_mono(path: &Path) -> Result<Vec<f32>> {
    lapid_nn::audio::load_audio(path, SAMPLE_RATE)
}
