//! Audio front-end: WAV loading, resampling, STFT, mel filterbanks and the Kaldi / lhotse
//! fbank (whole-signal and streaming).
//!
//! Signal processing runs on the host in f64 (like the numpy / torch references) and hands
//! the model f32 features.

mod fbank;
mod mel;
mod resample;
mod stft;
mod wav;

pub use fbank::{Fbank, OnlineFbank};
pub use mel::{MelFilterbank, MelNorm, MelScale, MelSpectrogram, hz_to_mel, mel_to_hz};
pub use resample::resample;
pub use stft::{PadMode, Stft, StftConfig, hann_window};
pub use wav::{Wav, load_audio, load_wav, mixdown, parse_wav};
