//! lapid-hub — ready-to-run models on lapid (Whisper speech recognition, ...).
//!
//! Each model lives in its own module (`whisper`, ...) with everything model-specific:
//! graphs, tokenizer, decoding, conversion from upstream checkpoints and verification
//! wiring. [`convert::convert`] routes a source checkpoint to its model's converter.
//!
//! ```no_run
//! use lapid_hub::Model;
//! let device = lapid::Device::best();
//! let model = lapid_hub::Whisper::from_hub("openai/whisper-small", "int8", &device)?;
//! println!("{:?}", model.info());
//! let out = model.transcribe_file("data/jfk.wav".as_ref(), &Default::default())?;
//! println!("{}", out.text);
//! # Ok::<(), lapid::Error>(())
//! ```
//!
//! # Model convention
//!
//! Every model type (`Whisper`, ...) has the same binding-friendly surface: concrete
//! argument types, no generics or lifetimes, errors with stable codes.
//!
//! ```text
//! impl Xxx {
//!     /// Load `repo` at `dtype` ("fp16", "int8", "int4", ...) from the lapid cache,
//!     /// converting from Hugging Face on first use (`model::hub_model_path`).
//!     pub fn from_hub(repo: &str, dtype: &str, device: &Device) -> Result<Xxx>;
//!     /// Load a converted .lapid file (mmap, zero copy until upload).
//!     pub fn load(path: &Path, device: &Device) -> Result<Xxx>;
//!     /// Load a .lapid image from memory (`Arc::new(vec)`, `Arc::new(include_bytes!(..))`
//!     /// as `lapid::format::Bytes`); zero copy, works on wasm.
//!     pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Xxx>;
//! }
//! impl Model for Xxx { fn info(&self) -> ModelInfo; fn device(&self) -> &Device; }
//! ```
//!
//! Implement `load` / `load_bytes` through one private `from_file(LapidFile, &Device)`
//! (`LapidFile::open` / `LapidFile::from_bytes`), and `from_hub` as
//! `Self::load(&model::hub_model_path(repo, dtype, convert_fn)?, device)`.
//! Store the `ModelInfo` (`ModelInfo::from_file(&file, task)`) at load time.
//! Developer entry points go in `<model>/cmd.rs`: `verify(reference, &cmd::ModelArgs)` and
//! `bench(input, &cmd::ModelArgs, runs)`, routed by name from [`cmd::verify`] /
//! [`cmd::bench`] (`lapid verify <model>`, `lapid bench <model>`). Apps are examples
//! (`examples/<task>.rs`), not CLI commands.
//!
//! Features: `download` (default) enables Hugging Face downloads; without it `from_hub`
//! only finds already-cached models ("download_disabled" otherwise), and `load` /
//! `load_bytes` always work.

pub mod audio;
pub mod cmd;
pub mod convert;
pub mod embeddinggemma;
pub mod gemma4;
pub mod lfm3;
pub mod model;
pub mod parakeet;
pub mod phonon;
pub mod qwen35;
pub mod whisper;

pub use embeddinggemma::{EmbedOptions, EmbeddingGemma};
pub use gemma4::Gemma4;
pub use lfm3::Lfm3;
pub use model::{Bytes, Model, ModelInfo};
pub use parakeet::Parakeet;
pub use phonon::{Phonon, SpeakOptions};
pub use qwen35::Qwen35;
pub use whisper::{Segment, Task, TranscribeOptions, Transcript, Whisper};

pub(crate) fn debug_enabled() -> bool {
    std::env::var("LAPID_DEBUG").is_ok_and(|v| !v.is_empty() && v != "0" && v != "false")
}
