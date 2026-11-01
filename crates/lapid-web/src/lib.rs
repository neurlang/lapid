//! lapid-web: lapid models in the browser (wasm-bindgen), on WebGPU with a CPU fallback.
//!
//! JS API (see `plans/web/README.md` (local demo, not in git)):
//! ```js
//! import init, { Whisper } from "./pkg/lapid.js";
//! await init();
//! const w = await Whisper.load(bytes /* Uint8Array of a .lapid file */, "auto" /* | "webgpu" | "cpu" */);
//! const { text, segments } = await w.transcribe(samples16k /* Float32Array */);
//! ```
//!
//! The lapid runtime is synchronous, but WebGPU buffer reads are asynchronous in browsers.
//! On WebGPU each model call runs inside `WebAssembly.promising` (JSPI): the backend's
//! readback waits through a `WebAssembly.Suspending` import that yields to the event loop,
//! so the whole (synchronous) Rust stack suspends until the GPU data is mapped. See
//! `wasm.rs` and `js/lapid_jspi.js`. Without JSPI the CPU backend is used.
//!
//! The target-independent part (`load_whisper`, `transcribe_json`) is the same code path the
//! native example `examples/transcribe.rs` runs, so it can be verified natively on
//! `lapid-webgpu`.

#[cfg(target_arch = "wasm32")]
mod wasm;

use std::sync::Arc;

use lapid::{Device, Result};
use lapid_hub::{TranscribeOptions, Whisper};

/// Load a `.lapid` Whisper model from memory onto `device`.
pub fn load_whisper(bytes: Vec<u8>, device: &Device) -> Result<Whisper> {
    Whisper::load_bytes(Arc::new(bytes), device)
}

/// Transcription options from the JS-facing arguments.
pub fn options(language: Option<String>, timestamps: Option<bool>) -> TranscribeOptions {
    let language = language.map(|l| l.trim().to_string()).filter(|l| !l.is_empty() && l != "auto");
    TranscribeOptions { language, timestamps: timestamps.unwrap_or(true), ..TranscribeOptions::default() }
}

/// Transcribe 16 kHz mono samples; the transcript as JSON `{text, language, segments}`.
pub fn transcribe_json(model: &Whisper, samples_16k: &[f32], opts: &TranscribeOptions) -> Result<String> {
    let t = model.transcribe(samples_16k, opts)?;
    serde_json::to_string(&t).map_err(|e| lapid::Error::new("internal", format!("transcript json: {e}")))
}
