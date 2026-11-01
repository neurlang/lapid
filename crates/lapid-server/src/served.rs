//! The adapter contract between the HTTP layer and hub models: [`Served`] exposes what a
//! loaded model can do (chat, completion, embedding, transcription, speech) through plain
//! request structs and callbacks. Adapters run on their model's worker thread, one request
//! at a time, so they may keep mutable state (sessions) without locking.

use lapid::{Error, Result};
use serde::Serialize;
use serde_json::Value;

/// What a model serves.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Capability {
    Chat,
    Completion,
    Embedding,
    Transcription,
    Speech,
}

/// Static description of a loaded model (`/v1/models`, `/api/info`).
#[derive(Clone, Debug, Serialize)]
pub struct ModelMeta {
    /// The id clients put in `"model"`.
    pub id: String,
    pub name: String,
    pub architecture: String,
    pub task: String,
    pub dtype: String,
    pub license: String,
    pub source: String,
    pub path: String,
    pub file_size: u64,
    pub device: String,
    pub capabilities: Vec<Capability>,
    pub load_ms: u64,
    /// Model-specific extras (voices, sample_rate, embedding_dim, prompts, ...), flattened.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

impl ModelMeta {
    pub fn has(&self, cap: Capability) -> bool {
        self.capabilities.contains(&cap)
    }
}

/// Sampling and length controls shared by chat and text completion. `None` = the model's
/// recommended value.
#[derive(Clone, Debug, Default)]
pub struct GenParams {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
    /// Stop strings (the matched string is not returned).
    pub stop: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct ChatParams {
    /// OpenAI messages normalized to `[{role, content: string, ...}]`.
    pub messages: Value,
    pub tools: Option<Value>,
    pub enable_thinking: Option<bool>,
    pub sampling: GenParams,
}

/// A piece of generated text.
#[derive(Clone, Debug, PartialEq)]
pub enum Delta {
    Content(String),
    Reasoning(String),
}

/// How a generation ended, with token counts and timing.
#[derive(Clone, Debug, Default, Serialize)]
pub struct GenSummary {
    /// "stop" | "length" | "cancelled".
    pub finish_reason: String,
    /// All prompt tokens (fed + reused from the session cache).
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub ttft_ms: f64,
    pub decode_ms: f64,
}

impl GenSummary {
    pub fn prefill_tps(&self) -> f64 {
        rate(self.prompt_tokens - self.cached_tokens.min(self.prompt_tokens), self.ttft_ms)
    }

    pub fn decode_tps(&self) -> f64 {
        rate(self.completion_tokens.saturating_sub(1), self.decode_ms)
    }

    pub fn timings(&self) -> Value {
        serde_json::json!({
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "generated_tokens": self.completion_tokens,
            "ttft_ms": round2(self.ttft_ms),
            "decode_ms": round2(self.decode_ms),
            "prefill_tps": round2(self.prefill_tps()),
            "decode_tps": round2(self.decode_tps()),
        })
    }
}

fn rate(n: usize, ms: f64) -> f64 {
    if ms > 0.0 { n as f64 / (ms / 1e3) } else { 0.0 }
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

#[derive(Clone, Debug, Default)]
pub struct EmbedParams {
    pub dimensions: Option<usize>,
    /// Prompt name ("query", "document", ...) for models with a prompt table.
    pub prompt_name: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Embeddings {
    pub vectors: Vec<Vec<f32>>,
    pub prompt_tokens: usize,
}

#[derive(Clone, Debug, Default)]
pub struct AsrParams {
    pub language: Option<String>,
    pub translate: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct AsrSegment {
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub tokens: Vec<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AsrWord {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Clone, Debug, Default)]
pub struct TranscriptOut {
    pub text: String,
    pub language: Option<String>,
    pub segments: Vec<AsrSegment>,
    pub words: Vec<AsrWord>,
}

#[derive(Clone, Debug, Default)]
pub struct TtsParams {
    pub voice: Option<String>,
    pub seed: Option<u64>,
    pub temperature: Option<f32>,
}

fn unsupported(what: &str) -> Error {
    Error::new("unsupported", format!("this model does not support {what}"))
}

/// A loaded model as the server sees it. Default methods fail with "unsupported".
pub trait Served {
    fn meta(&self) -> &ModelMeta;
    fn meta_mut(&mut self) -> &mut ModelMeta;

    /// Stream the assistant reply; `out` returns false when the client is gone (stop).
    fn chat(&mut self, _req: &ChatParams, _out: &mut dyn FnMut(Delta) -> bool) -> Result<GenSummary> {
        Err(unsupported("chat completions"))
    }

    /// Continue raw `prompt` text.
    fn complete(&mut self, _prompt: &str, _gen: &GenParams, _out: &mut dyn FnMut(Delta) -> bool) -> Result<GenSummary> {
        Err(unsupported("text completions"))
    }

    fn embed(&mut self, _texts: &[String], _opts: &EmbedParams) -> Result<Embeddings> {
        Err(unsupported("embeddings"))
    }

    /// Transcribe 16 kHz mono samples.
    fn transcribe(&mut self, _samples_16k: &[f32], _opts: &AsrParams) -> Result<TranscriptOut> {
        Err(unsupported("transcription"))
    }

    /// Stream PCM chunks (mono f32 at `meta().extra["sample_rate"]`); `out` returns false to stop.
    fn speak(&mut self, _text: &str, _opts: &TtsParams, _out: &mut dyn FnMut(&[f32]) -> bool) -> Result<()> {
        Err(unsupported("speech"))
    }

    /// Output sample rate of `speak` (0 when unsupported).
    fn sample_rate(&self) -> u32 {
        0
    }
}
