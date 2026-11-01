//! DeepSeek V4 (`deepseek_v4`; llama.cpp `deepseek4`): hyper-connections (4 residual streams
//! mixed through Sinkhorn-normalized matrices), shared-KV MQA with head dim 512 and attention
//! sinks, sliding-window + compressed attention (CSA: overlapping 4-token compressor with a
//! lightning-indexer top-k; HCA: 128-token compressor), YaRN RoPE on compressed layers,
//! grouped low-rank output projection, hash-routed and sqrt-softplus-routed MoE with a shared
//! expert. Streaming chat, conversion from upstream safetensors and GGUF, verification and
//! benchmark wiring. The MTP head is not used.
//!
//! Generation pieces (token stream, cancellation, stats) are shared with [`crate::lfm3`]; the
//! prompt format is the upstream `encoding_dsv4.py` (see [`encoding`]).
//!
//! ```no_run
//! use lapid_hub::deepseek4::{ChatOptions, Deepseek4, Message};
//! let device = lapid::Device::best();
//! let llm = Deepseek4::load("deepseek4.lapid".as_ref(), &device)?;
//! let mut session = llm.new_session();
//! for token in llm.chat_stream(&mut session, &[Message::user("Hi!")], &ChatOptions::default())? {
//!     print!("{}", token?.text);
//! }
//! # Ok::<(), lapid::Error>(())
//! ```

pub mod bench;
pub mod cmd;
pub mod config;
pub mod convert;
pub mod encoding;
mod model;
pub mod verify;

use std::path::Path;

use lapid::{Device, Error, LapidFile, Result};
use lapid_nn::{Loader, SamplingParams, Tokenizer};
use serde_json::Value;

use crate::lfm3::jinja;
use crate::model::{Bytes, Model, ModelInfo};
pub use crate::lfm3::{CancelToken, FinishReason, GenerateOptions, GenerationStats, Message, Token, TokenStream};
pub use config::{Deepseek4Config, Yarn};
pub use lapid_nn::Precision;
use model::{Deepseek4Model, State};

#[derive(Clone, Debug, Default)]
pub struct ChatOptions {
    /// Thinking mode (`<think>` opened after the assistant tag); chat mode closes it.
    pub thinking: bool,
    pub generate: GenerateOptions,
}

/// A sequence being generated: model state (raw windows, compressor / indexer states) plus
/// the tokens it consumed. Reused across chat turns when the new prompt extends them.
pub struct Session {
    state: State,
    tokens: Vec<u32>,
}

impl Session {
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn position(&self) -> usize {
        self.state.offset()
    }

    pub fn reset(&mut self) {
        self.state.clear();
        self.tokens.clear();
    }

    /// Keep the state when the consumed tokens are a strict prefix of `prompt` (the
    /// compressor states cannot rewind), else start over. Returns the reused prompt tokens.
    fn reuse_prefix(&mut self, prompt: &[u32]) -> usize {
        let n = self.tokens.len();
        if n > 0 && n < prompt.len() && prompt[..n] == self.tokens[..] {
            return n;
        }
        self.reset();
        0
    }
}

pub(crate) const TOKENIZER_BLOB: &str = "tokenizer/tokenizer.json";
/// Metadata object name in `.lapid` files.
pub(crate) const META: &str = "deepseek4";

fn meta_err(m: impl Into<String>) -> Error {
    Error::new("metadata", format!("deepseek4: {}", m.into()))
}

/// A loaded DeepSeek V4 model bound to a device.
pub struct Deepseek4 {
    device: Device,
    info: ModelInfo,
    model: Deepseek4Model,
    tokenizer: Tokenizer,
    generation: Value,
    stop: Vec<u32>,
}

impl Deepseek4 {
    /// Load `repo` (upstream safetensors layout, bf16 / f32 weights) from the lapid cache,
    /// converting it first if missing.
    pub fn from_hub(repo: &str, dtype: &str, device: &Device) -> Result<Deepseek4> {
        Deepseek4::load(&crate::model::hub_model_path(repo, dtype, convert::convert_hf)?, device)
    }

    pub fn load(path: &Path, device: &Device) -> Result<Deepseek4> {
        Deepseek4::load_with(path, device, Precision::Stored)
    }

    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Deepseek4> {
        Deepseek4::from_file(LapidFile::from_bytes(bytes)?, device, Precision::Stored)
    }

    /// Load with a weight precision (the fake-quant modes build verification references).
    pub fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Deepseek4> {
        Deepseek4::from_file(LapidFile::open(path)?, device, precision)
    }

    fn from_file(file: LapidFile, device: &Device, precision: Precision) -> Result<Deepseek4> {
        let meta = file.metadata.get(META).ok_or_else(|| meta_err("no \"deepseek4\" metadata object"))?;
        let cfg = Deepseek4Config::from_json(meta)?;
        let name = meta.get("tokenizer_blob").and_then(Value::as_str).unwrap_or(TOKENIZER_BLOB);
        let bytes = file.blob(name).ok_or_else(|| meta_err(format!("blob {name:?} missing")))?;
        let json = std::str::from_utf8(bytes).map_err(|_| meta_err(format!("blob {name:?} is not UTF-8")))?;
        let tokenizer = Tokenizer::from_tokenizer_json(json)?;
        let mut stop = cfg.eos.clone();
        if let Some(id) = tokenizer.token_to_id(encoding::EOS)
            && !stop.contains(&id)
        {
            stop.push(id);
        }
        let model = Deepseek4Model::load(&Loader::new(&file, device, precision), &cfg)?;
        Ok(Deepseek4 {
            device: device.clone(),
            info: ModelInfo::from_file(&file, "text-generation"),
            model,
            tokenizer,
            generation: meta.get("generation").cloned().unwrap_or(Value::Null),
            stop,
        })
    }

    pub fn config(&self) -> &Deepseek4Config {
        &self.model.cfg
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn stop_tokens(&self) -> &[u32] {
        &self.stop
    }

    /// Sampling from `generation_config.json` (V4: temperature 1.0, top_p 1.0).
    pub fn recommended_sampling(&self) -> SamplingParams {
        let g = &self.generation;
        let f = |k: &str, d: f64| g.get(k).and_then(Value::as_f64).unwrap_or(d) as f32;
        SamplingParams {
            temperature: f("temperature", 1.0),
            top_k: g.get("top_k").and_then(Value::as_u64).unwrap_or(0) as usize,
            top_p: f("top_p", 1.0),
            repetition_penalty: f("repetition_penalty", 1.0),
            seed: 0,
        }
    }

    pub fn render_chat(&self, messages: &[Message], thinking: bool) -> Result<String> {
        self.render_chat_json(&serde_json::to_string(messages)?, None, thinking)
    }

    /// Render messages given as JSON (OpenAI style; key order kept). `tools_json` (a list of
    /// OpenAI tools) is attached to the system message, where upstream V4 carries tools.
    pub fn render_chat_json(&self, messages_json: &str, tools_json: Option<&str>, thinking: bool) -> Result<String> {
        let mut messages = jinja::Value::from_json_str(messages_json)?;
        if let Some(t) = tools_json {
            messages = encoding::with_tools(messages, jinja::Value::from_json_str(t)?);
        }
        encoding::encode(&messages, thinking)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text, false)
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        self.tokenizer.decode(ids, skip_special)
    }

    pub fn new_session(&self) -> Session {
        Session { state: self.model.new_state(), tokens: Vec::new() }
    }

    /// Stream tokens after `prompt`, reusing the session's cached prefix.
    pub fn generate<'a>(&'a self, session: &'a mut Session, prompt: &[u32], opts: GenerateOptions) -> Result<TokenStream<'a>> {
        let cached = session.reuse_prefix(prompt);
        let (model, device) = (&self.model, &self.device);
        let step = move |ids: &[u32]| -> Result<Vec<f32>> {
            let logits = model.step(device, ids, &mut session.state)?;
            session.tokens.extend_from_slice(ids);
            Ok(logits)
        };
        TokenStream::new(step, &self.tokenizer, prompt, cached, opts, &self.stop)
    }

    pub fn chat_stream<'a>(&'a self, session: &'a mut Session, messages: &[Message], opts: &ChatOptions) -> Result<TokenStream<'a>> {
        let ids = self.encode(&self.render_chat(messages, opts.thinking)?)?;
        self.generate(session, &ids, opts.generate.clone())
    }

    pub fn chat(&self, messages: &[Message], opts: &ChatOptions) -> Result<String> {
        let mut session = self.new_session();
        let (text, _, _) = self.chat_stream(&mut session, messages, opts)?.collect_text()?;
        Ok(text)
    }

    /// Logits of the last position after feeding `ids` (low-level step).
    pub fn step(&self, session: &mut Session, ids: &[u32]) -> Result<Vec<f32>> {
        let logits = self.model.step(&self.device, ids, &mut session.state)?;
        session.tokens.extend_from_slice(ids);
        Ok(logits)
    }

    pub(crate) fn model(&self) -> &Deepseek4Model {
        &self.model
    }
}

impl Model for Deepseek4 {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
