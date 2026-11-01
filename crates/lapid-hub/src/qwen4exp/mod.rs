//! Qwen4-Exp (`qwen4_exp`, e.g. Qwen/Qwen3.8-Flash-Next): the Qwen3.5 MoE hybrid decoder
//! (Gated DeltaNet + gated attention, sparse MoE) with hyper-connection residual streams, QSA
//! sparse attention and PLE n-gram hash embeddings. Model graphs, streaming chat, conversion
//! from Hugging Face safetensors and GGUF (`qwen4exp`, ggml quantization kept), verification
//! and benchmark wiring.
//!
//! Configuration, the GGUF tokenizer rebuild and several conversion helpers come from
//! [`crate::qwen35`]; generation pieces (token stream, cancellation, stats) and the Jinja
//! chat-template interpreter from [`crate::lfm3`].
//!
//! ```no_run
//! use lapid_hub::qwen4exp::{ChatOptions, Message, Qwen4Exp};
//! let device = lapid::Device::best();
//! let llm = Qwen4Exp::load("qwen3.8-flash-next-q4_0.lapid".as_ref(), &device)?;
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
mod model;
pub mod verify;

use std::path::Path;

use lapid::{Device, Error, LapidFile, Result};
use lapid_nn::{Loader, SamplingParams, Tokenizer};
use serde_json::Value;

use crate::lfm3::jinja::{self, Template};
use crate::model::{Bytes, Model, ModelInfo};
pub use crate::lfm3::{CancelToken, FinishReason, GenerateOptions, GenerationStats, Message, Token, TokenStream};
pub use config::{GateAct, PleConfig, QsaConfig, Qwen4ExpConfig};
pub use lapid_nn::Precision;
use model::{Qwen4ExpModel, State};

#[derive(Clone, Debug, Default)]
pub struct ChatOptions {
    /// Template `enable_thinking` (`None` = the template default: thinking on for Qwen3.8).
    pub enable_thinking: Option<bool>,
    pub generate: GenerateOptions,
}

/// A sequence being generated: model state (KV cache, conv windows, delta-rule states) plus
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
    /// recurrent state cannot rewind), else start over. Returns the reused prompt tokens.
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
pub(crate) const TEMPLATE_BLOB: &str = "tokenizer/chat_template.jinja";
/// Metadata object name in `.lapid` files.
pub(crate) const META: &str = "qwen4exp";

fn ids_of(v: Option<&Value>) -> Vec<u32> {
    match v {
        Some(Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_u64).map(|x| x as u32).collect(),
        _ => Vec::new(),
    }
}

fn meta_err(m: impl Into<String>) -> Error {
    Error::new("metadata", format!("qwen4exp: {}", m.into()))
}

/// A loaded Qwen4-Exp model bound to a device.
pub struct Qwen4Exp {
    device: Device,
    info: ModelInfo,
    model: Qwen4ExpModel,
    tokenizer: Tokenizer,
    template: Template,
    special_tokens: Vec<(String, String)>,
    generation: Value,
    stop: Vec<u32>,
}

impl Qwen4Exp {
    /// Load `repo` (e.g. "Qwen/Qwen3.8-Flash-Next") from the lapid cache, converting it first if missing.
    pub fn from_hub(repo: &str, dtype: &str, device: &Device) -> Result<Qwen4Exp> {
        Qwen4Exp::load(&crate::model::hub_model_path(repo, dtype, convert::convert_hf)?, device)
    }

    pub fn load(path: &Path, device: &Device) -> Result<Qwen4Exp> {
        Qwen4Exp::load_with(path, device, Precision::Stored)
    }

    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Qwen4Exp> {
        Qwen4Exp::from_file(LapidFile::from_bytes(bytes)?, device, Precision::Stored)
    }

    /// Load with a weight precision (the fake-quant modes build verification references).
    pub fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Qwen4Exp> {
        Qwen4Exp::from_file(LapidFile::open(path)?, device, precision)
    }

    fn from_file(file: LapidFile, device: &Device, precision: Precision) -> Result<Qwen4Exp> {
        let meta = file.metadata.get(META).ok_or_else(|| meta_err("no \"qwen4exp\" metadata object"))?;
        let cfg = Qwen4ExpConfig::from_json(meta)?;
        let blob = |key: &str, default: &str| -> Result<String> {
            let name = meta.get(key).and_then(Value::as_str).unwrap_or(default);
            let bytes = file.blob(name).ok_or_else(|| meta_err(format!("blob {name:?} missing")))?;
            String::from_utf8(bytes.to_vec()).map_err(|_| meta_err(format!("blob {name:?} is not UTF-8")))
        };
        let tokenizer = Tokenizer::from_tokenizer_json(&blob("tokenizer_blob", TOKENIZER_BLOB)?)?;
        let template = Template::parse(&blob("chat_template_blob", TEMPLATE_BLOB)?)?;
        let mut special_tokens = Vec::new();
        if let Some(m) = meta.get("special_tokens").and_then(Value::as_object) {
            for (k, v) in m {
                if let Some(s) = v.as_str() {
                    special_tokens.push((k.clone(), s.to_string()));
                }
            }
        }
        let mut stop = cfg.base.eos.clone();
        for id in ids_of(meta.pointer("/generation/eos_token_id")) {
            if !stop.contains(&id) {
                stop.push(id);
            }
        }
        for t in ["<|im_end|>", "<|endoftext|>"] {
            if let Some(id) = tokenizer.token_to_id(t)
                && !stop.contains(&id)
            {
                stop.push(id);
            }
        }
        let model = Qwen4ExpModel::load(&Loader::new(&file, device, precision), &cfg)?;
        Ok(Qwen4Exp {
            device: device.clone(),
            info: ModelInfo::from_file(&file, "text-generation"),
            model,
            tokenizer,
            template,
            special_tokens,
            generation: meta.get("generation").cloned().unwrap_or(Value::Null),
            stop,
        })
    }

    pub fn config(&self) -> &Qwen4ExpConfig {
        &self.model.cfg
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn stop_tokens(&self) -> &[u32] {
        &self.stop
    }

    /// Sampling from `generation_config.json` (Qwen3.8-Flash-Next: temperature 1.0, top_p 0.95,
    /// top_k 20), else temperature 0.7, top_p 0.8, top_k 20.
    pub fn recommended_sampling(&self) -> SamplingParams {
        let g = &self.generation;
        let f = |k: &str, d: f64| g.get(k).and_then(Value::as_f64).unwrap_or(d) as f32;
        SamplingParams {
            temperature: f("temperature", 0.7),
            top_k: g.get("top_k").and_then(Value::as_u64).unwrap_or(20) as usize,
            top_p: f("top_p", 0.8),
            repetition_penalty: f("repetition_penalty", 1.0),
            seed: 0,
        }
    }

    pub fn render_chat(&self, messages: &[Message], add_generation_prompt: bool, enable_thinking: Option<bool>) -> Result<String> {
        let json = serde_json::to_string(messages)?;
        self.render_chat_json(&json, None, add_generation_prompt, enable_thinking)
    }

    /// Render HF-style messages / tools given as JSON (key order kept).
    pub fn render_chat_json(&self, messages_json: &str, tools_json: Option<&str>, add_generation_prompt: bool, enable_thinking: Option<bool>) -> Result<String> {
        let messages = jinja::Value::from_json_str(messages_json)?;
        let tools = match tools_json {
            Some(t) => jinja::Value::from_json_str(t)?,
            None => jinja::Value::None,
        };
        let mut globals: Vec<(&str, jinja::Value)> =
            vec![("messages", messages), ("tools", tools), ("add_generation_prompt", jinja::Value::Bool(add_generation_prompt))];
        for (k, v) in &self.special_tokens {
            globals.push((k.as_str(), jinja::Value::str(v)));
        }
        if let Some(t) = enable_thinking {
            globals.push(("enable_thinking", jinja::Value::Bool(t)));
        }
        self.template.render(&globals)
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
        let ids = self.encode(&self.render_chat(messages, true, opts.enable_thinking)?)?;
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

    pub(crate) fn model(&self) -> &Qwen4ExpModel {
        &self.model
    }
}

impl Model for Qwen4Exp {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
