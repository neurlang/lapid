//! Liquid AI LFM3 (hybrid short-conv + full / sliding attention LLM): model graphs,
//! streaming generation, chat template rendering (a minimal Jinja interpreter), conversion
//! from Hugging Face safetensors and GGUF, and the verification / benchmark wiring.
//!
//! ```no_run
//! use lapid_hub::lfm3::{ChatOptions, Lfm3, Message};
//! let device = lapid::Device::best();
//! let llm = Lfm3::from_hub("LiquidAI/LFM3-1.7B", "int8", &device)?;
//! let mut session = llm.new_session();
//! let messages = [Message::user("Write a haiku about rust.")];
//! for token in llm.chat_stream(&mut session, &messages, &ChatOptions::default())? {
//!     print!("{}", token?.text);
//! }
//! # Ok::<(), lapid::Error>(())
//! ```

pub mod bench;
pub mod cmd;
pub mod config;
pub mod convert;
mod generate;
pub mod jinja;
mod model;
pub mod verify;

use std::path::Path;

use lapid::{Device, Error, LapidFile, Result};
use lapid_nn::{Loader, SamplingParams, Tokenizer};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::model::{Bytes, Model, ModelInfo};
pub use config::{LayerType, Lfm3Config};
pub use generate::{CancelToken, FinishReason, GenerateOptions, GenerationStats, Session, Token, TokenStream};
pub use lapid_nn::Precision;
use jinja::Template;
use model::Lfm3Model;

/// One chat message. `reasoning_content` is the assistant's thinking (rendered as
/// `<think>…</think>` for assistant turns after the last user message).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl Message {
    pub fn new(role: &str, content: impl Into<String>) -> Message {
        Message { role: role.to_string(), content: content.into(), reasoning_content: None }
    }

    pub fn system(content: impl Into<String>) -> Message {
        Message::new("system", content)
    }

    pub fn user(content: impl Into<String>) -> Message {
        Message::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Message {
        Message::new("assistant", content)
    }

    fn to_value(&self) -> jinja::Value {
        let mut m = vec![("role".to_string(), jinja::Value::str(&self.role)), ("content".to_string(), jinja::Value::str(&self.content))];
        if let Some(r) = &self.reasoning_content {
            m.push(("reasoning_content".to_string(), jinja::Value::str(r)));
        }
        jinja::Value::map(m)
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChatOptions {
    /// Template `enable_thinking` (`None` = the template default: thinking on).
    pub enable_thinking: Option<bool>,
    pub generate: GenerateOptions,
}

/// A loaded LFM3 model bound to a device.
pub struct Lfm3 {
    device: Device,
    info: ModelInfo,
    model: Lfm3Model,
    tokenizer: Tokenizer,
    template: Template,
    /// Template globals: bos_token, eos_token, pad_token.
    special_tokens: Vec<(String, String)>,
    generation: Value,
    stop: Vec<u32>,
}

/// Names of the blobs in an LFM3 `.lapid` file.
pub(crate) const TOKENIZER_BLOB: &str = "tokenizer/tokenizer.json";
pub(crate) const TEMPLATE_BLOB: &str = "tokenizer/chat_template.jinja";

fn meta_err(m: impl Into<String>) -> Error {
    Error::new("metadata", format!("lfm3: {}", m.into()))
}

impl Lfm3 {
    /// Load `repo` (e.g. "LiquidAI/LFM3-1.7B") from the lapid cache, converting it from the
    /// Hugging Face hub first if missing.
    pub fn from_hub(repo: &str, dtype: &str, device: &Device) -> Result<Lfm3> {
        Lfm3::load(&crate::model::hub_model_path(repo, dtype, convert::convert_hf)?, device)
    }

    /// Load a converted `.lapid` file.
    pub fn load(path: &Path, device: &Device) -> Result<Lfm3> {
        Lfm3::load_with(path, device, Precision::Stored)
    }

    /// Load a `.lapid` image from memory (zero copy).
    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Lfm3> {
        Lfm3::from_file(LapidFile::from_bytes(bytes)?, device, Precision::Stored)
    }

    /// Load with a weight precision (the fake-quant modes build verification references).
    pub fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Lfm3> {
        Lfm3::from_file(LapidFile::open(path)?, device, precision)
    }

    fn from_file(file: LapidFile, device: &Device, precision: Precision) -> Result<Lfm3> {
        let meta = file.metadata.get("lfm3").ok_or_else(|| meta_err("no \"lfm3\" metadata object"))?;
        let cfg = Lfm3Config::from_json(meta)?;
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
        let mut stop = cfg.special.eos.clone();
        if let Some(id) = tokenizer.token_to_id("<|im_end|>")
            && !stop.contains(&id)
        {
            stop.push(id);
        }
        let model = Lfm3Model::load(&Loader::new(&file, device, precision), &cfg)?;
        Ok(Lfm3 {
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

    pub fn config(&self) -> &Lfm3Config {
        &self.model.cfg
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Default stop tokens (eos ids + `<|im_end|>`).
    pub fn stop_tokens(&self) -> &[u32] {
        &self.stop
    }

    /// Sampling recommended by the model's `generation_config.json` (temperature 0.1,
    /// top_k 50, repetition penalty 1.1 for LFM3-1.7B).
    pub fn recommended_sampling(&self) -> SamplingParams {
        let g = &self.generation;
        let f = |k: &str, d: f64| g.get(k).and_then(Value::as_f64).unwrap_or(d) as f32;
        let do_sample = g.get("do_sample").and_then(Value::as_bool).unwrap_or(false);
        SamplingParams {
            temperature: if do_sample { f("temperature", 1.0) } else { 0.0 },
            top_k: g.get("top_k").and_then(Value::as_u64).unwrap_or(0) as usize,
            top_p: f("top_p", 1.0),
            repetition_penalty: f("repetition_penalty", 1.0),
            seed: 0,
        }
    }

    /// Render the chat template for typed messages.
    pub fn render_chat(&self, messages: &[Message], add_generation_prompt: bool, enable_thinking: Option<bool>) -> Result<String> {
        let msgs = jinja::Value::List(messages.iter().map(Message::to_value).collect());
        self.render_values(msgs, jinja::Value::None, add_generation_prompt, enable_thinking)
    }

    /// Render the chat template for HF-style messages / tools given as JSON (any fields,
    /// key order kept), e.g. tool calls and multimodal content parts.
    pub fn render_chat_json(
        &self,
        messages_json: &str,
        tools_json: Option<&str>,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> Result<String> {
        let msgs = jinja::Value::from_json_str(messages_json)?;
        let tools = match tools_json {
            Some(t) => jinja::Value::from_json_str(t)?,
            None => jinja::Value::None,
        };
        self.render_values(msgs, tools, add_generation_prompt, enable_thinking)
    }

    fn render_values(
        &self,
        messages: jinja::Value,
        tools: jinja::Value,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> Result<String> {
        let mut globals: Vec<(&str, jinja::Value)> = vec![
            ("messages", messages),
            ("tools", tools),
            ("add_generation_prompt", jinja::Value::Bool(add_generation_prompt)),
        ];
        for (k, v) in &self.special_tokens {
            globals.push((k.as_str(), jinja::Value::str(v)));
        }
        if let Some(t) = enable_thinking {
            globals.push(("enable_thinking", jinja::Value::Bool(t)));
        }
        self.template.render(&globals)
    }

    /// Token ids of rendered chat text (special tokens in the text are recognized; nothing added).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text, false)
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        self.tokenizer.decode(ids, skip_special)
    }

    /// A fresh sequence (KV cache + conv state).
    pub fn new_session(&self) -> Session {
        Session::new(self.model.new_state())
    }

    /// Stream tokens after `prompt`. When the session already consumed a prefix of `prompt`
    /// (e.g. the earlier turns of a chat), only the rest is prefilled.
    pub fn generate<'a>(&'a self, session: &'a mut Session, prompt: &[u32], opts: GenerateOptions) -> Result<TokenStream<'a>> {
        let cached = session.reuse_prefix(prompt);
        let model = &self.model;
        let device = &self.device;
        let step = move |ids: &[u32]| -> Result<Vec<f32>> {
            let logits = model.step(device, ids, &mut session.state)?;
            session.tokens.extend_from_slice(ids);
            Ok(logits)
        };
        TokenStream::new(step, &self.tokenizer, prompt, cached, opts, &self.stop)
    }

    /// Stream the assistant reply to `messages` (template with the generation prompt).
    pub fn chat_stream<'a>(&'a self, session: &'a mut Session, messages: &[Message], opts: &ChatOptions) -> Result<TokenStream<'a>> {
        let text = self.render_chat(messages, true, opts.enable_thinking)?;
        let ids = self.encode(&text)?;
        self.generate(session, &ids, opts.generate.clone())
    }

    /// The full assistant reply (non-streaming convenience).
    pub fn chat(&self, messages: &[Message], opts: &ChatOptions) -> Result<String> {
        let mut session = self.new_session();
        let (text, _, _) = self.chat_stream(&mut session, messages, opts)?.collect_text()?;
        Ok(text)
    }

    /// Logits of the last position after feeding `ids` to `session` (low-level step).
    pub fn step(&self, session: &mut Session, ids: &[u32]) -> Result<Vec<f32>> {
        let logits = self.model.step(&self.device, ids, &mut session.state)?;
        session.tokens.extend_from_slice(ids);
        Ok(logits)
    }

    pub(crate) fn model(&self) -> &Lfm3Model {
        &self.model
    }
}

impl Model for Lfm3 {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
