//! Google Gemma 4 (text decoder of `google/gemma-4-E2B-it` & co.): model graphs (per-layer
//! embeddings, sliding / full attention with KV sharing), streaming chat generation, chat
//! template rendering (the LFM3 Jinja interpreter), conversion from Hugging Face safetensors
//! and GGUF (K-quants kept as stored), and the verification / benchmark wiring.
//!
//! ```no_run
//! use lapid_hub::gemma4::{ChatOptions, Gemma4, Message};
//! let device = lapid::Device::best();
//! let llm = Gemma4::from_hub("google/gemma-4-E2B-it", "int8", &device)?;
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
mod model;
pub mod verify;

use std::path::Path;

use lapid::{Device, Error, LapidFile, Result};
use lapid_nn::{Loader, SamplingParams, Tokenizer};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::lfm3::jinja::{self, Template};
use crate::model::{Bytes, Model, ModelInfo};
pub use crate::lfm3::{CancelToken, FinishReason, GenerateOptions, GenerationStats, Token, TokenStream};
pub use config::{Gemma4Config, LayerType};
pub use lapid_nn::Precision;
use model::{Gemma4Model, State};

/// One chat message (`role`: "system" | "user" | "assistant"; `reasoning_content` is the
/// assistant's thinking, which the template drops from earlier turns).
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
    /// Template `enable_thinking` (`None` = the template default: off).
    pub enable_thinking: Option<bool>,
    pub generate: GenerateOptions,
}

/// A sequence being generated: the KV cache plus the tokens it has consumed.
pub struct Session {
    state: State,
    tokens: Vec<u32>,
}

impl Session {
    /// Tokens fed to the model so far.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Absolute position of the next token.
    pub fn position(&self) -> usize {
        self.state.offset()
    }

    pub fn reset(&mut self) {
        self.state.clear();
        self.tokens.clear();
    }

    /// Keep the cache when the consumed tokens are a strict prefix of `prompt`, otherwise
    /// start over. Returns how many prompt tokens are already consumed.
    fn reuse_prefix(&mut self, prompt: &[u32]) -> usize {
        let n = self.tokens.len();
        if n > 0 && n < prompt.len() && prompt[..n] == self.tokens[..] {
            return n;
        }
        self.reset();
        0
    }
}

/// Parse the chat template. The LFM3 Jinja interpreter has no `dictsort` filter (Gemma 4's
/// template uses it only as `for k, v in m | dictsort`): it is rewritten to `items` and the
/// render inputs get every map sorted by key (case-insensitive, as `dictsort`). Safe for
/// templates that never iterate or dump maps any other way. Gap → candidate for the shared
/// Jinja interpreter (lapid-nn).
fn parse_template(source: &str) -> Result<(Template, bool)> {
    if !source.contains("dictsort") {
        return Ok((Template::parse(source)?, false));
    }
    let rewritten = source.replace("| dictsort", "| items").replace("|dictsort", "| items");
    if rewritten.contains("dictsort") {
        return Err(Error::new("template", "gemma4: unsupported use of the dictsort filter"));
    }
    Ok((Template::parse(&rewritten)?, true))
}

/// `v` with every map's keys sorted (case-insensitive, stable), recursively.
fn sort_maps(v: jinja::Value) -> jinja::Value {
    match v {
        jinja::Value::Map(m) => {
            let mut m: Vec<(String, jinja::Value)> = m.into_iter().map(|(k, x)| (k, sort_maps(x))).collect();
            m.sort_by_key(|(k, _)| k.to_lowercase());
            jinja::Value::Map(m)
        }
        jinja::Value::List(l) => jinja::Value::List(l.into_iter().map(sort_maps).collect()),
        other => other,
    }
}

/// A loaded Gemma 4 model bound to a device.
pub struct Gemma4 {
    device: Device,
    info: ModelInfo,
    model: Gemma4Model,
    tokenizer: Tokenizer,
    template: Template,
    /// The template used `dictsort` (rewritten to `items`; maps are pre-sorted by key).
    sort_maps: bool,
    special_tokens: Vec<(String, String)>,
    generation: Value,
    stop: Vec<u32>,
}

pub(crate) const TOKENIZER_BLOB: &str = "tokenizer/tokenizer.json";
pub(crate) const TEMPLATE_BLOB: &str = "tokenizer/chat_template.jinja";

fn meta_err(m: impl Into<String>) -> Error {
    Error::new("metadata", format!("gemma4: {}", m.into()))
}

impl Gemma4 {
    /// Load `repo` (e.g. "google/gemma-4-E2B-it") from the lapid cache, converting it from the
    /// Hugging Face hub first if missing.
    pub fn from_hub(repo: &str, dtype: &str, device: &Device) -> Result<Gemma4> {
        Gemma4::load(&crate::model::hub_model_path(repo, dtype, convert::convert_hf)?, device)
    }

    /// Load a converted `.lapid` file.
    pub fn load(path: &Path, device: &Device) -> Result<Gemma4> {
        Gemma4::load_with(path, device, Precision::Stored)
    }

    /// Load a `.lapid` image from memory (zero copy).
    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Gemma4> {
        Gemma4::from_file(LapidFile::from_bytes(bytes)?, device, Precision::Stored)
    }

    /// Load with a weight precision (the fake-quant modes build verification references).
    pub fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Gemma4> {
        Gemma4::from_file(LapidFile::open(path)?, device, precision)
    }

    fn from_file(file: LapidFile, device: &Device, precision: Precision) -> Result<Gemma4> {
        let meta = file.metadata.get("gemma4").ok_or_else(|| meta_err("no \"gemma4\" metadata object"))?;
        let cfg = Gemma4Config::from_json(meta)?;
        let blob = |key: &str, default: &str| -> Result<String> {
            let name = meta.get(key).and_then(Value::as_str).unwrap_or(default);
            let bytes = file.blob(name).ok_or_else(|| meta_err(format!("blob {name:?} missing")))?;
            String::from_utf8(bytes.to_vec()).map_err(|_| meta_err(format!("blob {name:?} is not UTF-8")))
        };
        let tokenizer = Tokenizer::from_tokenizer_json(&blob("tokenizer_blob", TOKENIZER_BLOB)?)?;
        let (template, sort_maps) = parse_template(&blob("chat_template_blob", TEMPLATE_BLOB)?)?;
        let mut special_tokens = Vec::new();
        if let Some(m) = meta.get("special_tokens").and_then(Value::as_object) {
            for (k, v) in m {
                if let Some(s) = v.as_str() {
                    special_tokens.push((k.clone(), s.to_string()));
                }
            }
        }
        let generation = meta.get("generation").cloned().unwrap_or(Value::Null);
        let mut stop = cfg.special.eos.clone();
        let gen_eos: Vec<u32> = match generation.get("eos_token_id") {
            Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect(),
            Some(v) => v.as_u64().map(|x| x as u32).into_iter().collect(),
            None => Vec::new(),
        };
        let turn_end = tokenizer.token_to_id("<turn|>");
        for id in gen_eos.into_iter().chain(turn_end) {
            if !stop.contains(&id) {
                stop.push(id);
            }
        }
        let model = Gemma4Model::load(&Loader::new(&file, device, precision), &cfg)?;
        Ok(Gemma4 {
            device: device.clone(),
            info: ModelInfo::from_file(&file, "text-generation"),
            model,
            tokenizer,
            template,
            sort_maps,
            special_tokens,
            generation,
            stop,
        })
    }

    pub fn config(&self) -> &Gemma4Config {
        &self.model.cfg
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Default stop tokens (eos ids, generation_config eos ids, `<turn|>`).
    pub fn stop_tokens(&self) -> &[u32] {
        &self.stop
    }

    /// Sampling recommended by `generation_config.json` (temperature 1.0, top_k 64, top_p 0.95).
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
        let (messages, tools) = if self.sort_maps { (sort_maps(messages), sort_maps(tools)) } else { (messages, tools) };
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

    /// Token ids of rendered chat text (special tokens recognized; nothing added).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text, false)
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        self.tokenizer.decode(ids, skip_special)
    }

    /// A fresh sequence (empty KV cache).
    pub fn new_session(&self) -> Session {
        Session { state: self.model.new_state(), tokens: Vec::new() }
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

    pub(crate) fn model(&self) -> &Gemma4Model {
        &self.model
    }
}

impl Model for Gemma4 {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
