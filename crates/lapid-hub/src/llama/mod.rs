//! Plain-transformer LLM families in one implementation, selected by config flags: Llama
//! (2 / 3.x incl. llama3 RoPE scaling, tied or untied embeddings; Mistral-style configs),
//! Qwen2 / Qwen2.5 (q/k/v bias), Qwen3 dense (q/k RMSNorm) and Qwen3-MoE (sparse experts via
//! `Tensor::matmul_id`). Model graphs, streaming chat with sessions, conversion from Hugging
//! Face safetensors and GGUF (`llama`, `qwen2`, `qwen3`, `qwen3moe`; K-quants kept as-is),
//! verification and benchmark wiring.
//!
//! Generation pieces (token stream, cancellation, stats) and the Jinja chat-template
//! interpreter are shared with [`crate::lfm3`].
//!
//! ```no_run
//! use lapid_hub::llama::{ChatOptions, Llama, Message};
//! let device = lapid::Device::best();
//! let llm = Llama::from_hub("Qwen/Qwen3-0.6B", "int8", &device)?;
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
mod template;
pub mod verify;

use std::path::Path;

use lapid::{Device, Error, LapidFile, Result};
use lapid_nn::{Loader, SamplingParams, Tokenizer};
use serde_json::Value;

use crate::lfm3::jinja;
use crate::model::{Bytes, Model, ModelInfo};
pub use crate::lfm3::{CancelToken, FinishReason, GenerateOptions, GenerationStats, Message, Token, TokenStream};
pub use config::{LlamaConfig, MoeConfig, RopeScaling};
pub use lapid_nn::Precision;
use model::{LlamaModel, State};
use template::ChatTemplate;

/// Template options of a chat turn.
#[derive(Clone, Debug, Default)]
pub struct ChatOptions {
    /// Template `enable_thinking` (Qwen3; `None` = the template default).
    pub enable_thinking: Option<bool>,
    /// Template `date_string` (Llama 3.x "Today Date"); `None` = today's UTC date.
    pub date_string: Option<String>,
    pub generate: GenerateOptions,
}

/// A sequence being generated: the KV cache plus the tokens it consumed. Reused across chat
/// turns when the new prompt extends them.
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

    /// Keep the cache when the consumed tokens are a strict prefix of `prompt`, else start
    /// over. Returns the reused prompt tokens.
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
pub(crate) const META: &str = "llama";

/// Tokens that end a turn in the supported families' chat formats.
const STOP_TOKENS: [&str; 5] = ["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>"];

fn meta_err(m: impl Into<String>) -> Error {
    Error::new("metadata", format!("llama: {}", m.into()))
}

/// A loaded Llama / Qwen2 / Qwen3 / Qwen3-MoE model bound to a device.
pub struct Llama {
    device: Device,
    info: ModelInfo,
    model: LlamaModel,
    tokenizer: Tokenizer,
    template: ChatTemplate,
    special_tokens: Vec<(String, String)>,
    generation: Value,
    stop: Vec<u32>,
}

impl Llama {
    /// Load `repo` (e.g. "Qwen/Qwen3-0.6B", "meta-llama/Llama-3.2-1B-Instruct") from the lapid
    /// cache, converting it first if missing.
    pub fn from_hub(repo: &str, dtype: &str, device: &Device) -> Result<Llama> {
        Llama::load(&crate::model::hub_model_path(repo, dtype, convert::convert_hf)?, device)
    }

    pub fn load(path: &Path, device: &Device) -> Result<Llama> {
        Llama::load_with(path, device, Precision::Stored)
    }

    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Llama> {
        Llama::from_file(LapidFile::from_bytes(bytes)?, device, Precision::Stored)
    }

    /// Load with a weight precision (the fake-quant modes build verification references).
    pub fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Llama> {
        Llama::from_file(LapidFile::open(path)?, device, precision)
    }

    fn from_file(file: LapidFile, device: &Device, precision: Precision) -> Result<Llama> {
        let meta = file.metadata.get(META).ok_or_else(|| meta_err("no \"llama\" metadata object"))?;
        let cfg = LlamaConfig::from_json(meta)?;
        let blob = |key: &str, default: &str| -> Result<String> {
            let name = meta.get(key).and_then(Value::as_str).unwrap_or(default);
            let bytes = file.blob(name).ok_or_else(|| meta_err(format!("blob {name:?} missing")))?;
            String::from_utf8(bytes.to_vec()).map_err(|_| meta_err(format!("blob {name:?} is not UTF-8")))
        };
        let tokenizer = Tokenizer::from_tokenizer_json(&blob("tokenizer_blob", TOKENIZER_BLOB)?)?;
        let template = ChatTemplate::parse(&blob("chat_template_blob", TEMPLATE_BLOB)?)?;
        let mut special_tokens = Vec::new();
        if let Some(m) = meta.get("special_tokens").and_then(Value::as_object) {
            for (k, v) in m {
                if let Some(s) = v.as_str() {
                    special_tokens.push((k.clone(), s.to_string()));
                }
            }
        }
        let generation = meta.get("generation").cloned().unwrap_or(Value::Null);
        let mut stop = cfg.eos.clone();
        for id in config::ids(generation.get("eos_token_id")).into_iter().chain(STOP_TOKENS.iter().filter_map(|t| tokenizer.token_to_id(t))) {
            if !stop.contains(&id) {
                stop.push(id);
            }
        }
        let model = LlamaModel::load(&Loader::new(&file, device, precision), &cfg)?;
        Ok(Llama {
            device: device.clone(),
            info: ModelInfo::from_file(&file, "text-generation"),
            model,
            tokenizer,
            template,
            special_tokens,
            generation,
            stop,
        })
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.model.cfg
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The chat template as stored in the file.
    pub fn chat_template_source(&self) -> &str {
        self.template.source()
    }

    pub fn stop_tokens(&self) -> &[u32] {
        &self.stop
    }

    /// Sampling from `generation_config.json` when the checkpoint has one (missing keys:
    /// temperature 0.7, top_p 0.8, top_k 20), else greedy-ish defaults of the same values.
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

    pub fn render_chat(&self, messages: &[Message], add_generation_prompt: bool, opts: &ChatOptions) -> Result<String> {
        let json = serde_json::to_string(messages)?;
        self.render_chat_json(&json, None, add_generation_prompt, opts)
    }

    /// Render HF-style messages / tools given as JSON (key order kept).
    pub fn render_chat_json(&self, messages_json: &str, tools_json: Option<&str>, add_generation_prompt: bool, opts: &ChatOptions) -> Result<String> {
        let messages = jinja::Value::from_json_str(messages_json)?;
        let tools = match tools_json {
            Some(t) => self.template.tools(jinja::Value::from_json_str(t)?),
            None => jinja::Value::None,
        };
        let date = opts.date_string.clone().unwrap_or_else(template::today);
        let mut globals: Vec<(&str, jinja::Value)> = vec![
            ("messages", messages),
            ("tools", tools),
            ("add_generation_prompt", jinja::Value::Bool(add_generation_prompt)),
            ("date_string", jinja::Value::Str(date)),
        ];
        for (k, v) in &self.special_tokens {
            globals.push((k.as_str(), jinja::Value::str(v)));
        }
        if let Some(t) = opts.enable_thinking {
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
        let ids = self.encode(&self.render_chat(messages, true, opts)?)?;
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

    pub(crate) fn model(&self) -> &LlamaModel {
        &self.model
    }
}

impl Model for Llama {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
