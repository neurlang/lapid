//! Chat side of a language model: typed [`Message`]s, the chat template (rendered with the
//! shared Jinja interpreter exactly like transformers' `apply_chat_template`), and the text
//! assets a converted `.lapid` carries (tokenizer, template, special tokens, generation
//! defaults, stop tokens) read from the model's metadata object.

use lapid::{Error, LapidFile, Result};
use lapid_convert::assets::{TEMPLATE_BLOB, TOKENIZER_BLOB};
use lapid_nn::{SamplingParams, Tokenizer};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::generate::GenerateOptions;
use super::jinja::{self, Template};

/// One chat message. `reasoning_content` is the assistant's thinking (templates render it
/// as e.g. `<think>…</think>` for assistant turns after the last user message).
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

    pub fn to_value(&self) -> jinja::Value {
        let mut m = vec![("role".to_string(), jinja::Value::str(&self.role)), ("content".to_string(), jinja::Value::str(&self.content))];
        if let Some(r) = &self.reasoning_content {
            m.push(("reasoning_content".to_string(), jinja::Value::str(r)));
        }
        jinja::Value::map(m)
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChatOptions {
    /// Template `enable_thinking` (`None` = the template default).
    pub enable_thinking: Option<bool>,
    pub generate: GenerateOptions,
}

/// A parsed chat template plus its special-token globals (`bos_token`, `eos_token`, ...).
#[derive(Clone, Debug)]
pub struct ChatTemplate {
    template: Template,
    globals: Vec<(String, String)>,
}

impl ChatTemplate {
    /// `special_tokens`: object of template globals (string values).
    pub fn new(source: &str, special_tokens: &Value) -> Result<ChatTemplate> {
        let mut globals = Vec::new();
        if let Some(m) = special_tokens.as_object() {
            for (k, v) in m {
                if let Some(s) = v.as_str() {
                    globals.push((k.clone(), s.to_string()));
                }
            }
        }
        Ok(ChatTemplate { template: Template::parse(source)?, globals })
    }

    /// Render typed messages.
    pub fn render(&self, messages: &[Message], add_generation_prompt: bool, enable_thinking: Option<bool>) -> Result<String> {
        let msgs = jinja::Value::List(messages.iter().map(Message::to_value).collect());
        self.render_values(msgs, jinja::Value::None, add_generation_prompt, enable_thinking, &[])
    }

    /// Render HF-style messages / tools given as JSON (any fields, key order kept).
    pub fn render_json(&self, messages_json: &str, tools_json: Option<&str>, add_generation_prompt: bool, enable_thinking: Option<bool>) -> Result<String> {
        let msgs = jinja::Value::from_json_str(messages_json)?;
        let tools = match tools_json {
            Some(t) => jinja::Value::from_json_str(t)?,
            None => jinja::Value::None,
        };
        self.render_values(msgs, tools, add_generation_prompt, enable_thinking, &[])
    }

    /// Render with explicit values and extra globals.
    pub fn render_values(
        &self,
        messages: jinja::Value,
        tools: jinja::Value,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
        extra: &[(&str, jinja::Value)],
    ) -> Result<String> {
        let mut globals: Vec<(&str, jinja::Value)> =
            vec![("messages", messages), ("tools", tools), ("add_generation_prompt", jinja::Value::Bool(add_generation_prompt))];
        for (k, v) in &self.globals {
            globals.push((k.as_str(), jinja::Value::str(v)));
        }
        if let Some(t) = enable_thinking {
            globals.push(("enable_thinking", jinja::Value::Bool(t)));
        }
        globals.extend(extra.iter().cloned());
        self.template.render(&globals)
    }
}

/// Tokenizer, chat template, generation defaults and stop tokens of a converted model.
pub struct TextAssets {
    pub tokenizer: Tokenizer,
    pub template: ChatTemplate,
    /// `generation_config.json`-style defaults.
    pub generation: Value,
    /// eos ids + end-of-turn tokens.
    pub stop: Vec<u32>,
}

fn meta_err(arch: &str, m: impl Into<String>) -> Error {
    Error::new("metadata", format!("{arch}: {}", m.into()))
}

/// Ids of `eos_token_id` given as int or list.
pub fn id_list(v: Option<&Value>) -> Vec<u32> {
    let one = |x: &Value| x.as_u64().and_then(|n| u32::try_from(n).ok());
    match v {
        Some(Value::Array(a)) => a.iter().filter_map(one).collect(),
        Some(x) => one(x).into_iter().collect(),
        None => Vec::new(),
    }
}

impl TextAssets {
    /// From the blobs and keys of metadata object `meta` (`tokenizer_blob`,
    /// `chat_template_blob`, `special_tokens`, `generation`, `eos_token_id`); `end_of_turn`
    /// names extra stop tokens (e.g. `<|im_end|>`).
    pub fn from_file(file: &LapidFile, arch: &str, meta: &Value, end_of_turn: &[&str]) -> Result<TextAssets> {
        let blob = |key: &str, default: &str| -> Result<String> {
            let name = meta.get(key).and_then(Value::as_str).unwrap_or(default);
            let bytes = file.blob(name).ok_or_else(|| meta_err(arch, format!("blob {name:?} missing")))?;
            String::from_utf8(bytes.to_vec()).map_err(|_| meta_err(arch, format!("blob {name:?} is not UTF-8")))
        };
        let tokenizer = Tokenizer::from_tokenizer_json(&blob("tokenizer_blob", TOKENIZER_BLOB)?)?;
        let template = ChatTemplate::new(&blob("chat_template_blob", TEMPLATE_BLOB)?, meta.get("special_tokens").unwrap_or(&Value::Null))?;
        let mut stop = id_list(meta.get("eos_token_id"));
        for name in end_of_turn {
            if let Some(id) = tokenizer.token_to_id(name)
                && !stop.contains(&id)
            {
                stop.push(id);
            }
        }
        Ok(TextAssets { tokenizer, template, generation: meta.get("generation").cloned().unwrap_or(Value::Null), stop })
    }

    /// Sampling recommended by the model's generation defaults (greedy unless `do_sample`).
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
}
