//! Qwen3-VL (image + text → text; e.g. `Qwen/Qwen3-VL-2B-Instruct`): the generic vision
//! encoder ([`crate::vision`]) with Qwen's merger and DeepStack features, a Qwen3 text decoder
//! with M-RoPE, streaming chat with images, conversion from Hugging Face safetensors and
//! from a GGUF pair (model + `mmproj`), and the verification / benchmark wiring.
//!
//! ```no_run
//! use lapid_hub::qwen3vl::{ChatOptions, Qwen3Vl};
//! use lapid_hub::vision::{Image, Message};
//! let device = lapid::Device::best();
//! let vlm = Qwen3Vl::from_hub("Qwen/Qwen3-VL-2B-Instruct", "int8", &device)?;
//! let img = Image::open("photo.jpg".as_ref())?;
//! let reply = vlm.chat(&[Message::user_with_images(vec![img], "What is in this picture?")], &ChatOptions::default())?;
//! println!("{reply}");
//! # Ok::<(), lapid::Error>(())
//! ```

pub mod bench;
pub mod cmd;
pub mod config;
pub mod convert;
pub(crate) mod model;
pub(crate) mod prompt;
pub mod verify;

use std::path::Path;

use lapid::{Device, Error, LapidFile, Result};
use lapid_nn::{Loader, SamplingParams, Tokenizer};
use serde_json::Value;

use crate::lfm3::jinja::{self, Template};
use crate::model::{Bytes, Model, ModelInfo};
use crate::vision::{Image, Message, Patches, VisionEncoder, VisionOutput};
pub use crate::lfm3::{CancelToken, FinishReason, GenerateOptions, GenerationStats, Token, TokenStream};
pub use config::{Qwen3VlConfig, TextConfig};
pub use lapid_nn::Precision;
use model::{ImageSpan, Pos3, State, TextModel};
pub use prompt::{Prepared, PreparedImage};

#[derive(Clone, Debug, Default)]
pub struct ChatOptions {
    pub generate: GenerateOptions,
    /// Override the processor's pixel budget per image (`max_pixels`), e.g. to bound memory.
    pub max_pixels: Option<usize>,
}

/// A sequence being generated: KV cache + M-RoPE cursor, the tokens and positions consumed,
/// and a fingerprint of every image already inside the cache.
pub struct Session {
    state: State,
    tokens: Vec<u32>,
    images: Vec<(usize, u64)>,
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
        self.images.clear();
    }

    /// Keep the cache when the consumed tokens are a strict prefix of the prompt and every
    /// cached image is the same picture; else start over. Returns the reused tokens.
    fn reuse_prefix(&mut self, p: &Prepared) -> usize {
        let n = self.tokens.len();
        let images_match = self.images.iter().all(|(s, h)| p.images.iter().any(|im| im.start == *s && im.hash == *h));
        let cut_image = p.images.iter().any(|im| im.start < n && im.start + im.tokens > n);
        if n > 0 && n < p.ids.len() && p.ids[..n] == self.tokens[..] && images_match && !cut_image {
            return n;
        }
        self.reset();
        0
    }
}

pub(crate) const TOKENIZER_BLOB: &str = "tokenizer/tokenizer.json";
pub(crate) const TEMPLATE_BLOB: &str = "tokenizer/chat_template.jinja";
pub(crate) const META: &str = "qwen3vl";

fn meta_err(m: impl Into<String>) -> Error {
    Error::new("metadata", format!("qwen3vl: {}", m.into()))
}

/// A loaded Qwen3-VL model bound to a device.
pub struct Qwen3Vl {
    device: Device,
    info: ModelInfo,
    cfg: Qwen3VlConfig,
    vision: VisionEncoder,
    model: TextModel,
    tokenizer: Tokenizer,
    template: Template,
    special_tokens: Vec<(String, String)>,
    generation: Value,
    stop: Vec<u32>,
}

impl Qwen3Vl {
    /// Load `repo` (e.g. "Qwen/Qwen3-VL-2B-Instruct") from the lapid cache, converting it first if missing.
    pub fn from_hub(repo: &str, dtype: &str, device: &Device) -> Result<Qwen3Vl> {
        Qwen3Vl::load(&crate::model::hub_model_path(repo, dtype, convert::convert_hf)?, device)
    }

    pub fn load(path: &Path, device: &Device) -> Result<Qwen3Vl> {
        Qwen3Vl::load_with(path, device, Precision::Stored)
    }

    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Qwen3Vl> {
        Qwen3Vl::from_file(LapidFile::from_bytes(bytes)?, device, Precision::Stored)
    }

    /// Load with a weight precision (the fake-quant modes build verification references).
    pub fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Qwen3Vl> {
        Qwen3Vl::from_file(LapidFile::open(path)?, device, precision)
    }

    fn from_file(file: LapidFile, device: &Device, precision: Precision) -> Result<Qwen3Vl> {
        let meta = file.metadata.get(META).ok_or_else(|| meta_err("no \"qwen3vl\" metadata object"))?;
        let cfg = Qwen3VlConfig::from_json(meta, None)?;
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
        let generation = meta.get("generation").cloned().unwrap_or(Value::Null);
        let mut stop = cfg.eos.clone();
        let gen_eos: Vec<u32> = match generation.get("eos_token_id") {
            Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect(),
            Some(v) => v.as_u64().map(|x| x as u32).into_iter().collect(),
            None => Vec::new(),
        };
        let named = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|t| tokenizer.token_to_id(t)).collect::<Vec<_>>();
        for id in gen_eos.into_iter().chain(named) {
            if !stop.contains(&id) {
                stop.push(id);
            }
        }
        let loader = Loader::new(&file, device, precision);
        let vision = VisionEncoder::load(&loader, &cfg.vision)?;
        let model = TextModel::load(&loader, &cfg.text)?;
        Ok(Qwen3Vl {
            device: device.clone(),
            info: ModelInfo::from_file(&file, "image-text-to-text"),
            cfg,
            vision,
            model,
            tokenizer,
            template,
            special_tokens,
            generation,
            stop,
        })
    }

    pub fn config(&self) -> &Qwen3VlConfig {
        &self.cfg
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn stop_tokens(&self) -> &[u32] {
        &self.stop
    }

    /// Sampling from `generation_config.json` (temperature 0.7, top_p 0.8, top_k 20 for the
    /// Instruct models).
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

    /// Render the chat template (images become `<|vision_start|><|image_pad|><|vision_end|>`).
    pub fn render_chat(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
        let msgs = jinja::Value::List(messages.iter().map(Message::to_value).collect());
        self.render_values(msgs, jinja::Value::None, add_generation_prompt)
    }

    /// Render HF-style messages / tools given as JSON (key order kept).
    pub fn render_chat_json(&self, messages_json: &str, tools_json: Option<&str>, add_generation_prompt: bool) -> Result<String> {
        let tools = match tools_json {
            Some(t) => jinja::Value::from_json_str(t)?,
            None => jinja::Value::None,
        };
        self.render_values(jinja::Value::from_json_str(messages_json)?, tools, add_generation_prompt)
    }

    fn render_values(&self, messages: jinja::Value, tools: jinja::Value, add_generation_prompt: bool) -> Result<String> {
        let mut globals: Vec<(&str, jinja::Value)> =
            vec![("messages", messages), ("tools", tools), ("add_generation_prompt", jinja::Value::Bool(add_generation_prompt))];
        for (k, v) in &self.special_tokens {
            globals.push((k.as_str(), jinja::Value::str(v)));
        }
        self.template.render(&globals)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text, false)
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        self.tokenizer.decode(ids, skip_special)
    }

    /// Preprocess an image into encoder patches (resize / normalize / patchify).
    pub fn preprocess(&self, img: &Image, max_pixels: Option<usize>) -> Result<Patches> {
        let mut p = self.cfg.processor.clone();
        if let Some(m) = max_pixels {
            p.max_pixels = m.max(p.min_pixels);
        }
        p.process(img)
    }

    /// Run the vision encoder on preprocessed patches (realized merged + DeepStack features).
    pub fn encode_patches(&self, patches: &Patches) -> Result<VisionOutput> {
        self.vision.encode(&self.device, patches)
    }

    /// Token ids, M-RoPE positions and image patches for `messages` (template with the
    /// generation prompt, image placeholders expanded).
    pub fn prepare(&self, messages: &[Message], max_pixels: Option<usize>) -> Result<Prepared> {
        let text = self.render_chat(messages, true)?;
        let ids = self.encode(&text)?;
        let patches = crate::vision::collect_images(messages)
            .into_iter()
            .map(|img| Ok((self.preprocess(img, max_pixels)?, prompt::image_hash(img))))
            .collect::<Result<Vec<_>>>()?;
        Prepared::new(&self.cfg, &ids, patches)
    }

    pub fn new_session(&self) -> Session {
        Session { state: self.model.new_state(), tokens: Vec::new(), images: Vec::new() }
    }

    /// Stream tokens after a prepared prompt, reusing the session's cached prefix. The images
    /// not yet in the cache are encoded first.
    pub fn generate_prepared<'a>(&'a self, session: &'a mut Session, p: Prepared, opts: GenerateOptions) -> Result<TokenStream<'a>> {
        let cached = session.reuse_prefix(&p);
        let mut outs: Vec<(usize, VisionOutput)> = Vec::new();
        for im in p.images.iter().filter(|im| im.start + im.tokens > cached) {
            outs.push((im.start, self.encode_patches(&im.patches)?));
            session.images.push((im.start, im.hash));
        }
        let (model, device) = (&self.model, &self.device);
        let (prompt_ids, positions) = (p.ids.clone(), p.positions);
        let step = move |ids: &[u32]| -> Result<Vec<f32>> {
            let at = session.tokens.len();
            let pos: Vec<Pos3> = if at + ids.len() <= positions.len() {
                positions[at..at + ids.len()].to_vec()
            } else {
                let p0 = session.state.next_pos();
                (0..ids.len()).map(|i| [p0 + i; 3]).collect()
            };
            let spans: Vec<ImageSpan<'_>> = outs
                .iter()
                .filter(|(s, o)| *s + o.merged.shape()[0] > at && *s < at + ids.len())
                .filter(|(s, _)| *s >= at)
                .map(|(s, o)| ImageSpan { start: s - at, out: o })
                .collect();
            let logits = model.step(device, ids, &pos, &spans, &mut session.state)?;
            session.tokens.extend_from_slice(ids);
            Ok(logits)
        };
        TokenStream::new(step, &self.tokenizer, &prompt_ids, cached, opts, &self.stop)
    }

    /// Stream tokens after a text-only prompt of token ids.
    pub fn generate<'a>(&'a self, session: &'a mut Session, prompt: &[u32], opts: GenerateOptions) -> Result<TokenStream<'a>> {
        let p = Prepared::new(&self.cfg, prompt, Vec::new())?;
        self.generate_prepared(session, p, opts)
    }

    /// Stream the assistant reply to `messages` (text and images).
    pub fn chat_stream<'a>(&'a self, session: &'a mut Session, messages: &[Message], opts: &ChatOptions) -> Result<TokenStream<'a>> {
        let p = self.prepare(messages, opts.max_pixels)?;
        self.generate_prepared(session, p, opts.generate.clone())
    }

    /// The full assistant reply (non-streaming convenience).
    pub fn chat(&self, messages: &[Message], opts: &ChatOptions) -> Result<String> {
        let mut session = self.new_session();
        let (text, _, _) = self.chat_stream(&mut session, messages, opts)?.collect_text()?;
        Ok(text)
    }

    /// Logits of the last position after feeding text `ids` to `session` (low-level step).
    pub fn step(&self, session: &mut Session, ids: &[u32]) -> Result<Vec<f32>> {
        let p0 = session.state.next_pos();
        let pos: Vec<Pos3> = (0..ids.len()).map(|i| [p0 + i; 3]).collect();
        let logits = self.model.step(&self.device, ids, &pos, &[], &mut session.state)?;
        session.tokens.extend_from_slice(ids);
        Ok(logits)
    }

    pub(crate) fn text_model(&self) -> &TextModel {
        &self.model
    }

    pub(crate) fn vision(&self) -> &VisionEncoder {
        &self.vision
    }
}

impl Model for Qwen3Vl {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests;
