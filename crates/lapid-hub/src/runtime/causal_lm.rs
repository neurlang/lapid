//! The causal LM pipeline. A model implements [`CausalLm`]: load its weights, make a state,
//! and build the lazy graph of one forward over new tokens (staging state, emitting named
//! taps for verification). [`Llm`] provides everything else: `.lapid` / hub loading, chunked
//! prefill + decode steps, sessions with prefix reuse, chat templates, sampling, stop tokens
//! and strings, token streaming, cancellation and timing, traced forwards for `lapid verify`,
//! lazy graphs for calibrated quantization.
//!
//! ```ignore
//! impl CausalLm for MyModel {
//!     const ARCH: &'static str = "mymodel";
//!     type State = LmCache;
//!     fn load(l: &Loader<'_>, meta: &Value) -> Result<Self> { ... }
//!     fn dims(&self) -> LmDims { ... }
//!     fn new_state(&self) -> LmCache { LmCache::new(KvCache::new(n_layers), 0) }
//!     fn forward(&self, ids: &[u32], st: &mut LmCache, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>> { ... }
//! }
//! pub type MyLlm = Llm<MyModel>;
//! ```

use std::path::Path;

use lapid::{Device, Error, LapidFile, Result, Tensor};
use lapid_nn::{Loader, Precision, SamplingParams, Tokenizer};
use serde_json::Value;

use super::chat::{ChatOptions, ChatTemplate, Message, TextAssets};
use super::generate::{GenerateOptions, TokenStream};
use super::session::{LmState, Session};
use crate::model::{Bytes, Model, ModelInfo};

/// Which logit rows a forward computes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Rows {
    /// The last position (decoding, prompt end).
    Last,
    /// Every position (verification).
    All,
    /// None (prefill chunks before the last).
    None,
}

/// Named intermediates collected during a traced forward (`embed`, `layers.{i}`,
/// `layers.{i}.op`, `norm`, ...; the pipeline adds `logits`). Off during generation.
#[derive(Default)]
pub struct Taps {
    list: Option<Vec<(String, Tensor)>>,
}

impl Taps {
    pub fn off() -> Taps {
        Taps { list: None }
    }

    pub fn on() -> Taps {
        Taps { list: Some(Vec::new()) }
    }

    pub fn enabled(&self) -> bool {
        self.list.is_some()
    }

    /// Record `t` under `name` (no-op when off; pass `format_args!(..)` to format lazily).
    pub fn tap(&mut self, name: impl std::fmt::Display, t: &Tensor) {
        if let Some(l) = &mut self.list {
            l.push((name.to_string(), t.clone()));
        }
    }

    pub fn into_vec(self) -> Vec<(String, Tensor)> {
        self.list.unwrap_or_default()
    }
}

/// Sizes the pipeline and the harness need.
#[derive(Copy, Clone, Debug)]
pub struct LmDims {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
}

/// A decoder-only language model: only its own math.
pub trait CausalLm: Sized + Send + Sync + 'static {
    /// `.lapid` metadata `architecture` and config object key.
    const ARCH: &'static str;
    type State: LmState;

    /// Build from weights; `meta` is the metadata object `ARCH`.
    fn load(l: &Loader<'_>, meta: &Value) -> Result<Self>;
    fn dims(&self) -> LmDims;
    fn new_state(&self) -> Self::State;
    /// Lazy graph for `ids` at positions `state.offset()..`, staging new state; returns the
    /// f32 logits of `rows` ([n_rows, vocab]), `None` for `Rows::None`.
    fn forward(&self, ids: &[u32], state: &mut Self::State, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>>;

    /// Prompt tokens per evaluation during prefill (bounds attention scratch memory).
    fn prefill_chunk(&self) -> usize {
        512
    }
    /// Token strings that end a turn besides `eos_token_id` (e.g. `<|im_end|>`).
    fn end_of_turn(&self) -> &'static [&'static str] {
        &[]
    }
    /// Reasoning block markers (open, close) for splitting thinking from content.
    fn think_markers(&self) -> (&'static str, &'static str) {
        ("<think>", "</think>")
    }
}

/// Evaluate `outputs` with the staged state and commit it (`n` tokens fed).
fn eval_commit<S: LmState>(device: &Device, outputs: &[&Tensor], state: &mut S, n: usize) -> Result<Vec<Tensor>> {
    let mut outs: Vec<&Tensor> = outputs.to_vec();
    outs.extend(state.staged());
    let mut realized = match device.eval(&outs) {
        Ok(r) if r.len() == outs.len() => r,
        Ok(_) => {
            state.discard();
            return Err(Error::new("backend", "eval returned the wrong number of outputs"));
        }
        Err(e) => {
            state.discard();
            return Err(e);
        }
    };
    let st = realized.split_off(outputs.len());
    state.commit(st, n)?;
    Ok(realized)
}

/// Feed `ids` (a prompt, chunked, or one token) and return the f32 logits of the last position.
pub fn step<M: CausalLm>(model: &M, device: &Device, ids: &[u32], state: &mut M::State) -> Result<Vec<f32>> {
    if ids.is_empty() {
        return Err(Error::new("invalid_argument", "step with no tokens"));
    }
    let chunks: Vec<&[u32]> = ids.chunks(model.prefill_chunk().max(1)).collect();
    let last = chunks.len() - 1;
    for (ci, chunk) in chunks.into_iter().enumerate() {
        let rows = if ci == last { Rows::Last } else { Rows::None };
        let out = match model.forward(chunk, state, rows, &mut Taps::off()) {
            Ok(o) => o,
            Err(e) => {
                state.discard();
                return Err(e);
            }
        };
        let outputs: Vec<&Tensor> = out.iter().collect();
        let realized = eval_commit(device, &outputs, state, chunk.len())?;
        if ci == last {
            let logits = realized.first().ok_or_else(|| Error::new("backend", "no logits realized"))?;
            return device.to_vec_f32(logits);
        }
    }
    Err(Error::new("internal", "prefill produced no logits"))
}

/// One uncached forward over `ids` from a fresh state with every tap realized (plus `logits`).
pub fn forward_traced<M: CausalLm>(model: &M, device: &Device, ids: &[u32], rows: Rows) -> Result<Vec<(String, Vec<f32>)>> {
    let mut state = model.new_state();
    let mut taps = Taps::on();
    let logits = model.forward(ids, &mut state, rows, &mut taps)?;
    let mut list = taps.into_vec();
    if let Some(l) = logits {
        list.push(("logits".to_string(), l));
    }
    let refs: Vec<&Tensor> = list.iter().map(|(_, t)| t).collect();
    let realized = device.eval(&refs)?;
    list.iter().zip(&realized).map(|((n, _), t)| Ok((n.clone(), device.to_vec_f32(t)?))).collect()
}

/// Lazy last-position logits of one uncached forward (nothing evaluated; calibration).
pub fn forward_graph<M: CausalLm>(model: &M, ids: &[u32]) -> Result<Tensor> {
    let mut state = model.new_state();
    model.forward(ids, &mut state, Rows::Last, &mut Taps::off())?.ok_or_else(|| Error::new("internal", "forward graph has no logits"))
}

/// A loaded causal LM bound to a device, with its tokenizer and chat template.
pub struct Llm<M: CausalLm> {
    device: Device,
    info: ModelInfo,
    model: M,
    text: TextAssets,
}

impl<M: CausalLm> Llm<M> {
    /// Load `repo` from the lapid cache, converting it (hub converter routing) first if missing.
    pub fn from_hub(repo: &str, dtype: &str, device: &Device) -> Result<Llm<M>> {
        let convert = |r: &str, d: lapid_convert::OutDType, out: &Path| crate::convert::convert(r, d.name(), Some(out)).map(|_| ());
        Llm::load(&crate::model::hub_model_path(repo, dtype, convert)?, device)
    }

    /// Load a converted `.lapid` file.
    pub fn load(path: &Path, device: &Device) -> Result<Llm<M>> {
        Llm::load_with(path, device, Precision::Stored)
    }

    /// Load a `.lapid` image from memory (zero copy).
    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Llm<M>> {
        Llm::from_file(LapidFile::from_bytes(bytes)?, device, Precision::Stored)
    }

    /// Load with a weight precision (the fake-quant modes build verification references).
    pub fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Llm<M>> {
        Llm::from_file(LapidFile::open(path)?, device, precision)
    }

    pub fn from_file(file: LapidFile, device: &Device, precision: Precision) -> Result<Llm<M>> {
        let meta = file
            .metadata
            .get(M::ARCH)
            .ok_or_else(|| Error::new("metadata", format!("{}: no {:?} metadata object", M::ARCH, M::ARCH)))?;
        let model = M::load(&Loader::new(&file, device, precision), meta)?;
        let text = TextAssets::from_file(&file, M::ARCH, meta, model.end_of_turn())?;
        Ok(Llm { device: device.clone(), info: ModelInfo::from_file(&file, "text-generation"), model, text })
    }

    /// The model's own math (config, ...).
    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.text.tokenizer
    }

    pub fn template(&self) -> &ChatTemplate {
        &self.text.template
    }

    /// Default stop tokens (eos ids + end-of-turn tokens).
    pub fn stop_tokens(&self) -> &[u32] {
        &self.text.stop
    }

    /// Sampling recommended by the model's generation defaults.
    pub fn recommended_sampling(&self) -> SamplingParams {
        self.text.recommended_sampling()
    }

    pub fn think_markers(&self) -> (&'static str, &'static str) {
        self.model.think_markers()
    }

    /// Render the chat template for typed messages.
    pub fn render_chat(&self, messages: &[Message], add_generation_prompt: bool, enable_thinking: Option<bool>) -> Result<String> {
        self.text.template.render(messages, add_generation_prompt, enable_thinking)
    }

    /// Render the chat template for HF-style messages / tools given as JSON.
    pub fn render_chat_json(&self, messages_json: &str, tools_json: Option<&str>, add_generation_prompt: bool, enable_thinking: Option<bool>) -> Result<String> {
        self.text.template.render_json(messages_json, tools_json, add_generation_prompt, enable_thinking)
    }

    /// Token ids of rendered chat text (special tokens in the text are recognized; nothing added).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.text.tokenizer.encode(text, false)
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        self.text.tokenizer.decode(ids, skip_special)
    }

    /// A fresh sequence.
    pub fn new_session(&self) -> Session<M::State> {
        Session::new(self.model.new_state())
    }

    /// Stream tokens after `prompt`; a cached prefix of `prompt` in `session` is not refed.
    pub fn generate<'a>(&'a self, session: &'a mut Session<M::State>, prompt: &[u32], opts: GenerateOptions) -> Result<TokenStream<'a>> {
        let cached = session.reuse_prefix(prompt);
        let (model, device) = (&self.model, &self.device);
        let step_fn = move |ids: &[u32]| -> Result<Vec<f32>> {
            let logits = step(model, device, ids, &mut session.state)?;
            session.tokens.extend_from_slice(ids);
            Ok(logits)
        };
        TokenStream::new(step_fn, &self.text.tokenizer, prompt, cached, opts, &self.text.stop)
    }

    /// Stream the assistant reply to `messages` (template with the generation prompt).
    pub fn chat_stream<'a>(&'a self, session: &'a mut Session<M::State>, messages: &[Message], opts: &ChatOptions) -> Result<TokenStream<'a>> {
        let ids = self.encode(&self.render_chat(messages, true, opts.enable_thinking)?)?;
        self.generate(session, &ids, opts.generate.clone())
    }

    /// The full assistant reply (non-streaming convenience).
    pub fn chat(&self, messages: &[Message], opts: &ChatOptions) -> Result<String> {
        let mut session = self.new_session();
        let (text, _, _) = self.chat_stream(&mut session, messages, opts)?.collect_text()?;
        Ok(text)
    }

    /// Logits of the last position after feeding `ids` to `session` (low-level step).
    pub fn step(&self, session: &mut Session<M::State>, ids: &[u32]) -> Result<Vec<f32>> {
        let logits = step(&self.model, &self.device, ids, &mut session.state)?;
        session.tokens.extend_from_slice(ids);
        Ok(logits)
    }

    /// Uncached forward over `ids` with every tap realized (verification).
    pub fn forward_traced(&self, ids: &[u32], rows: Rows) -> Result<Vec<(String, Vec<f32>)>> {
        forward_traced(&self.model, &self.device, ids, rows)
    }
}

impl<M: CausalLm> Model for Llm<M> {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
