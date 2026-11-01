//! Liquid AI LFM3 (hybrid short-conv + full / sliding attention LLM): only the model's math,
//! config, conversion tables and verification descriptor; the rest is `crate::runtime`.
//!
//! ```no_run
//! use lapid_hub::lfm3::{ChatOptions, Lfm3, Message};
//! let device = lapid::Device::best();
//! let llm = Lfm3::from_hub("LiquidAI/LFM3-1.7B", "int8", &device)?;
//! let mut session = llm.new_session();
//! for token in llm.chat_stream(&mut session, &[Message::user("Write a haiku about rust.")], &ChatOptions::default())? {
//!     print!("{}", token?.text);
//! }
//! # Ok::<(), lapid::Error>(())
//! ```

pub mod config;
pub mod convert;
pub mod model;

use lapid_convert::engine::ConvertSpec;
use lapid_verify::Descriptor;

pub use config::{LayerType, Lfm3Config};
pub use model::Lfm3Model;
pub use crate::runtime::{CancelToken, ChatOptions, FinishReason, GenerateOptions, GenerationStats, Message, Token, TokenStream, jinja};
use crate::runtime::{LmCache, LmDef, Llm, ModelEntry};

/// A loaded LFM3 model (the generic causal-LM pipeline over [`Lfm3Model`]).
pub type Lfm3 = Llm<Lfm3Model>;
pub type Session = crate::runtime::Session<LmCache>;

impl LmDef for Lfm3Model {
    const DESCRIPTOR: Descriptor = Descriptor {
        name: "lfm3",
        default_model: "LiquidAI/LFM3-1.7B",
        default_dtype: "int8",
        reference_dir: "data/reference/lfm3",
        reference_script: "tools/reference/lfm3_ref.py",
        reference_label: "PyTorch fp32 (HF Lfm3ForCausalLM, trust_remote_code)",
        strict_dtypes: &["fp32", "fp16"],
        bench_prompt_tokens: 512,
        bench_gen_tokens: 128,
    };
    const CONVERT: &'static ConvertSpec = &convert::CONVERT;
}

pub static ENTRY: ModelEntry = ModelEntry::causal_lm::<Lfm3Model>();

impl Lfm3 {
    pub fn config(&self) -> &Lfm3Config {
        &self.model().cfg
    }
}
