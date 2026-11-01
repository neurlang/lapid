//! Plain-transformer LLM families in one implementation, selected by config flags: Llama
//! (2 / 3.x incl. llama3 RoPE scaling, tied or untied embeddings; Mistral-style configs),
//! Qwen2 / Qwen2.5 (q/k/v bias), Qwen3 dense (q/k RMSNorm) and Qwen3-MoE (sparse experts).
//! Only the math, config, conversion tables and verification descriptor live here; loading,
//! sessions, chat, streaming, verify / bench and serving are `crate::runtime`.
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
//!
//! Llama 3.x templates print "Today Date"; pin it with
//! `ChatOptions { template_vars: vec![("date_string".into(), "26 Jul 2024".into())], .. }`
//! (default: the template's `strftime_now`, today's UTC date).

pub mod config;
pub mod convert;
pub mod model;

use lapid_convert::engine::ConvertSpec;
use lapid_verify::Descriptor;

pub use crate::runtime::{CancelToken, ChatOptions, FinishReason, GenerateOptions, GenerationStats, Message, Token, TokenStream};
pub use config::{LlamaConfig, MoeConfig, RopeScaling};
pub use lapid_nn::Precision;
pub use model::LlamaModel;
use crate::runtime::{LmCache, LmDef, Llm, ModelEntry};

/// A loaded Llama / Mistral / Qwen2 / Qwen3 / Qwen3-MoE model.
pub type Llama = Llm<LlamaModel>;
pub type Session = crate::runtime::Session<LmCache>;

impl LmDef for LlamaModel {
    const DESCRIPTOR: Descriptor = Descriptor {
        name: "llama",
        default_model: "Qwen/Qwen3-0.6B",
        default_dtype: "int8",
        reference_dir: "data/reference/llama/qwen3",
        reference_script: "tools/reference/llama_ref.py",
        reference_label: "PyTorch fp32 (HF transformers)",
        strict_dtypes: &["fp32", "fp16"],
        bench_prompt_tokens: 512,
        bench_gen_tokens: 128,
    };
    const CONVERT: &'static ConvertSpec = &convert::CONVERT;
}

pub static ENTRY: ModelEntry = ModelEntry::causal_lm::<LlamaModel>();

impl Llama {
    pub fn config(&self) -> &LlamaConfig {
        &self.model().cfg
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::cmd::ModelArgs;

    /// Every `data/reference/llama/<family>/tiny` (`tools/reference/llama_ref.py --tiny`,
    /// skipped when absent): HF checkpoint -> fp32 `.lapid` -> full verification on the CPU.
    #[test]
    fn tiny_references() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/reference/llama");
        let Ok(dirs) = std::fs::read_dir(&root) else { return };
        for d in dirs.flatten() {
            let tiny = d.path().join("tiny");
            if !tiny.join("result.json").is_file() {
                continue;
            }
            let args = ModelArgs { model: None, dtype: Some("fp32".into()), device: Some("cpu".into()) };
            let rep = crate::runtime::lm_verify::verify::<LlamaModel>(&LlamaModel::DESCRIPTOR, Some(&tiny), &args).expect("verify");
            assert_eq!(rep["ok"], serde_json::json!(true), "{}: {}", tiny.display(), serde_json::to_string_pretty(&rep).unwrap_or_default());
        }
    }
}
