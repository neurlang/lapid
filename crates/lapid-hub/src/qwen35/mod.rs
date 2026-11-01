//! Qwen3.5 (hybrid Gated DeltaNet + gated attention LLM; dense `qwen3_5` and MoE
//! `qwen3_5_moe`): only the math, config, conversion tables and verification descriptor;
//! loading, sessions, chat, streaming, verify / bench and serving are `crate::runtime`.
//!
//! ```no_run
//! use lapid_hub::qwen35::{ChatOptions, Message, Qwen35};
//! let device = lapid::Device::best();
//! let llm = Qwen35::from_hub("Qwen/Qwen3.5-0.8B", "int8", &device)?;
//! let mut session = llm.new_session();
//! for token in llm.chat_stream(&mut session, &[Message::user("Hi!")], &ChatOptions::default())? {
//!     print!("{}", token?.text);
//! }
//! # Ok::<(), lapid::Error>(())
//! ```

pub mod config;
pub mod convert;
pub mod model;

use lapid_convert::engine::ConvertSpec;
use lapid_verify::Descriptor;

pub use crate::runtime::{CancelToken, ChatOptions, FinishReason, GenerateOptions, GenerationStats, Message, Token, TokenStream};
pub use config::{LayerType, MoeConfig, Qwen35Config};
pub use lapid_nn::Precision;
pub use model::Qwen35Model;
use crate::runtime::{LmCache, LmDef, Llm, ModelEntry};

/// A loaded Qwen3.5 model.
pub type Qwen35 = Llm<Qwen35Model>;
pub type Session = crate::runtime::Session<LmCache>;

impl LmDef for Qwen35Model {
    const DESCRIPTOR: Descriptor = Descriptor {
        name: "qwen35",
        default_model: "Qwen/Qwen3.5-0.8B",
        default_dtype: "int8",
        reference_dir: "data/reference/qwen35",
        reference_script: "tools/reference/qwen35_ref.py",
        reference_label: "PyTorch fp32 (HF Qwen3_5 text decoder)",
        strict_dtypes: &["fp32", "fp16"],
        bench_prompt_tokens: 512,
        bench_gen_tokens: 128,
    };
    const CONVERT: &'static ConvertSpec = &convert::CONVERT;
}

pub static ENTRY: ModelEntry = ModelEntry::causal_lm::<Qwen35Model>();

impl Qwen35 {
    pub fn config(&self) -> &Qwen35Config {
        &self.model().cfg
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::cmd::ModelArgs;

    /// The tiny random MoE reference (`tools/reference/qwen35_ref.py`) on the CPU, when present.
    #[test]
    fn tiny_moe_reference() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/reference/qwen35/tiny_moe");
        if !dir.join("result.json").is_file() {
            return;
        }
        let args = ModelArgs { model: None, dtype: Some("fp32".into()), device: Some("cpu".into()) };
        let rep = crate::runtime::lm_verify::verify::<Qwen35Model>(&Qwen35Model::DESCRIPTOR, Some(&dir), &args).expect("verify");
        assert_eq!(rep["ok"], serde_json::json!(true), "{}", serde_json::to_string_pretty(&rep).unwrap_or_default());
    }
}
