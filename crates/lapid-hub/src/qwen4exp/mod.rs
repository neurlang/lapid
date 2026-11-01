//! Qwen4-Exp (`qwen4_exp`, e.g. Qwen/Qwen3.8-Flash-Next): the Qwen3.5 MoE hybrid decoder
//! (Gated DeltaNet + gated attention, sparse MoE; those blocks come from [`crate::qwen35`])
//! with hyper-connection residual streams, QSA sparse attention and PLE n-gram hash
//! embeddings. Only the math, config, conversion tables and verification descriptor;
//! loading, sessions, chat, streaming, verify / bench and serving are `crate::runtime`.
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

mod attn;
pub mod config;
pub mod convert;
mod hc;
pub mod model;
mod ple;

use lapid_convert::engine::ConvertSpec;
use lapid_verify::Descriptor;

pub use crate::runtime::{CancelToken, ChatOptions, FinishReason, GenerateOptions, GenerationStats, Message, Token, TokenStream};
pub use config::{GateAct, PleConfig, QsaConfig, Qwen4ExpConfig};
pub use lapid_nn::Precision;
pub use model::Qwen4ExpModel;
use crate::runtime::{LmDef, Llm, ModelEntry};

/// A loaded Qwen4-Exp model.
pub type Qwen4Exp = Llm<Qwen4ExpModel>;
pub type Session = crate::runtime::Session<model::State>;

impl LmDef for Qwen4ExpModel {
    const DESCRIPTOR: Descriptor = Descriptor {
        name: "qwen4exp",
        default_model: "data/reference/qwen4exp/tiny/qwen4exp-tiny",
        default_dtype: "fp32",
        reference_dir: "data/reference/qwen4exp/tiny",
        reference_script: "tools/reference/qwen4exp_ref.py",
        reference_label: "PyTorch fp32 (HF Qwen4Exp text decoder)",
        strict_dtypes: &["fp32", "fp16"],
        bench_prompt_tokens: 512,
        bench_gen_tokens: 128,
    };
    const CONVERT: &'static ConvertSpec = &convert::CONVERT;
}

pub static ENTRY: ModelEntry = ModelEntry::causal_lm::<Qwen4ExpModel>();

impl Qwen4Exp {
    pub fn config(&self) -> &Qwen4ExpConfig {
        &self.model().cfg
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::cmd::ModelArgs;

    /// The tiny reference model (`tools/reference/qwen4exp_ref.py`) on the CPU, when present.
    #[test]
    fn tiny_reference_cpu() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/reference/qwen4exp/tiny");
        if !dir.join("qwen4exp-tiny/config.json").is_file() {
            return;
        }
        let args = ModelArgs { model: None, dtype: Some("fp32".into()), device: Some("cpu".into()) };
        let rep = crate::runtime::lm_verify::verify::<Qwen4ExpModel>(&Qwen4ExpModel::DESCRIPTOR, Some(&dir), &args).expect("verify");
        assert_eq!(rep["ok"], serde_json::json!(true), "{}", serde_json::to_string_pretty(&rep["layers"]).unwrap_or_default());
    }
}
