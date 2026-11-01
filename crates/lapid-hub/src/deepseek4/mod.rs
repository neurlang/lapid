//! DeepSeek V4 (`deepseek_v4`; llama.cpp `deepseek4`): hyper-connections (4 residual streams
//! mixed through Sinkhorn-normalized matrices), shared-KV MQA with head dim 512 and attention
//! sinks, sliding-window + compressed attention (CSA: overlapping 4-token compressor with a
//! lightning-indexer top-k; HCA: 128-token compressor), YaRN RoPE on compressed layers,
//! grouped low-rank output projection, hash-routed and sqrt-softplus-routed MoE with a shared
//! expert. The MTP head is not used. Prompts follow the upstream `encoding_dsv4.py`
//! ([`encoding`]; V4 has no Jinja template). Only the model's math, config, conversion tables
//! and verification descriptor live here; the rest is `crate::runtime`.
//!
//! ```no_run
//! use lapid_hub::deepseek4::{ChatOptions, Deepseek4, Message};
//! let device = lapid::Device::best();
//! let llm = Deepseek4::load("deepseek4.lapid".as_ref(), &device)?;
//! let mut session = llm.new_session();
//! for token in llm.chat_stream(&mut session, &[Message::user("Hi!")], &ChatOptions::default())? {
//!     print!("{}", token?.text);
//! }
//! # Ok::<(), lapid::Error>(())
//! ```

pub mod config;
pub mod convert;
pub mod encoding;
pub mod model;

use lapid_convert::engine::ConvertSpec;
use lapid_verify::Descriptor;

pub use config::{Deepseek4Config, Yarn};
pub use model::Deepseek4Model;
pub use crate::runtime::{CancelToken, ChatOptions, FinishReason, GenerateOptions, GenerationStats, Message, Token, TokenStream};
use crate::runtime::{LmDef, Llm, ModelEntry};

/// A loaded DeepSeek V4 model (the generic causal-LM pipeline over [`Deepseek4Model`]).
pub type Deepseek4 = Llm<Deepseek4Model>;
pub type Session = crate::runtime::Session<model::State>;

impl LmDef for Deepseek4Model {
    /// The real checkpoints do not fit this machine: the reference is a random-weight model
    /// with the V4-Flash structure (`deepseek4_ref.py --tiny`).
    const DESCRIPTOR: Descriptor = Descriptor {
        name: "deepseek4",
        default_model: "data/reference/deepseek4/tiny/model",
        default_dtype: "fp32",
        reference_dir: "data/reference/deepseek4/tiny",
        reference_script: "tools/reference/deepseek4_ref.py",
        reference_label: "PyTorch fp32 (HF DeepseekV4ForCausalLM)",
        strict_dtypes: &["fp32", "fp16"],
        bench_prompt_tokens: 512,
        bench_gen_tokens: 128,
    };
    const CONVERT: &'static ConvertSpec = &convert::CONVERT;
}

pub static ENTRY: ModelEntry = ModelEntry::causal_lm::<Deepseek4Model>();

impl Deepseek4 {
    pub fn config(&self) -> &Deepseek4Config {
        &self.model().cfg
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    /// The tiny reference (`deepseek4_ref.py --tiny`), when present: convert (fp32) and verify on the CPU.
    #[test]
    fn tiny_reference_cpu() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/reference/deepseek4/tiny");
        if !root.join("result.json").is_file() || !root.join("model/config.json").is_file() {
            eprintln!("skipped: {} not found", root.display());
            return;
        }
        let out = std::env::temp_dir().join(format!("lapid-deepseek4-tiny-{}.lapid", std::process::id()));
        let model = root.join("model").display().to_string();
        convert::CONVERT.convert(&model, lapid_convert::OutDType::F32, &out).expect("convert");
        let args = crate::cmd::ModelArgs { model: Some(out.display().to_string()), dtype: Some("fp32".into()), device: Some("cpu".into()) };
        let report = (ENTRY.verify)(Some(&root), &args).expect("verify");
        let _ = std::fs::remove_file(&out);
        assert_eq!(report["ok"], serde_json::json!(true), "{}", serde_json::to_string_pretty(&report).unwrap_or_default());
    }
}
