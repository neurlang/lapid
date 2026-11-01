//! Qwen3-VL (image + text -> text, e.g. `Qwen/Qwen3-VL-2B-Instruct`): only the model's math,
//! config, conversion tables and verification descriptor; text and image chat, sessions,
//! verify and bench are `crate::runtime` (`Llm`, `runtime::vlm`).
//!
//! ```no_run
//! use lapid_hub::qwen3vl::{ChatOptions, Qwen3Vl};
//! use lapid_hub::vision::{Image, Message};
//! let device = lapid::Device::best();
//! let vlm = Qwen3Vl::from_hub("Qwen/Qwen3-VL-2B-Instruct", "int8", &device)?;
//! let img = Image::open("photo.jpg".as_ref())?;
//! let reply = vlm.mm_chat(&[Message::user_with_images(vec![img], "What is in this picture?")], &ChatOptions::default())?;
//! println!("{reply}");
//! # Ok::<(), lapid::Error>(())
//! ```

pub mod config;
pub mod convert;
pub mod model;

use std::path::Path;

use lapid::Result;
use lapid_convert::engine::ConvertSpec;
use lapid_verify::Descriptor;
use serde_json::Value;

pub use crate::runtime::{CancelToken, ChatOptions, FinishReason, GenerateOptions, GenerationStats, Token, TokenStream};
pub use config::{Qwen3VlConfig, TextConfig};
pub use model::Qwen3VlModel;
use crate::cmd::ModelArgs;
use crate::runtime::{LmDef, Llm, ModelEntry, vlm_verify};

/// A loaded Qwen3-VL model (text chat: `chat_stream`; with images: `mm_chat_stream`).
pub type Qwen3Vl = Llm<Qwen3VlModel>;
pub type Session = crate::runtime::Session<model::MropeCache>;

impl LmDef for Qwen3VlModel {
    const DESCRIPTOR: Descriptor = Descriptor {
        name: "qwen3vl",
        default_model: "Qwen/Qwen3-VL-2B-Instruct",
        default_dtype: "int8",
        reference_dir: "data/reference/qwen3vl",
        reference_script: "tools/reference/qwen3vl_ref.py",
        reference_label: "PyTorch fp32 (HF Qwen3VLForConditionalGeneration)",
        strict_dtypes: &["fp32", "fp16"],
        bench_prompt_tokens: 512,
        bench_gen_tokens: 128,
    };
    const CONVERT: &'static ConvertSpec = &convert::CONVERT;
}

fn verify(r: Option<&Path>, a: &ModelArgs) -> Result<Value> {
    vlm_verify::verify::<Qwen3VlModel>(&Qwen3VlModel::DESCRIPTOR, r, a)
}

fn bench(i: Option<&Path>, a: &ModelArgs, runs: usize) -> Result<Value> {
    vlm_verify::bench::<Qwen3VlModel>(&Qwen3VlModel::DESCRIPTOR, i, a, runs)
}

pub static ENTRY: ModelEntry = ModelEntry { verify, bench, ..ModelEntry::causal_lm::<Qwen3VlModel>() };

impl Qwen3Vl {
    pub fn config(&self) -> &Qwen3VlConfig {
        &self.model().cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tiny reference (`qwen3vl_ref.py --tiny`), when present: convert (fp32) and verify on the CPU.
    #[test]
    fn tiny_reference_cpu() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/reference/qwen3vl/tiny");
        if !root.join("result.json").is_file() || !root.join("model/config.json").is_file() {
            eprintln!("skipped: {} not found", root.display());
            return;
        }
        let out = std::env::temp_dir().join(format!("lapid-qwen3vl-tiny-{}.lapid", std::process::id()));
        super::convert::CONVERT.convert(&root.join("model").display().to_string(), lapid_convert::OutDType::F32, &out).expect("convert");
        let args = ModelArgs { model: Some(out.display().to_string()), dtype: Some("fp32".into()), device: Some("cpu".into()) };
        let report = (super::ENTRY.verify)(Some(&root), &args).expect("verify");
        let _ = std::fs::remove_file(&out);
        assert_eq!(report["ok"], serde_json::json!(true), "{report:#}");
    }
}
