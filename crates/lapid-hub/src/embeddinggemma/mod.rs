//! Google EmbeddingGemma (`google/embeddinggemma-300m`) sentence embeddings: a Gemma 3
//! bidirectional text encoder and two Dense layers; mean pooling, L2 normalization, task
//! prompts, Matryoshka truncation, batching, verification and serving are the shared
//! embedding pipeline (`runtime::embedding`).
//!
//! ```no_run
//! use lapid_hub::embeddinggemma::{EmbedOptions, EmbeddingGemma, cosine_similarity};
//! let device = lapid::Device::best();
//! let model = EmbeddingGemma::from_hub("google/embeddinggemma-300m", "fp16", &device)?;
//! let q = model.embed(&["Which planet is known as the Red Planet?"], &EmbedOptions::query())?;
//! let d = model.embed(&["Mars is often called the Red Planet."], &EmbedOptions::document())?;
//! println!("{:.3}", cosine_similarity(&q[0], &d[0]));
//! # Ok::<(), lapid::Error>(())
//! ```

pub mod config;
pub mod convert;
mod model;

use std::path::Path;

use lapid::{Device, Error, LapidFile, Result, Tensor};
use lapid_convert::engine::ConvertSpec;
use lapid_nn::{Loader, Tokenizer};
use lapid_verify::Descriptor;

pub use config::EmbeddingGemmaConfig;
pub use lapid_nn::Precision;
pub use crate::runtime::embedding::{EmbedOptions, Pooling, TextEmbedder, cosine_similarity, similarity_matrix};
use crate::model::{Bytes, Model, ModelInfo, hub_model_path};
use crate::runtime::embedding::{EmbedderDef, TokenBatch, embed};
use crate::runtime::{ModelEntry, Taps};

/// Default Hugging Face repo.
pub const DEFAULT_REPO: &str = "google/embeddinggemma-300m";

/// A loaded EmbeddingGemma model bound to a device.
pub struct EmbeddingGemma {
    device: Device,
    info: ModelInfo,
    cfg: EmbeddingGemmaConfig,
    tokenizer: Tokenizer,
    model: model::Model,
}

impl EmbedderDef for EmbeddingGemma {
    const DESCRIPTOR: Descriptor = Descriptor {
        name: "embeddinggemma",
        default_model: DEFAULT_REPO,
        default_dtype: "fp16",
        reference_dir: "data/reference/embeddinggemma",
        reference_script: "tools/reference/embeddinggemma_ref.py",
        reference_label: "PyTorch fp32 (sentence-transformers)",
        strict_dtypes: &["fp32", "fp16"],
        bench_prompt_tokens: 0,
        bench_gen_tokens: 0,
    };
    const CONVERT: &'static ConvertSpec = &convert::CONVERT;

    fn from_file(file: &LapidFile, device: &Device, precision: Precision) -> Result<Self> {
        let cfg = EmbeddingGemmaConfig::from_metadata(&file.metadata)?;
        let blob = file
            .blob(&cfg.tokenizer_blob)
            .ok_or_else(|| Error::new("missing_metadata", format!("tokenizer blob {:?} missing", cfg.tokenizer_blob)))?;
        let tokenizer = Tokenizer::from_bytes(blob)?;
        let model = model::Model::load(&Loader::new(file, device, precision), &cfg)?;
        Ok(EmbeddingGemma { device: device.clone(), info: ModelInfo::from_file(file, "feature-extraction"), cfg, tokenizer, model })
    }
}

pub static ENTRY: ModelEntry = ModelEntry::embedder::<EmbeddingGemma>();

impl TextEmbedder for EmbeddingGemma {
    fn device(&self) -> &Device {
        &self.device
    }
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text, true)
    }
    fn pad_id(&self) -> u32 {
        self.cfg.pad_token_id
    }
    fn max_seq_len(&self) -> usize {
        self.cfg.max_seq_length
    }
    fn dim(&self) -> usize {
        self.cfg.output_dim()
    }
    fn pooling(&self) -> Pooling {
        Pooling::Mean
    }
    fn normalize(&self) -> bool {
        self.cfg.normalize
    }
    fn prompt(&self, name: &str) -> Option<String> {
        self.cfg.prompts.get(name).cloned()
    }
    fn default_prompt_name(&self) -> Option<String> {
        self.cfg.default_prompt_name.clone()
    }
    fn prompt_names(&self) -> Vec<String> {
        self.cfg.prompts.keys().cloned().collect()
    }
    fn matryoshka_dims(&self) -> Vec<usize> {
        self.cfg.matryoshka_dims.clone()
    }
    fn encode(&self, batch: &TokenBatch, taps: &mut Taps) -> Result<Tensor> {
        self.model.encode(batch, taps)
    }
    fn project(&self, pooled: Tensor, taps: &mut Taps) -> Result<Tensor> {
        self.model.project(pooled, taps)
    }
}

impl EmbeddingGemma {
    /// Load `repo` from the lapid cache, converting it from the Hugging Face hub first if
    /// it is missing (`dtype`: "fp32", "fp16", "int8", "int4").
    pub fn from_hub(repo: &str, dtype: &str, device: &Device) -> Result<EmbeddingGemma> {
        EmbeddingGemma::load(&hub_model_path(repo, dtype, |r, d, out| convert::CONVERT.convert(r, d, out))?, device)
    }

    /// Load a converted `.lapid` file.
    pub fn load(path: &Path, device: &Device) -> Result<EmbeddingGemma> {
        crate::runtime::embedding::load(path, device, Precision::Stored)
    }

    /// Load from an in-memory `.lapid` image (zero copy).
    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<EmbeddingGemma> {
        EmbeddingGemma::from_file(&LapidFile::from_bytes(bytes)?, device, Precision::Stored)
    }

    pub fn config(&self) -> &EmbeddingGemmaConfig {
        &self.cfg
    }

    /// Embed `texts`: one L2-normalized vector per text, in input order.
    pub fn embed(&self, texts: &[&str], opts: &EmbedOptions) -> Result<Vec<Vec<f32>>> {
        embed(self, texts, opts)
    }
}

impl Model for EmbeddingGemma {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
