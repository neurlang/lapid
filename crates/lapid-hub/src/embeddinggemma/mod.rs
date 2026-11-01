//! Google EmbeddingGemma (`google/embeddinggemma-300m`) sentence embeddings: a Gemma 3
//! bidirectional text encoder, mean pooling, two Dense layers and L2 normalization
//! (sentence-transformers pipeline), with task prompts and Matryoshka truncation.
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
//!
//! Batching: texts are tokenized, sorted by length (longest first) and run in chunks of
//! `batch_size`, each right-padded to a multiple of [`PAD_MULTIPLE`] with padding masks,
//! so one call embeds many texts with few distinct graph shapes.

pub mod bench;
pub mod cmd;
pub mod config;
pub mod convert;
mod model;
pub mod verify;

use std::path::Path;

use lapid::{Device, Error, LapidFile, Result, Tensor};

use crate::model::{Bytes, Model as HubModel, ModelInfo, hub_model_path};
pub use config::EmbeddingGemmaConfig;
pub use lapid_nn::Precision;
use lapid_nn::{Loader, Tokenizer};
use model::{Batch, Model, l2_normalize};

/// Default Hugging Face repo.
pub const DEFAULT_REPO: &str = "google/embeddinggemma-300m";
/// Batches are padded to a multiple of this many tokens (fewer distinct graph shapes).
pub const PAD_MULTIPLE: usize = 8;

/// A traced intermediate: (name, host values, shape).
pub type TracedTensor = (String, Vec<f32>, Vec<usize>);

/// Options of [`EmbeddingGemma::embed`].
#[derive(Clone, Debug, PartialEq)]
pub struct EmbedOptions {
    /// Prompt name from the model's prompt table ("query", "document", "STS",
    /// "Clustering", ...). Ignored when `prompt` is set.
    pub prompt_name: Option<String>,
    /// Literal prompt prefix (e.g. "task: search result | query: ").
    pub prompt: Option<String>,
    /// Matryoshka output size (e.g. 768, 512, 256, 128); `None` = full size.
    pub dim: Option<usize>,
    /// Texts per forward pass.
    pub batch_size: usize,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        EmbedOptions { prompt_name: None, prompt: None, dim: None, batch_size: 32 }
    }
}

impl EmbedOptions {
    /// Prompt by name (see [`EmbeddingGemma::prompts`]).
    pub fn named(prompt_name: &str) -> EmbedOptions {
        EmbedOptions { prompt_name: Some(prompt_name.to_string()), ..Default::default() }
    }

    /// Retrieval query ("task: search result | query: ").
    pub fn query() -> EmbedOptions {
        EmbedOptions::named("query")
    }

    /// Retrieval document ("title: none | text: ").
    pub fn document() -> EmbedOptions {
        EmbedOptions::named("document")
    }

    pub fn with_dim(mut self, dim: usize) -> EmbedOptions {
        self.dim = Some(dim);
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> EmbedOptions {
        self.batch_size = batch_size;
        self
    }
}

/// Cosine similarity of two vectors (0 when either is zero or lengths differ).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
    }
    if na == 0.0 || nb == 0.0 { 0.0 } else { (dot / (na.sqrt() * nb.sqrt())) as f32 }
}

/// Pairwise cosine similarities: `m[i][j] = cos(a[i], b[j])`.
pub fn similarity_matrix(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    a.iter().map(|x| b.iter().map(|y| cosine_similarity(x, y)).collect()).collect()
}

/// Hugging Face pipeline tag.
pub const TASK: &str = "feature-extraction";

/// A loaded EmbeddingGemma model bound to a device.
pub struct EmbeddingGemma {
    device: Device,
    info: ModelInfo,
    cfg: EmbeddingGemmaConfig,
    tokenizer: Tokenizer,
    model: Model,
}

impl EmbeddingGemma {
    /// Load `repo` from the lapid cache, converting it from the Hugging Face hub first if
    /// it is missing (`dtype`: "fp32", "fp16", "int8", "int4").
    pub fn from_hub(repo: &str, dtype: &str, device: &Device) -> Result<EmbeddingGemma> {
        EmbeddingGemma::load(&hub_model_path(repo, dtype, convert::convert_hf)?, device)
    }

    /// Load a converted `.lapid` file.
    pub fn load(path: &Path, device: &Device) -> Result<EmbeddingGemma> {
        EmbeddingGemma::load_with(path, device, Precision::Stored)
    }

    /// Load from an in-memory `.lapid` image (zero copy).
    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<EmbeddingGemma> {
        EmbeddingGemma::from_file(&LapidFile::from_bytes(bytes)?, device, Precision::Stored)
    }

    /// Load with a weight precision (fake-quant modes build the references quantized
    /// kernels are verified against).
    pub fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<EmbeddingGemma> {
        let file = LapidFile::open(path)?;
        EmbeddingGemma::from_file(&file, device, precision)
    }

    fn from_file(file: &LapidFile, device: &Device, precision: Precision) -> Result<EmbeddingGemma> {
        let cfg = EmbeddingGemmaConfig::from_metadata(&file.metadata)?;
        let blob = file
            .blob(&cfg.tokenizer_blob)
            .ok_or_else(|| Error::new("missing_metadata", format!("tokenizer blob {:?} missing", cfg.tokenizer_blob)))?;
        let text = std::str::from_utf8(blob).map_err(|_| Error::new("tokenizer", "tokenizer blob is not UTF-8"))?;
        let tokenizer = Tokenizer::from_tokenizer_json(text)?;
        let model = Model::load(&Loader::new(file, device, precision), &cfg)?;
        Ok(EmbeddingGemma { device: device.clone(), info: ModelInfo::from_file(file, TASK), cfg, tokenizer, model })
    }

    pub fn config(&self) -> &EmbeddingGemmaConfig {
        &self.cfg
    }

    /// Full embedding size.
    pub fn dim(&self) -> usize {
        self.cfg.output_dim()
    }

    /// Prompt names and their prefixes.
    pub fn prompts(&self) -> &std::collections::BTreeMap<String, String> {
        &self.cfg.prompts
    }

    /// The prompt prefix `opts` selects ("" for none).
    pub fn prompt_for(&self, opts: &EmbedOptions) -> Result<String> {
        if let Some(p) = &opts.prompt {
            return Ok(p.clone());
        }
        let name = opts.prompt_name.as_ref().or(self.cfg.default_prompt_name.as_ref());
        match name {
            None => Ok(String::new()),
            Some(n) => self.cfg.prompts.get(n).cloned().ok_or_else(|| {
                let known: Vec<&str> = self.cfg.prompts.keys().map(String::as_str).collect();
                Error::new("invalid_argument", format!("unknown prompt name {n:?} (known: {})", known.join(", ")))
            }),
        }
    }

    /// Token ids of `prompt + text` (`<bos>` … `<eos>`), truncated to `max_seq_length`
    /// keeping the final `<eos>`.
    pub fn tokenize(&self, text: &str, prompt: &str) -> Result<Vec<u32>> {
        let mut ids = self.tokenizer.encode(&format!("{prompt}{text}"), true)?;
        let max = self.cfg.max_seq_length;
        if ids.len() > max {
            let last = ids[ids.len() - 1];
            ids.truncate(max - 1);
            ids.push(last);
        }
        Ok(ids)
    }

    /// Embed `texts`: one L2-normalized vector per text, in input order.
    pub fn embed(&self, texts: &[&str], opts: &EmbedOptions) -> Result<Vec<Vec<f32>>> {
        let prompt = self.prompt_for(opts)?;
        let ids: Vec<Vec<u32>> = texts.iter().map(|t| self.tokenize(t, &prompt)).collect::<Result<_>>()?;
        let refs: Vec<&[u32]> = ids.iter().map(Vec::as_slice).collect();
        self.embed_ids(&refs, opts.dim, opts.batch_size)
    }

    /// Embed pre-tokenized inputs (see [`EmbeddingGemma::tokenize`]).
    pub fn embed_ids(&self, ids: &[&[u32]], dim: Option<usize>, batch_size: usize) -> Result<Vec<Vec<f32>>> {
        let full = self.dim();
        let dim = dim.unwrap_or(full);
        if dim == 0 || dim > full {
            return Err(Error::new("invalid_argument", format!("dim {dim} outside 1..={full}")));
        }
        if batch_size == 0 {
            return Err(Error::new("invalid_argument", "batch_size must be >= 1"));
        }
        for (i, r) in ids.iter().enumerate() {
            if r.is_empty() || r.len() > self.cfg.max_seq_length {
                return Err(Error::new(
                    "invalid_argument",
                    format!("input {i}: {} tokens (allowed 1..={})", r.len(), self.cfg.max_seq_length),
                ));
            }
        }
        // Longest first: similar lengths share a batch, and a too-large batch fails early.
        let mut order: Vec<usize> = (0..ids.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(ids[i].len()));
        let mut out = vec![Vec::new(); ids.len()];
        for chunk in order.chunks(batch_size) {
            let rows: Vec<&[u32]> = chunk.iter().map(|&i| ids[i]).collect();
            let longest = rows.iter().map(|r| r.len()).max().unwrap_or(1);
            let seq = longest.div_ceil(PAD_MULTIPLE) * PAD_MULTIPLE;
            let batch = Batch::new(&rows, seq, self.cfg.pad_token_id)?;
            let graph = self.model.graph(&batch, false)?;
            let host = self.eval_host(&graph.embedding)?;
            for (r, &i) in chunk.iter().enumerate() {
                out[i] = finish(&host[r * full..(r + 1) * full], dim, self.cfg.normalize);
            }
        }
        Ok(out)
    }

    /// One padded batch with every intermediate, for verification: (name, [B, ...] host
    /// values, shape) in graph order, then the final normalized embeddings.
    pub fn embed_traced(&self, ids: &[&[u32]]) -> Result<(Vec<TracedTensor>, Vec<Vec<f32>>)> {
        let longest = ids.iter().map(|r| r.len()).max().unwrap_or(1);
        let batch = Batch::new(ids, longest, self.cfg.pad_token_id)?;
        let g = self.model.graph(&batch, true)?;
        let mut refs: Vec<&Tensor> = g.traced.iter().collect();
        refs.push(&g.embedding);
        let realized = self.device.eval(&refs)?;
        let mut traced = Vec::with_capacity(g.traced.len());
        for (lazy, t) in g.traced.iter().zip(&realized) {
            let name = lazy.name().unwrap_or("unnamed").to_string();
            traced.push((name, self.device.to_vec_f32(t)?, t.shape().to_vec()));
        }
        let last = realized.last().ok_or_else(|| Error::new("backend", "device.eval returned no outputs"))?;
        let host = self.device.to_vec_f32(last)?;
        let full = self.dim();
        let emb = host.chunks_exact(full).map(|r| finish(r, full, self.cfg.normalize)).collect();
        Ok((traced, emb))
    }

    fn eval_host(&self, t: &Tensor) -> Result<Vec<f32>> {
        let realized = self.device.eval(&[t])?;
        let r = realized.first().ok_or_else(|| Error::new("backend", "device.eval returned no outputs"))?;
        self.device.to_vec_f32(r)
    }
}

impl HubModel for EmbeddingGemma {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

/// Normalize, truncate to `dim` (Matryoshka) and re-normalize.
fn finish(row: &[f32], dim: usize, normalize: bool) -> Vec<f32> {
    let mut v = row.to_vec();
    if normalize {
        l2_normalize(&mut v);
    }
    if dim < v.len() {
        v.truncate(dim);
        if normalize {
            l2_normalize(&mut v);
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-7);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 2.0]).abs() < 1e-7);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
        let m = similarity_matrix(&[vec![1.0, 0.0]], &[vec![1.0, 0.0], vec![-1.0, 0.0]]);
        assert!((m[0][1] + 1.0).abs() < 1e-7);
    }

    #[test]
    fn matryoshka_truncation() {
        let v = finish(&[3.0, 4.0, 12.0], 2, true);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        assert_eq!(finish(&[3.0, 4.0], 2, false), vec![3.0, 4.0]);
    }

    #[test]
    fn options() {
        let o = EmbedOptions::query().with_dim(256).with_batch_size(4);
        assert_eq!(o.prompt_name.as_deref(), Some("query"));
        assert_eq!((o.dim, o.batch_size), (Some(256), 4));
    }
}
