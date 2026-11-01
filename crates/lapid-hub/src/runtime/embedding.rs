//! Text embedding pipeline: prompt prefix, tokenize, truncate, sort by length, batch and
//! right-pad, run the encoder, pool (mean / CLS / last token), optional dense projection,
//! L2 normalization and Matryoshka truncation (truncate to `dim`, then renormalize).
//!
//! How a model plugs in: implement [`TextEmbedder`]. `encode` builds the lazy encoder graph
//! of a padded [`TokenBatch`] (use [`TokenBatch::key_padding_mask`] for attention) and
//! returns token states `[B, S, D]`; `project` optionally maps pooled `[B, D]` to the
//! output (e.g. EmbeddingGemma's two Dense layers). [`embed`] does the rest, with pooling
//! as one batched matmul on the device.

use lapid::{Device, Error, Result, Tensor, bail};

/// How token states become one vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    /// Mean over real (unpadded) tokens.
    Mean,
    /// The first token.
    Cls,
    /// The last real token.
    LastToken,
}

/// The model's math and tokenizer.
pub trait TextEmbedder {
    fn device(&self) -> &Device;
    /// Token ids of `text` (the prompt prefix is already prepended), special tokens included.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>>;
    fn pad_id(&self) -> u32;
    fn max_seq_len(&self) -> usize;
    /// Output size (after `project`).
    fn dim(&self) -> usize;
    fn pooling(&self) -> Pooling;
    /// L2-normalize outputs (sentence-transformers `Normalize`).
    fn normalize(&self) -> bool {
        true
    }
    /// Named prompt prefixes ("query", "document", ...).
    fn prompt(&self, _name: &str) -> Option<String> {
        None
    }
    /// Prompt used when the options name none.
    fn default_prompt_name(&self) -> Option<String> {
        None
    }
    /// Lazy encoder graph: token states `[B, S, D]`.
    fn encode(&self, batch: &TokenBatch) -> Result<Tensor>;
    /// Optional head on pooled `[B, D]` (lazy) -> `[B, dim]`.
    fn project(&self, pooled: Tensor) -> Result<Tensor> {
        Ok(pooled)
    }
}

/// How over-long inputs are cut to `max_seq_len`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Truncation {
    /// Keep the head and the final token (e.g. `<eos>`).
    #[default]
    KeepLast,
    /// Keep the head.
    Head,
    /// Fail.
    Error,
}

/// Options of [`embed`].
#[derive(Clone, Debug, PartialEq)]
pub struct EmbedOptions {
    /// Prompt name (see [`TextEmbedder::prompt`]); ignored when `prompt` is set.
    pub prompt_name: Option<String>,
    /// Literal prompt prefix.
    pub prompt: Option<String>,
    /// Matryoshka output size; `None` = full size.
    pub dim: Option<usize>,
    /// Texts per forward pass.
    pub batch_size: usize,
    /// Batches are padded to a multiple of this many tokens (fewer distinct graph shapes).
    pub pad_multiple: usize,
    pub truncation: Truncation,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        EmbedOptions { prompt_name: None, prompt: None, dim: None, batch_size: 32, pad_multiple: 8, truncation: Truncation::KeepLast }
    }
}

/// One right-padded batch of token ids.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenBatch {
    /// Row-major `[B, S]` ids (padding = pad id).
    pub ids: Vec<i32>,
    /// Real token count per row (each >= 1).
    pub lengths: Vec<usize>,
    pub seq: usize,
}

impl TokenBatch {
    /// Right-pad `rows` to `seq` (>= the longest row) with `pad`.
    pub fn new(rows: &[&[u32]], seq: usize, pad: u32) -> Result<TokenBatch> {
        if rows.is_empty() {
            bail!("invalid_argument", "empty batch");
        }
        let mut ids = Vec::with_capacity(rows.len() * seq);
        let mut lengths = Vec::with_capacity(rows.len());
        for r in rows {
            if r.is_empty() || r.len() > seq {
                bail!("invalid_argument", "batch row of {} tokens does not fit 1..={seq}", r.len());
            }
            ids.extend(r.iter().map(|&t| t as i32));
            ids.extend(std::iter::repeat_n(pad as i32, seq - r.len()));
            lengths.push(r.len());
        }
        Ok(TokenBatch { ids, lengths, seq })
    }

    pub fn size(&self) -> usize {
        self.lengths.len()
    }

    /// Any row shorter than `seq`.
    pub fn padded(&self) -> bool {
        self.lengths.iter().any(|&l| l != self.seq)
    }

    /// Ids as an i32 tensor `[B, S]`.
    pub fn ids_tensor(&self) -> Result<Tensor> {
        Tensor::from_i32(&self.ids, &[self.size(), self.seq])
    }

    /// Additive key padding mask (`-inf` on padded keys), see `lapid_nn::attention`.
    pub fn key_padding_mask(&self) -> Result<Tensor> {
        lapid_nn::attention::key_padding_mask(&self.lengths, self.seq)
    }

    /// Pooling weights `[B, 1, S]`.
    pub fn pool_weights(&self, pooling: Pooling) -> Result<Tensor> {
        let s = self.seq;
        let mut data = vec![0.0f32; self.size() * s];
        for (b, &len) in self.lengths.iter().enumerate() {
            let row = &mut data[b * s..(b + 1) * s];
            match pooling {
                Pooling::Mean => row[..len].fill(1.0 / len as f32),
                Pooling::Cls => row[0] = 1.0,
                Pooling::LastToken => row[len - 1] = 1.0,
            }
        }
        Tensor::from_f32(&data, &[self.size(), 1, s])
    }
}

/// The prompt prefix `opts` selects ("" for none).
pub fn prompt_for<M: TextEmbedder + ?Sized>(model: &M, opts: &EmbedOptions) -> Result<String> {
    if let Some(p) = &opts.prompt {
        return Ok(p.clone());
    }
    match opts.prompt_name.clone().or_else(|| model.default_prompt_name()) {
        None => Ok(String::new()),
        Some(n) => model.prompt(&n).ok_or_else(|| Error::new("invalid_argument", format!("unknown prompt name {n:?}"))),
    }
}

/// Token ids of `prompt + text`, cut to `max_seq_len` by `truncation`.
pub fn tokenize<M: TextEmbedder + ?Sized>(model: &M, text: &str, prompt: &str, truncation: Truncation) -> Result<Vec<u32>> {
    let mut ids = model.tokenize(&format!("{prompt}{text}"))?;
    let max = model.max_seq_len();
    if ids.len() > max {
        match truncation {
            Truncation::KeepLast if max >= 1 => {
                let last = ids[ids.len() - 1];
                ids.truncate(max - 1);
                ids.push(last);
            }
            Truncation::KeepLast | Truncation::Head => ids.truncate(max),
            Truncation::Error => bail!("invalid_argument", "input of {} tokens exceeds {max}", ids.len()),
        }
    }
    Ok(ids)
}

/// Embed `texts`: one vector per text, in input order.
pub fn embed<M: TextEmbedder + ?Sized>(model: &M, texts: &[&str], opts: &EmbedOptions) -> Result<Vec<Vec<f32>>> {
    let prompt = prompt_for(model, opts)?;
    let ids: Vec<Vec<u32>> = texts.iter().map(|t| tokenize(model, t, &prompt, opts.truncation)).collect::<Result<_>>()?;
    let refs: Vec<&[u32]> = ids.iter().map(Vec::as_slice).collect();
    embed_ids(model, &refs, opts)
}

/// Lazy embedding graph of one batch (pooled and projected, before normalization) `[B, dim]`.
pub fn embedding_graph<M: TextEmbedder + ?Sized>(model: &M, batch: &TokenBatch) -> Result<Tensor> {
    let tokens = model.encode(batch)?;
    let shape = tokens.shape().to_vec();
    if shape.len() != 3 || shape[0] != batch.size() || shape[1] != batch.seq {
        bail!("shape_mismatch", "encoder gave {shape:?}, expected [{}, {}, D]", batch.size(), batch.seq);
    }
    let pooled = batch.pool_weights(model.pooling())?.matmul(&tokens)?.reshape(&[shape[0], shape[2]])?;
    model.project(pooled)
}

/// Embed pre-tokenized inputs (prompt already applied).
pub fn embed_ids<M: TextEmbedder + ?Sized>(model: &M, ids: &[&[u32]], opts: &EmbedOptions) -> Result<Vec<Vec<f32>>> {
    let full = model.dim();
    let dim = opts.dim.unwrap_or(full);
    if dim == 0 || dim > full {
        bail!("invalid_argument", "dim {dim} outside 1..={full}");
    }
    if opts.batch_size == 0 {
        bail!("invalid_argument", "batch_size must be >= 1");
    }
    let max = model.max_seq_len();
    for (i, r) in ids.iter().enumerate() {
        if r.is_empty() || r.len() > max {
            bail!("invalid_argument", "input {i}: {} tokens (allowed 1..={max})", r.len());
        }
    }
    // Longest first: similar lengths share a batch, and a too-large batch fails early.
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(ids[i].len()));
    let mut out = vec![Vec::new(); ids.len()];
    let multiple = opts.pad_multiple.max(1);
    for chunk in order.chunks(opts.batch_size) {
        let rows: Vec<&[u32]> = chunk.iter().map(|&i| ids[i]).collect();
        let longest = rows.iter().map(|r| r.len()).max().unwrap_or(1);
        let seq = longest.div_ceil(multiple) * multiple;
        let batch = TokenBatch::new(&rows, seq, model.pad_id())?;
        let graph = embedding_graph(model, &batch)?;
        let dev = model.device();
        let realized = dev.eval(&[&graph])?;
        let r = realized.first().ok_or_else(|| Error::new("backend", "device.eval returned no outputs"))?;
        let host = dev.to_vec_f32(r)?;
        if host.len() != chunk.len() * full {
            bail!("shape_mismatch", "embedding batch gave {} values, expected {}", host.len(), chunk.len() * full);
        }
        for (k, &i) in chunk.iter().enumerate() {
            out[i] = finish(&host[k * full..(k + 1) * full], dim, model.normalize());
        }
    }
    Ok(out)
}

/// `x / max(||x||, 1e-12)` (torch `F.normalize`), in place.
pub fn l2_normalize(x: &mut [f32]) {
    let n = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt().max(1e-12);
    for v in x {
        *v = (f64::from(*v) / n) as f32;
    }
}

/// Normalize, truncate to `dim` (Matryoshka) and re-normalize.
pub fn finish(row: &[f32], dim: usize, normalize: bool) -> Vec<f32> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Toy: one token per byte; token `t` has state `[t, 1, 0, 0]`; the head doubles.
    struct Toy {
        device: Device,
        pooling: Pooling,
        normalize: bool,
        max: usize,
    }

    impl Toy {
        fn new(pooling: Pooling, normalize: bool) -> Toy {
            Toy { device: Device::cpu(), pooling, normalize, max: 64 }
        }
    }

    impl TextEmbedder for Toy {
        fn device(&self) -> &Device {
            &self.device
        }
        fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text.bytes().map(|b| u32::from(b - b'0')).collect())
        }
        fn pad_id(&self) -> u32 {
            99
        }
        fn max_seq_len(&self) -> usize {
            self.max
        }
        fn dim(&self) -> usize {
            4
        }
        fn pooling(&self) -> Pooling {
            self.pooling
        }
        fn normalize(&self) -> bool {
            self.normalize
        }
        fn prompt(&self, name: &str) -> Option<String> {
            (name == "query").then(|| "9".to_string())
        }
        fn encode(&self, batch: &TokenBatch) -> Result<Tensor> {
            assert!(batch.seq.is_multiple_of(8));
            let mut x = Vec::new();
            for &id in &batch.ids {
                x.extend([id as f32, 1.0, 0.0, 0.0]);
            }
            Tensor::from_f32(&x, &[batch.size(), batch.seq, 4])
        }
        fn project(&self, pooled: Tensor) -> Result<Tensor> {
            pooled.mul_scalar(2.0)
        }
    }

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn pooling_modes_ignore_padding() {
        let texts = ["1", "123", "36"];
        let opts = EmbedOptions { batch_size: 2, ..Default::default() };
        let mean = embed(&Toy::new(Pooling::Mean, false), &texts, &opts).expect("embed");
        assert!(close(&mean[0], &[2.0, 2.0, 0.0, 0.0]));
        assert!(close(&mean[1], &[4.0, 2.0, 0.0, 0.0]));
        assert!(close(&mean[2], &[9.0, 2.0, 0.0, 0.0]));
        let cls = embed(&Toy::new(Pooling::Cls, false), &texts, &opts).expect("embed");
        assert!(close(&cls[1], &[2.0, 2.0, 0.0, 0.0]));
        let last = embed(&Toy::new(Pooling::LastToken, false), &texts, &opts).expect("embed");
        assert!(close(&last[1], &[6.0, 2.0, 0.0, 0.0]) && close(&last[2], &[12.0, 2.0, 0.0, 0.0]));
    }

    #[test]
    fn normalize_matryoshka_and_prompts() {
        let m = Toy::new(Pooling::Mean, true);
        // mean of "3" -> [3, 1, 0, 0] * 2, normalized
        let v = embed(&m, &["3"], &EmbedOptions::default()).expect("embed");
        let n = 10f32.sqrt();
        assert!(close(&v[0], &[3.0 / n, 1.0 / n, 0.0, 0.0]));
        let opts = EmbedOptions { dim: Some(1), ..Default::default() };
        assert!(close(&embed(&m, &["3"], &opts).expect("embed")[0], &[1.0]));
        // prompt "9" prepended: mean of [9, 3] = 6
        let q = EmbedOptions { prompt_name: Some("query".into()), ..Default::default() };
        let v = embed(&Toy::new(Pooling::Mean, false), &["3"], &q).expect("embed");
        assert!(close(&v[0], &[12.0, 2.0, 0.0, 0.0]));
        let bad = EmbedOptions { prompt_name: Some("nope".into()), ..Default::default() };
        assert!(embed(&m, &["3"], &bad).is_err());
        assert!(embed(&m, &["3"], &EmbedOptions { dim: Some(5), ..Default::default() }).is_err());
        assert!((cosine_similarity(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn truncation_and_batches() {
        let mut m = Toy::new(Pooling::LastToken, false);
        m.max = 3;
        assert_eq!(tokenize(&m, "12345", "", Truncation::KeepLast).expect("tok"), vec![1, 2, 5]);
        assert_eq!(tokenize(&m, "12345", "", Truncation::Head).expect("tok"), vec![1, 2, 3]);
        assert!(tokenize(&m, "12345", "", Truncation::Error).is_err());
        let v = embed(&m, &["12345"], &EmbedOptions::default()).expect("embed");
        assert!(close(&v[0], &[10.0, 2.0, 0.0, 0.0]));
        let b = TokenBatch::new(&[&[1, 2], &[3]], 4, 0).expect("batch");
        assert_eq!(b.ids, vec![1, 2, 0, 0, 3, 0, 0, 0]);
        assert!(b.padded() && TokenBatch::new(&[&[1, 2, 3]], 2, 0).is_err());
        assert!(close(&finish(&[3.0, 4.0, 12.0], 2, true), &[0.6, 0.8]));
    }
}
