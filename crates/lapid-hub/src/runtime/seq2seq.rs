//! Encoder-decoder (Whisper-style) pipeline: encode once, then an autoregressive decoder with
//! a self-attention KV cache and a per-input cross-attention memory, forced prefix tokens,
//! logit processors (suppression, forced tokens, timestamp rules) and greedy selection.
//!
//! How a model plugs in: implement [`Seq2Seq`]. `encode` returns the memory the decoder
//! attends to (e.g. Whisper's per-layer cross K/V), `new_cache` an empty self-attention
//! cache, and `decode` runs `ids` through the decoder (the whole prefix first, then one
//! token per step), extends the cache and returns host logits for the requested rows.
//! Model-specific rules (e.g. Whisper's ApplyTimestampRules) are [`LogitsProcessor`]s;
//! [`generate`] does the loop, limits, log-probabilities and prefix row inspection.

use lapid::{Error, Result, bail};
use lapid_nn::sampling::{argmax, log_softmax};

/// The model's math.
pub trait Seq2Seq {
    /// Encoder input (e.g. a log-mel window).
    type Input: ?Sized;
    /// What the decoder attends to (e.g. cross-attention K/V per layer).
    type Memory;
    /// Self-attention cache.
    type Cache;
    fn encode(&self, input: &Self::Input) -> Result<Self::Memory>;
    fn new_cache(&self, memory: &Self::Memory) -> Result<Self::Cache>;
    /// Positions held by `cache`.
    fn cache_len(&self, cache: &Self::Cache) -> usize;
    /// Maximum decoder positions (text context).
    fn max_positions(&self) -> usize;
    /// Run `ids` after the cached positions, extend `cache`, and return the logits of
    /// `rows` (indices into `ids`), concatenated: `[rows.len() * vocab]`.
    fn decode(&self, ids: &[u32], cache: &mut Self::Cache, memory: &Self::Memory, rows: &[usize]) -> Result<Vec<f32>>;
}

/// What a logits processor sees at one step.
#[derive(Clone, Copy, Debug)]
pub struct StepContext<'a> {
    /// The full sequence so far (prefix included).
    pub tokens: &'a [u32],
    /// Index of the first sampled token in `tokens` (= prefix length).
    pub sample_begin: usize,
}

impl StepContext<'_> {
    /// Tokens sampled so far.
    pub fn sampled(&self) -> &[u32] {
        self.tokens.get(self.sample_begin..).unwrap_or(&[])
    }

    /// Index of the token being chosen (0 = first sampled token).
    pub fn step(&self) -> usize {
        self.tokens.len().saturating_sub(self.sample_begin)
    }
}

/// Edits the logits of one step in place (mask with `-inf`).
pub trait LogitsProcessor {
    fn process(&mut self, logits: &mut [f32], ctx: &StepContext<'_>);
}

impl<F: FnMut(&mut [f32], &StepContext<'_>)> LogitsProcessor for F {
    fn process(&mut self, logits: &mut [f32], ctx: &StepContext<'_>) {
        self(logits, ctx)
    }
}

/// Set `ids` to `-inf` (ids outside the vocabulary are ignored).
pub fn mask_tokens(logits: &mut [f32], ids: &[u32]) {
    for &id in ids {
        if let Some(l) = logits.get_mut(id as usize) {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Set `[start, end)` to `-inf` (clamped to the vocabulary).
pub fn mask_range(logits: &mut [f32], start: usize, end: usize) {
    let end = end.min(logits.len());
    if start < end {
        logits[start..end].fill(f32::NEG_INFINITY);
    }
}

/// Suppressed at every step.
#[derive(Clone, Debug, Default)]
pub struct SuppressTokens(pub Vec<u32>);

impl LogitsProcessor for SuppressTokens {
    fn process(&mut self, logits: &mut [f32], _: &StepContext<'_>) {
        mask_tokens(logits, &self.0);
    }
}

/// Suppressed on the first sampled token only (e.g. Whisper's SuppressBlank).
#[derive(Clone, Debug, Default)]
pub struct SuppressAtBegin(pub Vec<u32>);

impl LogitsProcessor for SuppressAtBegin {
    fn process(&mut self, logits: &mut [f32], ctx: &StepContext<'_>) {
        if ctx.step() == 0 {
            mask_tokens(logits, &self.0);
        }
    }
}

/// Forces token `id` at sampled step `step` (`(step, id)` pairs).
#[derive(Clone, Debug, Default)]
pub struct ForceTokens(pub Vec<(usize, u32)>);

impl LogitsProcessor for ForceTokens {
    fn process(&mut self, logits: &mut [f32], ctx: &StepContext<'_>) {
        let step = ctx.step();
        if let Some(&(_, id)) = self.0.iter().find(|(s, _)| *s == step)
            && (id as usize) < logits.len()
        {
            let keep = logits[id as usize];
            logits.fill(f32::NEG_INFINITY);
            logits[id as usize] = if keep.is_finite() { keep } else { 0.0 };
        }
    }
}

/// Options of [`generate`].
#[derive(Clone, Debug)]
pub struct GenerateOptions {
    /// Forced initial tokens (start-of-transcript, language, task, prompt, ...). Not empty.
    pub prefix: Vec<u32>,
    /// End token: stops generation (not part of the returned tokens).
    pub eos: Option<u32>,
    /// Sampled tokens at most (Whisper: `n_text_ctx / 2`).
    pub max_new_tokens: usize,
    /// Prefix positions whose raw logits are returned from the first step (e.g. Whisper's
    /// no-speech probability at the start-of-transcript token).
    pub inspect_rows: Vec<usize>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        GenerateOptions { prefix: Vec::new(), eos: None, max_new_tokens: 224, inspect_rows: Vec::new() }
    }
}

/// Result of [`generate`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Generation {
    /// Sampled tokens after the prefix, up to (not including) `eos`.
    pub tokens: Vec<u32>,
    /// Sum of the chosen tokens' log-probabilities (after processing; `eos` included).
    pub sum_logprob: f64,
    /// Ended on `eos` (rather than a length limit).
    pub finished: bool,
    /// Raw logits of `inspect_rows`, in order.
    pub inspected: Vec<Vec<f32>>,
}

impl Generation {
    /// OpenAI's average: `sum_logprob / (tokens + 1)`.
    pub fn avg_logprob(&self) -> f32 {
        (self.sum_logprob / (self.tokens.len() + 1) as f64) as f32
    }
}

/// Encode `input` and generate.
pub fn run<M: Seq2Seq + ?Sized>(
    model: &M,
    input: &M::Input,
    opts: &GenerateOptions,
    processors: &mut [&mut dyn LogitsProcessor],
) -> Result<Generation> {
    let memory = model.encode(input)?;
    generate(model, &memory, opts, processors)
}

/// Raw next-token logits after `prefix` with a fresh cache (e.g. language detection).
pub fn next_logits<M: Seq2Seq + ?Sized>(model: &M, memory: &M::Memory, prefix: &[u32]) -> Result<Vec<f32>> {
    if prefix.is_empty() {
        bail!("invalid_argument", "seq2seq: empty prefix");
    }
    let mut cache = model.new_cache(memory)?;
    model.decode(prefix, &mut cache, memory, &[prefix.len() - 1])
}

/// Greedy autoregressive decoding over an encoded memory.
pub fn generate<M: Seq2Seq + ?Sized>(
    model: &M,
    memory: &M::Memory,
    opts: &GenerateOptions,
    processors: &mut [&mut dyn LogitsProcessor],
) -> Result<Generation> {
    let prefix = &opts.prefix;
    if prefix.is_empty() {
        bail!("invalid_argument", "seq2seq: empty prefix");
    }
    if let Some(&bad) = opts.inspect_rows.iter().find(|&&r| r >= prefix.len()) {
        bail!("invalid_argument", "seq2seq: inspect row {bad} outside the prefix of {}", prefix.len());
    }
    let n_ctx = model.max_positions();
    let mut tokens: Vec<u32> = prefix.clone();
    let mut pending: Vec<u32> = prefix.clone();
    let mut cache = model.new_cache(memory)?;
    let mut out = Generation::default();

    for i in 0..opts.max_new_tokens {
        if model.cache_len(&cache) + pending.len() > n_ctx {
            break;
        }
        let last = pending.len() - 1;
        let mut rows: Vec<usize> = Vec::new();
        if i == 0 {
            for &r in &opts.inspect_rows {
                if r != last && !rows.contains(&r) {
                    rows.push(r);
                }
            }
        }
        rows.push(last);
        let all = model.decode(&pending, &mut cache, memory, &rows)?;
        if all.is_empty() || !all.len().is_multiple_of(rows.len()) {
            bail!("shape_mismatch", "seq2seq: decoder gave {} logits for {} rows", all.len(), rows.len());
        }
        let vocab = all.len() / rows.len();
        let row = |r: usize| -> Result<Vec<f32>> {
            let k = rows.iter().position(|&x| x == r).ok_or_else(|| Error::new("internal", "seq2seq: row not decoded"))?;
            Ok(all[k * vocab..(k + 1) * vocab].to_vec())
        };
        if i == 0 {
            out.inspected = opts.inspect_rows.iter().map(|&r| row(r)).collect::<Result<_>>()?;
        }
        let mut logits = row(last)?;
        let ctx = StepContext { tokens: &tokens, sample_begin: prefix.len() };
        for p in processors.iter_mut() {
            p.process(&mut logits, &ctx);
        }
        let next = argmax(&logits) as u32;
        let lp = log_softmax(&logits);
        out.sum_logprob += f64::from(lp.get(next as usize).copied().unwrap_or(f32::NEG_INFINITY));
        tokens.push(next);
        pending = vec![next];
        if opts.eos == Some(next) {
            out.finished = true;
            break;
        }
        if tokens.len() > n_ctx {
            break;
        }
    }
    out.tokens = tokens[prefix.len()..].iter().copied().take_while(|&t| Some(t) != opts.eos).collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Toy: vocabulary 0..5 text + 5 = eos. After token `t` the logits favor `(t + 1) % 5`
    /// (3.0), then eos (1.0); the memory adds a bias to one token.
    struct Toy {
        n_ctx: usize,
    }

    const EOS: u32 = 5;

    impl Seq2Seq for Toy {
        type Input = [f32];
        type Memory = Option<u32>;
        type Cache = Vec<u32>;
        fn encode(&self, input: &[f32]) -> Result<Option<u32>> {
            Ok(input.first().map(|&x| x as u32))
        }
        fn new_cache(&self, _: &Option<u32>) -> Result<Vec<u32>> {
            Ok(Vec::new())
        }
        fn cache_len(&self, cache: &Vec<u32>) -> usize {
            cache.len()
        }
        fn max_positions(&self) -> usize {
            self.n_ctx
        }
        fn decode(&self, ids: &[u32], cache: &mut Vec<u32>, memory: &Option<u32>, rows: &[usize]) -> Result<Vec<f32>> {
            assert!(cache.is_empty() || ids.len() == 1, "prefix first, then one token per step");
            cache.extend_from_slice(ids);
            let mut out = Vec::new();
            for &r in rows {
                let mut l = vec![0.0f32; 6];
                l[((ids[r] + 1) % 5) as usize] = 3.0;
                l[EOS as usize] = 1.0;
                if let Some(b) = memory {
                    l[*b as usize] += 10.0;
                }
                out.extend(l);
            }
            Ok(out)
        }
    }

    fn opts(prefix: &[u32], max: usize) -> GenerateOptions {
        GenerateOptions { prefix: prefix.to_vec(), eos: Some(EOS), max_new_tokens: max, inspect_rows: Vec::new() }
    }

    #[test]
    fn greedy_with_prefix_and_limit() {
        let m = Toy { n_ctx: 100 };
        let g = generate(&m, &None, &opts(&[0], 4), &mut []).expect("generate");
        assert_eq!(g.tokens, vec![1, 2, 3, 4]);
        assert!(!g.finished);
        assert!(g.avg_logprob() < 0.0);
        // encode + memory bias: eos always wins
        let g = run(&m, &[5.0][..], &opts(&[0], 10), &mut []).expect("run");
        assert_eq!((g.tokens.clone(), g.finished), (vec![], true));
    }

    #[test]
    fn suppression_and_forcing() {
        let m = Toy { n_ctx: 100 };
        let mut sup = SuppressTokens(vec![3]);
        let g = generate(&m, &None, &opts(&[0], 10), &mut [&mut sup]).expect("generate");
        assert_eq!((g.tokens.clone(), g.finished), (vec![1, 2], true));

        let mut begin = SuppressAtBegin(vec![1]);
        let g = generate(&m, &None, &opts(&[0], 10), &mut [&mut begin]).expect("generate");
        assert_eq!(g.tokens, Vec::<u32>::new());

        let mut force = ForceTokens(vec![(1, 4)]);
        let g = generate(&m, &None, &opts(&[0], 4), &mut [&mut force]).expect("generate");
        assert_eq!(g.tokens, vec![1, 4, 0, 1]);

        // closures are processors; they see the whole sequence
        let mut seen = Vec::new();
        let mut log = |l: &mut [f32], c: &StepContext<'_>| {
            seen.push(c.sampled().to_vec());
            mask_range(l, 4, 6);
        };
        let g = generate(&m, &None, &opts(&[2], 3), &mut [&mut log]).expect("generate");
        assert_eq!(g.tokens, vec![3, 0, 1]);
        assert_eq!(seen, vec![vec![], vec![3], vec![3, 0]]);
    }

    #[test]
    fn context_limit_and_inspection() {
        let m = Toy { n_ctx: 4 };
        let mut o = opts(&[0, 1], 10);
        o.inspect_rows = vec![0, 1];
        let g = generate(&m, &None, &o, &mut []).expect("generate");
        assert_eq!(g.tokens, vec![2, 3, 4]);
        assert_eq!(g.inspected.len(), 2);
        assert_eq!(g.inspected[0][1], 3.0);
        assert_eq!(g.inspected[1][2], 3.0);
        assert_eq!(next_logits(&m, &None, &[3]).expect("logits")[4], 3.0);
        o.inspect_rows = vec![2];
        assert!(generate(&m, &None, &o, &mut []).is_err());
        assert!(generate(&m, &None, &opts(&[], 3), &mut []).is_err());
    }
}
