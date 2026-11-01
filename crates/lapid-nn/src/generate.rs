//! Autoregressive generation loop as an iterator.
//!
//! The model is a step callback: `step(new_tokens) -> logits of the last position`. It
//! receives the whole prompt first, then one sampled token per call, and keeps its own
//! state (typically a [`KvCache`](crate::attention::KvCache)).
//!
//! ```no_run
//! # use lapid_nn::generate::generate;
//! # use lapid_nn::Sampler;
//! # fn model_step(_: &[u32]) -> lapid::Result<Vec<f32>> { Ok(vec![0.0; 10]) }
//! # fn main() -> lapid::Result<()> {
//! let mut tokens = generate(&[1, 2, 3], model_step, Sampler::greedy(), 64, &[2]);
//! for token in &mut tokens {
//!     let token = token?;
//!     // stream / detokenize `token`
//! }
//! # Ok(())
//! # }
//! ```

use lapid::{Error, Result};

use crate::sampling::Sampler;

type Processor = Box<dyn FnMut(&mut [f32], &[u32])>;

pub struct Generate<F> {
    step: F,
    sampler: Sampler,
    tokens: Vec<u32>,
    prompt_len: usize,
    pending: Vec<u32>,
    stop: Vec<u32>,
    remaining: usize,
    done: bool,
    processor: Option<Processor>,
}

/// Generate up to `max_new_tokens` after `prompt`, stopping after any of `stop_tokens`
/// (the stop token is yielded, then the iterator ends). An error ends the iteration.
pub fn generate<F>(prompt: &[u32], step: F, sampler: Sampler, max_new_tokens: usize, stop_tokens: &[u32]) -> Generate<F>
where
    F: FnMut(&[u32]) -> Result<Vec<f32>>,
{
    Generate {
        step,
        sampler,
        tokens: prompt.to_vec(),
        prompt_len: prompt.len(),
        pending: prompt.to_vec(),
        stop: stop_tokens.to_vec(),
        remaining: max_new_tokens,
        done: false,
        processor: None,
    }
}

impl<F> Generate<F> {
    /// Edit logits before sampling (`f(logits, tokens_so_far)`), e.g. to suppress tokens.
    pub fn with_logits_processor(mut self, f: impl FnMut(&mut [f32], &[u32]) + 'static) -> Self {
        self.processor = Some(Box::new(f));
        self
    }

    /// Prompt plus generated tokens.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Generated tokens only.
    pub fn generated(&self) -> &[u32] {
        &self.tokens[self.prompt_len..]
    }
}

impl<F> Iterator for Generate<F>
where
    F: FnMut(&[u32]) -> Result<Vec<f32>>,
{
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Result<u32>> {
        if self.done || self.remaining == 0 {
            return None;
        }
        if self.pending.is_empty() {
            self.done = true;
            return Some(Err(Error::new("invalid_argument", "generate needs a non-empty prompt")));
        }
        let mut logits = match (self.step)(&self.pending) {
            Ok(l) => l,
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        };
        if logits.is_empty() {
            self.done = true;
            return Some(Err(Error::new("invalid_argument", "generate: step returned no logits")));
        }
        let _sample = tracing::trace_span!("sample").entered();
        if let Some(p) = &mut self.processor {
            p(&mut logits, &self.tokens);
        }
        let t = self.sampler.sample(&logits, &self.tokens);
        self.tokens.push(t);
        self.pending = vec![t];
        self.remaining -= 1;
        if self.stop.contains(&t) {
            self.done = true;
        }
        Some(Ok(t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Toy model: next token = (sum of all tokens seen) % 7, as one-hot logits.
    fn toy() -> impl FnMut(&[u32]) -> Result<Vec<f32>> {
        let mut seen: Vec<u32> = Vec::new();
        move |new: &[u32]| {
            seen.extend_from_slice(new);
            let mut l = vec![0.0f32; 7];
            l[(seen.iter().sum::<u32>() % 7) as usize] = 10.0;
            Ok(l)
        }
    }

    #[test]
    fn greedy_until_max_or_stop() {
        let mut g = generate(&[1, 2], toy(), Sampler::greedy(), 4, &[]);
        let out: Vec<u32> = g.by_ref().map(|t| t.unwrap()).collect();
        // sums: 3 -> 3, 6 -> 6, 12 -> 5, 17 -> 3
        assert_eq!(out, vec![3, 6, 5, 3]);
        assert_eq!(g.generated(), &[3, 6, 5, 3]);
        assert_eq!(g.tokens().len(), 6);

        let out: Vec<u32> = generate(&[1, 2], toy(), Sampler::greedy(), 10, &[6]).map(|t| t.unwrap()).collect();
        assert_eq!(out, vec![3, 6]);
    }

    #[test]
    fn processor_and_errors() {
        let g = generate(&[1, 2], toy(), Sampler::greedy(), 1, &[]).with_logits_processor(|l, _| l[3] = f32::NEG_INFINITY);
        let out: Vec<u32> = g.map(|t| t.unwrap()).collect();
        assert_ne!(out, vec![3]);

        let mut e = generate(&[], toy(), Sampler::greedy(), 3, &[]);
        assert!(e.next().unwrap().is_err());
        assert!(e.next().is_none());

        let failing = |_: &[u32]| -> Result<Vec<f32>> { Err(Error::new("backend", "boom")) };
        let mut f = generate(&[1], failing, Sampler::greedy(), 3, &[]);
        assert_eq!(f.next().unwrap().unwrap_err().code, "backend");
        assert!(f.next().is_none());
    }
}
