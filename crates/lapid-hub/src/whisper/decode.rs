//! Single-window decoding with OpenAI whisper semantics (`decoding.py`): greedy
//! (temperature 0), SuppressBlank, SuppressTokens, ApplyTimestampRules, language detection.

use lapid::Result;
use lapid_nn::sampling::{argmax, log_softmax, logsumexp};

use crate::whisper::Whisper;
use crate::whisper::model::decoder::CrossKv;

/// ApplyTimestampRules parameters.
#[derive(Clone, Debug)]
pub(crate) struct TimestampRules {
    pub eot: u32,
    pub no_timestamps: u32,
    pub timestamp_begin: u32,
    /// `round(max_initial_timestamp / time_precision)` = 50 for 1.0 s.
    pub max_initial_timestamp_index: Option<u32>,
}

/// Logit filters in OpenAI order: SuppressBlank, SuppressTokens, ApplyTimestampRules.
#[derive(Clone, Debug)]
pub(crate) struct LogitFilters {
    pub sample_begin: usize,
    /// Suppressed on the first sampled token (`encode(" ") + [eot]`).
    pub suppress_blank: Vec<u32>,
    /// Suppressed at every step.
    pub suppress: Vec<u32>,
    pub timestamps: Option<TimestampRules>,
}

fn mask(logits: &mut [f32], ids: &[u32]) {
    for &id in ids {
        if let Some(l) = logits.get_mut(id as usize) {
            *l = f32::NEG_INFINITY;
        }
    }
}

fn mask_range(logits: &mut [f32], start: usize, end: usize) {
    let end = end.min(logits.len());
    if start < end {
        logits[start..end].fill(f32::NEG_INFINITY);
    }
}

impl TimestampRules {
    /// `tokens` is the full sequence so far (initial tokens included).
    pub fn apply(&self, logits: &mut [f32], tokens: &[u32], sample_begin: usize) {
        let tb = self.timestamp_begin as usize;
        let eot = self.eot as usize;
        let n = logits.len();
        // <|notimestamps|> is handled by without_timestamps
        mask(logits, &[self.no_timestamps]);

        let sampled = tokens.get(sample_begin..).unwrap_or(&[]);
        let is_ts = |t: u32| t >= self.timestamp_begin;
        let last_was_timestamp = sampled.last().is_some_and(|&t| is_ts(t));
        let penultimate_was_timestamp = sampled.len() < 2 || is_ts(sampled[sampled.len() - 2]);
        if last_was_timestamp {
            if penultimate_was_timestamp {
                // has to be non-timestamp
                mask_range(logits, tb, n);
            } else {
                // cannot be normal text tokens
                mask_range(logits, 0, eot);
            }
        }

        if let Some(&last_ts) = sampled.iter().rev().find(|&&t| is_ts(t)) {
            // timestamps shouldn't decrease; force nonzero segment length
            let timestamp_last = if last_was_timestamp && !penultimate_was_timestamp {
                last_ts as usize
            } else {
                last_ts as usize + 1
            };
            mask_range(logits, tb, timestamp_last);
        }

        if tokens.len() == sample_begin {
            // suppress generating non-timestamp tokens at the beginning
            mask_range(logits, 0, tb);
            if let Some(max_idx) = self.max_initial_timestamp_index {
                let last_allowed = tb + max_idx as usize;
                mask_range(logits, last_allowed + 1, n);
            }
        }

        // if sum of probability over timestamps is above any other token, sample timestamp
        if tb < n {
            let logprobs = log_softmax(logits);
            let timestamp_logprob = logsumexp(&logprobs[tb..]);
            let max_text = logprobs[..tb].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if timestamp_logprob > max_text {
                mask_range(logits, 0, tb);
            }
        }
    }
}

impl LogitFilters {
    pub fn apply(&self, logits: &mut [f32], tokens: &[u32]) {
        if tokens.len() == self.sample_begin {
            mask(logits, &self.suppress_blank);
        }
        mask(logits, &self.suppress);
        if let Some(ts) = &self.timestamps {
            ts.apply(logits, tokens, self.sample_begin);
        }
    }
}

/// Result of decoding one window.
#[derive(Clone, Debug)]
pub(crate) struct WindowResult {
    /// Sampled tokens (after the initial tokens, before EOT).
    pub tokens: Vec<u32>,
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
}

impl Whisper {
    /// OpenAI `detect_language`: logits after `<|startoftranscript|>`, restricted to the
    /// language tokens; returns the most likely language token id.
    pub(crate) fn detect_language_token(&self, cross: &CrossKv) -> Result<Option<u32>> {
        if self.cfg.languages.is_empty() {
            return Ok(None);
        }
        let mut cache = self.decoder.new_cache();
        let logits =
            self.decoder.step(&self.device, &[self.cfg.special.sot], &mut cache, cross, &[0])?;
        let best = self
            .cfg
            .languages
            .iter()
            .filter_map(|&(_, id)| logits.get(id as usize).map(|&l| (id, l)))
            .fold(None::<(u32, f32)>, |acc, (id, l)| match acc {
                Some((_, bl)) if bl >= l => acc,
                _ => Some((id, l)),
            });
        Ok(best.map(|(id, _)| id))
    }

    /// Greedy decoding of one window (`DecodingTask.run` with n_group = 1).
    pub(crate) fn decode_window(
        &self,
        cross: &CrossKv,
        initial: &[u32],
        sot_index: usize,
        filters: &LogitFilters,
    ) -> Result<WindowResult> {
        let eot = self.cfg.special.eot;
        let n_ctx = self.decoder.n_ctx();
        let sample_len = n_ctx / 2;
        let mut tokens: Vec<u32> = initial.to_vec();
        let mut pending: Vec<u32> = initial.to_vec();
        let mut cache = self.decoder.new_cache();
        let mut sum_logprobs = 0.0f64;
        let mut no_speech_prob = f32::NAN;

        for i in 0..sample_len {
            if cache.offset() + pending.len() > n_ctx {
                break;
            }
            let last = pending.len() - 1;
            let want_no_speech = i == 0 && self.cfg.special.no_speech.is_some();
            let rows: Vec<usize> =
                if want_no_speech && sot_index != last { vec![sot_index, last] } else { vec![last] };
            let all = self.decoder.step(&self.device, &pending, &mut cache, cross, &rows)?;
            let vocab = all.len() / rows.len();
            if want_no_speech
                && let Some(ns) = self.cfg.special.no_speech {
                    let lp = log_softmax(&all[..vocab]);
                    no_speech_prob = lp.get(ns as usize).map_or(f32::NAN, |v| v.exp());
                }
            let mut logits = all[(rows.len() - 1) * vocab..].to_vec();
            filters.apply(&mut logits, &tokens);

            let next = argmax(&logits) as u32;
            let logprobs = log_softmax(&logits);
            if tokens.last() != Some(&eot) {
                sum_logprobs += logprobs[next as usize] as f64;
            }
            tokens.push(next);
            pending = vec![next];
            if next == eot || tokens.len() > n_ctx {
                break;
            }
        }

        let sampled: Vec<u32> = tokens[initial.len()..]
            .iter()
            .copied()
            .take_while(|&t| t != eot)
            .collect();
        let avg_logprob = (sum_logprobs / (sampled.len() + 1) as f64) as f32;
        Ok(WindowResult { tokens: sampled, avg_logprob, no_speech_prob })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // toy vocab: text 0..5, eot 5, specials 6..10, no_timestamps 9, timestamps 10..20
    fn rules() -> TimestampRules {
        TimestampRules { eot: 5, no_timestamps: 9, timestamp_begin: 10, max_initial_timestamp_index: Some(3) }
    }

    fn allowed(l: &[f32]) -> Vec<usize> {
        l.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, _)| i).collect()
    }

    #[test]
    fn first_token_must_be_early_timestamp() {
        let mut logits = vec![0.0f32; 20];
        rules().apply(&mut logits, &[6, 7, 8], 3);
        assert_eq!(allowed(&logits), vec![10, 11, 12, 13]);
    }

    #[test]
    fn timestamps_come_in_pairs() {
        // after a single timestamp following text: only timestamps / eot
        let mut logits = vec![0.0f32; 20];
        rules().apply(&mut logits, &[6, 7, 8, 10, 1, 2, 12], 3);
        let a = allowed(&logits);
        assert!(a.iter().all(|&i| i >= 5 && i != 9));
        // the closing timestamp may equal the last one (nonzero-length is enforced for opens)
        assert!(a.contains(&12) && !a.contains(&11));

        // after a pair of timestamps: no timestamps allowed
        let mut logits = vec![0.0f32; 20];
        rules().apply(&mut logits, &[6, 7, 8, 10, 1, 12, 12], 3);
        let a = allowed(&logits);
        assert!(a.iter().all(|&i| i < 10));
    }

    #[test]
    fn timestamps_do_not_decrease() {
        let mut logits = vec![0.0f32; 20];
        logits[1] = 5.0; // text clearly preferred, so the logprob rule does not kick in
        rules().apply(&mut logits, &[6, 7, 8, 10, 1, 14, 15, 2], 3);
        let a = allowed(&logits);
        // text allowed, timestamps must be > 15
        assert!(a.contains(&1));
        assert!(!a.contains(&15) && a.contains(&16));
    }

    #[test]
    fn timestamp_mass_beats_text() {
        // text token 0 has the highest single prob, but timestamps together dominate
        let mut logits = vec![f32::NEG_INFINITY; 20];
        logits[0] = 2.0;
        for l in logits.iter_mut().skip(10) {
            *l = 1.0;
        }
        rules().apply(&mut logits, &[6, 7, 8, 10, 1], 3);
        assert!(!logits[0].is_finite());
        assert!(logits[16].is_finite());
    }

    #[test]
    fn suppress_blank_only_at_start() {
        let f = LogitFilters { sample_begin: 3, suppress_blank: vec![1, 5], suppress: vec![2], timestamps: None };
        let mut l = vec![0.0f32; 8];
        f.apply(&mut l, &[6, 7, 8]);
        assert_eq!(allowed(&l), vec![0, 3, 4, 6, 7]);
        let mut l = vec![0.0f32; 8];
        f.apply(&mut l, &[6, 7, 8, 0]);
        assert_eq!(allowed(&l), vec![0, 1, 3, 4, 5, 6, 7]);
    }
}
