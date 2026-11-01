//! Whisper's own decoding rules on the shared seq2seq pipeline (`runtime::seq2seq`): OpenAI
//! `ApplyTimestampRules` as a logits processor, language detection, and the long-form
//! `transcribe()` seek loop (30 s windows, timestamp-based seeking, segments, prompt
//! conditioning on previous text).

use lapid::{Error, Result};
use lapid_nn::sampling::{log_softmax, logsumexp};

use super::mel::{self, CHUNK_SAMPLES, LongMel, N_FRAMES};
use super::{Segment, Task, TranscribeOptions, Transcript, Whisper};
use crate::runtime::seq2seq::{self, GenerateOptions, LogitsProcessor, StepContext, SuppressAtBegin, SuppressTokens, mask_range, mask_tokens};

/// OpenAI defaults.
const NO_SPEECH_THRESHOLD: f32 = 0.6;
const LOGPROB_THRESHOLD: f32 = -1.0;
const MAX_INITIAL_TIMESTAMP: f32 = 1.0;

/// OpenAI `ApplyTimestampRules`.
#[derive(Clone, Debug)]
pub(crate) struct TimestampRules {
    pub eot: u32,
    pub no_timestamps: u32,
    pub timestamp_begin: u32,
    /// `round(max_initial_timestamp / time_precision)` = 50 for 1.0 s.
    pub max_initial_timestamp_index: Option<u32>,
}

impl LogitsProcessor for TimestampRules {
    fn process(&mut self, logits: &mut [f32], ctx: &StepContext<'_>) {
        let (tb, eot, n) = (self.timestamp_begin as usize, self.eot as usize, logits.len());
        mask_tokens(logits, &[self.no_timestamps]);
        let sampled = ctx.sampled();
        let is_ts = |t: u32| t >= self.timestamp_begin;
        let last_was_timestamp = sampled.last().is_some_and(|&t| is_ts(t));
        let penultimate_was_timestamp = sampled.len() < 2 || is_ts(sampled[sampled.len() - 2]);
        if last_was_timestamp {
            // a pair of timestamps: has to be text; a single one: cannot be text
            if penultimate_was_timestamp { mask_range(logits, tb, n) } else { mask_range(logits, 0, eot) }
        }
        if let Some(&last_ts) = sampled.iter().rev().find(|&&t| is_ts(t)) {
            // timestamps do not decrease; force a nonzero segment length
            let last = if last_was_timestamp && !penultimate_was_timestamp { last_ts as usize } else { last_ts as usize + 1 };
            mask_range(logits, tb, last);
        }
        if sampled.is_empty() {
            mask_range(logits, 0, tb);
            if let Some(max_idx) = self.max_initial_timestamp_index {
                mask_range(logits, tb + max_idx as usize + 1, n);
            }
        }
        // timestamps win when their total probability beats any single text token
        if tb < n {
            let lp = log_softmax(logits);
            if logsumexp(&lp[tb..]) > lp[..tb].iter().copied().fold(f32::NEG_INFINITY, f32::max) {
                mask_range(logits, 0, tb);
            }
        }
    }
}

/// Mel windows: the HF single-window extractor for the first window of audio <= 30 s (what
/// the reference uses), OpenAI's global long-form spectrogram otherwise.
struct MelSource<'a> {
    model: &'a Whisper,
    samples: &'a [f32],
    long: Option<LongMel>,
}

impl MelSource<'_> {
    fn window(&mut self, seek: usize, size: usize) -> Vec<f32> {
        if seek == 0 && self.samples.len() <= CHUNK_SAMPLES {
            return self.model.log_mel(self.samples);
        }
        let (samples, mel) = (self.samples, &self.model.mel);
        self.long.get_or_insert_with(|| LongMel::new(samples, mel)).window(seek, size)
    }
}

impl Whisper {
    /// The language token to decode with (`None` for English-only models).
    fn language_token(&self, lang: Option<&str>, first: impl FnOnce() -> Result<Option<super::CrossKv>>) -> Result<(Option<u32>, Option<super::CrossKv>)> {
        let cfg = &self.cfg;
        match lang {
            Some(lang) if cfg.is_multilingual() => {
                let id = cfg.language_token(lang).ok_or_else(|| {
                    let known: Vec<&str> = cfg.languages.iter().map(|(c, _)| c.as_str()).collect();
                    Error::new("invalid_argument", format!("unknown language '{lang}'; known: {}", known.join(", ")))
                })?;
                Ok((Some(id), None))
            }
            Some(lang) => {
                let l = super::config::strip_lang(lang.trim()).to_ascii_lowercase();
                if l != "en" && l != "english" {
                    return Err(Error::new("invalid_argument", format!("this model is English-only; cannot use language '{lang}'")));
                }
                Ok((None, None))
            }
            None if cfg.is_multilingual() => match first()? {
                // OpenAI `detect_language`: logits after <|startoftranscript|>, language tokens only
                Some(memory) => {
                    let logits = seq2seq::next_logits(self, &memory, &[cfg.special.sot])?;
                    let best = cfg.languages.iter().filter_map(|&(_, id)| logits.get(id as usize).map(|&l| (id, l))).fold(
                        None::<(u32, f32)>,
                        |acc, (id, l)| match acc {
                            Some((_, bl)) if bl >= l => acc,
                            _ => Some((id, l)),
                        },
                    );
                    Ok((best.map(|(id, _)| id), Some(memory)))
                }
                None => Ok((None, None)),
            },
            None => Ok((None, None)),
        }
    }
}

pub(crate) fn transcribe(model: &Whisper, samples: &[f32], opts: &TranscribeOptions) -> Result<Transcript> {
    let (cfg, sp) = (&model.cfg, &model.cfg.special);
    let content_frames = mel::content_frames(samples.len());
    let mut mels = MelSource { model, samples, long: None };
    let (language_token, mut first) = model.language_token(opts.language.as_deref(), || {
        if content_frames == 0 {
            return Ok(None);
        }
        let mel = mels.window(0, content_frames.min(N_FRAMES));
        Ok(Some(seq2seq::Seq2Seq::encode(model, &mel[..])?))
    })?;
    let language = language_token.and_then(|t| cfg.language_code(t)).unwrap_or("en").to_string();
    if opts.task == Task::Translate && !cfg.is_multilingual() {
        return Err(Error::new("invalid_argument", "English-only models cannot translate"));
    }

    let mut sot_sequence = vec![sp.sot];
    if let Some(lang) = language_token {
        sot_sequence.extend([lang, if opts.task == Task::Translate { sp.translate } else { sp.transcribe }]);
    }
    if !opts.timestamps {
        sot_sequence.push(sp.no_timestamps);
    }
    let (time_precision, input_stride) = (cfg.time_precision(), cfg.input_stride());
    let max_initial = Some((MAX_INITIAL_TIMESTAMP / time_precision).round() as u32);
    let rules = TimestampRules { eot: sp.eot, no_timestamps: sp.no_timestamps, timestamp_begin: sp.timestamp_begin, max_initial_timestamp_index: max_initial };
    let max_prompt = model.decoder.n_ctx() / 2 - 1;
    let tb = sp.timestamp_begin;

    let (mut seek, mut all_tokens, mut segments) = (0usize, Vec::<u32>::new(), Vec::<Segment>::new());
    while seek < content_frames {
        let time_offset = (seek * mel::HOP_LENGTH) as f32 / mel::SAMPLE_RATE as f32;
        let segment_size = N_FRAMES.min(content_frames - seek);
        let segment_duration = (segment_size * mel::HOP_LENGTH) as f32 / mel::SAMPLE_RATE as f32;
        let memory = match first.take() {
            Some(m) if seek == 0 => m,
            _ => seq2seq::Seq2Seq::encode(model, &mels.window(seek, segment_size)[..])?,
        };
        // prompt: previous text (temperature is always 0, so it is never reset)
        let mut prefix: Vec<u32> = Vec::new();
        if !all_tokens.is_empty() {
            prefix.push(sp.prev);
            prefix.extend_from_slice(&all_tokens[all_tokens.len().saturating_sub(max_prompt)..]);
        }
        let sot_index = prefix.len();
        prefix.extend_from_slice(&sot_sequence);
        let gen_opts = GenerateOptions {
            prefix,
            eos: Some(sp.eot),
            max_new_tokens: model.decoder.n_ctx() / 2,
            inspect_rows: if sp.no_speech.is_some() { vec![sot_index] } else { Vec::new() },
        };
        let mut blank = SuppressAtBegin(model.suppress_blank.clone());
        let mut suppress = SuppressTokens(model.suppress.clone());
        let mut ts = rules.clone();
        let mut procs: Vec<&mut dyn LogitsProcessor> = vec![&mut blank, &mut suppress];
        if opts.timestamps {
            procs.push(&mut ts);
        }
        let g = seq2seq::generate(model, &memory, &gen_opts, &mut procs)?;
        let no_speech_prob = match (sp.no_speech, g.inspected.first()) {
            (Some(ns), Some(row)) => log_softmax(row).get(ns as usize).map_or(f32::NAN, |v| v.exp()),
            _ => f32::NAN,
        };
        if crate::debug_enabled() {
            eprintln!("[lapid-hub] window @{time_offset:.2}s: {} tokens", g.tokens.len());
        }
        // no voice activity
        if no_speech_prob > NO_SPEECH_THRESHOLD && g.avg_logprob() <= LOGPROB_THRESHOLD {
            seek += segment_size;
            continue;
        }

        let (previous_seek, tokens) = (seek, &g.tokens);
        let is_ts: Vec<bool> = tokens.iter().map(|&t| t >= tb).collect();
        let n = tokens.len();
        let single_timestamp_ending = n >= 2 && !is_ts[n - 2] && is_ts[n - 1];
        let consecutive: Vec<usize> = (0..n.saturating_sub(1)).filter(|&i| is_ts[i] && is_ts[i + 1]).map(|i| i + 1).collect();
        let new_segment = |start: f32, end: f32, toks: &[u32]| {
            let text: Vec<u32> = toks.iter().copied().filter(|&t| t < sp.eot).collect();
            Segment { start, end, text: model.tokenizer.decode(&text, true).trim().to_string(), tokens: toks.to_vec() }
        };
        let mut current = Vec::new();
        if !consecutive.is_empty() {
            let mut slices = consecutive;
            if single_timestamp_ending {
                slices.push(n);
            }
            let mut last_slice = 0usize;
            for &cur in &slices {
                let sliced = &tokens[last_slice..cur];
                if let (Some(&a), Some(&b)) = (sliced.first(), sliced.last()) {
                    let (s, e) = (a.saturating_sub(tb) as f32, b.saturating_sub(tb) as f32);
                    current.push(new_segment(time_offset + s * time_precision, time_offset + e * time_precision, sliced));
                }
                last_slice = cur;
            }
            if single_timestamp_ending {
                // no speech after the last timestamp
                seek += segment_size;
            } else {
                // skip the unfinished segment: seek to the last timestamp
                seek += tokens[last_slice - 1].saturating_sub(tb) as usize * input_stride;
            }
        } else {
            let mut duration = segment_duration;
            if let Some(&last_ts) = tokens.iter().rev().find(|&&t| t >= tb)
                && last_ts != tb
            {
                duration = (last_ts - tb) as f32 * time_precision;
            }
            current.push(new_segment(time_offset, time_offset + duration, tokens));
            seek += segment_size;
        }
        if seek <= previous_seek {
            seek = previous_seek + segment_size;
        }
        // instantaneous or empty segments carry no text
        for seg in current.into_iter().filter(|s| s.start != s.end && !s.text.is_empty()) {
            all_tokens.extend_from_slice(&seg.tokens);
            segments.push(seg);
        }
    }
    Ok(Transcript { text: model.tokenizer.decode(&all_tokens, true).trim().to_string(), language, segments })
}

#[cfg(test)]
mod tests {
    use super::*;

    // toy vocab: text 0..5, eot 5, specials 6..10, no_timestamps 9, timestamps 10..20
    fn apply(tokens: &[u32], logits: &mut [f32]) {
        let mut r = TimestampRules { eot: 5, no_timestamps: 9, timestamp_begin: 10, max_initial_timestamp_index: Some(3) };
        r.process(logits, &StepContext { tokens, sample_begin: 3 });
    }

    fn allowed(l: &[f32]) -> Vec<usize> {
        l.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, _)| i).collect()
    }

    #[test]
    fn timestamp_rules() {
        let mut l = vec![0.0f32; 20];
        apply(&[6, 7, 8], &mut l);
        assert_eq!(allowed(&l), vec![10, 11, 12, 13]); // early timestamp first
        let mut l = vec![0.0f32; 20];
        apply(&[6, 7, 8, 10, 1, 2, 12], &mut l);
        let a = allowed(&l); // a single timestamp after text: timestamps / eot, not decreasing
        assert!(a.iter().all(|&i| i >= 5 && i != 9) && a.contains(&12) && !a.contains(&11));
        let mut l = vec![0.0f32; 20];
        apply(&[6, 7, 8, 10, 1, 12, 12], &mut l);
        assert!(allowed(&l).iter().all(|&i| i < 10)); // after a pair: no timestamps
        let mut l = vec![0.0f32; 20];
        l[1] = 5.0;
        apply(&[6, 7, 8, 10, 1, 14, 15, 2], &mut l);
        let a = allowed(&l);
        assert!(a.contains(&1) && !a.contains(&15) && a.contains(&16));
        // timestamp mass beats the best text token
        let mut l = vec![f32::NEG_INFINITY; 20];
        l[0] = 2.0;
        l[10..].fill(1.0);
        apply(&[6, 7, 8, 10, 1], &mut l);
        assert!(!l[0].is_finite() && l[16].is_finite());
    }
}
