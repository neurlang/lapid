//! Long-form transcription: OpenAI `transcribe()` seek loop over 30 s windows with
//! timestamp-based seeking, segment building and prompt conditioning on previous text.

/// Timing clock: `std::time::Instant` natively; reads zero on wasm32 (no clock there).
#[derive(Clone, Copy)]
struct Instant(#[cfg(not(target_arch = "wasm32"))] std::time::Instant);

impl Instant {
    fn now() -> Instant {
        Instant(#[cfg(not(target_arch = "wasm32"))] std::time::Instant::now())
    }

    fn elapsed(&self) -> std::time::Duration {
        #[cfg(not(target_arch = "wasm32"))]
        return self.0.elapsed();
        #[cfg(target_arch = "wasm32")]
        std::time::Duration::ZERO
    }
}

use lapid::{Error, Result, Tensor};

use crate::whisper::decode::{LogitFilters, TimestampRules};
use crate::whisper::mel::{self, CHUNK_SAMPLES, LongMel, N_FRAMES};
use crate::whisper::{Segment, Task, TranscribeOptions, Transcript, Whisper};

/// OpenAI defaults.
const NO_SPEECH_THRESHOLD: f32 = 0.6;
const LOGPROB_THRESHOLD: f32 = -1.0;
const MAX_INITIAL_TIMESTAMP: f32 = 1.0;

/// Mel windows: the HF single-window extractor for the first window of ≤ 30 s audio (this
/// is what the reference uses), OpenAI's global long-form spectrogram otherwise.
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

pub(crate) fn transcribe(model: &Whisper, samples: &[f32], opts: &TranscribeOptions) -> Result<Transcript> {
    let cfg = &model.cfg;
    let sp = &cfg.special;
    let debug = crate::debug_enabled();
    let content_frames = mel::content_frames(samples.len());
    let mut mels = MelSource { model, samples, long: None };

    // ---- language ----
    let mut first_encoding: Option<Tensor> = None;
    let language_token: Option<u32> = match &opts.language {
        Some(lang) if cfg.is_multilingual() => Some(cfg.language_token(lang).ok_or_else(|| {
            let known: Vec<&str> = cfg.languages.iter().map(|(c, _)| c.as_str()).collect();
            Error::new(
                "invalid_argument",
                format!("unknown language '{lang}'; known: {}", known.join(", ")),
            )
        })?),
        Some(lang) => {
            let l = crate::whisper::model::config::strip_lang(lang.trim()).to_ascii_lowercase();
            if l != "en" && l != "english" {
                return Err(Error::new(
                    "invalid_argument",
                    format!("this model is English-only; cannot use language '{lang}'"),
                ));
            }
            None
        }
        None if cfg.is_multilingual() && content_frames > 0 => {
            let t0 = Instant::now();
            let enc = model.encode(&mels.window(0, content_frames.min(N_FRAMES)))?;
            let cross = model.decoder.cross_kv(&model.device, &enc)?;
            let tok = model.detect_language_token(&cross)?;
            first_encoding = Some(enc);
            if debug {
                let code = tok.and_then(|t| cfg.language_code(t)).unwrap_or("?");
                eprintln!("[lapid-hub] detected language '{code}' in {:.1} ms", ms(t0));
            }
            tok
        }
        None => None,
    };
    let language = language_token
        .and_then(|t| cfg.language_code(t))
        .unwrap_or("en")
        .to_string();
    if opts.task == Task::Translate && !cfg.is_multilingual() {
        return Err(Error::new("invalid_argument", "English-only models cannot translate"));
    }

    // ---- prompt pieces ----
    let mut sot_sequence = vec![sp.sot];
    if let Some(lang) = language_token {
        sot_sequence.push(lang);
        sot_sequence.push(match opts.task {
            Task::Transcribe => sp.transcribe,
            Task::Translate => sp.translate,
        });
    }
    if !opts.timestamps {
        sot_sequence.push(sp.no_timestamps);
    }
    let time_precision = cfg.time_precision();
    let input_stride = cfg.input_stride();
    let ts_rules = opts.timestamps.then(|| TimestampRules {
        eot: sp.eot,
        no_timestamps: sp.no_timestamps,
        timestamp_begin: sp.timestamp_begin,
        max_initial_timestamp_index: Some((MAX_INITIAL_TIMESTAMP / time_precision).round() as u32),
    });
    let max_prompt = model.decoder.n_ctx() / 2 - 1;

    // ---- seek loop ----
    let mut seek = 0usize;
    let mut all_tokens: Vec<u32> = Vec::new();
    let mut segments: Vec<Segment> = Vec::new();
    let prompt_reset_since = 0usize; // temperature is always 0: never reset

    while seek < content_frames {
        let t0 = Instant::now();
        let time_offset = (seek * mel::HOP_LENGTH) as f32 / mel::SAMPLE_RATE as f32;
        let segment_size = N_FRAMES.min(content_frames - seek);
        let segment_duration = (segment_size * mel::HOP_LENGTH) as f32 / mel::SAMPLE_RATE as f32;

        let encoded = match first_encoding.take() {
            Some(enc) if seek == 0 => enc,
            _ => model.encode(&mels.window(seek, segment_size))?,
        };
        let t_enc = ms(t0);
        let cross = model.decoder.cross_kv(&model.device, &encoded)?;

        let prompt = &all_tokens[prompt_reset_since..];
        let mut initial: Vec<u32> = Vec::with_capacity(prompt.len() + sot_sequence.len() + 1);
        if !prompt.is_empty() {
            initial.push(sp.prev);
            initial.extend_from_slice(&prompt[prompt.len().saturating_sub(max_prompt)..]);
        }
        let sot_index = initial.len();
        initial.extend_from_slice(&sot_sequence);
        let filters = LogitFilters {
            sample_begin: initial.len(),
            suppress_blank: model.suppress_blank.clone(),
            suppress: model.suppress.clone(),
            timestamps: ts_rules.clone(),
        };

        let result = model.decode_window(&cross, &initial, sot_index, &filters)?;
        if debug {
            eprintln!(
                "[lapid-hub] window @{time_offset:.2}s: encode {t_enc:.1} ms, total {:.1} ms, {} tokens",
                ms(t0),
                result.tokens.len()
            );
        }

        // no voice activity check
        let confident = result.avg_logprob > LOGPROB_THRESHOLD;
        if result.no_speech_prob > NO_SPEECH_THRESHOLD && !confident {
            seek += segment_size;
            continue;
        }

        let previous_seek = seek;
        let tokens = &result.tokens;
        let tb = sp.timestamp_begin;
        let is_ts: Vec<bool> = tokens.iter().map(|&t| t >= tb).collect();
        let n = tokens.len();
        let single_timestamp_ending = n >= 2 && !is_ts[n - 2] && is_ts[n - 1];
        let consecutive: Vec<usize> =
            (0..n.saturating_sub(1)).filter(|&i| is_ts[i] && is_ts[i + 1]).map(|i| i + 1).collect();

        let mut current: Vec<Segment> = Vec::new();
        let new_segment = |start: f32, end: f32, toks: &[u32]| {
            let text_tokens: Vec<u32> = toks.iter().copied().filter(|&t| t < sp.eot).collect();
            Segment {
                start,
                end,
                text: model.tokenizer.decode(&text_tokens, true).trim().to_string(),
                tokens: toks.to_vec(),
            }
        };

        if !consecutive.is_empty() {
            let mut slices = consecutive;
            if single_timestamp_ending {
                slices.push(n);
            }
            let mut last_slice = 0usize;
            for &current_slice in &slices {
                let sliced = &tokens[last_slice..current_slice];
                if let (Some(&first), Some(&last)) = (sliced.first(), sliced.last()) {
                    let start_pos = first.saturating_sub(tb) as f32;
                    let end_pos = last.saturating_sub(tb) as f32;
                    current.push(new_segment(
                        time_offset + start_pos * time_precision,
                        time_offset + end_pos * time_precision,
                        sliced,
                    ));
                }
                last_slice = current_slice;
            }
            if single_timestamp_ending {
                // single timestamp at the end means no speech after the last timestamp
                seek += segment_size;
            } else {
                // ignore the unfinished segment and seek to the last timestamp
                let last_timestamp_pos = tokens[last_slice - 1].saturating_sub(tb) as usize;
                seek += last_timestamp_pos * input_stride;
            }
        } else {
            let mut duration = segment_duration;
            if let Some(&last_ts) = tokens.iter().rev().find(|&&t| t >= tb)
                && last_ts != tb {
                    // no consecutive timestamps but it has a timestamp; use the last one
                    duration = (last_ts - tb) as f32 * time_precision;
                }
            current.push(new_segment(time_offset, time_offset + duration, tokens));
            seek += segment_size;
        }
        if seek <= previous_seek {
            // never loop on the same window
            seek = previous_seek + segment_size;
        }

        // instantaneous or empty segments carry no text
        for seg in current {
            if seg.start == seg.end || seg.text.is_empty() {
                continue;
            }
            all_tokens.extend_from_slice(&seg.tokens);
            segments.push(seg);
        }
    }

    Ok(Transcript {
        text: model.tokenizer.decode(&all_tokens, true).trim().to_string(),
        language,
        segments,
    })
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}
