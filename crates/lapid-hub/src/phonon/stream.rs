//! Phonon on the shared speech pipeline (`runtime::tts`): one graph evaluation per codec
//! frame runs the backbone step, the flow head and the Mimi decoder together, so every 80 ms
//! frame of PCM is available as soon as its latent is sampled. Chunking, end of speech,
//! fade-in, streaming and cancellation are [`crate::runtime::tts::SpeechStream`].

use lapid::{DType, Error, Result, Tensor, bail};
use lapid_nn::KvCache;
use lapid_nn::sampling::Rng;

use super::mimi::MimiState;
use super::{Phonon, SpeakOptions, text};
use crate::runtime::tts::{Frame, Segment, SpeechModel, StreamOptions};

/// Where the flow sampler's Gaussian noise comes from.
#[derive(Clone)]
pub(crate) enum Noise {
    /// Seeded N(0, 1) (Box-Muller over xoshiro256**), scaled by sqrt(temperature).
    Seeded(Rng),
    /// Pre-drawn noise, `latent_dim` values per frame, used as is (already scaled); frames
    /// past the end get zeros. Verification feeds the reference's noise this way.
    Fixed(Vec<f32>, usize),
}

impl Noise {
    fn draw(&mut self, n: usize, std: f32) -> Vec<f32> {
        match self {
            Noise::Seeded(rng) => {
                let mut out = Vec::with_capacity(n);
                while out.len() < n {
                    let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
                    let u2 = rng.next_f64();
                    let r = (-2.0 * u1.ln()).sqrt();
                    let a = std::f64::consts::TAU * u2;
                    out.push((r * a.cos()) as f32 * std);
                    if out.len() < n {
                        out.push((r * a.sin()) as f32 * std);
                    }
                }
                out
            }
            Noise::Fixed(v, pos) => {
                let out = (0..n).map(|i| v.get(*pos + i).copied().unwrap_or(0.0)).collect();
                *pos += n;
                out
            }
        }
    }
}

/// Utterance state: codec state, voice, conditioning, noise.
pub struct Utterance {
    mimi: MimiState,
    voice: Option<Tensor>,
    cond: Tensor,
    time_conds: Vec<Tensor>,
    noise: Noise,
    noise_std: f32,
    max_chunk_tokens: usize,
    max_frames: Option<usize>,
    /// Normalized latents and EOS logits of every frame (for verification).
    pub(crate) record: Option<(Vec<f32>, Vec<f32>)>,
}

/// Generation state of the text chunk being spoken.
pub struct ChunkState {
    cache: KvCache,
    /// Normalized latent of the previous frame [1, 1, ldim] (None before the prefill).
    prev: Option<Tensor>,
}

impl Phonon {
    /// Utterance state and stream options for `opts` with `noise`.
    pub(crate) fn utterance(&self, opts: &SpeakOptions, noise: Noise) -> Result<(Utterance, StreamOptions)> {
        let g = &self.cfg.generation;
        let name = opts.voice.as_deref().unwrap_or(&self.cfg.assets.default_voice);
        let voice = self.voices.get(name).cloned().ok_or_else(|| {
            Error::new("not_found", format!("unknown voice {name:?} (have: {})", self.voices().join(", ")))
        })?;
        let steps = opts.num_flow_steps.unwrap_or(g.num_flow_steps).max(1);
        let c = &self.cfg.conditions;
        let mut lazy = vec![self.lm.condition(c.num_speakers, opts.duration_delta.unwrap_or(c.duration_delta), c)?];
        for i in 0..steps {
            lazy.push(self.lm.flow.time_condition(i as f32 / steps as f32, (i + 1) as f32 / steps as f32)?);
        }
        let mut realized = self.device.eval(&lazy.iter().collect::<Vec<_>>())?;
        let time_conds = realized.split_off(1);
        let cond = realized.pop().ok_or_else(|| Error::new("internal", "eval returned nothing"))?;
        let utt = Utterance {
            mimi: self.mimi.new_state(),
            voice: Some(voice),
            cond,
            time_conds,
            noise,
            noise_std: opts.temperature.unwrap_or(g.temperature).max(0.0).sqrt(),
            max_chunk_tokens: opts.max_chunk_tokens.unwrap_or(50).max(1),
            max_frames: opts.max_frames,
            record: None,
        };
        let so = StreamOptions {
            frames_per_chunk: opts.frames_per_chunk.unwrap_or(1).max(1),
            eos_threshold: Some(opts.eos_threshold.unwrap_or(g.eos_threshold)),
            min_frames_before_eos: g.min_frames_before_eos,
            fade_in_samples: (g.fade_in_duration * self.cfg.mimi.sample_rate as f32) as usize,
            cancel: None,
        };
        Ok((utt, so))
    }
}

impl SpeechModel for Phonon {
    type Input = Vec<u32>;
    type Stream = Utterance;
    type SegmentState = ChunkState;
    type Options = SpeakOptions;

    fn sample_rate(&self) -> u32 {
        self.cfg.mimi.sample_rate
    }

    fn new_stream(&self, opts: &SpeakOptions) -> Result<(Utterance, StreamOptions)> {
        let seed = opts.seed.unwrap_or_else(|| {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)
        });
        self.utterance(opts, Noise::Seeded(Rng::new(seed)))
    }

    /// Prepared text split at sentence ends into chunks of at most `max_chunk_tokens`, each with
    /// a frame budget of `ceil((tokens / 3 + 2) * frame_rate)`.
    fn segments(&self, text: &str, utt: &Utterance) -> Result<Vec<Segment<Vec<u32>>>> {
        let g = &self.cfg.generation;
        let prepared = text::prepare(text);
        if prepared.is_empty() {
            bail!("invalid_argument", "nothing to say: the text is empty");
        }
        let mut out = Vec::new();
        for piece in text::split(&self.tokenizer, &prepared, utt.max_chunk_tokens)? {
            let ids = text::encode(&self.tokenizer, &piece)?;
            let short = text::word_count(&piece) <= g.short_text_max_words;
            let budget = ((ids.len() as f64 / 3.0 + 2.0) * self.cfg.mimi.frame_rate as f64).ceil() as usize;
            out.push(Segment {
                input: ids,
                max_frames: utt.max_frames.unwrap_or(budget).max(1),
                frames_after_eos: if short { g.frames_after_eos_short_text } else { g.frames_after_eos },
            });
        }
        Ok(out)
    }

    fn begin_segment(&self, _: &mut Utterance, _: &Vec<u32>) -> Result<ChunkState> {
        Ok(ChunkState { cache: self.lm.new_cache(), prev: None })
    }

    fn step(&self, u: &mut Utterance, ids: &Vec<u32>, st: &mut ChunkState, _frame: usize) -> Result<Frame> {
        let lm = &self.lm;
        let ldim = lm.latent_dim;
        let built = (|| -> Result<[Tensor; 3]> {
            let x = match &st.prev {
                None => lm.prompt_embeds(u.voice.as_ref(), ids, &u.cond)?,
                Some(prev) => lm.latent_embeds(prev, &u.cond)?,
            };
            let step = lm.step(&x, &mut st.cache, None)?;
            let noise = Tensor::from_f32(&u.noise.draw(ldim, u.noise_std), &[1, ldim])?;
            let latent = lm.flow.sample(&step.hidden, &noise, &u.time_conds)?;
            let audio_latent = lm.denormalize(&latent)?.reshape(&[1, 1, ldim])?;
            let pcm = self.mimi.decode(&audio_latent, &mut u.mimi, None)?;
            Ok([latent, step.eos, pcm])
        })();
        let r = built.and_then(|outs| {
            let mut all: Vec<&Tensor> = outs.iter().collect();
            all.extend(st.cache.staged());
            all.extend(u.mimi.staged());
            self.device.eval(&all)
        });
        let mut r = match r {
            Ok(r) => r,
            Err(e) => {
                st.cache.discard();
                u.mimi.discard();
                return Err(e);
            }
        };
        let n_lm = st.cache.staged().len();
        let mimi_states = r.split_off(3 + n_lm);
        st.cache.commit(r.split_off(3))?;
        u.mimi.commit(mimi_states)?;
        let dev = &self.device;
        let eos = dev.to_vec_f32(&r[1])?.first().copied().unwrap_or(f32::NEG_INFINITY);
        if let Some((lat, eo)) = u.record.as_mut() {
            lat.extend(dev.to_vec_f32(&r[0])?);
            eo.push(eos);
        }
        st.prev = Some(r[0].reshape(&[1, 1, ldim])?.cast(DType::F32)?);
        Ok(Frame { pcm: dev.to_vec_f32(&r[2])?, eos_logit: Some(eos), end: false })
    }
}
