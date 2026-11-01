//! Streaming synthesis: one graph evaluation per codec frame runs the backbone step, the
//! flow head and the Mimi decoder together, so every 80 ms frame of PCM is available as
//! soon as its latent is sampled.

use std::collections::VecDeque;
use std::time::Instant;

use lapid::{DType, Error, Result, Tensor};
use lapid_nn::sampling::Rng;

use super::Phonon;
use lapid_nn::KvCache;
use super::mimi::MimiState;

/// Where the flow sampler's Gaussian noise comes from.
#[derive(Clone)]
pub(crate) enum Noise {
    /// Seeded N(0, 1) (Box–Muller over xoshiro256**), scaled by sqrt(temperature).
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

/// One text chunk to synthesize.
pub(crate) struct TextChunk {
    pub ids: Vec<u32>,
    pub frames_after_eos: usize,
    pub max_frames: usize,
}

/// Generation state of the chunk being spoken.
struct ChunkState {
    chunk: TextChunk,
    cache: KvCache,
    /// Normalized latent of the previous frame [1, 1, ldim] (None before the prefill).
    prev: Option<Tensor>,
    frames: usize,
    /// Frames left after end of speech was detected.
    remaining: Option<usize>,
}

/// Timings of a stream (milliseconds since the stream was created).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct StreamStats {
    pub first_audio_ms: Option<f64>,
    pub frames: usize,
    pub samples: usize,
    pub elapsed_ms: f64,
}

/// Iterator of PCM chunks (mono f32 at [`Phonon::sample_rate`]); see [`Phonon::speak_stream`].
pub struct SpeechStream<'a> {
    model: &'a Phonon,
    chunks: VecDeque<TextChunk>,
    cur: Option<ChunkState>,
    mimi: MimiState,
    voice: Option<Tensor>,
    cond: Tensor,
    time_conds: Vec<Tensor>,
    noise: Noise,
    noise_std: f32,
    eos_threshold: f32,
    min_frames_before_eos: usize,
    frames_per_chunk: usize,
    fade_samples: usize,
    emitted: usize,
    started: Instant,
    stats: StreamStats,
    done: bool,
    /// Normalized latents and EOS logits of every frame (for verification).
    pub(crate) record: Option<(Vec<f32>, Vec<f32>)>,
}

/// Settings resolved from [`super::SpeakOptions`] and the model config.
pub(crate) struct StreamSetup {
    pub chunks: Vec<TextChunk>,
    pub voice: Option<Tensor>,
    pub cond: Tensor,
    pub time_conds: Vec<Tensor>,
    pub noise: Noise,
    pub noise_std: f32,
    pub eos_threshold: f32,
    pub min_frames_before_eos: usize,
    pub frames_per_chunk: usize,
    pub fade_samples: usize,
}

impl<'a> SpeechStream<'a> {
    pub(crate) fn new(model: &'a Phonon, s: StreamSetup) -> SpeechStream<'a> {
        SpeechStream {
            model,
            chunks: s.chunks.into(),
            cur: None,
            mimi: model.mimi.new_state(),
            voice: s.voice,
            cond: s.cond,
            time_conds: s.time_conds,
            noise: s.noise,
            noise_std: s.noise_std,
            eos_threshold: s.eos_threshold,
            min_frames_before_eos: s.min_frames_before_eos,
            frames_per_chunk: s.frames_per_chunk.max(1),
            fade_samples: s.fade_samples,
            emitted: 0,
            started: Instant::now(),
            stats: StreamStats::default(),
            done: false,
            record: None,
        }
    }

    pub fn stats(&self) -> StreamStats {
        let mut s = self.stats.clone();
        s.elapsed_ms = self.started.elapsed().as_secs_f64() * 1e3;
        s
    }

    /// One frame: Ok(None) when every chunk is finished.
    fn frame(&mut self) -> Result<Option<Vec<f32>>> {
        if self.cur.is_none() {
            match self.chunks.pop_front() {
                Some(chunk) => {
                    self.cur = Some(ChunkState {
                        chunk,
                        cache: self.model.lm.new_cache(),
                        prev: None,
                        frames: 0,
                        remaining: None,
                    })
                }
                None => return Ok(None),
            }
        }
        let lm = &self.model.lm;
        let st = self.cur.as_mut().ok_or_else(|| Error::new("internal", "no chunk state"))?;
        let ldim = lm.latent_dim;
        let x = match &st.prev {
            None => lm.prompt_embeds(self.voice.as_ref(), &st.chunk.ids, &self.cond)?,
            Some(prev) => lm.latent_embeds(prev, &self.cond)?,
        };
        let built = (|| -> Result<(Tensor, Tensor, Tensor)> {
            let step = lm.step(&x, &mut st.cache, None)?;
            let noise = Tensor::from_f32(&self.noise.draw(ldim, self.noise_std), &[1, ldim])?;
            let latent = lm.flow.sample(&step.hidden, &noise, &self.time_conds)?;
            let audio_latent = lm.denormalize(&latent)?.reshape(&[1, 1, ldim])?;
            let pcm = self.model.mimi.decode(&audio_latent, &mut self.mimi, None)?;
            Ok((latent, step.eos, pcm))
        })();
        let (latent, eos, pcm) = match built {
            Ok(v) => v,
            Err(e) => {
                st.cache.discard();
                self.mimi.discard();
                return Err(e);
            }
        };
        let n_lm = st.cache.staged().len();
        let mut outs: Vec<&Tensor> = vec![&latent, &eos, &pcm];
        outs.extend(st.cache.staged());
        outs.extend(self.mimi.staged());
        let mut r = match self.model.device.eval(&outs) {
            Ok(r) => r,
            Err(e) => {
                st.cache.discard();
                self.mimi.discard();
                return Err(e);
            }
        };
        let mimi_states = r.split_off(3 + n_lm);
        let lm_states = r.split_off(3);
        st.cache.commit(lm_states)?;
        self.mimi.commit(mimi_states)?;
        let dev = &self.model.device;
        let eos_logit = dev.to_vec_f32(&r[1])?.first().copied().unwrap_or(f32::NEG_INFINITY);
        let mut samples = dev.to_vec_f32(&r[2])?;
        if let Some((lat, eo)) = self.record.as_mut() {
            lat.extend(dev.to_vec_f32(&r[0])?);
            eo.push(eos_logit);
        }
        st.prev = Some(r[0].reshape(&[1, 1, ldim])?.cast(DType::F32)?);

        // HF PocketTTSEosThresholdCriteria: the crossing frame and `frames_after_eos` more.
        let can_end = st.frames >= self.min_frames_before_eos;
        st.frames += 1;
        if st.remaining.is_none() && can_end && eos_logit > self.eos_threshold {
            st.remaining = Some(st.chunk.frames_after_eos);
        }
        let finished = match st.remaining {
            Some(0) => true,
            Some(n) => {
                st.remaining = Some(n - 1);
                false
            }
            None => false,
        } || st.frames >= st.chunk.max_frames;
        if finished {
            self.cur = None;
        }

        // fade-in of the utterance start
        for (i, s) in samples.iter_mut().enumerate() {
            let pos = self.emitted + i;
            if pos >= self.fade_samples {
                break;
            }
            *s *= pos as f32 / (self.fade_samples.max(2) - 1) as f32;
        }
        self.emitted += samples.len();
        self.stats.frames += 1;
        self.stats.samples += samples.len();
        if self.stats.first_audio_ms.is_none() {
            self.stats.first_audio_ms = Some(self.started.elapsed().as_secs_f64() * 1e3);
        }
        Ok(Some(samples))
    }
}

impl Iterator for SpeechStream<'_> {
    type Item = Result<Vec<f32>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let mut out = Vec::new();
        for _ in 0..self.frames_per_chunk {
            match self.frame() {
                Ok(Some(pcm)) => out.extend(pcm),
                Ok(None) => {
                    self.done = true;
                    break;
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            }
        }
        if out.is_empty() { None } else { Some(Ok(out)) }
    }
}
