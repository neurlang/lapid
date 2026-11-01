//! Phonon's part of `lapid verify phonon` (the rest is `runtime::tts_harness`): every stage
//! on the *reference* inputs of `tools/reference/phonon_ref.py`, so errors do not compound:
//! * backbone prefill on the reference voice + ids: `lm.embeds`, `lm.layers.{i}`, `lm.out`;
//! * generation, teacher-forced: frame i is computed from the reference latents of frames
//!   `< i` and the reference noise of frame i: `gen.conditions`, `gen.eos`, `gen.latents`;
//! * codec on the reference codec latents, all frames in one call (`mimi.*` stages) and
//!   frame by frame through the streaming state (`mimi.*.streamed`: every stage's per-frame
//!   outputs concatenated in time, against the same reference stage);
//! * end to end: free-running synthesis on the reference noise (frames, waveform and the
//!   latent trajectory).

use lapid::{Error, Result, Tensor};
use lapid_verify::diff;
use lapid_verify::harness::{Reference, ids};
use serde_json::Value;

use super::stream::Noise;
use super::{Phonon, SpeakOptions};
use crate::runtime::tts::{ReferenceSpeech, speak_from};

/// Free-running latents vs the reference trajectory (relative L2).
pub const LATENT_TOLERANCE: f32 = 0.1;

/// Reference inputs of the per-stage runs.
pub(crate) struct Inputs {
    voice: Vec<f32>,
    ids: Vec<u32>,
    noise: Vec<f32>,
    latents: Vec<f32>,
    audio_latents: Vec<f32>,
    n: usize,
}

fn first(mut v: Vec<Tensor>) -> Result<Tensor> {
    v.drain(..).next().ok_or_else(|| Error::new("internal", "eval returned nothing"))
}

/// All per-stage outputs of one model on the reference inputs, by name.
pub(crate) fn run_model(tts: &Phonon, inp: &Inputs) -> Result<Vec<(String, Vec<f32>)>> {
    let dev = &tts.device;
    let lm = &tts.lm;
    let ldim = lm.latent_dim;
    let dim = tts.cfg.lm.audio_prompt_dim;
    let mut out: Vec<(String, Vec<f32>)> = Vec::new();

    let c = &tts.cfg.conditions;
    let cond = first(dev.eval(&[&lm.condition(c.num_speakers, c.duration_delta, c)?])?)?;
    let steps = tts.cfg.generation.num_flow_steps.max(1);
    let tcs: Vec<Tensor> = (0..steps)
        .map(|i| lm.flow.time_condition(i as f32 / steps as f32, (i + 1) as f32 / steps as f32))
        .collect::<Result<_>>()?;
    let tcs = dev.eval(&tcs.iter().collect::<Vec<_>>())?;
    let voice = dev.upload(&Tensor::from_f32(&inp.voice, &[inp.voice.len() / dim, dim])?)?;

    // ---- backbone: prefill traced, then teacher-forced frames
    let mut cache = lm.new_cache();
    let (mut conds, mut eos, mut lats) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..inp.n {
        let x = if i == 0 {
            lm.prompt_embeds(Some(&voice), &inp.ids, &cond)?
        } else {
            let prev = Tensor::from_f32(&inp.latents[(i - 1) * ldim..i * ldim], &[1, 1, ldim])?;
            lm.latent_embeds(&prev, &cond)?
        };
        let mut traced = Vec::new();
        if i == 0 {
            traced.push(x.named("lm.embeds"));
        }
        let step = lm.step(&x, &mut cache, (i == 0).then_some(&mut traced))?;
        let noise = Tensor::from_f32(&inp.noise[i * ldim..(i + 1) * ldim], &[1, ldim])?;
        let latent = lm.flow.sample(&step.hidden, &noise, &tcs)?;
        let mut outs: Vec<&Tensor> = vec![&step.hidden, &step.eos, &latent];
        outs.extend(traced.iter());
        let n_out = outs.len();
        outs.extend(cache.staged());
        let mut r = dev.eval(&outs)?;
        cache.commit(r.split_off(n_out))?;
        conds.extend(dev.to_vec_f32(&r[0])?);
        eos.extend(dev.to_vec_f32(&r[1])?);
        lats.extend(dev.to_vec_f32(&r[2])?);
        for (t, real) in traced.iter().zip(&r[3..]) {
            out.push((t.name().unwrap_or("lm.unnamed").to_string(), dev.to_vec_f32(real)?));
        }
    }
    out.push(("gen.conditions".into(), conds));
    out.push(("gen.eos".into(), eos));
    out.push(("gen.latents".into(), lats));

    // ---- codec: all frames in one call (stages), then frame by frame (streaming state)
    let al = Tensor::from_f32(&inp.audio_latents, &[1, inp.n, ldim])?;
    let mut state = tts.mimi.new_state();
    let mut traced = Vec::new();
    let audio = tts.mimi.decode(&al, &mut state, Some(&mut traced))?;
    let mut outs: Vec<&Tensor> = traced.iter().collect();
    outs.push(&audio);
    let r = dev.eval(&outs)?;
    for (t, real) in traced.iter().zip(&r) {
        out.push((t.name().unwrap_or("mimi.unnamed").to_string(), dev.to_vec_f32(real)?));
    }
    out.push(("mimi.audio".into(), dev.to_vec_f32(&r[r.len() - 1])?));
    let mut state = tts.mimi.new_state();
    let mut streamed = Vec::new();
    type Frames = Vec<(Vec<usize>, Vec<f32>)>;
    let mut per_stage: Vec<(String, Frames)> = Vec::new();
    for i in 0..inp.n {
        let al = Tensor::from_f32(&inp.audio_latents[i * ldim..(i + 1) * ldim], &[1, 1, ldim])?;
        let mut tr = Vec::new();
        let pcm = tts.mimi.decode(&al, &mut state, Some(&mut tr))?;
        let mut outs = vec![&pcm];
        outs.extend(state.staged());
        let ns = outs.len();
        outs.extend(tr.iter());
        let mut r = dev.eval(&outs)?;
        let trr = r.split_off(ns);
        for (t, real) in tr.iter().zip(&trr) {
            let name = t.name().unwrap_or("").to_string();
            let v = dev.to_vec_f32(real)?;
            let shape = real.shape().to_vec();
            match per_stage.iter_mut().find(|(n, _)| *n == name) {
                Some((_, frames)) => frames.push((shape, v)),
                None => per_stage.push((name, vec![(shape, v)])),
            }
        }
        state.commit(r.split_off(1))?;
        streamed.extend(dev.to_vec_f32(&r[0])?);
    }
    out.push(("mimi.audio.streamed".into(), streamed));
    for (name, frames) in per_stage {
        // concatenate along time: [1, C, l] on the last axis, [1, l, C] (transformer) on axis 1
        let time_major = name == "mimi.transformer";
        let mut full = Vec::new();
        if time_major {
            for (_, v) in &frames {
                full.extend_from_slice(v);
            }
        } else {
            let c = frames[0].0[1];
            for ch in 0..c {
                for (sh, v) in &frames {
                    let l = sh[2];
                    full.extend_from_slice(&v[ch * l..(ch + 1) * l]);
                }
            }
        }
        out.push((format!("{name}.streamed"), full));
    }
    Ok(out)
}

impl Inputs {
    pub(crate) fn from_reference(r: &Reference) -> Result<Inputs> {
        Ok(Inputs {
            voice: r.acts.get("voice")?,
            ids: ids(&r.result, "text_ids")?,
            noise: r.acts.get("gen.noise")?,
            latents: r.acts.get("gen.latents")?,
            audio_latents: r.acts.get("gen.audio_latents")?,
            n: r.result.get("n_frames").and_then(Value::as_u64).unwrap_or(0) as usize,
        })
    }
}

/// Free-running synthesis of the reference text on the reference noise.
pub(crate) fn reference_speech(tts: &Phonon, r: &Reference) -> Result<ReferenceSpeech> {
    let text = r.result.get("text").and_then(Value::as_str).ok_or_else(|| Error::new("reference", "result.json: missing text"))?;
    let temperature = r.result.get("temperature").and_then(Value::as_f64).map(|t| t as f32);
    let inputs = Inputs::from_reference(r)?;
    let opts = SpeakOptions { temperature, ..Default::default() };
    let (mut utt, so) = tts.utterance(&opts, Noise::Fixed(inputs.noise.clone(), 0))?;
    utt.record = Some((Vec::new(), Vec::new()));
    let mut stream = speak_from(tts, text, utt, so)?;
    let mut audio = Vec::new();
    for pcm in stream.by_ref() {
        audio.extend(pcm?);
    }
    let (latents, eos) = stream.state().record.clone().unwrap_or_default();
    let latents_rel = if latents.len() == inputs.latents.len() { Some(diff(&latents, &inputs.latents)?.rel) } else { None };
    Ok(ReferenceSpeech {
        audio,
        frames: eos.len(),
        first_audio_ms: stream.stats().first_audio_ms,
        checks: vec![("latents".into(), latents_rel, LATENT_TOLERANCE)],
    })
}
