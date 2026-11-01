//! `lapid verify phonon`: compare lapid's Phonon with the HF transformers port, stage by
//! stage (reference data from `tools/reference/phonon_ref.py`).
//!
//! Every stage runs on the *reference* inputs so errors do not compound:
//! * tokens: our text preparation + tokenizer vs the reference prompt's text ids;
//! * backbone prefill on the reference voice + ids: `lm.embeds`, `lm.layers.{i}`, `lm.out`;
//! * generation, teacher-forced: frame i is computed from the reference latents of frames
//!   `< i` and the reference noise of frame i: `gen.conditions`, `gen.eos`, `gen.latents`;
//! * codec on the reference codec latents, all frames in one call (`mimi.*` stages) and
//!   frame by frame through the streaming state (`mimi.*.streamed`: every stage's per-frame
//!   outputs concatenated in time, against the same reference stage);
//! * end to end: `speak_stream` free-running on the reference noise — frame count and the
//!   final waveform.
//!
//! Quantized models (int8 / int4) are checked against fake-quant references (the same
//! weights dequantized to f32, CPU; per tensor the closer of f32 / int8 activations), and
//! their loss against PyTorch fp32 is reported separately (as for Whisper).

use std::path::{Path, PathBuf};
use std::time::Instant;

use lapid::{Device, Error, Result, Tensor};
use lapid_convert::{OutDType, default_model_path};
use lapid_verify::timing::ms;
use lapid_verify::{Activations, Comparison, KERNEL_TOLERANCE, Stages, diff, tolerance};
use serde_json::{Value, json};

use super::stream::Noise;
use super::{Phonon, Precision, SpeakOptions};

/// End to end, generation runs free (each frame from our own previous latents) on the
/// reference noise, so per-step rounding differences compound through the autoregression;
/// the waveform is compared through its log-magnitude spectrogram (phase-insensitive) and
/// the latent trajectory. Raw waveform error is reported, not judged.
pub const SPECTRAL_TOLERANCE: f32 = 0.1;
pub const LATENT_TOLERANCE: f32 = 0.1;

/// Relative L2 distance of the log-magnitude spectrograms (n_fft 1024, hop 256).
fn spectral_rel(ours: &[f32], reference: &[f32]) -> Result<f32> {
    let stft = lapid_nn::audio::Stft::new(lapid_nn::audio::StftConfig::new(1024, 256))?;
    let spec = |x: &[f32]| -> Vec<f32> {
        let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        stft.spectrogram(&x64, 1.0).iter().map(|&m| (m + 1e-4).ln() as f32).collect()
    };
    Ok(diff(&spec(ours), &spec(reference))?.rel)
}

pub(crate) fn resolve_model(model: &str, dtype: &str) -> Result<(PathBuf, Option<f64>)> {
    let p = Path::new(model);
    if p.extension().is_some_and(|e| e == "lapid") || p.is_file() {
        return Ok((p.to_path_buf(), None));
    }
    let path = default_model_path(model, OutDType::parse(dtype)?);
    if path.is_file() {
        return Ok((path, None));
    }
    let t = Instant::now();
    super::convert::convert_hf(model, OutDType::parse(dtype)?, &path)?;
    Ok((path, Some(ms(t))))
}

struct Reference {
    text: String,
    text_ids: Vec<u32>,
    n_frames: usize,
    temperature: f32,
}

fn read_reference(dir: &Path) -> Result<Reference> {
    let path = dir.join("result.json");
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| Error::new("io", format!("{}: {e} (run tools/reference/phonon_ref.py)", path.display())))?;
    let v: Value = serde_json::from_str(&raw)?;
    let bad = |k: &str| Error::new("reference", format!("result.json: missing or invalid {k:?}"));
    let text_ids = v
        .get("text_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("text_ids"))?
        .iter()
        .map(|x| x.as_u64().map(|n| n as u32).ok_or_else(|| bad("text_ids")))
        .collect::<Result<_>>()?;
    Ok(Reference {
        text: v.get("text").and_then(Value::as_str).ok_or_else(|| bad("text"))?.to_string(),
        text_ids,
        n_frames: v.get("n_frames").and_then(Value::as_u64).ok_or_else(|| bad("n_frames"))? as usize,
        temperature: v.get("temperature").and_then(Value::as_f64).ok_or_else(|| bad("temperature"))? as f32,
    })
}

/// Reference inputs of the per-stage runs.
struct Inputs {
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
fn run_model(tts: &Phonon, inp: &Inputs) -> Result<Vec<(String, Vec<f32>)>> {
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

fn reference_name(name: &str) -> &str {
    name.strip_suffix(".streamed").unwrap_or(name)
}

fn compare(ours: &[(String, Vec<f32>)], reference: &dyn Fn(&str) -> Result<Vec<f32>>, tol: f32) -> Result<Comparison> {
    let mut c = Comparison::new();
    for (name, v) in ours {
        match reference(reference_name(name)) {
            Ok(r) if r.len() == v.len() => {
                c.compare(name, v, &r, tol);
            }
            Ok(r) => {
                c.layers.push(lapid_verify::LayerReport::failed(name.clone()));
                c.notes.push(format!("{name}: {} values vs reference {}", v.len(), r.len()));
            }
            Err(e) => {
                c.layers.push(lapid_verify::LayerReport::failed(name.clone()));
                c.notes.push(format!("{name}: {}", e.message));
            }
        }
    }
    Ok(c)
}

/// Per tensor, the closer of the two fake-quant references.
fn closest(w: &Comparison, a: &Comparison) -> Comparison {
    let mut m = Comparison::new();
    for (x, y) in w.layers.iter().zip(&a.layers) {
        m.layers.push(if y.rel < x.rel || (x.rel.is_nan() && !y.rel.is_nan()) { y.clone() } else { x.clone() });
    }
    m.notes = w.notes.clone();
    m
}

fn lookup(outs: &[(String, Vec<f32>)]) -> impl Fn(&str) -> Result<Vec<f32>> + '_ {
    move |name| {
        outs.iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| Error::new("missing_tensor", format!("no {name:?}")))
    }
}

/// Full verification as JSON. Check `"ok"`.
pub fn verify(reference_dir: &Path, model: &str, dtype: &str, device: Option<&str>) -> Result<Value> {
    let reference = read_reference(reference_dir)?;
    let acts = Activations::open(&reference_dir.join("activations.safetensors"))?;
    let dev = lapid_verify::open_device(device)?;
    let mut stages = Stages::new();
    let (path, convert_ms) = resolve_model(model, dtype)?;
    if let Some(t) = convert_ms {
        stages.record("convert", t);
    }
    let file_dtype =
        lapid::format::Reader::open(&path)?.metadata.get("dtype").and_then(Value::as_str).unwrap_or(dtype).to_string();
    let quantized = matches!(file_dtype.as_str(), "int8" | "int4");
    let quality_tol = tolerance(&file_dtype)?;
    let tts = stages.time("load", || Phonon::load(&path, &dev))?;

    // ---- tokens
    let (prepared, ids) = tts.tokenize(&reference.text)?;
    let tokens_match = ids == reference.text_ids;

    let n = reference.n_frames;
    let inputs = Inputs {
        voice: acts.get("voice")?,
        ids: reference.text_ids.clone(),
        noise: acts.get("gen.noise")?,
        latents: acts.get("gen.latents")?,
        audio_latents: acts.get("gen.audio_latents")?,
        n,
    };
    let ours = stages.time("run", || run_model(&tts, &inputs))?;
    // The reference has no raw output of the last backbone block (HF exposes it normed).
    let ours: Vec<(String, Vec<f32>)> = ours.into_iter().filter(|(n, _)| acts.has(reference_name(n))).collect();
    let pytorch = |name: &str| acts.get(name);
    let quality = compare(&ours, &pytorch, quality_tol)?;

    let mut kernel = None;
    let mut kernel_json = Value::Null;
    if quantized {
        let cpu = Device::cpu();
        let (fw, fa) = stages.time("fake_quant", || -> Result<_> {
            let w = run_model(&Phonon::load_with(&path, &cpu, Precision::FakeQuantWeights)?, &inputs)?;
            let a = run_model(&Phonon::load_with(&path, &cpu, Precision::FakeQuantActivations)?, &inputs)?;
            Ok((w, a))
        })?;
        let kw = compare(&ours, &lookup(&fw), KERNEL_TOLERANCE)?;
        let ka = compare(&ours, &lookup(&fa), KERNEL_TOLERANCE)?;
        let k = closest(&kw, &ka);
        kernel_json = json!({
            "reference": "fake-quant: same weights dequantized to f32 on cpu; per tensor the closer of f32 and int8 activations",
            "tolerance": KERNEL_TOLERANCE,
            "comparison": k.to_json(),
        });
        kernel = Some(k);
    }

    // ---- end to end on the reference noise
    let opts = SpeakOptions { temperature: Some(reference.temperature), ..Default::default() };
    let t = Instant::now();
    let mut stream = tts.stream_with(&reference.text, &opts, Noise::Fixed(inputs.noise.clone(), 0))?;
    stream.record = Some((Vec::new(), Vec::new()));
    let mut audio = Vec::new();
    for pcm in stream.by_ref() {
        audio.extend(pcm?);
    }
    let first_audio_ms = stream.stats().first_audio_ms;
    stages.record("speak", ms(t));
    let frames = stream.record.as_ref().map_or(0, |(_, e)| e.len());
    let ref_audio = acts.get("audio")?;
    let frames_match = frames == n;
    let audio_diff = if audio.len() == ref_audio.len() { Some(diff(&audio, &ref_audio)?) } else { None };
    let spectral = if audio.len() == ref_audio.len() { Some(spectral_rel(&audio, &ref_audio)?) } else { None };
    let free_latents = stream.record.as_ref().map(|(l, _)| l.clone()).unwrap_or_default();
    let latents_rel =
        if free_latents.len() == inputs.latents.len() { Some(diff(&free_latents, &inputs.latents)?.rel) } else { None };
    let strict = matches!(file_dtype.as_str(), "fp32" | "fp16");
    let audio_ok = !strict
        || (frames_match
            && spectral.is_some_and(|d| d <= SPECTRAL_TOLERANCE)
            && latents_rel.is_some_and(|d| d <= LATENT_TOLERANCE));

    let correctness = kernel.as_ref().unwrap_or(&quality);
    let ok = tokens_match && correctness.ok() && audio_ok;
    let mut value = json!({
        "ok": ok,
        "device": dev.description(),
        "model": model,
        "model_file": path.display().to_string(),
        "dtype": file_dtype,
        "correctness_reference": if quantized { "fake-quant" } else { "pytorch-fp32" },
        "tolerance": if quantized { KERNEL_TOLERANCE } else { quality_tol },
        "first_failing_layer": correctness.first_failing(),
        "layers": correctness.layers,
        "tokens_match": tokens_match,
        "prepared_text": prepared,
        "tokens": ids,
        "reference_tokens": reference.text_ids,
        "end_to_end": {
            "frames": frames,
            "reference_frames": n,
            "frames_match": frames_match,
            "latents_rel": latents_rel,
            "latents_tolerance": LATENT_TOLERANCE,
            "spectral_rel": spectral,
            "spectral_tolerance": SPECTRAL_TOLERANCE,
            "waveform_rel": audio_diff.map(|d| d.rel),
            "waveform_max_abs": audio_diff.map(|d| d.max_abs),
            "must_match": strict,
            "ok": audio_ok,
            "first_audio_ms": first_audio_ms,
        },
        "kernel": kernel_json,
        "quality": {
            "reference": "PyTorch fp32 (HF GradiumPhononTTSForConditionalGeneration + PocketTTSAudioTokenizer)",
            "informational": quantized,
            "tolerance": quality_tol,
            "within_tolerance": quality.ok(),
            "max_rel": quality.max_rel(),
            "layers": quality.layers,
        },
        "notes": correctness.notes,
        "timings_ms": stages.to_json(),
        "reference_dir": reference_dir.display().to_string(),
    });
    if !tokens_match {
        lapid_verify::merge_object(
            &mut value,
            json!({"tokens_note": "text preparation / tokenizer differ from the reference"}),
        );
    }
    Ok(value)
}
