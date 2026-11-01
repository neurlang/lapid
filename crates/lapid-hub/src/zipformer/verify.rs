//! Zipformer on the generic audio harness: the verification descriptor (fbank, every chunk
//! on the reference fbank with all caches traced, tokens overall and per chunk, end to end
//! from odd-sized audio pieces) and the streaming benchmark.

use std::path::Path;
use std::time::Instant;

use lapid::{Device, Error, Result};
use lapid_convert::engine::ConvertSpec;
use lapid_nn::Precision;
use lapid_verify::Descriptor;
use serde_json::{Map, Value, json};

use super::config::{LOG_EPS, PAD_FRAMES, TAIL_FRAMES};
use lapid_nn::audio::Fbank;
use super::stream::greedy;
use super::{StreamOptions, Zipformer};
use crate::runtime::audio::{AsrModel, AudioDef, BenchRun, Check, EndToEnd, Row, Sample, Stages, find_path};
use crate::runtime::serve::{AsrParams, AsrSegment, TranscriptOut};
use crate::runtime::transducer::{GreedyDecoder, Transducer};

/// Tolerance of our fbank against lhotse (f64 vs f32 math).
const FBANK_TOLERANCE: f32 = 1e-4;
/// Samples per `accept_waveform` call: odd in verification, 100 ms in benchmarks.
const VERIFY_PIECE: usize = 1013;
const BENCH_PIECE: usize = 1600;

/// Row of a traced / state tensor: per-kind aggregation over chunks and layers.
fn row_of(name: &str) -> String {
    match name.strip_prefix("state.") {
        Some(rest) if rest != "embed" => format!("state.{}", rest.rsplit('.').next().unwrap_or(rest)),
        _ => name.to_string(),
    }
}

fn num(s: &Sample, k: &str) -> Result<usize> {
    s.result[k].as_u64().map(|v| v as usize).ok_or_else(|| Error::new("reference", format!("result.json: no {k:?}")))
}

fn stream_options(s: &Sample) -> Result<StreamOptions> {
    Ok(StreamOptions { chunk_size: Some(num(s, "chunk_size")?), left_context_frames: Some(num(s, "left_context_frames")?), endpoint: None })
}

impl AudioDef for Zipformer {
    const DESCRIPTOR: Descriptor = Descriptor {
        name: "zipformer",
        default_model: "data/hf/zipformer2-en-streaming-2023-05-17",
        default_dtype: "fp16",
        reference_dir: "data/reference/zipformer",
        reference_script: "tools/reference/zipformer_ref.py",
        reference_label: "icefall fp32 streaming decode (PyTorch)",
        strict_dtypes: &["fp32", "fp16", "int8"],
        bench_prompt_tokens: 0,
        bench_gen_tokens: 0,
    };
    const CONVERT: &'static ConvertSpec = &super::convert::CONVERT;
    const QUALITY_REQUIRED: bool = true;

    fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Self> {
        Zipformer::load_with(path, device, precision)
    }

    fn sample_rate(&self) -> u32 {
        self.config().sample_rate
    }

    fn sample_model(s: &Sample) -> Option<String> {
        let m = s.result["model"].as_str()?;
        Some(find_path(&s.dir, m).map_or_else(|| m.to_string(), |p| p.display().to_string()))
    }

    /// Chunks `chunk.{c}.{name}` aggregated per row like [`AudioDef::stages`].
    fn reference_row(s: &Sample, name: &str) -> Result<Vec<f32>> {
        if name == "fbank" {
            return s.get("fbank");
        }
        let states = s.u32s("/states_chunks").unwrap_or_default();
        let mut out = Vec::new();
        for c in 0..num(s, "chunks")? {
            let p = format!("chunk.{c}.");
            for n in s.acts.names().iter().filter(|n| n.starts_with(&p)) {
                let rest = &n[p.len()..];
                if row_of(rest) == name && (!rest.starts_with("state.") || states.contains(&(c as u32))) {
                    out.extend(s.get(n)?);
                }
            }
        }
        Ok(out)
    }

    /// Fbank from the audio; chunk by chunk on the reference fbank (icefall's schedule and
    /// tail padding): encoder output of every chunk, stack outputs of the traced chunks,
    /// every cache after the state chunks; greedy tokens overall and per chunk.
    fn stages(&self, s: &Sample, audio: &[f32]) -> Result<Stages> {
        let (m, cfg) = (self.model(), self.config());
        let fbank = Fbank::new(cfg.feature_dim, cfg.sample_rate).compute(audio);
        let mut rows = vec![Row::front("fbank", fbank, FBANK_TOLERANCE)];
        let (chunk, f) = (num(s, "chunk_size")?, cfg.feature_dim);
        let mut st = m.chunk_state(chunk, num(s, "left_context_frames")?)?;
        let mut feats = s.get("fbank")?;
        feats.extend(std::iter::repeat_n(LOG_EPS, (PAD_FRAMES + TAIL_FRAMES) * f));
        let (total, window, states) = (feats.len() / f, 2 * chunk + PAD_FRAMES, s.u32s("/states_chunks").unwrap_or_default());
        let mut dec = GreedyDecoder::new(m, greedy())?;
        let mut agg: Vec<(String, Vec<f32>)> = vec![("encoder_out".into(), Vec::new())];
        let mut chunk_tokens: Vec<Vec<u32>> = Vec::new();
        let (mut n, mut c) = (0, 0);
        while n < total {
            let mut x = feats[n * f..(n + window).min(total) * f].to_vec();
            x.resize(window * f, LOG_EPS);
            let (enc, tr) = m.run_chunk(&x, &mut st, true)?;
            let tr = tr.ok_or_else(|| Error::new("internal", "no chunk trace"))?;
            chunk_tokens.push(dec.decode_chunk(m, &enc)?.iter().map(|s| s.token).collect());
            agg[0].1.extend(m.device().to_vec_f32(&enc)?);
            let state_rows = if states.contains(&(c as u32)) { &tr.state[..] } else { &[][..] };
            for (name, v) in tr.traced.iter().filter(|(n, _)| s.acts.has(&format!("chunk.{c}.{n}"))).chain(state_rows) {
                let row = row_of(name);
                match agg.iter_mut().find(|(k, _)| *k == row) {
                    Some((_, acc)) => acc.extend(v),
                    None => agg.push((row, v.clone())),
                }
            }
            n += 2 * chunk;
            c += 1;
        }
        // only the projected encoder output is compared with the fake-quant model
        rows.extend(agg.into_iter().map(|(k, v)| if k == "encoder_out" { Row::new(k, v) } else { Row::new(k, v).quality_only() }));
        let ref_chunks: Vec<Vec<u32>> = s.result["chunk_tokens"].as_array().into_iter().flatten().map(|a| a.as_array().into_iter().flatten().filter_map(Value::as_u64).map(|v| v as u32).collect()).collect();
        let tokens: Vec<u32> = chunk_tokens.concat();
        let random_int8 = s.result["random"].as_bool() == Some(true) && s.dtype == "int8";
        let checks = vec![
            Check::new("tokens_match", tokens == s.u32s("/tokens")?, s.strict && !random_int8),
            // lower precisions may move a token by one frame across a chunk boundary
            Check::new("chunk_tokens_match", chunk_tokens == ref_chunks, s.dtype == "fp32"),
        ];
        Ok(Stages { rows, argmax: Vec::new(), checks })
    }

    fn end_to_end(&self, s: &Sample, audio: &[f32]) -> Result<EndToEnd> {
        let mut st = self.stream(&stream_options(s)?)?;
        for piece in audio.chunks(VERIFY_PIECE) {
            st.accept_waveform(piece)?;
        }
        st.input_finished()?;
        let r = st.result();
        let random_int8 = s.result["random"].as_bool() == Some(true) && s.dtype == "int8";
        let reference_text = s.text("text");
        let ok = r.tokens == s.u32s("/tokens")? && r.text == reference_text;
        let info = json!({"chunk_size": num(s, "chunk_size")?, "left_context_frames": num(s, "left_context_frames")?, "tokens": r.tokens.len()});
        Ok(EndToEnd { text: r.text, reference_text, checks: vec![Check::new("e2e_match", ok, s.strict && !random_int8)], info: info.as_object().cloned().unwrap_or_default() })
    }

    /// The WAV fed in 100 ms pieces as fast as possible: wall time, stage times, and the
    /// first partial (audio fed when the first token appeared, its time, that call's compute).
    fn bench_run(&self, audio: &[f32]) -> Result<BenchRun> {
        let sr = f64::from(self.config().sample_rate);
        let chunk = std::env::var("LAPID_ZIPFORMER_CHUNK").ok().and_then(|v| v.parse().ok());
        let mut s = self.stream(&StreamOptions { chunk_size: chunk, left_context_frames: None, endpoint: None })?;
        let t = Instant::now();
        let (mut fed, mut first) = (0usize, None::<(f64, f64, f64)>);
        let mut note = |out: &[super::StreamResult], fed: usize, tc: Instant| {
            if first.is_none()
                && let Some(r) = out.iter().find(|r| !r.tokens.is_empty())
            {
                first = Some((fed as f64 / sr, f64::from(r.timestamps[0]), tc.elapsed().as_secs_f64() * 1e3));
            }
        };
        for piece in audio.chunks(BENCH_PIECE) {
            let tc = Instant::now();
            let out = s.accept_waveform(piece)?;
            fed += piece.len();
            note(&out, fed, tc);
        }
        let tc = Instant::now();
        let out = s.input_finished()?;
        note(&out, fed, tc);
        let total = t.elapsed().as_secs_f64() * 1e3;
        let (st, (fa, ft, fc)) = (s.stats(), first.unwrap_or((f64::NAN, f64::NAN, f64::NAN)));
        Ok(BenchRun {
            times: vec![("total", total), ("fbank", st.frontend_ms), ("encoder", st.encoder_ms), ("decode", st.decode_ms)],
            values: vec![
                ("chunks", st.chunks as f64),
                ("encoder_ms_per_chunk", if st.chunks > 0 { st.encoder_ms / st.chunks as f64 } else { 0.0 }),
                ("first_partial_audio_s", fa),
                ("first_token_time_s", ft),
                ("first_partial_compute_ms", fc),
            ],
            text: s.result().text,
        })
    }

    fn bench_info(&self) -> Map<String, Value> {
        let chunk: Option<usize> = std::env::var("LAPID_ZIPFORMER_CHUNK").ok().and_then(|v| v.parse().ok());
        let v = json!({"chunk_size": chunk.unwrap_or(self.config().chunk_size), "left_context_frames": self.config().left_context_frames});
        v.as_object().cloned().unwrap_or_default()
    }
}

impl AsrModel for Zipformer {
    /// One stream over the whole signal; segments at sherpa-onnx endpoints.
    fn recognize(&self, samples: &[f32], _: &AsrParams) -> Result<TranscriptOut> {
        let mut s = self.stream(&StreamOptions::default())?;
        let mut finals = s.accept_waveform(samples)?;
        finals.extend(s.input_finished()?);
        let segments: Vec<AsrSegment> = finals
            .into_iter()
            .filter(|r| r.is_final && !r.tokens.is_empty())
            .map(|r| AsrSegment { start: r.start, end: r.timestamps.last().copied().unwrap_or(r.start) + 0.04, text: r.text, tokens: r.tokens })
            .collect();
        let text = segments.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ");
        Ok(TranscriptOut { text, language: None, segments, words: Vec::new() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_names() {
        assert_eq!(row_of("state.3.1.key"), "state.key");
        assert_eq!(row_of("state.embed"), "state.embed");
        assert_eq!(row_of("stack.2"), "stack.2");
    }

    /// The tiny random model (data/reference/zipformer/tiny) on the CPU, when present.
    #[test]
    fn tiny_reference_cpu() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/reference/zipformer/tiny");
        if !dir.join("result.json").is_file() {
            return;
        }
        let args = crate::cmd::ModelArgs { model: Some(dir.join("model").display().to_string()), dtype: Some("fp32".into()), device: Some("cpu".into()) };
        let v = crate::runtime::audio_verify::verify::<Zipformer>(Some(&dir), &args).expect("verify");
        assert_eq!(v["ok"], json!(true), "{v:#}");
    }
}
