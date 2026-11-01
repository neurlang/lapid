//! Audio models (speech recognition, voice activity) on the shared framework.
//!
//! A model implements [`AudioDef`]: its descriptor and converter, loading at a precision and
//! the model-specific halves of verification ([`AudioDef::stages`] on the reference inputs,
//! [`AudioDef::end_to_end`] from the audio) and benchmarks ([`AudioDef::bench_run`]). The
//! generic drivers ([`super::audio_verify`], [`super::audio_bench`]) do the rest: sample
//! discovery, model resolution, fake-quant references, tolerances, reports and statistics.
//! Speech recognizers also implement [`AsrModel`] and are served through [`AsrServed`]
//! (`lapid serve`, `lapid run`, [`crate::load`]). Registry entry: [`asr_entry`] or
//! [`audio_entry`].

use std::path::{Path, PathBuf};

use lapid::{Device, Result, bail};
use lapid_convert::engine::ConvertSpec;
use lapid_nn::Precision;
use lapid_verify::{Activations, Descriptor};
use serde_json::{Map, Value, json};

use super::registry::ModelEntry;
use super::serve::{AnyModel, AsrParams, Capability, LoadOptions, ModelMeta, TranscriptOut};

/// One reference sample: a directory with `result.json`, `activations.safetensors` and the
/// audio it names.
pub struct Sample {
    pub dir: PathBuf,
    pub result: Value,
    pub acts: Activations,
    /// `result.json` `"audio"`, resolved.
    pub audio: PathBuf,
    /// Dtype of the file under test.
    pub dtype: String,
    /// The dtype is one of the descriptor's `strict_dtypes` (outputs must be exact).
    pub strict: bool,
}

impl Sample {
    pub fn get(&self, name: &str) -> Result<Vec<f32>> {
        self.acts.get(name)
    }

    /// Unsigned integer array `result.json` `ptr` (JSON pointer).
    pub fn u32s(&self, ptr: &str) -> Result<Vec<u32>> {
        let a = self.result.pointer(ptr).and_then(Value::as_array);
        let a = a.ok_or_else(|| lapid::Error::new("reference", format!("{}: result.json has no array {ptr}", self.dir.display())))?;
        a.iter().map(|x| x.as_u64().map(|v| v as u32).ok_or_else(|| lapid::Error::new("reference", format!("{ptr}: non-integer value")))).collect()
    }

    /// String `result.json` `key` (trimmed, empty when missing).
    pub fn text(&self, key: &str) -> String {
        self.result.get(key).and_then(Value::as_str).unwrap_or("").trim().to_string()
    }
}

/// `rel` as given (absolute or from the working directory), else relative to `dir` or one of
/// its ancestors (reference files name paths relative to themselves or to the repo root).
pub fn find_path(dir: &Path, rel: &str) -> Option<PathBuf> {
    let from_dir = dir.ancestors().map(|a| a.join(rel)).find(|q| q.exists());
    from_dir.or_else(|| Some(PathBuf::from(rel)).filter(|p| p.exists()))
}

/// A named output compared with the reference.
pub struct Row {
    pub name: String,
    pub values: Vec<f32>,
    /// Fixed tolerance (front ends such as log-mel features) instead of the dtype's; such
    /// rows are not compared with the fake-quant model.
    pub tol: Option<f32>,
    /// Compared with the fake-quant model for quantized files.
    pub kernel: bool,
}

impl Row {
    pub fn new(name: impl Into<String>, values: Vec<f32>) -> Row {
        Row { name: name.into(), values, tol: None, kernel: true }
    }

    /// A front-end row (fixed tolerance).
    pub fn front(name: impl Into<String>, values: Vec<f32>, tol: f32) -> Row {
        Row { name: name.into(), values, tol: Some(tol), kernel: false }
    }

    /// Compared with the PyTorch reference only.
    pub fn quality_only(mut self) -> Row {
        self.kernel = false;
        self
    }
}

/// Per-step argmax agreement of row `row` (steps of `width` values; argmax taken separately
/// over the segments cut at `splits`, e.g. token and duration logits of a TDT joint).
pub struct Argmax {
    pub row: String,
    pub width: usize,
    pub splits: Vec<usize>,
}

/// A pass / fail result; `required` ones decide `ok`.
pub struct Check {
    pub name: String,
    pub ok: bool,
    pub required: bool,
}

impl Check {
    pub fn new(name: impl Into<String>, ok: bool, required: bool) -> Check {
        Check { name: name.into(), ok, required }
    }
}

/// Model outputs on a sample's reference inputs.
#[derive(Default)]
pub struct Stages {
    pub rows: Vec<Row>,
    pub argmax: Vec<Argmax>,
    pub checks: Vec<Check>,
}

/// End-to-end result on a sample's audio.
#[derive(Default)]
pub struct EndToEnd {
    pub text: String,
    pub reference_text: String,
    pub checks: Vec<Check>,
    /// Extra report fields (token lists, language, ...).
    pub info: Map<String, Value>,
}

/// Timings and values of one benchmark run.
#[derive(Default)]
pub struct BenchRun {
    /// Stage times in ms (reported as `<name>_ms`); `"total"` is required.
    pub times: Vec<(&'static str, f64)>,
    /// Other per-run values (tokens, chunks, latencies, ...).
    pub values: Vec<(&'static str, f64)>,
    pub text: String,
}

/// An audio model on the shared verification and benchmark harness.
pub trait AudioDef: Sized + 'static {
    const DESCRIPTOR: Descriptor;
    const CONVERT: &'static ConvertSpec;
    /// Default `lapid bench` input.
    const BENCH_AUDIO: &'static str = "data/jfk.wav";
    /// The PyTorch comparison must pass for quantized files too (else it is informational
    /// and the fake-quant comparison decides).
    const QUALITY_REQUIRED: bool = false;

    fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Self>;
    fn sample_rate(&self) -> u32;
    /// The model a sample names itself, used when `--model` is not given.
    fn sample_model(_s: &Sample) -> Option<String> {
        None
    }
    /// Reference values of row `name`.
    fn reference_row(s: &Sample, name: &str) -> Result<Vec<f32>> {
        s.get(name)
    }
    /// Stage outputs on the sample's reference inputs (each stage fed the reference input,
    /// so errors do not compound); `audio` (at [`sample_rate`](Self::sample_rate)) for the
    /// front-end rows.
    fn stages(&self, s: &Sample, audio: &[f32]) -> Result<Stages>;
    /// End to end from the sample's audio (at [`sample_rate`](Self::sample_rate)).
    fn end_to_end(&self, s: &Sample, audio: &[f32]) -> Result<EndToEnd>;
    /// The benchmark input from the loaded audio (e.g. tiled to a fixed length).
    fn bench_input(&self, audio: Vec<f32>) -> Vec<f32> {
        audio
    }
    /// One benchmark run.
    fn bench_run(&self, audio: &[f32]) -> Result<BenchRun>;
    /// Extra fields of the bench report.
    fn bench_info(&self) -> Map<String, Value> {
        Map::new()
    }
}

/// Speech recognition for serving.
pub trait AsrModel {
    /// Transcribe mono samples at the model's sample rate.
    fn recognize(&self, samples: &[f32], opts: &AsrParams) -> Result<TranscriptOut>;
}

/// [`AnyModel`] adapter of a speech recognizer (input: 16 kHz mono, resampled as needed).
pub struct AsrServed<M> {
    pub model: M,
    pub meta: ModelMeta,
    rate: u32,
}

impl<M: AudioDef + AsrModel> AsrServed<M> {
    pub fn new(model: M, mut meta: ModelMeta) -> AsrServed<M> {
        let rate = model.sample_rate();
        meta.capabilities = vec![Capability::Transcription];
        meta.extra.insert("sample_rate".into(), json!(rate));
        AsrServed { model, meta, rate }
    }
}

impl<M: AudioDef + AsrModel> AnyModel for AsrServed<M> {
    fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    fn meta_mut(&mut self) -> &mut ModelMeta {
        &mut self.meta
    }

    fn transcribe(&mut self, samples_16k: &[f32], opts: &AsrParams) -> Result<TranscriptOut> {
        if self.rate == 16_000 {
            return self.model.recognize(samples_16k, opts);
        }
        self.model.recognize(&crate::audio::resample(samples_16k, 16_000, self.rate), opts)
    }
}

fn serve_asr<M: AudioDef + AsrModel>(path: &Path, device: &Device, meta: ModelMeta, _: &LoadOptions) -> Result<Box<dyn AnyModel>> {
    Ok(Box::new(AsrServed::new(M::load_with(path, device, Precision::Stored)?, meta)))
}

fn no_serve<M: AudioDef>(path: &Path, _: &Device, _: ModelMeta, _: &LoadOptions) -> Result<Box<dyn AnyModel>> {
    bail!("unsupported_model", "{}: {} models have no serving capability", path.display(), M::DESCRIPTOR.name)
}

fn verify<M: AudioDef>(r: Option<&Path>, a: &crate::cmd::ModelArgs) -> Result<Value> {
    super::audio_verify::verify::<M>(r, a)
}

fn bench<M: AudioDef>(i: Option<&Path>, a: &crate::cmd::ModelArgs, runs: usize) -> Result<Value> {
    super::audio_bench::bench::<M>(i, a, runs)
}

/// Registry entry of a speech recognizer: generic verify, bench and serving.
pub const fn asr_entry<M: AudioDef + AsrModel>() -> ModelEntry {
    ModelEntry { name: M::DESCRIPTOR.name, convert: M::CONVERT, verify: verify::<M>, bench: bench::<M>, serve: serve_asr::<M>, calibrate: None }
}

/// Registry entry of an audio model that is not served (e.g. voice activity detection).
pub const fn audio_entry<M: AudioDef>() -> ModelEntry {
    ModelEntry { name: M::DESCRIPTOR.name, convert: M::CONVERT, verify: verify::<M>, bench: bench::<M>, serve: no_serve::<M>, calibrate: None }
}
