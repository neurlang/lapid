//! Phonon checkpoints (Gradium's runtime package on the Hugging Face hub, e.g.
//! `gradium/phonon-40m-en`) → one `.lapid` file, and voice files → `.lapid` extras.
//!
//! The model file holds three components — `flow_lm` (backbone, conditioners, flow head),
//! `mimi` (codec decoder) and `voices` (voice-prompt latents, `voices.<name>` [frames, 512],
//! the package's `default-voice.safetensors` as `voices.default`) — and blobs: the
//! tokenizer (`tokenizer.json`, `tokenizer.model`) and the original JSON configs / chat
//! template. Names follow the checkpoint, except that the fused attention `in_proj`
//! weights are split into `q_proj` / `k_proj` / `v_proj` (rows in the original interleaved
//! RoPE layout) and training-only tensors (`flow.w_s_t.*`, `learnt_padding`) are dropped.
//! Precision: transformer / flow-head Linear weights follow the dtype; the text embedding at
//! most int8; convolutions f16; norms, biases, latent statistics and voices f32.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use lapid::{Error, Result, bail};
use lapid_convert::policy::Role;
use lapid_convert::safetensors::SafeTensors;
use lapid_convert::writer::{ModelWriter, file_hash, read_json};
use lapid_convert::{OutDType, download_hf, download_hf_optional};
use serde_json::{Value, json};

use super::config::*;

pub(crate) const TOKENIZER_BLOB: &str = "tokenizer/tokenizer.json";
pub(crate) const TOKENIZER_MODEL_BLOB: &str = "tokenizer/tokenizer.model";
pub(crate) const VOICE_PREFIX: &str = "voices.";
const VOICE_KEY: &str = "speaker_wavs";

/// Source files of a checkpoint (repo download or local directory).
struct Files {
    config: PathBuf,
    weights: PathBuf,
    voice: PathBuf,
    tokenizer: PathBuf,
    optional: Vec<(&'static str, PathBuf)>,
    readme: Option<PathBuf>,
    repo: String,
    remote: bool,
}

const OPTIONAL: [&str; 4] =
    ["tokenizer.model", "generation_config.json", "processor_config.json", "chat_template.jinja"];

fn locate(src: &str) -> Result<Files> {
    let dir = Path::new(src);
    if dir.is_dir() {
        let req = |f: &str| {
            Some(dir.join(f))
                .filter(|p| p.is_file())
                .ok_or_else(|| Error::new("not_found", format!("{}: no {f}", dir.display())))
        };
        let optional = OPTIONAL.iter().filter_map(|f| Some((*f, dir.join(f))).filter(|(_, p)| p.is_file())).collect();
        return Ok(Files {
            config: req("config.json")?,
            weights: req("model.safetensors")?,
            voice: req("default-voice.safetensors")?,
            tokenizer: req("tokenizer.json")?,
            optional,
            readme: Some(dir.join("README.md")).filter(|p| p.is_file()),
            repo: dir.display().to_string(),
            remote: false,
        });
    }
    let repo = src.trim_matches('/');
    let mut optional = Vec::new();
    for f in OPTIONAL {
        if let Some(p) = download_hf_optional(repo, f)? {
            optional.push((f, p));
        }
    }
    Ok(Files {
        config: download_hf(repo, "config.json")?,
        weights: download_hf(repo, "model.safetensors")?,
        voice: download_hf(repo, "default-voice.safetensors")?,
        tokenizer: download_hf(repo, "tokenizer.json")?,
        optional,
        readme: download_hf_optional(repo, "README.md")?,
        repo: repo.to_string(),
        remote: true,
    })
}

/// Whether an HF `config.json` is a Phonon runtime package.
pub fn is_phonon_config(cfg: &Value) -> bool {
    cfg.get("model_type").and_then(Value::as_str) == Some("gradium_phonon_tts")
}

/// Dtype-policy role of a Phonon tensor.
pub(crate) fn role(name: &str, shape: &[usize]) -> Role {
    if shape.len() <= 1 || name.starts_with(VOICE_PREFIX) || name.ends_with(".bias") || name.contains("norm") {
        Role::Param
    } else if name.ends_with("transcript_in_segment.embed.weight") {
        Role::LinearInt8
    } else if shape.len() == 2 && !name.ends_with(".embed.weight") {
        Role::Linear
    } else {
        Role::Dense
    }
}

fn f(v: &Value, path: &str) -> Result<f64> {
    v.pointer(path)
        .and_then(Value::as_f64)
        .ok_or_else(|| Error::new("missing_metadata", format!("config: missing number {path}")))
}

fn u(v: &Value, path: &str) -> Result<usize> {
    v.pointer(path)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| Error::new("missing_metadata", format!("config: missing integer {path}")))
}

fn opt_f(v: Option<&Value>, key: &str, default: f64) -> f64 {
    v.and_then(|v| v.get(key)).and_then(Value::as_f64).unwrap_or(default)
}

fn opt_u(v: Option<&Value>, key: &str, default: usize) -> usize {
    v.and_then(|v| v.get(key)).and_then(Value::as_u64).map_or(default, |x| x as usize)
}

/// `PhononConfig` from the package's JSON files.
pub(crate) fn build_config(cfg: &Value, generation: Option<&Value>, processor: Option<&Value>) -> Result<PhononConfig> {
    let ratios: Vec<usize> = cfg
        .pointer("/mimi/ratios")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect())
        .ok_or_else(|| Error::new("missing_metadata", "config: missing mimi.ratios"))?;
    let sample_rate = u(cfg, "/mimi/sample_rate")?;
    let frame_rate = f(cfg, "/mimi/frame_rate")?;
    let hop: usize = ratios.iter().product();
    let samples_per_frame = (sample_rate as f64 / frame_rate).round() as usize;
    if hop == 0 || !samples_per_frame.is_multiple_of(hop) {
        bail!("unsupported_model", "mimi: frame size {samples_per_frame} is not a multiple of the hop {hop}");
    }
    let fh = cfg.get("flow_head_config");
    Ok(PhononConfig {
        lm: LmConfig {
            d_model: u(cfg, "/hidden_size")?,
            n_heads: u(cfg, "/num_attention_heads")?,
            n_layers: u(cfg, "/num_hidden_layers")?,
            ffn_dim: u(cfg, "/intermediate_size")?,
            rope_theta: cfg.pointer("/rope_parameters/rope_theta").and_then(Value::as_f64).unwrap_or(10000.0) as f32,
            norm_eps: opt_f(Some(cfg), "layer_norm_eps", 1e-5) as f32,
            latent_dim: u(cfg, "/latent_dim")?,
            vocab_size: u(cfg, "/vocab_size")?,
            audio_prompt_dim: opt_u(Some(cfg), "audio_prompt_dim", 512),
        },
        flow: FlowConfig {
            dim: opt_u(fh, "hidden_size", u(cfg, "/flow_lm/flow_dim")?),
            depth: opt_u(fh, "num_hidden_layers", u(cfg, "/flow_lm/flow_depth")?),
            frequency_embedding_size: opt_u(fh, "frequency_embedding_size", 256),
            norm_eps: opt_f(fh, "layer_norm_eps", 1e-6) as f32,
            time_norm_eps: opt_f(fh, "rms_norm_eps", 1e-5) as f32,
            flow_type: fh.and_then(|h| h.get("flow_type")).and_then(Value::as_str).unwrap_or("lsd").to_string(),
        },
        conditions: ConditionConfig {
            num_speakers: opt_u(Some(cfg), "num_speakers", 1),
            duration_delta: opt_f(Some(cfg), "duration_delta", 0.0) as f32,
            duration_delta_dim: opt_u(Some(cfg), "duration_delta_embedding_dim", 128),
            duration_delta_scale: opt_f(Some(cfg), "duration_delta_scale", 1000.0) as f32,
            duration_delta_max_period: opt_f(Some(cfg), "duration_delta_max_period", 10000.0) as f32,
        },
        mimi: MimiConfig {
            sample_rate: sample_rate as u32,
            frame_rate: frame_rate as f32,
            dimension: u(cfg, "/mimi/dimension")?,
            latent_dim: u(cfg, "/mimi/quantizer_dimension")?,
            n_filters: u(cfg, "/mimi/n_filters")?,
            n_residual_layers: u(cfg, "/mimi/n_residual_layers")?,
            ratios,
            kernel_size: u(cfg, "/mimi/kernel_size")?,
            last_kernel_size: u(cfg, "/mimi/last_kernel_size")?,
            residual_kernel_size: u(cfg, "/mimi/residual_kernel_size")?,
            dilation_base: u(cfg, "/mimi/dilation_base")?,
            compress: u(cfg, "/mimi/compress")?,
            upsample_stride: samples_per_frame / hop,
            channels: u(cfg, "/mimi/channels")?,
            transformer: MimiTransformerConfig {
                d_model: u(cfg, "/mimi/transformer_d_model")?,
                n_heads: u(cfg, "/mimi/transformer_num_heads")?,
                n_layers: u(cfg, "/mimi/transformer_num_layers")?,
                ffn_dim: u(cfg, "/mimi/transformer_dim_feedforward")?,
                rope_theta: f(cfg, "/mimi/transformer_max_period")? as f32,
                window: u(cfg, "/mimi/transformer_context")? + 1,
                norm_eps: 1e-5,
            },
        },
        generation: GenerationConfig {
            temperature: opt_f(generation, "temperature", 0.3) as f32,
            eos_threshold: opt_f(generation, "eos_threshold", opt_f(Some(cfg), "eos_threshold", 4.0)) as f32,
            num_flow_steps: opt_u(generation, "num_flow_steps", 1).max(1),
            min_frames_before_eos: opt_u(generation, "min_frames_before_eos", 0),
            frames_after_eos: opt_u(processor, "frames_after_eos", 2),
            frames_after_eos_short_text: opt_u(processor, "frames_after_eos_short_text", 4),
            short_text_max_words: opt_u(processor, "short_text_max_words", 4),
            fade_in_duration: opt_f(processor, "fade_in_duration", 0.005) as f32,
        },
        tokens: TokenIds {
            pad: opt_u(Some(cfg), "pad_token_id", 3) as u32,
            audio_bos: opt_u(Some(cfg), "audio_bos_token_id", 4) as u32,
            audio_prompt: opt_u(Some(cfg), "audio_prompt_token_id", 5) as u32,
            audio_latent: opt_u(Some(cfg), "audio_latent_token_id", 6) as u32,
        },
        assets: Assets {
            tokenizer: TOKENIZER_BLOB.to_string(),
            tokenizer_model: None,
            voice_prefix: VOICE_PREFIX.to_string(),
            default_voice: "default".to_string(),
        },
    })
}

/// `license:` from the model card front matter.
fn card_license(readme: Option<&Path>) -> Option<String> {
    let text = std::fs::read_to_string(readme?).ok()?;
    let front = text.strip_prefix("---")?.split("\n---").next()?;
    front.lines().find_map(|l| l.trim().strip_prefix("license:").map(|s| s.trim().to_string()))
}

/// Tensors written for one checkpoint tensor: (name, shape, data). Splits fused `in_proj`.
/// (name, shape, data) of one written tensor.
type Mapped = (String, Vec<usize>, Vec<f32>);

fn map_tensor(name: &str, shape: Vec<usize>, data: Vec<f32>) -> Result<Vec<Mapped>> {
    if name.starts_with("flow_lm.flow.w_s_t.") || name.ends_with(".learnt_padding") || name.ends_with("num_ema_updates")
    {
        return Ok(vec![]);
    }
    let name = name.replace("in_proj_weight", "in_proj.weight");
    if let Some(prefix) = name.strip_suffix(".in_proj.weight") {
        if shape.len() != 2 || !shape[0].is_multiple_of(3) {
            bail!("shape_mismatch", "{name}: fused in_proj must be [3*d, d], got {shape:?}");
        }
        let (rows, cols) = (shape[0] / 3, shape[1]);
        return Ok(["q_proj", "k_proj", "v_proj"]
            .iter()
            .enumerate()
            .map(|(i, p)| {
                (
                    format!("{prefix}.{p}.weight"),
                    vec![rows, cols],
                    data[i * rows * cols..(i + 1) * rows * cols].to_vec(),
                )
            })
            .collect());
    }
    Ok(vec![(name, shape, data)])
}

/// Voice-prompt latents [frames, dim] from a voice safetensors (`speaker_wavs` [1, dim, frames]).
pub fn read_voice_safetensors(path: &Path) -> Result<(Vec<f32>, Vec<usize>)> {
    let st = SafeTensors::open(path)?;
    let shape = st.shape(VOICE_KEY)?;
    let (dim, frames) = match shape.as_slice() {
        [1, d, t] | [d, t] => (*d, *t),
        s => bail!("shape_mismatch", "{}: {VOICE_KEY} must be [1, dim, frames], got {s:?}", path.display()),
    };
    let data = st.to_f32(VOICE_KEY)?;
    let mut out = vec![0.0f32; data.len()];
    for c in 0..dim {
        for t in 0..frames {
            out[t * dim + c] = data[c * frames + t];
        }
    }
    Ok((out, vec![frames, dim]))
}

/// Tensors every Phonon checkpoint must provide (after mapping).
fn expected_tensors(cfg: &PhononConfig) -> Vec<String> {
    let mut v: Vec<String> = [
        "flow_lm.bos_emb",
        "flow_lm.emb_mean",
        "flow_lm.emb_std",
        "flow_lm.input_linear.weight",
        "flow_lm.out_norm.weight",
        "flow_lm.out_eos.weight",
        "flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight",
        "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        "flow_lm.flow_net.cond_embed.weight",
        "flow_lm.flow_net.input_proj.weight",
        "flow_lm.flow_net.final_layer.linear.weight",
        "mimi.quantizer.output_proj.weight",
        "mimi.upsample.convtr.convtr.weight",
        "voices.default",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for i in 0..cfg.lm.n_layers {
        for p in
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "linear1", "linear2"]
        {
            v.push(format!("flow_lm.transformer.layers.{i}.{p}.weight"));
        }
    }
    for i in 0..cfg.mimi.transformer.n_layers {
        v.push(format!("mimi.decoder_transformer.transformer.layers.{i}.self_attn.q_proj.weight"));
        v.push(format!("mimi.decoder_transformer.transformer.layers.{i}.layer_scale_1.scale"));
    }
    for i in 0..cfg.flow.depth {
        v.push(format!("flow_lm.flow_net.res_blocks.{i}.mlp.0.weight"));
    }
    v
}

/// Convert a Phonon package (repo id or local directory) to `.lapid` at `dtype`.
pub fn convert_hf(src: &str, dtype: OutDType, out: &Path) -> Result<()> {
    let files = locate(src)?;
    let cfg_json = read_json(&files.config)?;
    if !is_phonon_config(&cfg_json) {
        bail!("unsupported_model", "{}: config.json model_type is not \"gradium_phonon_tts\"", files.repo);
    }
    let opt = |name: &str| files.optional.iter().find(|(n, _)| *n == name).map(|(_, p)| p.clone());
    let generation = opt("generation_config.json").as_deref().map(read_json).transpose()?;
    let processor = opt("processor_config.json").as_deref().map(read_json).transpose()?;
    let mut cfg = build_config(&cfg_json, generation.as_ref(), processor.as_ref())?;
    if opt("tokenizer.model").is_some() {
        cfg.assets.tokenizer_model = Some(TOKENIZER_MODEL_BLOB.to_string());
    }

    let license = card_license(files.readme.as_deref()).unwrap_or_else(|| "cc-by-nc-4.0".to_string());
    let name = files.repo.trim_end_matches('/').rsplit('/').next().unwrap_or("phonon").to_string();
    let mut source = json!({"repo": files.repo, "file_hash": file_hash(&files.weights)?});
    if files.remote {
        source["revision"] = json!("main");
    }
    let mut meta = lapid_convert::writer::metadata(&name, "phonon", dtype, &license, source);
    meta["phonon"] = serde_json::to_value(&cfg)?;
    meta["components"] = json!({
        "flow_lm": {"architecture": "phonon-flow-lm", "config": "phonon", "license": license, "tensor_prefix": "flow_lm."},
        "mimi": {"architecture": "mimi-decoder", "config": "phonon", "license": license, "tensor_prefix": "mimi."},
        "voices": {"architecture": "phonon-voices", "config": "phonon", "license": license, "tensor_prefix": VOICE_PREFIX},
    });

    let st = SafeTensors::open(&files.weights)?;
    let names = st.names();
    let mut w = ModelWriter::create(out, meta, dtype, role, names.len() + 8)?;
    w.add_blob(TOKENIZER_BLOB, "application/json", &read(&files.tokenizer)?)?;
    for (n, p) in &files.optional {
        let (blob, media) = match *n {
            "tokenizer.model" => (TOKENIZER_MODEL_BLOB.to_string(), "application/x-sentencepiece"),
            "chat_template.jinja" => (format!("config/{n}"), "text/plain"),
            _ => (format!("config/{n}"), "application/json"),
        };
        w.add_blob(&blob, media, &read(p)?)?;
    }
    w.add_blob("config/config.json", "application/json", &read(&files.config)?)?;
    for n in &names {
        for (name, shape, data) in map_tensor(n, st.shape(n)?, st.to_f32(n)?)? {
            w.add_f32(&name, &shape, &data)?;
        }
    }
    let (voice, shape) = read_voice_safetensors(&files.voice)?;
    if shape[1] != cfg.lm.audio_prompt_dim {
        bail!("shape_mismatch", "default voice has width {}, model expects {}", shape[1], cfg.lm.audio_prompt_dim);
    }
    w.add_f32(&format!("{VOICE_PREFIX}default"), &shape, &voice)?;
    w.finish(&expected_tensors(&cfg))
}

fn read(p: &Path) -> Result<Vec<u8>> {
    std::fs::read(p).map_err(|e| Error::new("io", format!("{}: {e}", p.display())))
}

/// Write voices (each a `speaker_wavs` safetensors, e.g. from `gradium/phonon-voices`) into
/// a `.lapid` extras file (`kind: "extra"`, extends component `voices`); tensors
/// `voices.<name>`; `source_repo` is recorded as the source (e.g. "gradium/phonon-voices"). Load it with [`super::Phonon::add_voices`].
pub fn convert_voices(voices: &[(String, PathBuf)], source_repo: &str, license: &str, out: &Path) -> Result<()> {
    if voices.is_empty() {
        bail!("invalid_argument", "no voices to convert");
    }
    let source = json!({"repo": source_repo, "files": voices.iter().map(|(n, p)| json!({"voice": n, "file": p.display().to_string()})).collect::<Vec<_>>()});
    let mut meta = lapid_convert::writer::metadata("phonon-voices", "phonon", OutDType::F32, license, source);
    meta["kind"] = json!("extra");
    meta["extends"] = json!({"architecture": "phonon", "component": "voices"});
    let mut w = ModelWriter::create(out, meta, OutDType::F32, role, voices.len())?;
    let mut seen = BTreeSet::new();
    let mut expected = Vec::new();
    for (name, path) in voices {
        if !seen.insert(name.clone()) {
            bail!("invalid_argument", "duplicate voice name {name:?}");
        }
        let (data, shape) = read_voice_safetensors(path)?;
        let t = format!("{VOICE_PREFIX}{name}");
        w.add_f32(&t, &shape, &data)?;
        expected.push(t);
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles() {
        assert_eq!(role("flow_lm.transformer.layers.0.linear1.weight", &[2048, 512]), Role::Linear);
        assert_eq!(role("flow_lm.out_norm.weight", &[512]), Role::Param);
        assert_eq!(role("voices.default", &[125, 512]), Role::Param);
        assert_eq!(role("mimi.decoder.model.0.conv.weight", &[512, 512, 7]), Role::Dense);
        assert_eq!(role("flow_lm.condition_provider.conditioners.num_speakers.embed.weight", &[32, 16]), Role::Dense);
        assert_eq!(
            role("flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight", &[4001, 512]),
            Role::LinearInt8
        );
    }

    #[test]
    fn in_proj_split() {
        let data: Vec<f32> = (0..6 * 2).map(|x| x as f32).collect();
        let out = map_tensor("a.self_attn.in_proj.weight", vec![6, 2], data).expect("map");
        assert_eq!(out.len(), 3);
        assert_eq!(out[1].0, "a.self_attn.k_proj.weight");
        assert_eq!(out[1].2, vec![4.0, 5.0, 6.0, 7.0]);
        assert!(map_tensor("flow_lm.flow.w_s_t.0.bias", vec![1], vec![0.0]).expect("skip").is_empty());
    }
}
