//! Phonon hyper-parameters, stored in the `.lapid` metadata as `metadata["phonon"]` (written
//! by [`super::convert`] from the checkpoint's `config.json`, `generation_config.json` and
//! `processor_config.json`).

use lapid::{Error, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The flow language model backbone (a causal transformer over voice, text and latents).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LmConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub ffn_dim: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub latent_dim: usize,
    pub vocab_size: usize,
    /// Width of the voice-prompt latents (the speaker codec's), projected to `d_model`.
    pub audio_prompt_dim: usize,
}

/// The LSD flow head sampling one latent per frame from the backbone state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlowConfig {
    pub dim: usize,
    pub depth: usize,
    pub frequency_embedding_size: usize,
    pub norm_eps: f32,
    pub time_norm_eps: f32,
    /// `"lsd"` (two time conditions): the only flow type released checkpoints use.
    pub flow_type: String,
}

/// Constant conditions summed onto every audio-latent position.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConditionConfig {
    pub num_speakers: usize,
    pub duration_delta: f32,
    pub duration_delta_dim: usize,
    pub duration_delta_scale: f32,
    pub duration_delta_max_period: f32,
}

/// The Mimi decoder transformer (runs at the codec's inner frame rate).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MimiTransformerConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub ffn_dim: usize,
    pub rope_theta: f32,
    /// Keys visible to a query: `0 <= pos_q - pos_k < window` (= context + 1, as the
    /// reference runtime / HF `sliding_window`).
    pub window: usize,
    pub norm_eps: f32,
}

/// The Mimi decoder: latent projection, depthwise transposed-conv upsampling, transformer,
/// SEANet decoder.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MimiConfig {
    pub sample_rate: u32,
    pub frame_rate: f32,
    pub dimension: usize,
    pub latent_dim: usize,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    pub ratios: Vec<usize>,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub residual_kernel_size: usize,
    pub dilation_base: usize,
    pub compress: usize,
    /// Inner frames per latent frame (`sample_rate / frame_rate / prod(ratios)`).
    pub upsample_stride: usize,
    pub channels: usize,
    pub transformer: MimiTransformerConfig,
}

impl MimiConfig {
    /// Audio samples per latent frame.
    pub fn frame_size(&self) -> usize {
        self.upsample_stride * self.ratios.iter().product::<usize>()
    }
}

/// Generation defaults (generation_config.json + processor_config.json).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub temperature: f32,
    pub eos_threshold: f32,
    pub num_flow_steps: usize,
    pub min_frames_before_eos: usize,
    pub frames_after_eos: usize,
    pub frames_after_eos_short_text: usize,
    pub short_text_max_words: usize,
    pub fade_in_duration: f32,
}

/// Placeholder token ids of the prompt layout.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenIds {
    pub pad: u32,
    pub audio_bos: u32,
    pub audio_prompt: u32,
    pub audio_latent: u32,
}

/// Names of the non-tensor assets inside the `.lapid` file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Assets {
    /// HF `tokenizer.json` (Unigram + byte fallback).
    pub tokenizer: String,
    /// SentencePiece `tokenizer.model` (same vocabulary), if shipped.
    pub tokenizer_model: Option<String>,
    /// Tensor-name prefix of the voices (`voices.<name>`, [frames, audio_prompt_dim]).
    pub voice_prefix: String,
    pub default_voice: String,
}

/// Everything the runtime needs, `metadata["phonon"]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhononConfig {
    pub lm: LmConfig,
    pub flow: FlowConfig,
    pub conditions: ConditionConfig,
    pub mimi: MimiConfig,
    pub generation: GenerationConfig,
    pub tokens: TokenIds,
    pub assets: Assets,
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

/// `PhononConfig` from the package's JSON files (`config.json`, `generation_config.json`,
/// `processor_config.json`).
pub fn from_hf(cfg: &Value, generation: Option<&Value>, processor: Option<&Value>) -> Result<PhononConfig> {
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
            tokenizer: "tokenizer/tokenizer.json".to_string(),
            tokenizer_model: None,
            voice_prefix: "voices.".to_string(),
            default_voice: "default".to_string(),
        },
    })
}

impl PhononConfig {
    pub fn from_metadata(metadata: &Value) -> Result<PhononConfig> {
        let v = metadata
            .get("phonon")
            .ok_or_else(|| Error::new("metadata", "not a phonon model: metadata has no 'phonon' object"))?;
        let cfg: PhononConfig =
            serde_json::from_value(v.clone()).map_err(|e| Error::new("metadata", format!("phonon metadata: {e}")))?;
        cfg.check()?;
        Ok(cfg)
    }

    fn check(&self) -> Result<()> {
        let bad = |m: String| Err(Error::new("metadata", format!("phonon metadata: {m}")));
        if self.lm.n_heads == 0 || !self.lm.d_model.is_multiple_of(self.lm.n_heads) {
            return bad(format!("d_model {} not divisible by {} heads", self.lm.d_model, self.lm.n_heads));
        }
        let t = &self.mimi.transformer;
        if t.n_heads == 0 || !t.d_model.is_multiple_of(t.n_heads) || t.window == 0 {
            return bad("invalid mimi transformer shape".into());
        }
        if self.mimi.ratios.is_empty() || self.mimi.upsample_stride == 0 || self.mimi.ratios.contains(&0) {
            return bad("invalid mimi ratios / upsample stride".into());
        }
        if self.flow.flow_type != "lsd" {
            return bad(format!("flow type {:?} is not supported (lsd)", self.flow.flow_type));
        }
        if self.conditions.duration_delta_dim < 4 || !self.conditions.duration_delta_dim.is_multiple_of(2) {
            return bad("duration_delta_dim must be even and >= 4".into());
        }
        Ok(())
    }
}
