//! Phonon hyper-parameters, stored in the `.lapid` metadata as `metadata["phonon"]` (written
//! by [`super::convert`] from the checkpoint's `config.json`, `generation_config.json` and
//! `processor_config.json`).

use lapid::{Error, Result};
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
    /// `"lsd"` (two time conditions) — the only flow type released checkpoints use.
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
