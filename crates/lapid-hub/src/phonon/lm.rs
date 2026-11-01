//! The flow language model: one causal transformer over the stream
//! `[voice prompt frames | text tokens | BOS latent | generated latents...]`, an
//! end-of-speech head and the flow head that samples each next latent.
//!
//! Inputs: voice frames go through `speaker_wavs.output_proj` (no bias), text tokens through
//! the `transcript_in_segment` table, latents (normalized space, the learnt `bos_emb` first)
//! through `input_linear`, plus the constant condition vector (`num_speakers` lookup +
//! sinusoidal `duration_delta`, each projected) added to every latent position.

use lapid::{DType, Result, Tensor};
use lapid_nn::{Embedding, LayerNorm, Linear, Loader};

use lapid_nn::KvCache;

use super::config::PhononConfig;
use super::flow::FlowHead;
use super::transformer::{Transformer, TransformerSpec};

pub(crate) struct FlowLm {
    text_embed: Embedding,
    voice_proj: Linear,
    input_linear: Linear,
    transformer: Transformer,
    out_norm: LayerNorm,
    out_eos: Linear,
    speakers_embed: Embedding,
    speakers_proj: Linear,
    duration_proj: Linear,
    /// Learnt BOS latent [1, 1, ldim].
    bos: Tensor,
    /// Latent statistics [ldim] (realized f32): audio latent = latent * std + mean.
    pub emb_mean: Tensor,
    pub emb_std: Tensor,
    pub flow: FlowHead,
    pub latent_dim: usize,
    pub d_model: usize,
}

/// One backbone step's outputs (lazy).
pub(crate) struct LmStep {
    /// Normalized state of the last position [1, D] (conditions the flow head).
    pub hidden: Tensor,
    /// End-of-speech logit of the last position [1, 1].
    pub eos: Tensor,
}

const COND: &str = "flow_lm.condition_provider.conditioners";

impl FlowLm {
    pub fn load(loader: &Loader, cfg: &PhononConfig) -> Result<FlowLm> {
        let lm = &cfg.lm;
        let spec = TransformerSpec {
            n_layers: lm.n_layers,
            n_heads: lm.n_heads,
            d_model: lm.d_model,
            rope_theta: lm.rope_theta,
            norm_eps: lm.norm_eps,
            window: None,
            layer_scale: false,
        };
        let bos = loader.tensor_f32("flow_lm.bos_emb")?.reshape(&[1, 1, lm.latent_dim])?;
        Ok(FlowLm {
            text_embed: loader.embedding(&format!("{COND}.transcript_in_segment.embed"))?,
            voice_proj: loader.linear(&format!("{COND}.speaker_wavs.output_proj"), false)?,
            input_linear: loader.linear("flow_lm.input_linear", false)?,
            transformer: Transformer::load(loader, "flow_lm.transformer", &spec)?,
            out_norm: loader.layer_norm("flow_lm.out_norm", lm.norm_eps)?,
            out_eos: loader.linear("flow_lm.out_eos", true)?,
            speakers_embed: loader.embedding(&format!("{COND}.num_speakers.embed"))?,
            speakers_proj: loader.linear(&format!("{COND}.num_speakers.output_proj"), false)?,
            duration_proj: loader.linear(&format!("{COND}.duration_delta.output_proj"), false)?,
            bos,
            emb_mean: loader.tensor_f32("flow_lm.emb_mean")?,
            emb_std: loader.tensor_f32("flow_lm.emb_std")?,
            flow: FlowHead::load(loader, &cfg.flow)?,
            latent_dim: lm.latent_dim,
            d_model: lm.d_model,
        })
    }

    pub fn new_cache(&self) -> KvCache {
        self.transformer.new_cache()
    }

    /// The condition vector summed onto every latent position [1, 1, D].
    pub fn condition(
        &self,
        num_speakers: usize,
        duration_delta: f32,
        dd: &super::config::ConditionConfig,
    ) -> Result<Tensor> {
        let sp = self.speakers_embed.lookup(&[num_speakers as u32])?.cast(DType::F32)?;
        let sp = self.speakers_proj.forward(&sp)?;
        let half = dd.duration_delta_dim / 2;
        let phase0 = dd.duration_delta_scale * duration_delta;
        let phases: Vec<f32> =
            (0..half).map(|i| phase0 / dd.duration_delta_max_period.powf(i as f32 / (half - 1) as f32)).collect();
        let mut emb: Vec<f32> = phases.iter().map(|p| p.cos()).collect();
        emb.extend(phases.iter().map(|p| p.sin()));
        let du = self.duration_proj.forward(&Tensor::from_f32(&emb, &[1, 1, emb.len()])?)?;
        sp.add(&du)
    }

    /// Prompt embeddings [1, L, D]: voice frames [F, prompt_dim] (optional), text ids, BOS.
    pub fn prompt_embeds(&self, voice: Option<&Tensor>, text: &[u32], cond: &Tensor) -> Result<Tensor> {
        let mut parts = Vec::new();
        if let Some(v) = voice {
            parts.push(self.voice_proj.forward(&v.unsqueeze(0)?)?);
        }
        parts.push(self.text_embed.lookup(text)?.cast(DType::F32)?);
        parts.push(self.input_linear.forward(&self.bos)?.add(cond)?);
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 1)
    }

    /// Embedding of normalized latents [1, n, ldim].
    pub fn latent_embeds(&self, latents: &Tensor, cond: &Tensor) -> Result<Tensor> {
        self.input_linear.forward(latents)?.add(cond)
    }

    /// Backbone over x [1, S, D] (appending to `cache`): state and EOS logit of the last
    /// position. `trace` receives `lm.layers.{i}` and `lm.out` (all positions).
    pub fn step(&self, x: &Tensor, cache: &mut KvCache, mut trace: Option<&mut Vec<Tensor>>) -> Result<LmStep> {
        let h = self.transformer.forward(x, cache, trace.as_deref_mut().map(|t| ("lm.layers", t)))?;
        let h = self.out_norm.forward(&h)?;
        if let Some(t) = trace {
            t.push(h.named("lm.out"));
        }
        let s = h.shape()[1];
        let last = h.slice(1, s - 1, s)?.reshape(&[1, self.d_model])?;
        let eos = self.out_eos.forward(&last)?;
        Ok(LmStep { hidden: last, eos })
    }

    /// Normalized latent → codec latent: `x * std + mean`.
    pub fn denormalize(&self, latent: &Tensor) -> Result<Tensor> {
        latent.mul(&self.emb_std)?.add(&self.emb_mean)
    }
}
