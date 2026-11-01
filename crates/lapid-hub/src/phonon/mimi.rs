//! Streaming Mimi decoder: codec latents [B, T, 32] → 24 kHz audio, frame by frame.
//!
//! `quantizer.output_proj` (1x1 conv) → depthwise transposed-conv upsampling (12.5 Hz →
//! 200 Hz) → 2-layer transformer (sliding window, layer scale) → SEANet decoder (causal
//! convs with streaming state, ELU, transposed-conv upsampling 6·5·4, residual blocks).
//! All state (conv context, transposed-conv partial sums, windowed KV cache) lives in
//! [`MimiState`]; a chunk of any number of frames yields `frames * 1920` samples, and the
//! concatenation equals one full causal decode.

use lapid::{Device, Result, Tensor};
use lapid_nn::{Conv1d, Loader};

use lapid_nn::KvCache;

use super::config::MimiConfig;
use super::convtr::ConvTranspose1d;
use super::transformer::{Transformer, TransformerSpec};

/// ELU (alpha 1): max(x, 0) + exp(min(x, 0)) - 1.
fn elu(x: &Tensor) -> Result<Tensor> {
    let pos = x.binary_scalar(lapid::BinaryOp::Max, 0.0)?;
    let neg = x.binary_scalar(lapid::BinaryOp::Min, 0.0)?.exp()?.add_scalar(-1.0)?;
    pos.add(&neg)
}

enum Layer {
    Elu,
    /// Causal conv; index of its state slot.
    Conv(Conv1d, usize),
    /// Transposed conv; index of its state slot.
    ConvTr(ConvTranspose1d, usize),
    /// x + conv2(elu(conv1(elu(x)))); state slots of both convs.
    Res(Conv1d, usize, Conv1d, usize),
}

/// Streaming state of one decode stream.
pub(crate) struct MimiState {
    convs: Vec<Option<Tensor>>,
    cache: KvCache,
}

impl MimiState {
    /// Lazy state tensors to evaluate with the step outputs (then [`commit`](Self::commit)).
    pub fn staged(&self) -> Vec<&Tensor> {
        let mut v: Vec<&Tensor> = self.convs.iter().flatten().collect();
        v.extend(self.cache.staged());
        v
    }

    pub fn commit(&mut self, realized: Vec<Tensor>) -> Result<()> {
        let n = self.convs.iter().flatten().count();
        if realized.len() < n {
            lapid::bail!("internal", "mimi state commit: {} tensors for {n} conv states", realized.len());
        }
        let mut it = realized.into_iter();
        for s in self.convs.iter_mut() {
            if s.is_some() {
                *s = it.next();
            }
        }
        self.cache.commit(it.collect())
    }

    pub fn discard(&mut self) {
        self.cache.discard();
    }
}

pub(crate) struct MimiDecoder {
    latent_proj: Conv1d,
    upsample: ConvTranspose1d,
    up_slot: usize,
    transformer: Transformer,
    layers: Vec<(usize, Layer)>,
    n_states: usize,
}

impl MimiDecoder {
    pub fn load(loader: &Loader, cfg: &MimiConfig, device: &Device) -> Result<MimiDecoder> {
        let t = &cfg.transformer;
        let spec = TransformerSpec {
            n_layers: t.n_layers,
            n_heads: t.n_heads,
            d_model: t.d_model,
            rope_theta: t.rope_theta,
            norm_eps: t.norm_eps,
            window: Some(t.window),
            layer_scale: true,
        };
        let transformer = Transformer::load(loader, "mimi.decoder_transformer.transformer", &spec)?;
        let up_w = loader.tensor("mimi.upsample.convtr.convtr.weight")?;
        let upsample = ConvTranspose1d::new(&up_w, None, cfg.upsample_stride, cfg.dimension, device)?;

        let mut n_states = 0usize;
        let mut slot = || {
            n_states += 1;
            n_states - 1
        };
        let up_slot = slot();
        let p = "mimi.decoder.model";
        let conv = |idx: usize, sub: &str, dilation: usize| -> Result<Conv1d> {
            Ok(loader.conv1d(&format!("{p}.{idx}{sub}.conv"), true)?.with_dilation(dilation))
        };
        let mut layers = Vec::new();
        let mut idx = 0usize;
        layers.push((idx, Layer::Conv(conv(idx, "", 1)?, slot())));
        idx += 1;
        for &ratio in &cfg.ratios {
            layers.push((idx, Layer::Elu));
            idx += 1;
            let w = loader.tensor(&format!("{p}.{idx}.convtr.weight"))?;
            let b = loader.tensor(&format!("{p}.{idx}.convtr.bias"))?;
            layers.push((idx, Layer::ConvTr(ConvTranspose1d::new(&w, Some(b), ratio, 1, device)?, slot())));
            idx += 1;
            for j in 0..cfg.n_residual_layers {
                let c1 = conv(idx, ".block.1", cfg.dilation_base.pow(j as u32))?;
                let c2 = conv(idx, ".block.3", 1)?;
                let (s1, s2) = (slot(), slot());
                layers.push((idx, Layer::Res(c1, s1, c2, s2)));
                idx += 1;
            }
        }
        layers.push((idx, Layer::Elu));
        idx += 1;
        layers.push((idx, Layer::Conv(conv(idx, "", 1)?, slot())));
        Ok(MimiDecoder {
            latent_proj: loader.conv1d("mimi.quantizer.output_proj", false)?,
            upsample,
            up_slot,
            transformer,
            layers,
            n_states,
        })
    }

    pub fn new_state(&self) -> MimiState {
        MimiState { convs: vec![None; self.n_states], cache: self.transformer.new_cache() }
    }

    /// Decode codec latents [B, T, ldim] (any T >= 1) → audio [B, 1, T * frame_size].
    /// With `trace`, stage outputs are pushed: `mimi.latent_proj` [B, D, T], `mimi.upsample`
    /// [B, D, 16T], `mimi.transformer` [B, 16T, D], `mimi.seanet.{idx}` [B, C, L].
    pub fn decode(
        &self,
        latents: &Tensor,
        state: &mut MimiState,
        mut trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        let mut push = |name: String, t: &Tensor| {
            if let Some(tr) = trace.as_deref_mut() {
                tr.push(t.named(name));
            }
        };
        let x = self.latent_proj.forward(&latents.permute(&[0, 2, 1])?)?; // [B, D, T]
        push("mimi.latent_proj".into(), &x);
        let x = self.upsample.forward_stream(&x, &mut state.convs[self.up_slot])?;
        push("mimi.upsample".into(), &x);
        let h = self.transformer.forward(&x.permute(&[0, 2, 1])?, &mut state.cache, None)?;
        push("mimi.transformer".into(), &h);
        let mut x = h.permute(&[0, 2, 1])?;
        for (idx, layer) in &self.layers {
            x = match layer {
                Layer::Elu => elu(&x)?,
                Layer::Conv(c, s) => c.forward_stream(&x, &mut state.convs[*s])?,
                Layer::ConvTr(c, s) => c.forward_stream(&x, &mut state.convs[*s])?,
                Layer::Res(c1, s1, c2, s2) => {
                    let h = c1.forward_stream(&elu(&x)?, &mut state.convs[*s1])?;
                    let h = c2.forward_stream(&elu(&h)?, &mut state.convs[*s2])?;
                    x.add(&h)?
                }
            };
            push(format!("mimi.seanet.{idx}"), &x);
        }
        Ok(x)
    }
}
