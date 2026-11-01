//! `Zipformer2.streaming_forward` + `encoder_embed` + `joiner.encoder_proj` for one chunk.
//!
//! Stack `i` runs at 50 / ds_i Hz: `SimpleDownsample` (softmax-weighted sum of ds frames),
//! its layers with `left / ds_i` cached frames, `SimpleUpsample` (repeat) and the
//! `out_combiner` bypass. Stack inputs are cut / zero-padded to the stack width
//! (`convert_num_channels`); the output takes each channel from the most recent stack that
//! has it, then the final `SimpleDownsample(2)` and the joiner's encoder projection.
//!
//! A [`Plan`] holds what is constant for a (chunk, left context) pair: relative position
//! embeddings per stack and chunk-conv scales per layer, uploaded once.

use lapid::{DType, Device, Result, Tensor, bail};
use lapid_nn::{Linear, Loader};

use super::embed::Embed;
use super::layer::{Layer, LayerCtx, LayerState};
use super::{bypass, host, linear, param};
use crate::zipformer::config::ZipformerConfig;

/// Encoder state carried across chunks.
#[derive(Clone)]
pub(crate) struct EncoderState {
    /// One per layer, stacks in order.
    pub layers: Vec<LayerState>,
    /// ConvNeXt cache [3, W, 128].
    pub embed: Tensor,
    /// Encoder frames (50 Hz) processed so far (drives the key padding mask).
    pub processed: usize,
}

impl EncoderState {
    /// Every state tensor: embed cache, then 6 per layer.
    pub fn tensors(&self) -> Vec<&Tensor> {
        let mut v = vec![&self.embed];
        for l in &self.layers {
            v.extend(l.tensors());
        }
        v
    }

    /// Rebuild from `tensors()` order (realized results).
    pub fn from_tensors(v: &[Tensor], processed: usize) -> Result<EncoderState> {
        let Some((embed, rest)) = v.split_first() else { bail!("internal", "empty encoder state") };
        if rest.len() % 6 != 0 {
            bail!("internal", "encoder state: {} layer tensors", rest.len());
        }
        let layers = rest.chunks(6).map(LayerState::from_tensors).collect::<Result<_>>()?;
        Ok(EncoderState { layers, embed: embed.clone(), processed })
    }
}

struct Stack {
    layers: Vec<Layer>,
    ds: usize,
    /// softmax(downsample.bias) as [ds, 1] (ds > 1).
    down: Option<Tensor>,
    combiner: Option<Tensor>,
    dim: usize,
}

/// Constants of one streaming configuration.
pub(crate) struct Plan {
    pub chunk: usize,
    pub left: usize,
    /// Per stack: relative positions [L_i + 2 T_i - 1, pos_dim].
    pos: Vec<Tensor>,
    /// Per layer: chunk-conv scales [T_i, C_i] of its two conv modules.
    scales: Vec<(Tensor, Tensor)>,
}

pub(crate) struct Encoder {
    embed: Embed,
    stacks: Vec<Stack>,
    /// softmax(downsample_output.bias) [2, 1].
    out_down: Tensor,
    proj: Linear,
    cfg: ZipformerConfig,
}

/// One chunk's lazy graph.
pub(crate) struct ChunkGraph {
    /// Projected encoder frames [chunk / 2, joiner_dim].
    pub out: Tensor,
    pub state: EncoderState,
    /// "embed", "stack.{i}" (with `trace`).
    pub traced: Vec<Tensor>,
}

/// icefall `CompactRelPositionalEncoding` rows for offsets -(T+L-1) ..= T-1: [L + 2T - 1, d].
pub(crate) fn rel_positions(t: usize, l: usize, d: usize) -> Vec<f32> {
    let n = 2 * t + l - 1;
    let cl = (d as f64).sqrt();
    let length_scale = d as f64 / (2.0 * std::f64::consts::PI);
    let mut pe = vec![0.0f32; n * d];
    for r in 0..n {
        let x = r as f64 - (t + l - 1) as f64;
        let xc = cl * x.signum() * ((x.abs() + cl).ln() - cl.ln());
        let xa = (xc / length_scale).atan();
        for i in 0..d / 2 {
            let a = xa * (i + 1) as f64;
            pe[r * d + 2 * i] = a.cos() as f32;
            pe[r * d + 2 * i + 1] = a.sin() as f32;
        }
        pe[r * d + d - 1] = 1.0;
    }
    pe
}

/// Host softmax of a small bias vector as an uploaded [n, 1] tensor.
fn softmax_weights(l: &Loader<'_>, name: &str, n: usize) -> Result<Tensor> {
    let b = host(l, name)?;
    if b.len() != n {
        bail!("shape_mismatch", "{name}: {} values, expected {n}", b.len());
    }
    let m = b.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = b.iter().map(|&v| f64::from(v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    let w: Vec<f32> = e.iter().map(|v| (v / s) as f32).collect();
    l.device.upload(&Tensor::from_f32(&w, &[n, 1])?)
}

/// `SimpleDownsample`: x [T, C] (T divisible by ds) -> [T / ds, C].
fn downsample(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let (t, c, ds) = (x.shape()[0], x.shape()[1], w.shape()[0]);
    if t % ds != 0 {
        bail!("invalid_argument", "zipformer downsample: {t} frames not a multiple of {ds}");
    }
    x.contiguous()?.reshape(&[t / ds, ds, c])?.mul(w)?.sum(1, false)
}

/// `SimpleUpsample`: [T, C] -> [T ds, C] (each frame repeated).
fn upsample(x: &Tensor, ds: usize) -> Result<Tensor> {
    let (t, c) = (x.shape()[0], x.shape()[1]);
    x.reshape(&[t, 1, c])?.expand(&[t, ds, c])?.contiguous()?.reshape(&[t * ds, c])
}

/// `convert_num_channels`: cut or zero-pad the last axis to `c`.
fn to_channels(x: &Tensor, c: usize) -> Result<Tensor> {
    let have = x.shape()[1];
    if c <= have { x.slice(1, 0, c) } else { x.pad(1, 0, c - have, 0.0) }
}

impl Encoder {
    pub fn load(l: &Loader<'_>, cfg: &ZipformerConfig) -> Result<Encoder> {
        let mut stacks = Vec::with_capacity(cfg.num_stacks());
        for i in 0..cfg.num_stacks() {
            let ds = cfg.downsampling_factor[i];
            let base = if ds > 1 { format!("encoder.encoders.{i}.encoder") } else { format!("encoder.encoders.{i}") };
            let layers = (0..cfg.num_encoder_layers[i])
                .map(|j| Layer::load(l, &format!("{base}.layers.{j}"), cfg, i))
                .collect::<Result<_>>()?;
            let (down, combiner) = if ds > 1 {
                (
                    Some(softmax_weights(l, &format!("encoder.encoders.{i}.downsample.bias"), ds)?),
                    Some(param(l, &format!("encoder.encoders.{i}.out_combiner.bypass_scale"), cfg.encoder_dim[i])?),
                )
            } else {
                (None, None)
            };
            stacks.push(Stack { layers, ds, down, combiner, dim: cfg.encoder_dim[i] });
        }
        Ok(Encoder {
            embed: Embed::load(l, cfg)?,
            stacks,
            out_down: softmax_weights(l, "encoder.downsample_output.bias", 2)?,
            proj: linear(l, "joiner.encoder_proj", cfg.joiner_dim, cfg.max_dim())?,
            cfg: cfg.clone(),
        })
    }

    /// Constants for `chunk` / `left` (encoder frames at 50 Hz), uploaded to `device`.
    pub fn plan(&self, chunk: usize, left: usize, device: &Device) -> Result<Plan> {
        self.cfg.check_stream(chunk, left)?;
        let d = self.cfg.pos_dim;
        let mut pos = Vec::new();
        let mut scales = Vec::new();
        for s in &self.stacks {
            let (t, l) = (chunk / s.ds, left / s.ds);
            pos.push(device.upload(&Tensor::from_f32(&rel_positions(t, l, d), &[l + 2 * t - 1, d])?)?);
            for layer in &s.layers {
                let (a, b) = layer.chunk_scales(t);
                scales.push((device.upload(&Tensor::from_f32(&a, &[t, s.dim])?)?, device.upload(&Tensor::from_f32(&b, &[t, s.dim])?)?));
            }
        }
        Ok(Plan { chunk, left, pos, scales })
    }

    /// Zero state for `plan`.
    pub fn init_state(&self, plan: &Plan) -> EncoderState {
        let z = |shape: &[usize]| Tensor::zeros(shape, DType::F32);
        let mut layers = Vec::new();
        for (i, s) in self.stacks.iter().enumerate() {
            let l = plan.left / s.ds;
            let c = &self.cfg;
            for _ in &s.layers {
                let pad = c.cnn_module_kernel[i] / 2;
                layers.push(LayerState {
                    key: z(&[l, c.num_heads[i] * c.query_head_dim[i]]),
                    nonlin: z(&[l, c.nonlin_dim(i)]),
                    val1: z(&[l, c.num_heads[i] * c.value_head_dim[i]]),
                    val2: z(&[l, c.num_heads[i] * c.value_head_dim[i]]),
                    conv1: z(&[pad, s.dim]),
                    conv2: z(&[pad, s.dim]),
                });
            }
        }
        EncoderState { layers, embed: self.embed.init_cache(), processed: 0 }
    }

    /// Additive key padding mask of stack `ds` ([1, L/ds + T/ds]) while the cache is not full:
    /// cached position `k` (oldest first) is masked when `k ds < left - processed`, i.e. not
    /// yet filled (icefall's flipped mask subsampled by ds).
    fn mask(plan: &Plan, ds: usize, processed: usize) -> Result<Option<Tensor>> {
        if processed >= plan.left {
            return Ok(None);
        }
        let (l, t) = (plan.left / ds, plan.chunk / ds);
        let limit = plan.left - processed;
        let m: Vec<f32> = (0..l + t).map(|k| if k < l && k * ds < limit { -1000.0 } else { 0.0 }).collect();
        Ok(Some(Tensor::from_f32(&m, &[1, l + t])?))
    }

    /// Graph of one chunk: fbank frames [2 chunk + 13, F] and the previous state.
    pub fn chunk(&self, plan: &Plan, feats: &Tensor, state: &EncoderState, trace: bool) -> Result<ChunkGraph> {
        let mut traced = Vec::new();
        let (mut x, embed) = self.embed.forward(feats, &state.embed)?;
        if x.shape()[0] != plan.chunk {
            bail!("invalid_argument", "zipformer chunk: embed gave {} frames, plan expects {}", x.shape()[0], plan.chunk);
        }
        if trace {
            x = x.named("embed");
            traced.push(x.clone());
        }
        let mut outputs = Vec::with_capacity(self.stacks.len());
        let mut layers_out = Vec::with_capacity(state.layers.len());
        let mut li = 0;
        for (i, s) in self.stacks.iter().enumerate() {
            let orig = to_channels(&x, s.dim)?;
            let mut y = match &s.down {
                Some(w) => downsample(&orig, w)?,
                None => orig.clone(),
            };
            let mask = Self::mask(plan, s.ds, state.processed)?;
            for layer in &s.layers {
                let (a, b) = &plan.scales[li];
                let ctx = LayerCtx { pos: &plan.pos[i], mask: mask.as_ref(), scale1: a, scale2: b };
                let (out, st) = layer.forward(&y, &ctx, &state.layers[li])?;
                y = out;
                layers_out.push(st);
                li += 1;
            }
            if let Some(scale) = &s.combiner {
                let up = upsample(&y, s.ds)?.slice(0, 0, orig.shape()[0])?;
                y = bypass(&orig, &up, scale)?;
            }
            if trace {
                y = y.named(format!("stack.{i}"));
                traced.push(y.clone());
            }
            outputs.push(y.clone());
            x = y;
        }
        let full = self.full_dim(&outputs)?;
        let out = self.proj.forward(&downsample(&full, &self.out_down)?)?.named("encoder_out");
        let state = EncoderState { layers: layers_out, embed, processed: state.processed + plan.chunk };
        Ok(ChunkGraph { out, state, traced })
    }

    /// `_get_full_dim_output`.
    fn full_dim(&self, outputs: &[Tensor]) -> Result<Tensor> {
        let n = self.stacks.len();
        let mut pieces = vec![outputs[n - 1].clone()];
        let mut cur = self.stacks[n - 1].dim;
        for i in (0..n - 1).rev() {
            let d = self.stacks[i].dim;
            if d > cur {
                pieces.push(outputs[i].slice(1, cur, d)?);
                cur = d;
            }
        }
        let refs: Vec<&Tensor> = pieces.iter().collect();
        Tensor::cat(&refs, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_are_symmetric_and_biased() {
        let (t, l, d) = (3, 2, 8);
        let pe = rel_positions(t, l, d);
        let n = 2 * t + l - 1;
        let row = |x: isize| &pe[((x + (t + l - 1) as isize) as usize) * d..][..d];
        // offset 0: atan(0) = 0 -> cos 1, sin 0; last column 1
        assert_eq!(row(0)[0], 1.0);
        assert_eq!(row(0)[1], 0.0);
        assert!(pe.chunks(d).all(|r| r[d - 1] == 1.0));
        // cos even, sin odd in the offset
        assert!((row(2)[2] - row(-2)[2]).abs() < 1e-7 && (row(2)[3] + row(-2)[3]).abs() < 1e-7);
        assert_eq!(pe.len(), n * d);
    }
}
