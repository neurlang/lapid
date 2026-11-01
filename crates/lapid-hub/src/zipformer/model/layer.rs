//! `Zipformer2EncoderLayer.streaming_forward` on one chunk x [T, C] with cached left context.
//!
//! ```text
//! attn = softmax(q k^T + rel_shift(p pos^T) + key_padding_mask)    k, pos: [L cached ‖ T]
//! x += ff1(x); x += nonlin_attn(x, attn[0]); x += self_attn1(x, attn); x += conv1(x)
//! x += ff2(x); x = bypass_mid(orig, x); x += self_attn2(x, attn); x += conv2(x)
//! x += ff3(x); x = bypass(orig, norm(x))
//! ```
//!
//! The attention weights are explicit [H, T, L + T] (they feed three modules); the position
//! scores use the flat-view relative shift `y[i, j] = s[i, T - 1 - i + j]`.

use lapid::{Result, Tensor, bail};
use lapid_nn::speech::{DepthwiseConv, rel_shift};
use lapid_nn::{Linear, Loader};

use super::{BiasNorm, bypass, host, linear, param, swoosh_l, swoosh_r};
use crate::zipformer::config::ZipformerConfig;

/// Cached left context of one layer (time-major).
#[derive(Clone)]
pub(crate) struct LayerState {
    /// [L, H * query_head_dim]
    pub key: Tensor,
    /// [L, nonlin_dim]
    pub nonlin: Tensor,
    /// [L, H * value_head_dim] (self_attn1 / self_attn2)
    pub val1: Tensor,
    pub val2: Tensor,
    /// [kernel / 2, C] (conv_module1 / conv_module2)
    pub conv1: Tensor,
    pub conv2: Tensor,
}

pub(crate) const STATE_NAMES: [&str; 6] = ["key", "nonlin", "val1", "val2", "conv1", "conv2"];

impl LayerState {
    pub fn tensors(&self) -> [&Tensor; 6] {
        [&self.key, &self.nonlin, &self.val1, &self.val2, &self.conv1, &self.conv2]
    }

    pub fn from_tensors(v: &[Tensor]) -> Result<LayerState> {
        match v {
            [key, nonlin, val1, val2, conv1, conv2] => Ok(LayerState {
                key: key.clone(),
                nonlin: nonlin.clone(),
                val1: val1.clone(),
                val2: val2.clone(),
                conv1: conv1.clone(),
                conv2: conv2.clone(),
            }),
            _ => bail!("internal", "layer state needs 6 tensors, got {}", v.len()),
        }
    }
}

/// Per-chunk inputs shared by the layers of one stack.
pub(crate) struct LayerCtx<'a> {
    /// Relative positions -(T+L-1) .. T-1: [L + 2T - 1, pos_dim].
    pub pos: &'a Tensor,
    /// Additive key padding mask [1, L + T] (0 or -1000), `None` once the cache is full.
    pub mask: Option<&'a Tensor>,
    /// Chunk-conv scales [T, C] of this layer's two conv modules.
    pub scale1: &'a Tensor,
    pub scale2: &'a Tensor,
}

struct FeedForward {
    inp: Linear,
    out: Linear,
}

impl FeedForward {
    fn load(l: &Loader<'_>, p: &str, c: usize, hidden: usize) -> Result<FeedForward> {
        Ok(FeedForward { inp: linear(l, &format!("{p}.in_proj"), hidden, c)?, out: linear(l, &format!("{p}.out_proj"), c, hidden)? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.out.forward(&swoosh_l(&self.inp.forward(x)?)?)
    }
}

/// `ConvolutionModule` with `ChunkCausalDepthwiseConv1d`.
struct ConvModule {
    inp: Linear,
    /// Causal half-kernel over [cache ‖ chunk], and the full kernel within the chunk.
    causal: DepthwiseConv,
    chunk: DepthwiseConv,
    /// `chunkwise_conv_scale` [2, C, K] (host).
    edge: Vec<f32>,
    out: Linear,
    c: usize,
    k: usize,
}

impl ConvModule {
    fn load(l: &Loader<'_>, p: &str, c: usize, k: usize) -> Result<ConvModule> {
        let half = k.div_ceil(2);
        let causal = l.tensor_f32(&format!("{p}.depthwise_conv.causal_conv.weight"))?;
        l.expect_shape(&causal, &format!("{p}.depthwise_conv.causal_conv.weight"), &[c, half])?;
        let chunk = l.tensor_f32(&format!("{p}.depthwise_conv.chunkwise_conv.weight"))?;
        l.expect_shape(&chunk, &format!("{p}.depthwise_conv.chunkwise_conv.weight"), &[c, k])?;
        let edge = host(l, &format!("{p}.depthwise_conv.chunkwise_conv_scale"))?;
        if edge.len() != 2 * c * k {
            bail!("shape_mismatch", "{p}.depthwise_conv.chunkwise_conv_scale: {} values, expected 2 x {c} x {k}", edge.len());
        }
        Ok(ConvModule {
            inp: linear(l, &format!("{p}.in_proj"), 2 * c, c)?,
            causal: DepthwiseConv::new(&causal, Some(param(l, &format!("{p}.depthwise_conv.causal_conv.bias"), c)?))?,
            chunk: DepthwiseConv::new(&chunk, Some(param(l, &format!("{p}.depthwise_conv.chunkwise_conv.bias"), c)?))?,
            edge,
            out: linear(l, &format!("{p}.out_proj"), c, c)?,
            c,
            k,
        })
    }

    /// `_get_chunk_scale(T)` time-major: [T, C] = 1 + left edge + right edge.
    fn chunk_scale(&self, t: usize) -> Vec<f32> {
        let (c, k) = (self.c, self.k);
        let left = |ch: usize, i: usize| self.edge[ch * k + i];
        let right = |ch: usize, i: usize| self.edge[c * k + ch * k + i];
        let mut out = vec![1.0f32; t * c];
        for ti in 0..t {
            for ch in 0..c {
                let mut s = 1.0;
                if t < k {
                    s += left(ch, ti) + right(ch, k - t + ti);
                } else {
                    if ti < k {
                        s += left(ch, ti);
                    }
                    if ti >= t - k {
                        s += right(ch, ti - (t - k));
                    }
                }
                out[ti * c + ch] = s;
            }
        }
        out
    }

    /// x [T, C], cache [K/2, C], scale [T, C] -> (y [T, C], new cache).
    fn forward(&self, x: &Tensor, cache: &Tensor, scale: &Tensor) -> Result<(Tensor, Tensor)> {
        let (t, c, pad) = (x.shape()[0], self.c, self.k / 2);
        let h = self.inp.forward(x)?;
        let h = h.slice(1, 0, c)?.mul(&lapid_nn::ops::sigmoid(&h.slice(1, c, 2 * c)?)?)?;
        let xc = Tensor::cat(&[cache, &h], 0)?; // [pad + T, C]
        let new_cache = xc.slice(0, t, t + pad)?.contiguous()?;
        let causal = self.causal.forward(&xc, 0, (0, 0))?;
        let chunk = self.chunk.forward(&h, 0, (pad, pad))?;
        let y = chunk.mul(scale)?.add(&causal)?;
        Ok((self.out.forward(&swoosh_r(&y)?)?, new_cache))
    }
}

/// `RelPositionMultiheadAttentionWeights`.
struct AttnWeights {
    inp: Linear,
    pos: Linear,
    heads: usize,
    qd: usize,
    pd: usize,
}

/// [T, H * d] -> [H, T, d] (contiguous).
fn heads(x: &Tensor, h: usize) -> Result<Tensor> {
    let (t, hd) = (x.shape()[0], x.shape()[1]);
    x.contiguous()?.reshape(&[t, h, hd / h])?.permute(&[1, 0, 2])?.contiguous()
}

impl AttnWeights {
    /// (attention weights [H, T, L + T], new key cache [L, H * qd]).
    fn forward(&self, x: &Tensor, key_cache: &Tensor, ctx: &LayerCtx<'_>) -> Result<(Tensor, Tensor)> {
        let (t, l) = (x.shape()[0], key_cache.shape()[0]);
        let (hq, hp) = (self.heads * self.qd, self.heads * self.pd);
        let proj = self.inp.forward(x)?;
        let q = heads(&proj.slice(1, 0, hq)?, self.heads)?;
        let kf = Tensor::cat(&[key_cache, &proj.slice(1, hq, 2 * hq)?], 0)?; // [L + T, hq]
        let new_key = kf.slice(0, t, t + l)?.contiguous()?;
        let k = heads(&kf, self.heads)?;
        let scores = q.matmul_t(&k)?; // [H, T, L + T]
        let p = heads(&proj.slice(1, 2 * hq, 2 * hq + hp)?, self.heads)?; // [H, T, pd]
        let pe = heads(&self.pos.forward(ctx.pos)?, self.heads)?; // [H, n, pd]
        let pos_scores = rel_shift(&p.matmul_t(&pe)?, l + t)?;
        let mut scores = scores.add(&pos_scores)?;
        if let Some(m) = ctx.mask {
            scores = scores.add(m)?;
        }
        Ok((scores.softmax()?, new_key))
    }
}

/// `SelfAttention` (values with cached left context).
struct SelfAttention {
    inp: Linear,
    out: Linear,
    heads: usize,
}

impl SelfAttention {
    fn forward(&self, x: &Tensor, attn: &Tensor, cache: &Tensor) -> Result<(Tensor, Tensor)> {
        let (t, l) = (x.shape()[0], cache.shape()[0]);
        let vf = Tensor::cat(&[cache, &self.inp.forward(x)?], 0)?;
        let new_cache = vf.slice(0, t, t + l)?.contiguous()?;
        let o = attn.matmul(&heads(&vf, self.heads)?)?; // [H, T, vd]
        let hv = o.shape()[0] * o.shape()[2];
        let o = o.permute(&[1, 0, 2])?.contiguous()?.reshape(&[t, hv])?;
        Ok((self.out.forward(&o)?, new_cache))
    }
}

/// `NonlinAttention` (single head over the first head's weights).
struct NonlinAttention {
    inp: Linear,
    out: Linear,
    hidden: usize,
}

impl NonlinAttention {
    fn forward(&self, x: &Tensor, attn: &Tensor, cache: &Tensor) -> Result<(Tensor, Tensor)> {
        let (t, l, hc) = (x.shape()[0], cache.shape()[0], self.hidden);
        let h = self.inp.forward(x)?;
        let s = h.slice(1, 0, hc)?.tanh()?;
        let v = h.slice(1, hc, 2 * hc)?.mul(&s)?;
        let y = h.slice(1, 2 * hc, 3 * hc)?;
        let vf = Tensor::cat(&[cache, &v], 0)?; // [L + T, hc]
        let new_cache = vf.slice(0, t, t + l)?.contiguous()?;
        let a0 = attn.slice(0, 0, 1)?.squeeze(0)?.contiguous()?; // [T, L + T]
        let o = a0.matmul(&vf)?.mul(&y)?;
        Ok((self.out.forward(&o)?, new_cache))
    }
}

pub(crate) struct Layer {
    weights: AttnWeights,
    nonlin: NonlinAttention,
    sa1: SelfAttention,
    sa2: SelfAttention,
    ff1: FeedForward,
    ff2: FeedForward,
    ff3: FeedForward,
    conv1: ConvModule,
    conv2: ConvModule,
    norm: BiasNorm,
    bypass: Tensor,
    bypass_mid: Tensor,
}

impl Layer {
    pub fn load(l: &Loader<'_>, p: &str, cfg: &ZipformerConfig, i: usize) -> Result<Layer> {
        let c = cfg.encoder_dim[i];
        let (h, qd, vd, pd) = (cfg.num_heads[i], cfg.query_head_dim[i], cfg.value_head_dim[i], cfg.pos_head_dim[i]);
        let (ff, k, hc) = (cfg.feedforward_dim[i], cfg.cnn_module_kernel[i], cfg.nonlin_dim(i));
        let sa = |n: &str| -> Result<SelfAttention> {
            Ok(SelfAttention { inp: linear(l, &format!("{p}.{n}.in_proj"), h * vd, c)?, out: linear(l, &format!("{p}.{n}.out_proj"), c, h * vd)?, heads: h })
        };
        Ok(Layer {
            weights: AttnWeights {
                inp: linear(l, &format!("{p}.self_attn_weights.in_proj"), h * (2 * qd + pd), c)?,
                pos: linear(l, &format!("{p}.self_attn_weights.linear_pos"), h * pd, cfg.pos_dim)?,
                heads: h,
                qd,
                pd,
            },
            nonlin: NonlinAttention {
                inp: linear(l, &format!("{p}.nonlin_attention.in_proj"), 3 * hc, c)?,
                out: linear(l, &format!("{p}.nonlin_attention.out_proj"), c, hc)?,
                hidden: hc,
            },
            sa1: sa("self_attn1")?,
            sa2: sa("self_attn2")?,
            ff1: FeedForward::load(l, &format!("{p}.feed_forward1"), c, ff * 3 / 4)?,
            ff2: FeedForward::load(l, &format!("{p}.feed_forward2"), c, ff)?,
            ff3: FeedForward::load(l, &format!("{p}.feed_forward3"), c, ff * 5 / 4)?,
            conv1: ConvModule::load(l, &format!("{p}.conv_module1"), c, k)?,
            conv2: ConvModule::load(l, &format!("{p}.conv_module2"), c, k)?,
            norm: BiasNorm::load(l, &format!("{p}.norm"), c)?,
            bypass: param(l, &format!("{p}.bypass.bypass_scale"), c)?,
            bypass_mid: param(l, &format!("{p}.bypass_mid.bypass_scale"), c)?,
        })
    }

    /// Chunk-conv scales [T, C] of conv_module1 / conv_module2 (host).
    pub fn chunk_scales(&self, t: usize) -> (Vec<f32>, Vec<f32>) {
        (self.conv1.chunk_scale(t), self.conv2.chunk_scale(t))
    }

    pub fn forward(&self, x: &Tensor, ctx: &LayerCtx<'_>, st: &LayerState) -> Result<(Tensor, LayerState)> {
        let orig = x;
        let (attn, key) = self.weights.forward(x, &st.key, ctx)?;
        let x = x.add(&self.ff1.forward(x)?)?;
        let (na, nonlin) = self.nonlin.forward(&x, &attn, &st.nonlin)?;
        let x = x.add(&na)?;
        let (a1, val1) = self.sa1.forward(&x, &attn, &st.val1)?;
        let x = x.add(&a1)?;
        let (c1, conv1) = self.conv1.forward(&x, &st.conv1, ctx.scale1)?;
        let x = x.add(&c1)?;
        let x = x.add(&self.ff2.forward(&x)?)?;
        let x = bypass(orig, &x, &self.bypass_mid)?;
        let (a2, val2) = self.sa2.forward(&x, &attn, &st.val2)?;
        let x = x.add(&a2)?;
        let (c2, conv2) = self.conv2.forward(&x, &st.conv2, ctx.scale2)?;
        let x = x.add(&c2)?;
        let x = x.add(&self.ff3.forward(&x)?)?;
        let x = bypass(orig, &self.norm.forward(&x)?, &self.bypass)?;
        Ok((x, LayerState { key, nonlin, val1, val2, conv1, conv2 }))
    }
}
