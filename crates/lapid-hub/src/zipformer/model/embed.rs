//! `Conv2dSubsampling.streaming_forward`: fbank [T, F] -> [(T - 7) / 2 - 3, D].
//!
//! Channels-last [T, F, C] activations:
//! * Conv2d(1 → 8, 3×3, freq pad 1) → SwooshR → Conv2d(8 → 32, 3×3, stride 2) → SwooshR →
//!   Conv2d(32 → 128, 3×3, stride (1, 2)) → SwooshR, each as 9 taps concatenated (im2col) and
//!   one matmul (weights stored [Cout, (kt, kf, Cin)] by the converter);
//! * ConvNeXt: depthwise 7×7 (freq pad 3) over [3 cached frames ‖ chunk] as a fused 49-tap
//!   chain, point-wise 128 → 384 → SwooshL → 128, plus the bypass; the 3 frames before the
//!   3 look-ahead frames become the new cache;
//! * flatten (channel-major like PyTorch's `(c, f)`) → Linear → BiasNorm.

use lapid::{Result, Tensor, bail};
use lapid_nn::speech::{DepthwiseConv, taps_2d};
use lapid_nn::{Linear, Loader};

use super::{BiasNorm, linear, param, swoosh_l, swoosh_r};
use crate::zipformer::config::{EMBED_CHANNELS, EMBED_LAYER1, EMBED_LAYER2, ZipformerConfig};

/// ConvNeXt kernel (7×7) and its time cache (3 frames).
pub(crate) const CONVNEXT_K: usize = 7;
pub(crate) const CONVNEXT_CACHE: usize = CONVNEXT_K / 2;

/// 3×3 conv taps of x [T, F, C] (no time padding): [To, Fo, 9 C] in (kt, kf, c) order.
fn im2col3(x: &Tensor, t_stride: usize, f_stride: usize, f_pad: usize) -> Result<Tensor> {
    let (t, f) = (x.shape()[0], x.shape()[1]);
    if t < 3 {
        bail!("invalid_argument", "zipformer embed: {t} frames, need at least 3");
    }
    let to = (t - 3) / t_stride + 1;
    let fo = (f + 2 * f_pad - 3) / f_stride + 1;
    let xp = if f_pad > 0 { x.pad(1, f_pad, f_pad, 0.0)? } else { x.clone() };
    let taps = taps_2d(&xp, 0, (3, 3), (t_stride, f_stride), (to, fo))?;
    let refs: Vec<&Tensor> = taps.iter().collect();
    Tensor::cat(&refs, 2)
}

pub(crate) struct Embed {
    conv0: Linear,
    conv1: Linear,
    conv2: Linear,
    /// ConvNeXt depthwise 7×7 (weight [128, 49]).
    dw: DepthwiseConv,
    pw1: Linear,
    pw2: Linear,
    out: Linear,
    norm: BiasNorm,
    feature_dim: usize,
    width: usize,
}

impl Embed {
    pub fn load(l: &Loader<'_>, cfg: &ZipformerConfig) -> Result<Embed> {
        let p = "encoder_embed";
        let c = EMBED_CHANNELS;
        let width = cfg.embed_width();
        let dw = l.tensor_f32(&format!("{p}.convnext.depthwise_conv.weight"))?;
        l.expect_shape(&dw, &format!("{p}.convnext.depthwise_conv.weight"), &[c, CONVNEXT_K * CONVNEXT_K])?;
        Ok(Embed {
            conv0: linear(l, &format!("{p}.conv.0"), EMBED_LAYER1, 9)?,
            conv1: linear(l, &format!("{p}.conv.4"), EMBED_LAYER2, 9 * EMBED_LAYER1)?,
            conv2: linear(l, &format!("{p}.conv.7"), c, 9 * EMBED_LAYER2)?,
            dw: DepthwiseConv::new(&dw, Some(param(l, &format!("{p}.convnext.depthwise_conv.bias"), c)?))?,
            pw1: linear(l, &format!("{p}.convnext.pointwise_conv1"), 3 * c, c)?,
            pw2: linear(l, &format!("{p}.convnext.pointwise_conv2"), c, 3 * c)?,
            out: linear(l, &format!("{p}.out"), cfg.encoder_dim[0], c * width)?,
            norm: BiasNorm::load(l, &format!("{p}.out_norm"), cfg.encoder_dim[0])?,
            feature_dim: cfg.feature_dim,
            width,
        })
    }

    /// Zero ConvNeXt cache [3, W, 128].
    pub fn init_cache(&self) -> Tensor {
        Tensor::zeros(&[CONVNEXT_CACHE, self.width, EMBED_CHANNELS], lapid::DType::F32)
    }

    /// fbank chunk [T, F] + cache [3, W, 128] -> (embeddings [(T - 7) / 2 - 3, D], new cache).
    pub fn forward(&self, feats: &Tensor, cache: &Tensor) -> Result<(Tensor, Tensor)> {
        let t = feats.shape()[0];
        if feats.shape() != [t, self.feature_dim] || t < 7 + 2 * CONVNEXT_CACHE + 2 {
            bail!("invalid_argument", "zipformer embed: features {:?}, expected [>= 15, {}]", feats.shape(), self.feature_dim);
        }
        let x = feats.reshape(&[t, self.feature_dim, 1])?;
        let x = swoosh_r(&self.conv0.forward(&im2col3(&x, 1, 1, 1)?)?)?;
        let x = swoosh_r(&self.conv1.forward(&im2col3(&x, 2, 2, 0)?)?)?;
        let x = swoosh_r(&self.conv2.forward(&im2col3(&x, 1, 2, 0)?)?)?; // [T3, W, 128]
        let t3 = x.shape()[0];
        let t_out = t3 - CONVNEXT_CACHE;
        let bypass = x.slice(0, 0, t_out)?;
        let xc = Tensor::cat(&[cache, &x], 0)?; // [T3 + 3, W, 128]
        // icefall keeps the 3 frames before the look-ahead: [T_out, T_out + 3) of the concat.
        let new_cache = xc.slice(0, t_out, t_out + CONVNEXT_CACHE)?.contiguous()?;
        let pad = CONVNEXT_K / 2;
        let xp = xc.pad(1, pad, pad, 0.0)?;
        let y = self.dw.apply(&taps_2d(&xp, 0, (CONVNEXT_K, CONVNEXT_K), (1, 1), (t_out, self.width))?)?;
        let y = self.pw2.forward(&swoosh_l(&self.pw1.forward(&y)?)?)?;
        let x = bypass.add(&y)?; // [T_out, W, 128]
        let flat = x.permute(&[0, 2, 1])?.contiguous()?.reshape(&[t_out, EMBED_CHANNELS * self.width])?;
        let out = self.norm.forward(&self.out.forward(&flat)?)?;
        Ok((out, new_cache))
    }
}
