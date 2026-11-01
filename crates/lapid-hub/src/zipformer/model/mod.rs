//! Zipformer2 streaming transducer (icefall `egs/librispeech/ASR/zipformer`) as lazy lapid
//! graphs, batch 1, time-major `[T, C]` activations:
//!
//! * [`embed`]: `Conv2dSubsampling` (3 convs + ConvNeXt with a 3-frame causal cache),
//! * [`layer`]: `Zipformer2EncoderLayer.streaming_forward` (relative-position attention
//!   weights shared by nonlinear attention and two self-attentions, chunk-causal conv
//!   modules, three feed-forwards, BiasNorm, bypasses) with cached left context,
//! * [`encoder`]: the multi-rate stacks (SimpleDownsample / SimpleUpsample, out combiner),
//!   the full-dim output and the final /2,
//! * [`decoder`]: stateless decoder (embedding + grouped context conv) and joiner.
//!
//! Inference-only modules of icefall (Balancer, Whiten, dropout, ScaleGrad) are identities.

pub(crate) mod decoder;
pub(crate) mod embed;
pub(crate) mod encoder;
pub(crate) mod layer;

use lapid::{BinaryOp, Error, Result, Tensor, UnaryOp};
use lapid_nn::{Linear, Loader};

/// First tensor of an eval result.
pub(crate) fn first(v: Vec<Tensor>) -> Result<Tensor> {
    v.into_iter().next().ok_or_else(|| Error::new("backend", "device.eval returned no outputs"))
}

/// `{prefix}.weight` [out, in] + `{prefix}.bias` (when present), shape-checked.
pub(crate) fn linear(l: &Loader<'_>, prefix: &str, out: usize, inp: usize) -> Result<Linear> {
    let bias = l.contains(&format!("{prefix}.bias"));
    let lin = l.linear(prefix, bias)?;
    l.expect_shape(&lin.weight, &format!("{prefix}.weight"), &[out, inp])?;
    Ok(lin)
}

/// A (small) weight copied to the host as f32.
pub(crate) fn host(l: &Loader<'_>, name: &str) -> Result<Vec<f32>> {
    let t = l.tensor_f32(name)?;
    l.device.to_vec_f32(&t)
}

/// A per-channel parameter [C] on the device (f32).
pub(crate) fn param(l: &Loader<'_>, name: &str, c: usize) -> Result<Tensor> {
    let t = l.tensor_f32(name)?;
    l.expect_shape(&t, name, &[c])?;
    Ok(t)
}

/// `log(1 + exp(z))` without overflow: `max(z, 0) + log(1 + exp(-|z|))`.
fn softplus(z: &Tensor) -> Result<Tensor> {
    let tail = z.unary(UnaryOp::Abs)?.neg()?.exp()?.add_scalar(1.0)?.log()?;
    z.binary_scalar(BinaryOp::Max, 0.0)?.add(&tail)
}

/// icefall SwooshL: `log(1 + exp(x - 4)) - 0.08 x - 0.035`.
pub(crate) fn swoosh_l(x: &Tensor) -> Result<Tensor> {
    softplus(&x.add_scalar(-4.0)?)?.sub(&x.mul_scalar(0.08)?)?.add_scalar(-0.035)
}

/// icefall SwooshR: `log(1 + exp(x - 1)) - 0.08 x - 0.313261687`.
pub(crate) fn swoosh_r(x: &Tensor) -> Result<Tensor> {
    softplus(&x.add_scalar(-1.0)?)?.sub(&x.mul_scalar(0.08)?)?.add_scalar(-0.313_261_687)
}

/// icefall BiasNorm over the last axis: `x * mean((x - bias)^2)^-0.5 * exp(log_scale)`.
#[derive(Clone)]
pub(crate) struct BiasNorm {
    bias: Tensor,
    scale: f64,
}

impl BiasNorm {
    pub fn load(l: &Loader<'_>, prefix: &str, c: usize) -> Result<BiasNorm> {
        let log_scale = host(l, &format!("{prefix}.log_scale"))?;
        let ls = log_scale.first().copied().ok_or_else(|| Error::new("missing_tensor", format!("{prefix}.log_scale is empty")))?;
        Ok(BiasNorm { bias: param(l, &format!("{prefix}.bias"), c)?, scale: f64::from(ls).exp() })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.sub(&self.bias)?;
        let ms = d.mul(&d)?.mean(-1, true)?;
        x.mul(&ms.unary(UnaryOp::Rsqrt)?)?.mul_scalar(self.scale)
    }
}

/// icefall BypassModule (inference): `orig + (x - orig) * scale`.
pub(crate) fn bypass(orig: &Tensor, x: &Tensor, scale: &Tensor) -> Result<Tensor> {
    orig.add(&x.sub(orig)?.mul(scale)?)
}

/// Per-channel weights of a depthwise kernel `w` [C, K] (realized f32) -> K tensors [C].
pub(crate) fn kernel_taps(w: &Tensor) -> Result<Vec<Tensor>> {
    let k = w.shape().get(1).copied().unwrap_or(0);
    (0..k).map(|t| w.slice(1, t, t + 1)?.squeeze(1)).collect()
}

/// Σ taps[t] * weights[t] (+ bias): one elementwise chain the compiler fuses.
pub(crate) fn weighted_sum(taps: &[Tensor], weights: &[Tensor], bias: Option<&Tensor>) -> Result<Tensor> {
    let mut acc: Option<Tensor> = None;
    for (x, w) in taps.iter().zip(weights) {
        let term = x.mul(w)?;
        acc = Some(match acc {
            Some(a) => a.add(&term)?,
            None => term,
        });
    }
    let acc = acc.ok_or_else(|| Error::new("internal", "weighted sum of zero taps"))?;
    match bias {
        Some(b) => acc.add(b),
        None => Ok(acc),
    }
}

/// `x[.., start + i*stride, ..]` for `i in 0..n` along `axis` (zero-padding the tail when the
/// last window would run past the end; the padding is never selected).
pub(crate) fn strided(x: &Tensor, axis: usize, start: usize, n: usize, stride: usize) -> Result<Tensor> {
    if stride == 1 {
        return x.slice(axis as isize, start, start + n);
    }
    let len = x.shape()[axis];
    let need = start + n * stride;
    let x = if need > len { x.pad(axis as isize, 0, need - len, 0.0)? } else { x.clone() };
    let w = x.slice(axis as isize, start, need)?;
    let mut shape = w.shape().to_vec();
    shape[axis] = n;
    shape.insert(axis + 1, stride);
    w.contiguous()?.reshape(&shape)?.slice(axis as isize + 1, 0, 1)?.squeeze(axis as isize + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(t: &Tensor) -> Vec<f32> {
        let dev = lapid::Device::cpu();
        let r = dev.eval(&[t]).expect("eval");
        dev.to_vec_f32(&r[0]).expect("vec")
    }

    #[test]
    fn swoosh_matches_formula() {
        let v: Vec<f32> = vec![-100.0, -5.0, -1.0, 0.0, 0.5, 3.0, 10.0, 200.0];
        let x = Tensor::from_f32(&v, &[v.len()]).expect("x");
        let sp = |z: f64| if z > 30.0 { z } else { (1.0 + z.exp()).ln() };
        let l = eval(&swoosh_l(&x).expect("l"));
        let r = eval(&swoosh_r(&x).expect("r"));
        for (i, &a) in v.iter().enumerate() {
            let a = a as f64;
            let wl = sp(a - 4.0) - 0.08 * a - 0.035;
            let wr = sp(a - 1.0) - 0.08 * a - 0.313261687;
            assert!((l[i] as f64 - wl).abs() < 1e-4 * wl.abs().max(1.0), "swoosh_l({a}) {} vs {wl}", l[i]);
            assert!((r[i] as f64 - wr).abs() < 1e-4 * wr.abs().max(1.0), "swoosh_r({a}) {} vs {wr}", r[i]);
        }
    }

    #[test]
    fn strided_picks_every_other() {
        let x = Tensor::from_f32(&(0..7).map(|i| i as f32).collect::<Vec<_>>(), &[7, 1]).expect("x");
        assert_eq!(eval(&strided(&x, 0, 2, 3, 2).expect("s")), vec![2.0, 4.0, 6.0]);
        assert_eq!(eval(&strided(&x, 0, 0, 4, 2).expect("s")), vec![0.0, 2.0, 4.0, 6.0]);
    }
}
