//! Convolutions as shifted views ("taps") of the input: a depthwise kernel becomes one fused
//! elementwise multiply-add chain over its taps, a small dense kernel an im2col
//! (`Tensor::cat` of the taps) followed by one matmul. Used for the depthwise and 2-D
//! subsampling convolutions of conformer-style speech encoders, whose activations are kept
//! channels-last so every tap is a plain slice.

use lapid::{Error, Result, Tensor};

/// `x[.., start + i*stride, ..]` for `i in 0..n` along `axis`: a window slice viewed as
/// `[n, stride]` and its first column. The tail is zero-padded when the last window would run
/// past the end (the padding is never selected).
pub fn strided(x: &Tensor, axis: usize, start: usize, n: usize, stride: usize) -> Result<Tensor> {
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
    w.reshape(&shape)?.slice(axis as isize + 1, 0, 1)?.squeeze(axis as isize + 1)
}

/// The `k` stride-1 taps of a 1-D kernel along `axis` (already padded): tap `i` is
/// `x[.., i .. i + n, ..]`.
pub fn taps_1d(x: &Tensor, axis: usize, k: usize, n: usize) -> Result<Vec<Tensor>> {
    (0..k).map(|i| x.slice(axis as isize, i, i + n)).collect()
}

/// The `kh * kw` taps of a 2-D kernel over axes `axis` and `axis + 1` of `x` (already
/// padded), with strides `stride` and `out` output positions per axis; ordered (kt, kf)
/// row-major like a PyTorch `[Cout, Cin, kh, kw]` weight.
pub fn taps_2d(x: &Tensor, axis: usize, kernel: (usize, usize), stride: (usize, usize), out: (usize, usize)) -> Result<Vec<Tensor>> {
    let mut taps = Vec::with_capacity(kernel.0 * kernel.1);
    for kt in 0..kernel.0 {
        let rows = strided(x, axis, kt, out.0, stride.0)?;
        for kf in 0..kernel.1 {
            taps.push(strided(&rows, axis + 1, kf, out.1, stride.1)?);
        }
    }
    Ok(taps)
}

/// Per-channel weights of a depthwise kernel: `w` [C, K] (realized f32) -> K tensors [C].
pub fn kernel_taps(w: &Tensor) -> Result<Vec<Tensor>> {
    let k = w.shape().get(1).copied().unwrap_or(0);
    (0..k).map(|t| w.slice(1, t, t + 1)?.squeeze(1)).collect()
}

/// Σ_t taps[t] * weights[t] (+ bias), broadcasting [C] weights over the last axis. Kept as
/// one elementwise chain so the compiler fuses it into a single kernel.
pub fn weighted_sum(taps: &[Tensor], weights: &[Tensor], bias: Option<&Tensor>) -> Result<Tensor> {
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

/// Depthwise convolution over channels-last activations (channels on the last axis), as a
/// fused multiply-add chain of its taps. The kernel is stored `[C, K]` (1-D, or a 2-D
/// kernel flattened row-major).
#[derive(Clone)]
pub struct DepthwiseConv {
    /// K per-tap weights [C] (f32).
    pub weights: Vec<Tensor>,
    /// [C] (f32).
    pub bias: Option<Tensor>,
}

impl DepthwiseConv {
    /// `w` [C, K] realized f32, `bias` [C].
    pub fn new(w: &Tensor, bias: Option<Tensor>) -> Result<DepthwiseConv> {
        Ok(DepthwiseConv { weights: kernel_taps(w)?, bias })
    }

    pub fn kernel_size(&self) -> usize {
        self.weights.len()
    }

    /// Σ taps[t] * w[t] + bias over precomputed taps (e.g. [`taps_2d`]).
    pub fn apply(&self, taps: &[Tensor]) -> Result<Tensor> {
        weighted_sum(taps, &self.weights, self.bias.as_ref())
    }

    /// Stride-1 1-D convolution along `axis`, zero-padded by `pad` = (left, right).
    pub fn forward(&self, x: &Tensor, axis: usize, pad: (usize, usize)) -> Result<Tensor> {
        let k = self.kernel_size();
        let xp = if pad == (0, 0) { x.clone() } else { x.pad(axis as isize, pad.0, pad.1, 0.0)? };
        let n = (x.shape()[axis] + pad.0 + pad.1 + 1).saturating_sub(k);
        self.apply(&taps_1d(&xp, axis, k, n)?)
    }
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
    fn strided_picks_every_other() {
        let x = Tensor::from_f32(&(0..7).map(|i| i as f32).collect::<Vec<_>>(), &[7, 1]).expect("x");
        assert_eq!(eval(&strided(&x, 0, 2, 3, 2).expect("s")), vec![2.0, 4.0, 6.0]);
        assert_eq!(eval(&strided(&x, 0, 0, 4, 2).expect("s")), vec![0.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn taps_2d_match_naive_conv_indexing() {
        // x[t, f] = 100 t + f + 1 on a 5 x 4 grid, padded by one on every side; tap (kt, kf)
        // at output (i, j) of a 3x3 stride-2 conv reads x[2i + kt - 1, 2j + kf - 1].
        let (t, f) = (5usize, 4usize);
        let data: Vec<f32> = (0..t * f).map(|i| (100 * (i / f) + i % f) as f32 + 1.0).collect();
        let x = Tensor::from_f32(&data, &[1, t, f, 1]).expect("x");
        let xp = x.pad(1, 1, 1, 0.0).expect("pad").pad(2, 1, 0, 0.0).expect("pad");
        let (to, fo) = (3, 2);
        let taps = taps_2d(&xp, 1, (3, 3), (2, 2), (to, fo)).expect("taps");
        for (k, tap) in taps.iter().enumerate() {
            let (kt, kf) = (k / 3, k % 3);
            let got = eval(tap);
            for i in 0..to {
                for j in 0..fo {
                    let (ti, fj) = ((2 * i + kt) as isize - 1, (2 * j + kf) as isize - 1);
                    let inside = ti >= 0 && fj >= 0 && ti < t as isize && fj < f as isize;
                    let want = if inside { data[ti as usize * f + fj as usize] } else { 0.0 };
                    assert_eq!(got[i * fo + j], want, "tap {k} at ({i}, {j})");
                }
            }
        }
    }

    #[test]
    fn depthwise_same_padding() {
        // 2 channels, kernel 3, x[t, c] = t + 10 c
        let x = Tensor::from_f32(&[0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0], &[4, 2]).expect("x");
        let w = Tensor::from_f32(&[1.0, 2.0, 3.0, 0.0, 1.0, 0.0], &[2, 3]).expect("w");
        let conv = DepthwiseConv::new(&w, Some(Tensor::from_f32(&[0.5, -1.0], &[2]).expect("b"))).expect("conv");
        let y = eval(&conv.forward(&x, 0, (1, 1)).expect("y"));
        // channel 0: 1 x[t-1] + 2 x[t] + 3 x[t+1] + 0.5; channel 1: x[t] - 1
        assert_eq!(y, vec![3.5, 9.0, 8.5, 10.0, 14.5, 11.0, 8.5, 12.0]);
    }
}
