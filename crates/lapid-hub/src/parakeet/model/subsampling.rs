//! FastConformer 8× conv subsampling (`dw_striding`): Conv2d(1→C, 3×3, s2, p1) + ReLU, then
//! per extra stage depthwise Conv2d(C, 3×3, s2, p1) + point-wise 1×1 Conv2d + ReLU, then
//! Linear(C · F/8 → d_model) over the flattened (channel, frequency) features.
//!
//! Activations are channels-last [B, T, F, C], so convolutions are im2col + matmul (the full
//! conv), a fused 9-tap multiply-add chain (depthwise) and plain `Linear` (point-wise).
//!
//! HF masks each conv output beyond the valid length; feeding exactly the valid mel frames
//! (no padding frame) is equivalent: a masked frame is a zero, and so is the conv padding.

use lapid::{Error, Result, Tensor};
use lapid_nn::speech::{DepthwiseConv, taps_2d};
use lapid_nn::{Linear, Loader};

use super::linear_checked;
use crate::parakeet::config::ParakeetConfig;
use crate::parakeet::convert::subsampling_layers;

const K: usize = 3;
const STRIDE: usize = 2;
const PAD: usize = 1;

/// Output length of a k3 s2 p1 conv.
pub(crate) fn conv_out(len: usize) -> usize {
    (len + 2 * PAD - K) / STRIDE + 1
}

/// Encoder frames for `mel_frames` valid mel frames.
pub(crate) fn encoder_frames(mel_frames: usize, stages: usize) -> usize {
    (0..stages).fold(mel_frames, |l, _| if l == 0 { 0 } else { conv_out(l) })
}

/// The 9 taps of a 3×3 stride-2 pad-1 conv over x [B, T, F, C]: each [B, To, Fo, C],
/// ordered (kt, kf) row-major like the weight [C, Cin, 3, 3].
fn taps3x3(x: &Tensor) -> Result<(Vec<Tensor>, usize, usize)> {
    let (t, f) = (x.shape()[1], x.shape()[2]);
    let (to, fo) = (conv_out(t), conv_out(f));
    // Each tap reads [k, k + 2*n) of the padded axis; pad the tail so the last window fits.
    let right = |len: usize, n: usize| (K - 1 + n * STRIDE).saturating_sub(len + PAD);
    let xp = x.pad(1, PAD, right(t, to), 0.0)?.pad(2, PAD, right(f, fo), 0.0)?;
    Ok((taps_2d(&xp, 1, (K, K), (STRIDE, STRIDE), (to, fo))?, to, fo))
}

struct Stage {
    /// Depthwise 3×3 (weight [C, 9] f32, bias [C]).
    dw: DepthwiseConv,
    pw: Linear,
}

pub(crate) struct Subsampling {
    /// Full conv weight [C, 9] and bias [C].
    conv0: Linear,
    stages: Vec<Stage>,
    linear: Linear,
    n_mels: usize,
    channels: usize,
}

impl Subsampling {
    pub fn load(l: &Loader<'_>, cfg: &ParakeetConfig) -> Result<Subsampling> {
        let c = cfg.subsampling_channels;
        let p = "encoder.subsampling.layers";
        let mut conv0 = None;
        let mut stages = Vec::new();
        let mut pending_dw: Option<DepthwiseConv> = None;
        for (i, depthwise) in subsampling_layers(cfg.subsampling_stages()) {
            if i == 0 {
                conv0 = Some(linear_checked(l, &format!("{p}.0"), c, K * K)?);
            } else if depthwise {
                let w = l.tensor_f32(&format!("{p}.{i}.weight"))?;
                l.expect_shape(&w, &format!("{p}.{i}.weight"), &[c, K * K])?;
                pending_dw = Some(DepthwiseConv::new(&w, Some(l.tensor_f32(&format!("{p}.{i}.bias"))?))?);
            } else {
                let dw = pending_dw
                    .take()
                    .ok_or_else(|| Error::new("internal", "subsampling: point-wise conv without depthwise"))?;
                stages.push(Stage { dw, pw: linear_checked(l, &format!("{p}.{i}"), c, c)? });
            }
        }
        let conv0 = conv0.ok_or_else(|| Error::new("internal", "subsampling: no first conv"))?;
        let linear = linear_checked(l, "encoder.subsampling.linear", cfg.d_model, c * cfg.subsampled_mels())?;
        Ok(Subsampling { conv0, stages, linear, n_mels: cfg.n_mels, channels: c })
    }

    /// mel [T, n_mels] (frames-major, normalized) -> [1, T', d_model].
    pub fn forward(&self, mel: &[f32], frames: usize) -> Result<Tensor> {
        if frames == 0 || mel.len() != frames * self.n_mels {
            return Err(Error::new(
                "invalid_argument",
                format!("subsampling: mel has {} values, expected {frames} frames x {} mels", mel.len(), self.n_mels),
            ));
        }
        let x = Tensor::from_f32(mel, &[1, frames, self.n_mels, 1])?;
        let (taps, _, _) = taps3x3(&x)?;
        let refs: Vec<&Tensor> = taps.iter().collect();
        let cols = Tensor::cat(&refs, 3)?; // [1, To, Fo, 9]
        let mut x = lapid_nn::ops::relu(&self.conv0.forward(&cols)?)?; // [1, To, Fo, C]
        for s in &self.stages {
            let (taps, _, _) = taps3x3(&x)?;
            let y = s.dw.apply(&taps)?;
            x = lapid_nn::ops::relu(&s.pw.forward(&y)?)?;
        }
        // [1, T', F', C] -> [1, T', C, F'] -> [1, T', C * F'] (HF flattens channel-major).
        let (t, f) = (x.shape()[1], x.shape()[2]);
        let x = x.permute(&[0, 1, 3, 2])?.contiguous()?.reshape(&[1, t, self.channels * f])?;
        self.linear.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths() {
        assert_eq!(encoder_frames(1100, 3), 138);
        assert_eq!(encoder_frames(2499, 3), 313);
        assert_eq!(encoder_frames(8, 3), 1);
        assert_eq!(encoder_frames(1, 3), 1);
        assert_eq!(encoder_frames(0, 3), 0);
        assert_eq!(conv_out(128), 64);
    }

    #[test]
    fn taps_match_naive_conv_indexing() {
        // x[t, f] = 100 t + f on a 5 x 4 grid; tap (kt, kf) at output (i, j) reads
        // x[2i + kt - 1, 2j + kf - 1] (zero outside).
        let (t, f) = (5usize, 4usize);
        let data: Vec<f32> = (0..t * f).map(|i| (100 * (i / f) + i % f) as f32 + 1.0).collect();
        let x = Tensor::from_f32(&data, &[1, t, f, 1]).expect("x");
        let (taps, to, fo) = taps3x3(&x).expect("taps");
        assert_eq!((to, fo), (3, 2));
        let dev = lapid::Device::cpu();
        for (k, tap) in taps.iter().enumerate() {
            let (kt, kf) = (k / 3, k % 3);
            let r = dev.eval(&[tap]).expect("eval");
            let got = dev.to_vec_f32(&r[0]).expect("vec");
            for i in 0..to {
                for j in 0..fo {
                    let (ti, fj) = ((2 * i + kt) as isize - 1, (2 * j + kf) as isize - 1);
                    let want = if ti < 0 || fj < 0 || ti >= t as isize || fj >= f as isize {
                        0.0
                    } else {
                        data[ti as usize * f + fj as usize]
                    };
                    assert_eq!(got[i * fo + j], want, "tap {k} at ({i}, {j})");
                }
            }
        }
    }
}
