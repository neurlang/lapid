//! Gemma 4 rotary embedding (copied from `lfm3/model/rope.rs`, where `lfm3::model` is
//! private): half-split (`rotate_half`) over the whole head with "zero-tail" partial
//! rotation — HF "proportional" RoPE. The first `partial * head_dim / 2` frequencies follow
//! `theta^(-2i / head_dim)`, the rest are zero (pass through). `partial = 1` is plain RoPE.
//!
//! Second user after LFM3 → candidate for lapid-nn as `RopeLayout::HalfSplitZeroTail`.

use lapid::{Result, Tensor, bail};

#[derive(Clone, Debug)]
pub(crate) struct ZeroTailRope {
    head_dim: usize,
    /// Non-zero inverse frequencies (f32, computed like HF); the tail up to head_dim / 2 is 0.
    inv_freq: Vec<f32>,
}

impl ZeroTailRope {
    pub fn new(head_dim: usize, theta: f32, partial: f32) -> ZeroTailRope {
        let rope_angles = ((partial as f64 * head_dim as f64) as usize / 2).min(head_dim / 2);
        // HF: 1.0 / (base ** (torch.arange(0, 2 * n, 2, dtype=float) / dim)) in float32.
        let inv_freq = (0..rope_angles)
            .map(|i| {
                let e = (2 * i) as f32 / head_dim as f32;
                1.0f32 / theta.powf(e)
            })
            .collect();
        ZeroTailRope { head_dim, inv_freq }
    }

    /// (cos, sin) [S, head_dim / 2] for positions `offset..offset + s` (f32 angles as HF).
    pub fn cos_sin(&self, offset: usize, s: usize) -> Result<(Tensor, Tensor)> {
        let half = self.head_dim / 2;
        let mut cos = vec![1.0f32; s * half];
        let mut sin = vec![0.0f32; s * half];
        for r in 0..s {
            let p = (offset + r) as f32;
            for (i, &f) in self.inv_freq.iter().enumerate() {
                let a = p * f;
                cos[r * half + i] = (a as f64).cos() as f32;
                sin[r * half + i] = (a as f64).sin() as f32;
            }
        }
        Ok((Tensor::from_f32(&cos, &[s, half])?, Tensor::from_f32(&sin, &[s, half])?))
    }

    /// Rotate x [B, H, S, head_dim] (computed in f32, returned in x's dtype).
    pub fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        if x.rank() != 4 || x.shape()[3] != self.head_dim {
            bail!("shape_mismatch", "rope expects [B, H, S, {}], got {:?}", self.head_dim, x.shape());
        }
        let s = x.shape()[2];
        let half = self.head_dim / 2;
        let (cos, sin) = self.cos_sin(offset, s)?;
        let xf = x.cast(lapid::DType::F32)?;
        let x1 = xf.slice(3, 0, half)?;
        let x2 = xf.slice(3, half, self.head_dim)?;
        let o1 = x1.mul(&cos)?.sub(&x2.mul(&sin)?)?;
        let o2 = x2.mul(&cos)?.add(&x1.mul(&sin)?)?;
        Tensor::cat(&[&o1, &o2], 3)?.cast(x.dtype())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_tail_frequencies() {
        let r = ZeroTailRope::new(128, 1e6, 0.5);
        assert_eq!(r.inv_freq.len(), 32);
        assert_eq!(r.inv_freq[0], 1.0);
        let full = ZeroTailRope::new(128, 1e4, 1.0);
        assert_eq!(full.inv_freq.len(), 64);
        let (c, s) = r.cos_sin(5, 2).expect("tables");
        assert_eq!(c.shape(), &[2, 64]);
        assert_eq!(s.shape(), &[2, 64]);
    }
}
