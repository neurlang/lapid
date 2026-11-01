//! DeepSeek V4 RoPE: interleaved pairs `(x[2i], x[2i+1])` on the *last* `rope_dim` channels of
//! a head (`[nope | rope]` layout), optional YaRN frequency interpolation (compressed layers),
//! and the inverse rotation applied to the attention output (K = V carries the rotation).
//!
//! Tables (cos / sin per position) are computed on the host in f32 like HF
//! (`position * inv_freq` in f32, then cos / sin) and uploaded with the graph.

use lapid::{Result, Tensor, bail};

use crate::deepseek4::config::Yarn;

#[derive(Clone)]
pub(crate) struct TailRope {
    dim: usize,
    inv_freq: Vec<f32>,
}

/// HF `_compute_yarn_parameters` (truncate = true, attention factor forced to 1.0) /
/// upstream `precompute_freqs_cis`.
fn yarn_inv_freq(dim: usize, base: f32, y: &Yarn) -> Vec<f32> {
    let half = dim / 2;
    let correction = |rot: f32| -> f64 {
        dim as f64 * (y.original_max_positions as f64 / (rot as f64 * 2.0 * std::f64::consts::PI)).ln() / (2.0 * (base as f64).ln())
    };
    let low = correction(y.beta_fast).floor().max(0.0);
    let mut high = correction(y.beta_slow).ceil().min(dim as f64 - 1.0);
    if low == high {
        high += 0.001;
    }
    (0..half)
        .map(|i| {
            let extra = 1.0 / base.powf((2 * i) as f32 / dim as f32);
            let inter = extra / y.factor;
            let ramp = (((i as f64 - low) / (high - low)).clamp(0.0, 1.0)) as f32;
            let extrapolation = 1.0 - ramp;
            inter * (1.0 - extrapolation) + extra * extrapolation
        })
        .collect()
}

impl TailRope {
    pub fn new(dim: usize, base: f32, yarn: Option<&Yarn>) -> TailRope {
        let inv_freq = match yarn {
            Some(y) => yarn_inv_freq(dim, base, y),
            None => (0..dim / 2).map(|i| 1.0 / base.powf((2 * i) as f32 / dim as f32)).collect(),
        };
        TailRope { dim, inv_freq }
    }

    /// (cos, sin) `[n, dim / 2]` for absolute `positions`.
    pub fn tables(&self, positions: &[usize]) -> Result<(Tensor, Tensor)> {
        let half = self.dim / 2;
        let (mut c, mut s) = (Vec::with_capacity(positions.len() * half), Vec::with_capacity(positions.len() * half));
        for &p in positions {
            for &f in &self.inv_freq {
                let a = p as f32 * f;
                c.push(a.cos());
                s.push(a.sin());
            }
        }
        Ok((Tensor::from_f32(&c, &[positions.len(), half])?, Tensor::from_f32(&s, &[positions.len(), half])?))
    }

    /// Rotate the last `dim` channels of `x` `[n, H, D]` (one position per row n) by the
    /// tables of [`TailRope::tables`]; `inverse` rotates by `-angle`.
    pub fn apply(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, inverse: bool) -> Result<Tensor> {
        if x.rank() != 3 {
            bail!("shape_mismatch", "deepseek4 rope expects [n, H, D], got {:?}", x.shape());
        }
        let (n, h, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (r, half) = (self.dim, self.dim / 2);
        if d < r || cos.shape() != [n, half] {
            bail!("shape_mismatch", "deepseek4 rope: x {:?}, tables {:?}, rope dim {r}", x.shape(), cos.shape());
        }
        let rot = x.slice(2, d - r, d)?.reshape(&[n, h, half, 2])?;
        let a = rot.slice(3, 0, 1)?;
        let b = rot.slice(3, 1, 2)?;
        let c = cos.reshape(&[n, 1, half, 1])?;
        let s = sin.reshape(&[n, 1, half, 1])?;
        let s = if inverse { s.neg()? } else { s };
        let ra = a.mul(&c)?.sub(&b.mul(&s)?)?;
        let rb = a.mul(&s)?.add(&b.mul(&c)?)?;
        let rot = Tensor::cat(&[&ra, &rb], 3)?.reshape(&[n, h, r])?;
        if d == r {
            return Ok(rot);
        }
        Tensor::cat(&[&x.slice(2, 0, d - r)?, &rot], 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yarn_matches_plain_at_high_freq() {
        let y = Yarn { factor: 16.0, original_max_positions: 65536, beta_fast: 32.0, beta_slow: 1.0 };
        let a = TailRope::new(64, 160000.0, Some(&y));
        let b = TailRope::new(64, 160000.0, None);
        // the fastest-rotating pairs are extrapolated (unchanged), the slowest interpolated (/ factor)
        assert!((a.inv_freq[0] - b.inv_freq[0]).abs() < 1e-9);
        assert!((a.inv_freq[31] * 16.0 - b.inv_freq[31]).abs() < 1e-9 * b.inv_freq[31].max(1.0));
    }
}
