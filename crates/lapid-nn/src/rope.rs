//! Rotary position embeddings.
//!
//! The cos / sin tables are computed on the host the way Hugging Face does (float32
//! `inv_freq = 1 / theta^(2i / rotary_dim)`, float32 `position * inv_freq`) so positions far
//! into a long context round exactly like the PyTorch reference. They are tiny leaves
//! ([n, rotary_dim / 2]) that broadcast over batch and heads.

use lapid::{Result, Tensor, bail};

/// How rotated pairs are laid out inside a head.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RopeLayout {
    /// Pairs (i, i + rot/2) — HF `rotate_half` (Llama, Gemma, LFM, NeoX).
    HalfSplit,
    /// Pairs (2i, 2i + 1) — GPT-J / Kyutai (Mimi) style.
    Interleaved,
}

/// Position scaling applied before the rotation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum RopeScaling {
    None,
    /// HF "linear": positions are divided by `factor`.
    Linear { factor: f32 },
}

#[derive(Clone, Debug)]
pub struct Rope {
    pub head_dim: usize,
    /// Leading dims of each head that are rotated (`head_dim` = full rotary; partial rotary
    /// passes the rest through unchanged).
    pub rotary_dim: usize,
    pub theta: f32,
    pub scaling: RopeScaling,
    pub layout: RopeLayout,
}

impl Rope {
    /// Full rotary, half-split layout, no scaling.
    pub fn new(head_dim: usize, theta: f32) -> Rope {
        Rope { head_dim, rotary_dim: head_dim, theta, scaling: RopeScaling::None, layout: RopeLayout::HalfSplit }
    }

    pub fn with_rotary_dim(mut self, rotary_dim: usize) -> Rope {
        self.rotary_dim = rotary_dim;
        self
    }

    /// Partial rotary as a fraction of the head (`partial_rotary_factor`), rounded down to
    /// an even count.
    pub fn with_partial(mut self, fraction: f32) -> Rope {
        self.rotary_dim = ((self.head_dim as f32 * fraction) as usize) & !1;
        self
    }

    pub fn with_scaling(mut self, scaling: RopeScaling) -> Rope {
        self.scaling = scaling;
        self
    }

    pub fn with_layout(mut self, layout: RopeLayout) -> Rope {
        self.layout = layout;
        self
    }

    fn check(&self) -> Result<()> {
        if self.rotary_dim == 0 || !self.rotary_dim.is_multiple_of(2) || self.rotary_dim > self.head_dim {
            bail!(
                "invalid_argument",
                "rope: rotary_dim {} must be even, > 0 and <= head_dim {}",
                self.rotary_dim,
                self.head_dim
            );
        }
        if !(self.theta.is_finite() && self.theta > 0.0) {
            bail!("invalid_argument", "rope: theta must be positive, got {}", self.theta);
        }
        if let RopeScaling::Linear { factor } = self.scaling
            && !(factor.is_finite() && factor > 0.0) {
                bail!("invalid_argument", "rope: linear scaling factor must be positive, got {factor}");
            }
        Ok(())
    }

    /// Host angles [positions.len() * rotary_dim/2] (row-major, f32 arithmetic like HF).
    pub fn angles(&self, positions: &[usize]) -> Vec<f32> {
        let half = self.rotary_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / self.theta.powf((2 * i) as f32 / self.rotary_dim as f32))
            .collect();
        let mut out = Vec::with_capacity(positions.len() * half);
        for &p in positions {
            let mut t = p as f32;
            if let RopeScaling::Linear { factor } = self.scaling {
                t /= factor;
            }
            out.extend(inv_freq.iter().map(|&f| t * f));
        }
        out
    }

    /// (cos, sin) f32 tensors of shape [positions.len(), rotary_dim / 2].
    pub fn cos_sin(&self, positions: &[usize]) -> Result<(Tensor, Tensor)> {
        self.check()?;
        let a = self.angles(positions);
        let cos: Vec<f32> = a.iter().map(|&x| (x as f64).cos() as f32).collect();
        let sin: Vec<f32> = a.iter().map(|&x| (x as f64).sin() as f32).collect();
        let shape = [positions.len(), self.rotary_dim / 2];
        Ok((Tensor::from_f32(&cos, &shape)?, Tensor::from_f32(&sin, &shape)?))
    }

    /// Rotate x [B, H, S, head_dim] for positions `offset..offset + S`.
    pub fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let s = x.shape().get(2).copied().unwrap_or(0);
        let positions: Vec<usize> = (offset..offset + s).collect();
        self.apply_positions(x, &positions)
    }

    /// Rotate x [B, H, S, head_dim] with explicit positions (len S).
    pub fn apply_positions(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        if x.rank() != 4 || x.shape()[3] != self.head_dim {
            bail!("shape_mismatch", "rope expects [B, H, S, {}], got {:?}", self.head_dim, x.shape());
        }
        if positions.len() != x.shape()[2] {
            bail!("shape_mismatch", "rope: {} positions for {} rows", positions.len(), x.shape()[2]);
        }
        let (cos, sin) = self.cos_sin(positions)?;
        let (cos, sin) = (cos.cast(x.dtype())?, sin.cast(x.dtype())?);
        let (rot, half) = (self.rotary_dim, self.rotary_dim / 2);
        let xr = x.slice(3, 0, rot)?;
        let rotated = match self.layout {
            RopeLayout::HalfSplit => {
                let x1 = xr.slice(3, 0, half)?;
                let x2 = xr.slice(3, half, rot)?;
                let o1 = x1.mul(&cos)?.sub(&x2.mul(&sin)?)?;
                let o2 = x2.mul(&cos)?.add(&x1.mul(&sin)?)?;
                Tensor::cat(&[&o1, &o2], 3)?
            }
            RopeLayout::Interleaved => {
                let sh = xr.shape().to_vec();
                let pairs = xr.reshape(&[sh[0], sh[1], sh[2], half, 2])?;
                let x1 = pairs.slice(4, 0, 1)?;
                let x2 = pairs.slice(4, 1, 2)?;
                let n = positions.len();
                let (c, s) = (cos.reshape(&[n, half, 1])?, sin.reshape(&[n, half, 1])?);
                let o1 = x1.mul(&c)?.sub(&x2.mul(&s)?)?;
                let o2 = x2.mul(&c)?.add(&x1.mul(&s)?)?;
                Tensor::cat(&[&o1, &o2], 4)?.reshape(&sh)?
            }
        };
        if rot == self.head_dim {
            return Ok(rotated);
        }
        let pass = x.slice(3, rot, self.head_dim)?;
        Tensor::cat(&[&rotated, &pass], 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{assert_close, eval, rand_tensor};
    use lapid::DType;

    /// Naive reference on host data [B, H, S, D].
    fn rope_ref(x: &[f32], shape: [usize; 4], rope: &Rope, offset: usize) -> Vec<f32> {
        let [b, h, s, d] = shape;
        let half = rope.rotary_dim / 2;
        let positions: Vec<usize> = (offset..offset + s).collect();
        let ang = rope.angles(&positions);
        let mut out = x.to_vec();
        for bi in 0..b {
            for hi in 0..h {
                for si in 0..s {
                    let row = &x[((bi * h + hi) * s + si) * d..][..d];
                    let o = &mut out[((bi * h + hi) * s + si) * d..][..d];
                    for i in 0..half {
                        let (c, sn) = ((ang[si * half + i] as f64).cos() as f32, (ang[si * half + i] as f64).sin() as f32);
                        let (i1, i2) = match rope.layout {
                            RopeLayout::HalfSplit => (i, i + half),
                            RopeLayout::Interleaved => (2 * i, 2 * i + 1),
                        };
                        o[i1] = row[i1] * c - row[i2] * sn;
                        o[i2] = row[i2] * c + row[i1] * sn;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn matches_reference_all_variants() {
        let shape = [2, 3, 5, 8];
        let (x, xv) = rand_tensor(&shape, 21);
        let variants = [
            Rope::new(8, 10_000.0),
            Rope::new(8, 1_000_000.0).with_layout(RopeLayout::Interleaved),
            Rope::new(8, 10_000.0).with_partial(0.5),
            Rope::new(8, 10_000.0).with_rotary_dim(4).with_layout(RopeLayout::Interleaved),
            Rope::new(8, 10_000.0).with_scaling(RopeScaling::Linear { factor: 8.0 }),
        ];
        for (n, rope) in variants.iter().enumerate() {
            for offset in [0usize, 37] {
                let got = eval(&rope.apply(&x, offset).unwrap());
                let want = rope_ref(&xv, shape, rope, offset);
                assert_close(&got, &want, 1e-5);
                if rope.rotary_dim < 8 {
                    // pass-through dims unchanged
                    assert_eq!(got[4..8], xv[4..8], "variant {n}");
                }
            }
        }
    }

    #[test]
    fn position_zero_is_identity_and_norm_preserved() {
        let (x, xv) = rand_tensor(&[1, 1, 1, 6], 22);
        let rope = Rope::new(6, 10_000.0);
        assert_close(&eval(&rope.apply(&x, 0).unwrap()), &xv, 1e-7);
        let y = eval(&rope.apply(&x, 1234).unwrap());
        let n0: f32 = xv.iter().map(|v| v * v).sum();
        let n1: f32 = y.iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-4);
    }

    #[test]
    fn relative_property() {
        // <rope(q, m), rope(k, n)> depends only on m - n.
        let (q, _) = rand_tensor(&[1, 1, 1, 8], 23);
        let (k, _) = rand_tensor(&[1, 1, 1, 8], 24);
        let rope = Rope::new(8, 10_000.0);
        let dot = |m: usize, n: usize| -> f32 {
            let a = eval(&rope.apply(&q, m).unwrap());
            let b = eval(&rope.apply(&k, n).unwrap());
            a.iter().zip(&b).map(|(x, y)| x * y).sum()
        };
        assert!((dot(5, 2) - dot(13, 10)).abs() < 1e-4);
    }

    #[test]
    fn rejects_bad_config() {
        let x = Tensor::zeros(&[1, 1, 2, 8], DType::F32);
        assert!(Rope::new(8, 10_000.0).with_rotary_dim(3).apply(&x, 0).is_err());
        assert!(Rope::new(6, 10_000.0).apply(&x, 0).is_err());
        assert!(Rope::new(8, 10_000.0).apply_positions(&x, &[0]).is_err());
    }
}
