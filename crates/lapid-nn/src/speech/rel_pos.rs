//! Relative-position self-attention of speech encoders.
//!
//! * [`rel_shift`]: the Transformer-XL relative shift (flat-view trick), for full
//!   (`[.., T, 2T-1]` scores) and cached-left-context (`[.., T, 2T+L-1]`) attention.
//! * [`RelPosAttention`]: Transformer-XL / conformer `RelPositionMultiHeadAttention`
//!   (NeMo FastConformer, HF `ParakeetEncoderAttention`) with sinusoidal relative positions
//!   ([`sinusoidal_rel_positions`]):
//!
//! ```text
//! scores = ((q + u) k^T + rel_shift((q + v) p^T)) / sqrt(d_head),   p = linear_pos(pos_emb)
//! ```
//!
//! The position term `bd` is handed to the fused attention composite as its additive mask
//! (pre-scaled), so `softmax(qu k^T * scale + bd * scale) v` runs as one attention kernel.

use lapid::{Result, Tensor, bail};

use crate::Linear;
use crate::attention::{merge_heads, split_heads};

/// Relative shift over the last two axes: x [.., T, n] -> [.., T, k] with
/// `y[i, j] = x[i, T - 1 - i + j]` (`k <= n - 1`; scores over relative offsets
/// `T - 1 + L .. -(T - 1)` give key positions `0 .. L + T`). Flattened, row i starts at
/// `(T - 1) + i (n - 1)`, so the result is a window of the flat view reshaped to [T, n - 1]
/// and cut to k columns.
pub fn rel_shift(x: &Tensor, k: usize) -> Result<Tensor> {
    let s = x.shape();
    let r = s.len();
    if r < 2 {
        bail!("shape_mismatch", "rel_shift expects [.., T, n], got {s:?}");
    }
    let (t, n) = (s[r - 2], s[r - 1]);
    if t == 1 {
        return x.slice(r as isize - 1, 0, k);
    }
    if k + 1 > n {
        bail!("shape_mismatch", "rel_shift: scores {s:?} cannot give {k} key positions");
    }
    let lead = &s[..r - 2];
    let flat = x.contiguous()?.reshape(&[lead, &[t * n]].concat())?;
    let win = flat.slice(r as isize - 2, t - 1, t - 1 + t * (n - 1))?.contiguous()?;
    win.reshape(&[lead, &[t, n - 1]].concat())?.slice(r as isize - 1, 0, k)
}

/// Sinusoidal relative positions T-1 … -(T-1): [1, 2T-1, d] with interleaved (sin, cos)
/// pairs, computed like HF / NeMo (`inv_freq` and `pos * inv_freq` in f32).
pub fn sinusoidal_rel_positions(t: usize, d: usize) -> Result<Tensor> {
    let n = 2 * t - 1;
    let half = d / 2;
    let inv: Vec<f32> = (0..half).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / d as f32)).collect();
    let mut data = vec![0.0f32; n * d];
    for r in 0..n {
        let pos = (t as isize - 1 - r as isize) as f32;
        for (i, &f) in inv.iter().enumerate() {
            let a = (pos * f) as f64;
            data[r * d + 2 * i] = a.sin() as f32;
            data[r * d + 2 * i + 1] = a.cos() as f32;
        }
    }
    Tensor::from_f32(&data, &[1, n, d])
}

/// Transformer-XL relative-position multi-head self-attention (batch 1, full context).
pub struct RelPosAttention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    /// `linear_pos` (no bias).
    pub pos: Linear,
    /// Content / position biases u, v as [1, H, 1, Dh] f32.
    pub bias_u: Tensor,
    pub bias_v: Tensor,
    pub n_head: usize,
    pub scale: f32,
}

impl RelPosAttention {
    /// `bias_u` / `bias_v` given as [H, Dh] (f32).
    pub fn new(q: Linear, k: Linear, v: Linear, o: Linear, pos: Linear, bias_u: &Tensor, bias_v: &Tensor, n_head: usize) -> Result<RelPosAttention> {
        let d = q.out_features();
        if n_head == 0 || d % n_head != 0 {
            bail!("shape_mismatch", "relative attention: {d} channels for {n_head} heads");
        }
        let dh = d / n_head;
        Ok(RelPosAttention {
            q,
            k,
            v,
            o,
            pos,
            bias_u: bias_u.reshape(&[1, n_head, 1, dh])?,
            bias_v: bias_v.reshape(&[1, n_head, 1, dh])?,
            n_head,
            scale: 1.0 / (dh as f32).sqrt(),
        })
    }

    /// x [1, T, d], pos_emb [1, 2T-1, d] -> [1, T, d].
    pub fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let h = self.n_head;
        let q = split_heads(&self.q.forward(x)?, h)?;
        let k = split_heads(&self.k.forward(x)?, h)?;
        let v = split_heads(&self.v.forward(x)?, h)?;
        let p = split_heads(&self.pos.forward(pos_emb)?, h)?; // [1, H, 2T-1, Dh]
        let q_u = q.add(&self.bias_u)?;
        let q_v = q.add(&self.bias_v)?;
        let t = x.shape()[1];
        let bd = rel_shift(&q_v.matmul_t(&p)?, t)?.mul_scalar(f64::from(self.scale))?; // [1, H, T, T]
        let a = Tensor::attention(&q_u, &k, &v, Some(&bd), self.scale, false)?;
        self.o.forward(&merge_heads(&a)?)
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
    fn shift_full_context() {
        for t in [1usize, 2, 3, 5] {
            let p = 2 * t - 1;
            let data: Vec<f32> = (0..2 * t * p).map(|i| i as f32).collect();
            let x = Tensor::from_f32(&data, &[1, 2, t, p]).expect("x");
            let y = eval(&rel_shift(&x, t).expect("shift"));
            for hh in 0..2 {
                for i in 0..t {
                    for j in 0..t {
                        let want = data[hh * t * p + i * p + j + t - 1 - i];
                        assert_eq!(y[hh * t * t + i * t + j], want, "t {t} h {hh} ({i}, {j})");
                    }
                }
            }
        }
    }

    #[test]
    fn shift_with_left_context() {
        let (h, t, l) = (2usize, 3usize, 4usize);
        let n = 2 * t + l - 1;
        let data: Vec<f32> = (0..h * t * n).map(|i| i as f32).collect();
        let x = Tensor::from_f32(&data, &[h, t, n]).expect("x");
        let y = eval(&rel_shift(&x, l + t).expect("shift"));
        for hh in 0..h {
            for i in 0..t {
                for j in 0..l + t {
                    assert_eq!(y[(hh * t + i) * (l + t) + j], data[hh * t * n + i * n + t - 1 - i + j]);
                }
            }
        }
    }

    #[test]
    fn positions_interleaved() {
        let pe = eval(&sinusoidal_rel_positions(3, 4).expect("pe"));
        // row 0 = position 2, row 2 = position 0 -> (sin 0, cos 0) = (0, 1)
        assert!((pe[0] - 2f32.sin()).abs() < 1e-6 && (pe[1] - 2f32.cos()).abs() < 1e-6);
        assert_eq!(&pe[2 * 4..2 * 4 + 2], &[0.0, 1.0]);
        assert!((pe[4 * 4] - (-2f32).sin()).abs() < 1e-6);
    }
}
