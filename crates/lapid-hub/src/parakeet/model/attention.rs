//! Transformer-XL style relative-position multi-head attention (FastConformer `RelPositionMHA`,
//! HF `ParakeetEncoderAttention`):
//!
//! ```text
//! scores = ((q + u) k^T + rel_shift((q + v) p^T)) / sqrt(d_head),   p = linear_pos(pos_emb)
//! ```
//!
//! The position term `bd` is computed with a matmul, shifted with a flat-view trick and
//! handed to the fused attention composite as its additive mask (pre-scaled), so
//! `softmax(qu k^T * scale + bd * scale) v` runs as one attention kernel.

use lapid::{Error, Result, Tensor};
use lapid_nn::Loader;
use lapid_nn::attention::{merge_heads, split_heads};
use lapid_nn::Linear;

use super::linear_checked;

/// Sinusoidal relative positions T-1 … -(T-1): [1, 2T-1, d] with interleaved (sin, cos)
/// pairs, computed like HF (`inv_freq` and `pos * inv_freq` in f32).
pub(crate) fn rel_pos_embedding(t: usize, d: usize) -> Result<Tensor> {
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

/// Transformer-XL relative shift of x [B, H, T, 2T-1] -> [B, H, T, T]:
/// `y[i, j] = x[i, j + T - 1 - i]`. Flattened, row i starts at `(T-1) + i (2T-2)`, so the
/// result is a window of the flat view reshaped to [T, 2T-2] and cut to T columns.
pub(crate) fn rel_shift(x: &Tensor) -> Result<Tensor> {
    let s = x.shape();
    if s.len() != 4 || s[3] + 1 != 2 * s[2] {
        return Err(Error::new("shape_mismatch", format!("rel_shift expects [B, H, T, 2T-1], got {s:?}")));
    }
    let (b, h, t, p) = (s[0], s[1], s[2], s[3]);
    if t == 1 {
        return Ok(x.clone());
    }
    let flat = x.contiguous()?.reshape(&[b, h, t * p])?;
    let start = t - 1;
    let win = flat.slice(2, start, start + t * (2 * t - 2))?.contiguous()?;
    win.reshape(&[b, h, t, 2 * t - 2])?.slice(3, 0, t)
}

pub(crate) struct RelPosAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    pos: Linear,
    /// [1, H, 1, Dh] f32
    bias_u: Tensor,
    bias_v: Tensor,
    n_head: usize,
    scale: f32,
}

impl RelPosAttention {
    pub fn load(l: &Loader<'_>, prefix: &str, d: usize, n_head: usize) -> Result<RelPosAttention> {
        let dh = d / n_head;
        let bias = |n: &str| -> Result<Tensor> {
            let name = format!("{prefix}.{n}");
            let t = l.tensor_f32(&name)?;
            l.expect_shape(&t, &name, &[n_head, dh])?;
            t.reshape(&[1, n_head, 1, dh])
        };
        Ok(RelPosAttention {
            q: linear_checked(l, &format!("{prefix}.q_proj"), d, d)?,
            k: linear_checked(l, &format!("{prefix}.k_proj"), d, d)?,
            v: linear_checked(l, &format!("{prefix}.v_proj"), d, d)?,
            o: linear_checked(l, &format!("{prefix}.o_proj"), d, d)?,
            pos: linear_checked(l, &format!("{prefix}.relative_k_proj"), d, d)?,
            bias_u: bias("bias_u")?,
            bias_v: bias("bias_v")?,
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
        let bd = rel_shift(&q_v.matmul_t(&p)?)?.mul_scalar(f64::from(self.scale))?; // [1, H, T, T]
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
    fn shift_matches_definition() {
        for t in [1usize, 2, 3, 5] {
            let p = 2 * t - 1;
            let data: Vec<f32> = (0..2 * t * p).map(|i| i as f32).collect();
            let x = Tensor::from_f32(&data, &[1, 2, t, p]).expect("x");
            let y = eval(&rel_shift(&x).expect("shift"));
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
    fn positions_interleaved() {
        let pe = eval(&rel_pos_embedding(3, 4).expect("pe"));
        // row 0 = position 2, row 2 = position 0 -> (sin 0, cos 0) = (0, 1)
        assert!((pe[0] - 2f32.sin()).abs() < 1e-6 && (pe[1] - 2f32.cos()).abs() < 1e-6);
        assert_eq!(&pe[2 * 4..2 * 4 + 2], &[0.0, 1.0]);
        assert!((pe[4 * 4] - (-2f32).sin()).abs() < 1e-6);
    }
}
