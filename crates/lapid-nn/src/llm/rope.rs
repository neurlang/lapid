//! RoPE variants of decoder LLMs, as inverse-frequency tables plus host-built cos / sin tables
//! (f32 `position * inv_freq` like HF, then cos / sin):
//!
//! * [`scaled_inv_freq`]: default, HF "linear", HF "llama3" and per-frequency divisors
//!   (llama.cpp `rope_freqs.weight`); [`yarn_inv_freq`]: YaRN (DeepSeek V4 compressed layers).
//! * [`tables`] + [`rotate_half`]: half-split rotation of the whole head (Llama / Qwen).
//! * [`MRope`]: multimodal RoPE (Qwen2/3-VL): every token has (t, h, w) positions and each
//!   frequency is driven by one of them; [`mrope_positions`] is HF `get_rope_index`.
//! * [`TailRope`]: interleaved pairs on the last `dim` channels of a head, with the inverse
//!   rotation (DeepSeek V4).

use lapid::{Result, Tensor, bail};

/// Half-split rotation of x `[B, H, S, Dh]` by cos / sin `[S, Dh / 2]` (shared with the vision
/// towers).
pub use crate::vision::apply_rope_tables as rotate_half;
use crate::vision::rope_tables as cos_sin;

/// Position scaling of the rotary embedding (as inverse-frequency divisors).
#[derive(Clone, Debug, PartialEq)]
pub enum RopeScaling {
    None,
    /// HF "linear": positions divided by `factor` (= inverse frequencies divided by it).
    Linear { factor: f32 },
    /// HF "llama3": low frequencies divided by `factor`, a smooth ramp in between.
    Llama3 { factor: f32, low_freq_factor: f32, high_freq_factor: f32, original_max_positions: usize },
    /// Per-frequency divisors (llama.cpp `rope_freqs.weight`, how GGUF stores llama3 scaling).
    FreqFactors(Vec<f32>),
}

/// YaRN frequency interpolation.
#[derive(Clone, Debug, PartialEq)]
pub struct Yarn {
    pub factor: f32,
    pub original_max_positions: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

/// `1 / theta^(2i / dim)` for i < dim / 2 (f32 like HF).
pub fn inv_freq(dim: usize, theta: f32) -> Vec<f32> {
    (0..dim / 2).map(|i| 1.0f32 / theta.powf((2 * i) as f32 / dim as f32)).collect()
}

/// Inverse frequencies with `scaling` applied (computed like HF).
pub fn scaled_inv_freq(dim: usize, theta: f32, scaling: &RopeScaling) -> Vec<f32> {
    let base = inv_freq(dim, theta);
    match scaling {
        RopeScaling::None => base,
        RopeScaling::Linear { factor } => base.iter().map(|f| f / factor).collect(),
        RopeScaling::FreqFactors(ff) => base.iter().zip(ff).map(|(f, s)| f / s).collect(),
        RopeScaling::Llama3 { factor, low_freq_factor, high_freq_factor, original_max_positions } => {
            let old = *original_max_positions as f32;
            let low_wavelen = old / low_freq_factor;
            let high_wavelen = old / high_freq_factor;
            base.iter()
                .map(|&f| {
                    let wavelen = 2.0 * std::f32::consts::PI / f;
                    let scaled = if wavelen > low_wavelen { f / factor } else { f };
                    if wavelen >= high_wavelen && wavelen <= low_wavelen {
                        let smooth = (old / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                        (1.0 - smooth) * scaled / factor + smooth * scaled
                    } else {
                        scaled
                    }
                })
                .collect()
        }
    }
}

/// HF `_compute_yarn_parameters` (truncate = true, attention factor 1.0) / DeepSeek
/// `precompute_freqs_cis`.
pub fn yarn_inv_freq(dim: usize, base: f32, y: &Yarn) -> Vec<f32> {
    let correction = |rot: f32| -> f64 { dim as f64 * (y.original_max_positions as f64 / (rot as f64 * 2.0 * std::f64::consts::PI)).ln() / (2.0 * (base as f64).ln()) };
    let low = correction(y.beta_fast).floor().max(0.0);
    let mut high = correction(y.beta_slow).ceil().min(dim as f64 - 1.0);
    if low == high {
        high += 0.001;
    }
    inv_freq(dim, base)
        .into_iter()
        .enumerate()
        .map(|(i, extra)| {
            let extrapolation = 1.0 - (((i as f64 - low) / (high - low)).clamp(0.0, 1.0)) as f32;
            extra / y.factor * (1.0 - extrapolation) + extra * extrapolation
        })
        .collect()
}

/// (cos, sin) `[n, inv_freq.len()]` for `positions`.
pub fn tables(inv_freq: &[f32], positions: &[usize]) -> Result<(Tensor, Tensor)> {
    let a: Vec<f32> = positions.iter().flat_map(|&p| inv_freq.iter().map(move |f| p as f32 * f)).collect();
    cos_sin(&a, positions.len(), inv_freq.len())
}

/// Multimodal RoPE frequency assignment: `section` = (t, h, w) frequency counts over
/// head_dim / 2; `interleaved` (Qwen3-VL) cycles THWTHW.. over the first 3 h / 3 w
/// frequencies, otherwise (Qwen2-VL) the streams take contiguous blocks.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MRope {
    pub section: [usize; 3],
    pub interleaved: bool,
}

impl MRope {
    /// Which position stream (0 = t, 1 = h, 2 = w) drives frequency `i`.
    pub fn stream(&self, i: usize) -> usize {
        let [t, h, w] = self.section;
        match self.interleaved {
            true if i % 3 == 1 && i < 3 * h => 1,
            true if i % 3 == 2 && i < 3 * w => 2,
            true => 0,
            false if i < t => 0,
            false if i < t + h => 1,
            false => 2,
        }
    }

    /// (cos, sin) `[n, inv_freq.len()]` of (t, h, w) positions.
    pub fn tables(&self, inv_freq: &[f32], pos: &[[usize; 3]]) -> Result<(Tensor, Tensor)> {
        let a: Vec<f32> = pos.iter().flat_map(|p| inv_freq.iter().enumerate().map(move |(i, f)| p[self.stream(i)] as f32 * f)).collect();
        cos_sin(&a, pos.len(), inv_freq.len())
    }
}

/// M-RoPE positions of `n` tokens with image runs `(start, len, (grid_h, grid_w))` from cursor
/// `c`, and the next cursor: text tokens get t = h = w = cursor, an image of `H x W` merged
/// tokens gets (c, c + row, c + col) and moves the cursor to `c + max(H, W)`.
pub fn mrope_positions(n: usize, media: &[(usize, usize, (usize, usize))], mut c: usize) -> Result<(Vec<[usize; 3]>, usize)> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        match media.iter().find(|m| m.0 == out.len()) {
            Some(&(_, len, (gh, gw))) => {
                if len != gh * gw {
                    bail!("invalid_argument", "mrope: image run of {len} tokens for a {gh}x{gw} grid");
                }
                out.extend((0..gh).flat_map(|r| (0..gw).map(move |col| [c, c + r, c + col])));
                c += gh.max(gw);
            }
            None => {
                out.push([c; 3]);
                c += 1;
            }
        }
    }
    out.truncate(n);
    Ok((out, c))
}

/// Interleaved pairs `(x[2i], x[2i+1])` on the last `dim` channels of a head (`[nope | rope]`
/// layout); [`TailRope::apply`] can rotate by `-angle` (DeepSeek V4 un-rotates the attention
/// output, as K = V carries the rotation).
#[derive(Clone)]
pub struct TailRope {
    dim: usize,
    inv_freq: Vec<f32>,
}

impl TailRope {
    pub fn new(dim: usize, base: f32, yarn: Option<&Yarn>) -> TailRope {
        let inv_freq = match yarn {
            Some(y) => yarn_inv_freq(dim, base, y),
            None => inv_freq(dim, base),
        };
        TailRope { dim, inv_freq }
    }

    /// (cos, sin) `[n, dim / 2]` for absolute `positions`.
    pub fn tables(&self, positions: &[usize]) -> Result<(Tensor, Tensor)> {
        tables(&self.inv_freq, positions)
    }

    /// Rotate the last `dim` channels of `x` `[n, H, D]` (one position per row n) by the
    /// tables of [`TailRope::tables`]; `inverse` rotates by `-angle`.
    pub fn apply(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, inverse: bool) -> Result<Tensor> {
        if x.rank() != 3 {
            bail!("shape_mismatch", "tail rope expects [n, H, D], got {:?}", x.shape());
        }
        let (n, h, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (r, half) = (self.dim, self.dim / 2);
        if d < r || cos.shape() != [n, half] {
            bail!("shape_mismatch", "tail rope: x {:?}, tables {:?}, rope dim {r}", x.shape(), cos.shape());
        }
        let rot = x.slice(2, d - r, d)?.reshape(&[n, h, half, 2])?;
        let (a, b) = (rot.slice(3, 0, 1)?, rot.slice(3, 1, 2)?);
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
    fn llama3_scales_low_frequencies() {
        let s = RopeScaling::Llama3 { factor: 32.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max_positions: 8192 };
        let f = scaled_inv_freq(64, 500000.0, &s);
        assert_eq!(f.len(), 32);
        assert_eq!(f[0], 1.0);
        let plain = 1.0f32 / 500000f32.powf(62.0 / 64.0);
        assert!((f[31] - plain / 32.0).abs() < 1e-12);
    }

    #[test]
    fn yarn_matches_plain_at_high_freq() {
        let y = Yarn { factor: 16.0, original_max_positions: 65536, beta_fast: 32.0, beta_slow: 1.0 };
        let (a, b) = (yarn_inv_freq(64, 160000.0, &y), inv_freq(64, 160000.0));
        // the fastest-rotating pairs are extrapolated (unchanged), the slowest interpolated (/ factor)
        assert!((a[0] - b[0]).abs() < 1e-9);
        assert!((a[31] * 16.0 - b[31]).abs() < 1e-9 * b[31].max(1.0));
    }

    #[test]
    fn mrope_streams_and_positions() {
        let m = MRope { section: [24, 20, 20], interleaved: true };
        // interleaved: 0 t, 1 h, 2 w, ..., 58 h (< 60), 59 w, 60.. t
        assert_eq!([0, 1, 2, 58, 59, 61].map(|i| m.stream(i)), [0, 1, 2, 1, 2, 0]);
        // 2 text tokens, a 2x3 image, 1 text token.
        let (p, next) = mrope_positions(9, &[(2, 6, (2, 3))], 0).expect("positions");
        assert_eq!((p[1], p[2], p[7], p[8], next), ([1; 3], [2, 2, 2], [2, 3, 4], [5; 3], 6));
    }
}
