//! 2-D rotary embeddings over patch grids, and RoPE from host angle tables (also used for
//! multi-axis text RoPE such as Qwen3-VL's M-RoPE).
//!
//! Two vision layouts:
//! * [`grid_rope_tables`] (Qwen2-VL / Qwen3-VL ViT): one `rotate_half` rotation over the whole
//!   head; of its `Dh/2` angles the first `Dh/4` follow the row, the rest the column.
//! * [`axial_rope`] (Gemma 4 vision): the head is split in two halves, each rotated on its own
//!   (`rotate_half` within the half), the first by the column (x), the second by the row (y).

use lapid::{Result, Tensor};

use crate::Rope;

/// Rotate x [B, H, S, Dh] by host tables cos / sin [S, Dh/2] (HF `rotate_half` pairing:
/// element i with i + Dh/2, both using angle i).
pub fn apply_rope_tables(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dh = x.shape().last().copied().unwrap_or(0);
    let half = dh / 2;
    let (cos, sin) = (cos.cast(x.dtype())?, sin.cast(x.dtype())?);
    let x1 = x.slice(3, 0, half)?;
    let x2 = x.slice(3, half, dh)?;
    let o1 = x1.mul(&cos)?.sub(&x2.mul(&sin)?)?;
    let o2 = x2.mul(&cos)?.add(&x1.mul(&sin)?)?;
    Tensor::cat(&[&o1, &o2], 3)
}

/// cos / sin tables [n, Dh/2] from host angles.
pub fn rope_tables(angles: &[f32], n: usize, half: usize) -> Result<(Tensor, Tensor)> {
    let cos: Vec<f32> = angles.iter().map(|&a| f64::from(a).cos() as f32).collect();
    let sin: Vec<f32> = angles.iter().map(|&a| f64::from(a).sin() as f32).collect();
    Ok((Tensor::from_f32(&cos, &[n, half])?, Tensor::from_f32(&sin, &[n, half])?))
}

/// 2-D RoPE tables [N, Dh/2] for patches at `pos` (row, col): first Dh/4 angles from the row,
/// the rest from the column, frequencies `theta^(-2j / (Dh/2))`.
pub fn grid_rope_tables(pos: &[(usize, usize)], head_dim: usize, theta: f32) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let q = half / 2;
    let inv: Vec<f32> = (0..q).map(|j| 1.0 / theta.powf((2 * j) as f32 / half as f32)).collect();
    let mut a = Vec::with_capacity(pos.len() * half);
    for (r, c) in pos {
        a.extend(inv.iter().map(|f| *r as f32 * f));
        a.extend(inv.iter().map(|f| *c as f32 * f));
    }
    rope_tables(&a, pos.len(), half)
}

/// Axial RoPE on t [B, H, N, Dh]: `rope` (head dim Dh/2) rotates the first half of each head
/// at the column `xs` and the second half at the row `ys`.
pub fn axial_rope(rope: &Rope, t: &Tensor, xs: &[usize], ys: &[usize]) -> Result<Tensor> {
    let hd = 2 * rope.head_dim;
    let a = rope.apply_positions(&t.slice(3, 0, hd / 2)?, xs)?;
    Tensor::cat(&[&a, &rope.apply_positions(&t.slice(3, hd / 2, hd)?, ys)?], 3)
}
