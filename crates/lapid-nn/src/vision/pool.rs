//! Patch-grid resampling as small host-built matrices: bilinear interpolation of a learned
//! square position table to the image's grid, and k x k average pooling of patch features.

use lapid::{DType, Result, Tensor, bail};

use super::patches::block_order_positions;
use crate::Embedding;

/// Bilinear (align_corners) taps of a `side x side` grid resampled to `gh x gw`, per patch in
/// merge-block order: 4 (index, weight) pairs (HF `fast_pos_embed_interpolate`).
pub fn bilinear_taps(gh: usize, gw: usize, side: usize, merge: usize) -> (Vec<u32>, Vec<f32>) {
    let axis = |i: usize, size: usize| -> [(usize, f32); 2] {
        let src = if size > 1 { i as f32 * (side - 1) as f32 / (size - 1) as f32 } else { 0.0 };
        let f = src.floor();
        let d = src - f;
        let i0 = (f as usize).min(side - 1);
        let i1 = (f as usize + 1).min(side - 1);
        [(i0, 1.0 - d), (i1, d)]
    };
    let pos = block_order_positions(gh, gw, merge);
    let (mut idx, mut wts) = (Vec::with_capacity(pos.len() * 4), Vec::with_capacity(pos.len() * 4));
    for (r, c) in pos {
        let (rt, ct) = (axis(r, gh), axis(c, gw));
        for (ri, rw) in rt {
            for (ci, cw) in ct {
                idx.push((ri * side + ci) as u32);
                wts.push(rw.max(0.0) * cw.max(0.0));
            }
        }
    }
    (idx, wts)
}

/// Position table [side * side, D] bilinearly resampled to a `gh x gw` grid: [gh * gw, D] f32
/// rows in merge-block order.
pub fn interpolate_pos_table(table: &Tensor, gh: usize, gw: usize, side: usize, merge: usize) -> Result<Tensor> {
    let (n, d) = (gh * gw, table.shape()[1]);
    let (idx, w) = bilinear_taps(gh, gw, side, merge);
    let g = Embedding::new(table.clone()).lookup(&idx)?.cast(DType::F32)?.reshape(&[n * 4, d])?;
    g.mul(&Tensor::from_f32(&w, &[n * 4, 1])?)?.reshape(&[n, 4, d])?.sum(1, false)
}

/// k x k average pooling of patches at `positions` (x, y) as an [M, N] matrix: output token
/// `x / k + blocks_w * (y / k)` (blocks row-major, `blocks_w` blocks per row), weights 1/k^2.
pub fn block_mean_matrix(positions: &[[usize; 2]], k: usize, blocks_w: usize, m: usize) -> Result<Tensor> {
    let n = positions.len();
    let mut w = vec![0.0f32; m * n];
    for (j, p) in positions.iter().enumerate() {
        let block = p[0] / k + blocks_w * (p[1] / k);
        if block >= m {
            bail!("shape_mismatch", "vision: patch {j} pools to block {block} of {m}");
        }
        w[block * n + j] = 1.0 / (k * k) as f32;
    }
    Tensor::from_f32(&w, &[m, n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taps_corners() {
        // 2x2 grid from a 3x3 table, align_corners: exact corners.
        let (idx, w) = bilinear_taps(2, 2, 3, 2);
        assert_eq!(idx.len(), 16);
        // patch (0,0): all weight on index 0
        let s: f32 = idx[..4].iter().zip(&w[..4]).filter(|(i, _)| **i == 0).map(|(_, w)| *w).sum();
        assert!((s - 1.0).abs() < 1e-6);
        // patch (1,1) (4th in block order): weight on index 8
        let s: f32 = idx[12..].iter().zip(&w[12..]).filter(|(i, _)| **i == 8).map(|(_, w)| *w).sum();
        assert!((s - 1.0).abs() < 1e-6);
    }
}
