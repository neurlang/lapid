//! Resize rules and patch cutting for patch-based vision encoders.
//!
//! Two layouts are in use: Qwen2-VL / Qwen3-VL style (`smart_resize`, patches in
//! spatial-merge-block order, each patch flattened (channel, row, column) like a `[D, C, P, P]`
//! convolution kernel) and Gemma 4 style (aspect-preserving resize to a patch budget, patches
//! in raster order, each flattened (row, column, channel)). Either way the patch embedding is
//! one matmul over `[n, C * P * P]` rows.

use lapid::{Error, Result, bail};

/// Patches of one image: `data` [n, C*P*P], the patch grid (rows, cols).
#[derive(Clone, Debug)]
pub struct Patches {
    pub data: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
    /// Values per patch (C * P * P).
    pub dim: usize,
}

impl Patches {
    pub fn len(&self) -> usize {
        self.grid_h * self.grid_w
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Output tokens after an `m x m` spatial merge.
    pub fn merged_tokens(&self, m: usize) -> usize {
        self.len() / (m * m).max(1)
    }
}

/// HF `smart_resize`: (height, width) divisible by `factor`, pixel count within
/// [`min_pixels`, `max_pixels`], aspect ratio kept (Python `round` = ties to even).
pub fn smart_resize(height: usize, width: usize, factor: usize, min_pixels: usize, max_pixels: usize) -> Result<(usize, usize)> {
    if height == 0 || width == 0 || factor == 0 {
        return Err(Error::new("invalid_argument", "smart_resize: empty image"));
    }
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    if h.max(w) / h.min(w) > 200.0 {
        return Err(Error::new("invalid_argument", format!("image aspect ratio of {width}x{height} exceeds 200")));
    }
    let mut hb = (h / f).round_ties_even() * f;
    let mut wb = (w / f).round_ties_even() * f;
    if hb * wb > max_pixels as f64 {
        let beta = (h * w / max_pixels as f64).sqrt();
        hb = f.max((h / beta / f).floor() * f);
        wb = f.max((w / beta / f).floor() * f);
    } else if hb * wb < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    Ok((hb as usize, wb as usize))
}

/// HF `get_aspect_ratio_preserving_size` (Gemma 4): (height, width) with at most
/// `max_patches` patches of side `patch`, both sides multiples of `k * patch` (`k`: pooling
/// kernel).
pub fn aspect_preserving_size(height: usize, width: usize, patch: usize, max_patches: usize, k: usize) -> Result<(usize, usize)> {
    let target_px = (max_patches * patch * patch) as f64;
    let factor = (target_px / (height * width) as f64).sqrt();
    let side = k * patch;
    let mut th = ((factor * height as f64) / side as f64).floor() as usize * side;
    let mut tw = ((factor * width as f64) / side as f64).floor() as usize * side;
    let max_side = (max_patches / (k * k)) * side;
    match (th, tw) {
        (0, 0) => bail!("invalid_argument", "vision: image {height}x{width} resizes to 0x0"),
        (0, _) => (th, tw) = (side, ((width as f64 / height as f64).floor() as usize * side).min(max_side)),
        (_, 0) => (th, tw) = (((height as f64 / width as f64).floor() as usize * side).min(max_side), side),
        _ => {}
    }
    if (th * tw) as f64 > target_px {
        bail!("invalid_argument", "vision: {height}x{width} -> {th}x{tw} exceeds {max_patches} patches");
    }
    Ok((th, tw))
}

/// Cut a planar `[3, H, W]` image into `[n, 3*P*P]` patches ordered by merge block
/// (block row, block col, row in block, col in block), each patch (channel, row, column).
pub fn patchify(chw: &[f32], h: usize, w: usize, p: usize, m: usize) -> Patches {
    let (gh, gw) = (h / p, w / p);
    let dim = 3 * p * p;
    let mut data = Vec::with_capacity(gh * gw * dim);
    for br in 0..gh / m {
        for bc in 0..gw / m {
            for ir in 0..m {
                for ic in 0..m {
                    let (r, c) = (br * m + ir, bc * m + ic);
                    for ch in 0..3 {
                        for y in 0..p {
                            let row = ch * h * w + (r * p + y) * w + c * p;
                            data.extend_from_slice(&chw[row..row + p]);
                        }
                    }
                }
            }
        }
    }
    Patches { data, grid_h: gh, grid_w: gw, dim }
}

/// Cut a planar `[3, H, W]` image into `[n, 3*P*P]` patches in raster order (row-major over
/// the grid), each patch flattened (row, column, channel).
pub fn patchify_raster(chw: &[f32], h: usize, w: usize, p: usize) -> Patches {
    let (rows, cols, dim) = (h / p, w / p, 3 * p * p);
    let mut data = vec![0.0f32; rows * cols * dim];
    for r in 0..rows {
        for c in 0..cols {
            let base = (r * cols + c) * dim;
            for (iy, ix, ch) in (0..p).flat_map(|y| (0..p).flat_map(move |x| (0..3).map(move |ch| (y, x, ch)))) {
                data[base + (iy * p + ix) * 3 + ch] = chw[ch * h * w + (r * p + iy) * w + c * p + ix];
            }
        }
    }
    Patches { data, grid_h: rows, grid_w: cols, dim }
}

/// (row, col) of every patch in merge-block order.
pub fn block_order_positions(gh: usize, gw: usize, m: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(gh * gw);
    for br in 0..gh / m {
        for bc in 0..gw / m {
            for ir in 0..m {
                for ic in 0..m {
                    v.push((br * m + ir, bc * m + ic));
                }
            }
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_hf() {
        // Values from transformers' smart_resize (factor 32, min 65536, max 16777216).
        assert_eq!(smart_resize(240, 320, 32, 65_536, 16_777_216).ok(), Some((256, 320)));
        assert_eq!(smart_resize(76, 100, 32, 65_536, 16_777_216).ok(), Some((224, 320)));
        assert_eq!(smart_resize(4000, 3000, 32, 65_536, 1_000_000).ok(), Some((1152, 864)));
    }

    #[test]
    fn aspect_preserving_matches_hf() {
        // 640x488 (w x h) at 280 soft tokens -> 672 x 912 (HF Gemma 4).
        assert_eq!(aspect_preserving_size(488, 640, 16, 2520, 3).expect("size"), (672, 912));
        assert_eq!(aspect_preserving_size(110, 150, 16, 630, 3).expect("size"), (336, 432));
    }

    #[test]
    fn patch_order() {
        let (h, w, p) = (4, 4, 1);
        let chw: Vec<f32> = (0..3 * h * w).map(|i| i as f32).collect();
        let pt = patchify(&chw, h, w, p, 2);
        assert_eq!((pt.grid_h, pt.grid_w, pt.dim), (4, 4, 3));
        // second patch = (row 0, col 1); fourth = (1, 1); fifth = (0, 2)
        assert_eq!(pt.data[3], 1.0);
        assert_eq!(pt.data[9], 5.0);
        assert_eq!(pt.data[12], 2.0);
        assert_eq!(block_order_positions(4, 4, 2)[4], (0, 2));
        // Raster: patch 5 = (row 1, col 1), channel 2 of its only pixel (5) -> 2 * 16 + 5.
        assert_eq!(patchify_raster(&chw, h, w, p).data[5 * 3 + 2], 37.0);
    }
}
