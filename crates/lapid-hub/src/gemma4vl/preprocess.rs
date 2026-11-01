//! Gemma 4 image preprocessing (HF `Gemma4ImageProcessor` + `Gemma4Processor`): an
//! aspect-ratio-preserving resize to the largest size with at most `max_soft_tokens * k^2`
//! patches whose sides are multiples of `k * patch_size`, rescale to [0, 1], patchify
//! (row-major patches, each patch `(row, col, channel)` flattened), and patch positions (x, y).

use lapid::{Result, bail};

use super::config::ImageProcessorConfig;
use crate::vision::Image;

/// One image ready for the vision tower.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedImage {
    /// [n_patches, 3 * p * p], HF `pixel_values` (real patches only).
    pub patches: Vec<f32>,
    /// (x, y) = (column, row) of every patch.
    pub positions: Vec<[usize; 2]>,
    /// Patch grid (rows, columns).
    pub grid: (usize, usize),
    /// Soft tokens (`<|image|>` placeholders) after pooling.
    pub n_soft: usize,
    pub patch_size: usize,
}

impl PreparedImage {
    pub fn n_patches(&self) -> usize {
        self.positions.len()
    }

    /// From HF `pixel_values` / `image_position_ids` (real patches only), e.g. a reference.
    pub fn from_patches(patches: Vec<f32>, positions: Vec<[usize; 2]>, patch_size: usize, k: usize) -> Result<PreparedImage> {
        let n = positions.len();
        if n == 0 || patches.len() != n * 3 * patch_size * patch_size {
            bail!("shape_mismatch", "gemma4 vision: {} values for {n} patches of {patch_size}x{patch_size}", patches.len());
        }
        let cols = positions.iter().map(|p| p[0]).max().unwrap_or(0) + 1;
        let rows = positions.iter().map(|p| p[1]).max().unwrap_or(0) + 1;
        if rows * cols != n || rows % k != 0 || cols % k != 0 {
            bail!("shape_mismatch", "gemma4 vision: patch grid {rows}x{cols} vs {n} patches (pool {k})");
        }
        Ok(PreparedImage { patches, positions, grid: (rows, cols), n_soft: n / (k * k), patch_size })
    }
}

/// HF `get_aspect_ratio_preserving_size` (height, width).
pub fn target_size(height: usize, width: usize, patch: usize, max_patches: usize, k: usize) -> Result<(usize, usize)> {
    let total = (height * width) as f64;
    let target_px = (max_patches * patch * patch) as f64;
    let factor = (target_px / total).sqrt();
    let side = k * patch;
    let mut th = ((factor * height as f64) / side as f64).floor() as usize * side;
    let mut tw = ((factor * width as f64) / side as f64).floor() as usize * side;
    if th == 0 && tw == 0 {
        bail!("invalid_argument", "gemma4 vision: image {height}x{width} resizes to 0x0");
    }
    let max_side = (max_patches / (k * k)) * side;
    if th == 0 {
        th = side;
        tw = ((width as f64 / height as f64).floor() as usize * side).min(max_side);
    } else if tw == 0 {
        tw = side;
        th = ((height as f64 / width as f64).floor() as usize * side).min(max_side);
    }
    if (th * tw) as f64 > target_px {
        bail!("invalid_argument", "gemma4 vision: {height}x{width} -> {th}x{tw} exceeds {max_patches} patches");
    }
    Ok((th, tw))
}

/// Resize, rescale and patchify `img`.
pub fn prepare(img: &Image, pc: &ImageProcessorConfig) -> Result<PreparedImage> {
    let (p, k) = (pc.patch_size, pc.pooling_kernel_size);
    if p == 0 || k == 0 {
        bail!("metadata", "gemma4 vision: patch size {p}, pooling {k}");
    }
    let img = if pc.do_resize {
        let (th, tw) = target_size(img.height, img.width, p, pc.max_soft_tokens * k * k, k)?;
        img.resize(tw, th, pc.resample)?
    } else {
        img.clone()
    };
    if img.height % (p * k) != 0 || img.width % (p * k) != 0 {
        bail!("invalid_argument", "gemma4 vision: {}x{} is not a multiple of {}", img.height, img.width, p * k);
    }
    let (mean, std) = if pc.do_normalize { (pc.image_mean, pc.image_std) } else { ([0.0; 3], [1.0; 3]) };
    let chw = img.to_chw(pc.rescale_factor, mean, std);
    let (h, w) = (img.height, img.width);
    let (rows, cols) = (h / p, w / p);
    let pp = 3 * p * p;
    let mut patches = vec![0.0f32; rows * cols * pp];
    let mut positions = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let base = (r * cols + c) * pp;
            for iy in 0..p {
                for ix in 0..p {
                    let (y, x) = (r * p + iy, c * p + ix);
                    for ch in 0..3 {
                        patches[base + (iy * p + ix) * 3 + ch] = chw[ch * h * w + y * w + x];
                    }
                }
            }
            positions.push([c, r]);
        }
    }
    Ok(PreparedImage { patches, positions, grid: (rows, cols), n_soft: rows * cols / (k * k), patch_size: p })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_hf() {
        // 640x488 (w x h) at 280 soft tokens -> 672 x 912 (HF).
        assert_eq!(target_size(488, 640, 16, 2520, 3).expect("size"), (672, 912));
        assert_eq!(target_size(110, 150, 16, 630, 3).expect("size"), (336, 432));
        // Very wide image: height collapses to one block.
        let (h, w) = target_size(10, 4000, 16, 2520, 3).expect("wide");
        assert_eq!(h, 48);
        assert!(w <= 280 * 48);
    }

    #[test]
    fn patch_layout() {
        let rgb: Vec<u8> = (0..96 * 48 * 3).map(|i| (i % 251) as u8).collect();
        let img = Image::new(96, 48, rgb.clone()).expect("img");
        let pc = ImageProcessorConfig { do_resize: false, ..Default::default() };
        let p = prepare(&img, &pc).expect("prep");
        assert_eq!(p.grid, (3, 6));
        assert_eq!(p.n_soft, 2);
        assert_eq!(p.positions[7], [1, 1]);
        // Patch 7 (row 1, col 1), pixel (2, 5), channel 1.
        let (y, x) = (16 + 2, 16 + 5);
        let want = f32::from(rgb[(y * 96 + x) * 3 + 1]) * (1.0 / 255.0);
        assert_eq!(p.patches[7 * 768 + (2 * 16 + 5) * 3 + 1], want);
    }
}
