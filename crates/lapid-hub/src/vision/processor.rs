//! Image preprocessing for patch-based vision encoders (Qwen2-VL / Qwen3-VL style
//! `smart_resize` → rescale / normalize → patches in spatial-merge-block order).
//!
//! Patches are `[n, C * P * P]` rows in (channel, row, column) order — the flattening of a
//! `[D, C, P, P]` convolution kernel, so the patch embedding is one matmul. Video-style temporal
//! patches (Qwen's Conv3d over 2 identical frames for a still image) are folded into the
//! kernel at conversion time (sum over the temporal axis), so images carry one frame.

use lapid::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::image::{Filter, Image};

/// Preprocessing parameters (HF `preprocessor_config.json` / mmproj `clip.vision.*`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageProcessor {
    pub patch_size: usize,
    /// Spatial merge (patches per side merged into one output token).
    pub merge_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub rescale: f32,
    /// PIL resample id (3 = bicubic).
    pub resample: u64,
}

impl Default for ImageProcessor {
    fn default() -> Self {
        ImageProcessor {
            patch_size: 16,
            merge_size: 2,
            min_pixels: 65_536,
            max_pixels: 16_777_216,
            mean: [0.5; 3],
            std: [0.5; 3],
            rescale: 1.0 / 255.0,
            resample: 3,
        }
    }
}

fn triple(v: Option<&Value>, default: [f32; 3]) -> [f32; 3] {
    match v.and_then(Value::as_array) {
        Some(a) if a.len() == 3 => {
            let mut o = default;
            for (i, x) in a.iter().enumerate() {
                o[i] = x.as_f64().unwrap_or(f64::from(default[i])) as f32;
            }
            o
        }
        _ => default,
    }
}

/// Patches of one image: `data` [n, C*P*P] in merge-block order, the patch grid (rows, cols).
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

impl ImageProcessor {
    /// From an HF `preprocessor_config.json` (Qwen2-VL / Qwen3-VL keys).
    pub fn from_hf(v: &Value) -> ImageProcessor {
        let d = ImageProcessor::default();
        let u = |k: &str| v.get(k).and_then(Value::as_u64).map(|x| x as usize);
        let size = v.get("size");
        let size_u = |k: &str| size.and_then(|s| s.get(k)).and_then(Value::as_u64).map(|x| x as usize);
        ImageProcessor {
            patch_size: u("patch_size").unwrap_or(d.patch_size),
            merge_size: u("merge_size").unwrap_or(d.merge_size),
            min_pixels: size_u("shortest_edge").or_else(|| u("min_pixels")).unwrap_or(d.min_pixels),
            max_pixels: size_u("longest_edge").or_else(|| u("max_pixels")).unwrap_or(d.max_pixels),
            mean: triple(v.get("image_mean"), d.mean),
            std: triple(v.get("image_std"), d.std),
            rescale: v.get("rescale_factor").and_then(Value::as_f64).map_or(d.rescale, |x| x as f32),
            resample: v.get("resample").and_then(Value::as_u64).unwrap_or(d.resample),
        }
    }

    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    pub fn from_json(v: &Value) -> Result<ImageProcessor> {
        serde_json::from_value(v.clone()).map_err(|e| Error::new("metadata", format!("image processor: {e}")))
    }

    /// Size the image is resized to (height, width).
    pub fn target_size(&self, height: usize, width: usize) -> Result<(usize, usize)> {
        smart_resize(height, width, self.patch_size * self.merge_size, self.min_pixels, self.max_pixels)
    }

    /// Resize, normalize and cut into patches (merge-block order).
    pub fn process(&self, img: &Image) -> Result<Patches> {
        let (h, w) = self.target_size(img.height, img.width)?;
        let resized = img.resize(w, h, Filter::from_pil(self.resample))?;
        let chw = resized.to_chw(self.rescale, self.mean, self.std);
        Ok(patchify(&chw, h, w, self.patch_size, self.merge_size))
    }
}

/// Cut a planar `[3, H, W]` image into `[n, 3*P*P]` patches ordered by merge block
/// (block row, block col, row in block, col in block).
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
    }
}
