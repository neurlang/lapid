//! Image preprocessing settings of the generic encoder (Qwen2-VL / Qwen3-VL style
//! `smart_resize`, rescale / normalize, patches in spatial-merge-block order; the math is
//! `lapid_nn::vision`).
//!
//! Video-style temporal patches (Qwen's Conv3d over 2 identical frames for a still image) are
//! folded into the kernel at conversion time (sum over the temporal axis), so images carry one
//! frame.

use lapid::{Error, Result};
pub use lapid_nn::vision::Patches;
use lapid_nn::vision::{Filter, Image, patchify, smart_resize};
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
