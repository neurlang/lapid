//! Images for vision models: decoding (PNG / JPEG through the `image` crate built with only
//! its pure-Rust codecs), RGB conversion, antialiased resampling bit-exact with
//! torchvision (`antialias=True` on uint8, as the HF fast processors) and conversion to normalized planar f32.

use std::path::Path;

use lapid::{Error, Result};

/// An 8-bit RGB image, row-major, `rgb.len() == width * height * 3`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

/// Resampling filter (PIL names).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Filter {
    Bilinear,
    Bicubic,
    Lanczos,
}

impl Filter {
    /// From a HF processor `resample` value (PIL enum: 1 lanczos, 2 bilinear, 3 bicubic).
    pub fn from_pil(v: u64) -> Filter {
        match v {
            1 => Filter::Lanczos,
            2 => Filter::Bilinear,
            _ => Filter::Bicubic,
        }
    }

    fn support(self) -> f64 {
        match self {
            Filter::Bilinear => 1.0,
            Filter::Bicubic => 2.0,
            Filter::Lanczos => 3.0,
        }
    }

    fn eval(self, x: f64) -> f64 {
        let x = x.abs();
        match self {
            Filter::Bilinear => (1.0 - x).max(0.0),
            Filter::Bicubic => {
                let a = -0.5;
                if x < 1.0 {
                    ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * a
                } else {
                    0.0
                }
            }
            Filter::Lanczos => {
                if x < 3.0 {
                    sinc(x) * sinc(x / 3.0)
                } else {
                    0.0
                }
            }
        }
    }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Per output index: first input index and normalized float weights (PIL `precompute_coeffs`).
fn coeffs(in_size: usize, out_size: usize, filter: Filter) -> Vec<(usize, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let fscale = scale.max(1.0);
    let support = filter.support() * fscale;
    (0..out_size)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5).floor().max(0.0)) as usize;
            let xmax = ((center + support + 0.5).floor() as usize).min(in_size);
            let mut w: Vec<f64> = (xmin..xmax).map(|x| filter.eval((x as f64 - center + 0.5) / fscale)).collect();
            let total: f64 = w.iter().sum();
            if total != 0.0 {
                w.iter_mut().for_each(|v| *v /= total);
            }
            (xmin, w)
        })
        .collect()
}

/// [`coeffs`] in torchvision's uint8 fixed point: int16-range weights with the largest
/// precision (<= 22 bits) that keeps the biggest weight below 2^15. Returns the precision.
fn int_coeffs(in_size: usize, out_size: usize, filter: Filter) -> (Vec<(usize, Vec<i64>)>, u32) {
    let float = coeffs(in_size, out_size, filter);
    let mx = float.iter().flat_map(|(_, w)| w.iter()).fold(0.0f64, |m, v| m.max(v.abs()));
    let mut prec = 0u32;
    while prec < 22 && mx * f64::from(1u32 << (prec + 1)) < 32768.0 {
        prec += 1;
    }
    let one = f64::from(1u32 << prec);
    let ints = float
        .into_iter()
        .map(|(x0, w)| (x0, w.into_iter().map(|v| if v < 0.0 { (-0.5 + v * one) as i64 } else { (0.5 + v * one) as i64 }).collect()))
        .collect();
    (ints, prec)
}

fn round_shift(s: i64, prec: u32) -> u8 {
    let r = if prec == 0 { s } else { (s + (1i64 << (prec - 1))) >> prec };
    r.clamp(0, 255) as u8
}

impl Image {
    pub fn new(width: usize, height: usize, rgb: Vec<u8>) -> Result<Image> {
        if rgb.len() != width * height * 3 || width == 0 || height == 0 {
            return Err(Error::new("invalid_argument", format!("image: {} bytes for {width}x{height} RGB", rgb.len())));
        }
        Ok(Image { width, height, rgb })
    }

    /// Decode a PNG or JPEG (any color type; alpha is dropped, gray expanded) to RGB.
    pub fn decode(bytes: &[u8]) -> Result<Image> {
        let img = image::load_from_memory(bytes).map_err(|e| Error::new("image", format!("image decode: {e}")))?;
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        Image::new(w, h, rgb.into_raw())
    }

    /// Read and decode an image file.
    pub fn open(path: &Path) -> Result<Image> {
        let bytes = std::fs::read(path).map_err(|e| Error::new("io", format!("{}: {e}", path.display())))?;
        Image::decode(&bytes)
    }

    /// Antialiased resize, bit-exact with torchvision `resize(uint8, antialias=True)` (the HF
    /// fast image processors; PIL uses the same scheme): separable filters with int16
    /// fixed-point weights, horizontal pass first, rounded and clamped to 8 bits after each pass.
    pub fn resize(&self, width: usize, height: usize, filter: Filter) -> Result<Image> {
        if width == 0 || height == 0 {
            return Err(Error::new("invalid_argument", "image: resize to an empty size"));
        }
        let mut cur = self.clone();
        if width != cur.width {
            let (cx, p) = int_coeffs(cur.width, width, filter);
            let mut out = vec![0u8; width * cur.height * 3];
            for y in 0..cur.height {
                let row = &cur.rgb[y * cur.width * 3..(y + 1) * cur.width * 3];
                for (x, (x0, w)) in cx.iter().enumerate() {
                    for c in 0..3 {
                        let s: i64 = w.iter().enumerate().map(|(k, wk)| wk * i64::from(row[(x0 + k) * 3 + c])).sum();
                        out[(y * width + x) * 3 + c] = round_shift(s, p);
                    }
                }
            }
            cur = Image { width, height: cur.height, rgb: out };
        }
        if height != cur.height {
            let (cy, p) = int_coeffs(cur.height, height, filter);
            let mut out = vec![0u8; width * height * 3];
            for (y, (y0, w)) in cy.iter().enumerate() {
                for x in 0..width * 3 {
                    let s: i64 = w.iter().enumerate().map(|(k, wk)| wk * i64::from(cur.rgb[(y0 + k) * width * 3 + x])).sum();
                    out[y * width * 3 + x] = round_shift(s, p);
                }
            }
            cur = Image { width, height, rgb: out };
        }
        Ok(cur)
    }

    /// Planar `[3, H, W]` f32: `(v * rescale - mean[c]) / std[c]`.
    pub fn to_chw(&self, rescale: f32, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
        let n = self.width * self.height;
        let mut out = vec![0.0f32; 3 * n];
        for i in 0..n {
            for c in 0..3 {
                out[c * n + i] = (f32::from(self.rgb[i * 3 + c]) * rescale - mean[c]) / std[c];
            }
        }
        out
    }

    /// FNV-1a over size and pixels (identifies an image in a session's cache).
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let bytes = (self.width as u64).to_le_bytes().into_iter().chain((self.height as u64).to_le_bytes());
        for b in bytes.chain(self.rgb.iter().copied()) {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_identity_and_constant() {
        let img = Image::new(4, 3, vec![77; 36]).expect("img");
        assert_eq!(img.resize(4, 3, Filter::Bicubic).expect("same"), img);
        let r = img.resize(9, 2, Filter::Bicubic).expect("resize");
        assert!(r.rgb.iter().all(|&v| v == 77));
        let r = img.resize(2, 7, Filter::Bilinear).expect("resize");
        assert_eq!((r.width, r.height), (2, 7));
    }

    #[test]
    fn coeffs_sum_to_one() {
        for (i, o) in [(10, 3), (3, 10), (224, 448)] {
            for (_, w) in coeffs(i, o, Filter::Bicubic) {
                assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            }
            let (c, p) = int_coeffs(i, o, Filter::Bicubic);
            assert!(p > 10 && p <= 22);
            assert!(c.iter().flat_map(|(_, w)| w).all(|v| v.abs() < 32768));
        }
    }

    #[test]
    fn png_roundtrip() {
        let mut buf = Vec::new();
        let px: Vec<u8> = (0..2 * 2 * 3).map(|i| (i * 20) as u8).collect();
        image::RgbImage::from_raw(2, 2, px.clone())
            .expect("raw")
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .expect("encode");
        let img = Image::decode(&buf).expect("decode");
        assert_eq!(img.rgb, px);
    }
}
