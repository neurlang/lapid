//! Report types and numeric comparison metrics.

use lapid::{Error, Result};
use serde::Serialize;

/// One compared tensor (a layer output, the mel spectrogram or the logits).
#[derive(Debug, Clone, Serialize)]
pub struct LayerReport {
    pub name: String,
    pub max_abs: f32,
    pub mean_abs: f32,
    /// ‖a − b‖₂ / ‖b‖₂ with `b` the reference.
    pub rel: f32,
    pub ok: bool,
}

/// Error statistics of `ours` against `reference`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Diff {
    pub max_abs: f32,
    pub mean_abs: f32,
    pub rel: f32,
}

/// Compare two equally sized buffers. Accumulates in f64. Any NaN/inf in `ours` yields a
/// NaN `rel`, which never passes a tolerance check.
pub fn diff(ours: &[f32], reference: &[f32]) -> Result<Diff> {
    if ours.len() != reference.len() {
        return Err(Error::new(
            "shape_mismatch",
            format!("compared buffers differ in size: ours {} vs reference {}", ours.len(), reference.len()),
        ));
    }
    if ours.is_empty() {
        return Ok(Diff { max_abs: 0.0, mean_abs: 0.0, rel: 0.0 });
    }
    let mut max_abs = 0f64;
    let mut sum_abs = 0f64;
    let mut err2 = 0f64;
    let mut ref2 = 0f64;
    let mut finite = true;
    for (&a, &b) in ours.iter().zip(reference) {
        if !a.is_finite() {
            finite = false;
        }
        let d = (a as f64 - b as f64).abs();
        if d > max_abs || d.is_nan() {
            max_abs = d;
        }
        sum_abs += d;
        err2 += d * d;
        ref2 += (b as f64) * (b as f64);
    }
    let rel = if !finite {
        f64::NAN
    } else if ref2 > 0.0 {
        (err2 / ref2).sqrt()
    } else if err2 == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    Ok(Diff {
        max_abs: max_abs as f32,
        mean_abs: (sum_abs / ours.len() as f64) as f32,
        rel: rel as f32,
    })
}

impl LayerReport {
    pub fn new(name: impl Into<String>, d: Diff, tolerance: f32) -> LayerReport {
        LayerReport {
            name: name.into(),
            max_abs: d.max_abs,
            mean_abs: d.mean_abs,
            rel: d.rel,
            ok: d.rel <= tolerance,
        }
    }

    /// A tensor that could not be compared (missing on one side, wrong size, ...).
    pub fn failed(name: impl Into<String>) -> LayerReport {
        LayerReport { name: name.into(), max_abs: f32::NAN, mean_abs: f32::NAN, rel: f32::NAN, ok: false }
    }
}

/// Relative-error tolerance per layer of a model at `dtype` against an fp32 reference of
/// the original model (quality budget: includes quantization loss).
pub fn tolerance(dtype: &str) -> Result<f32> {
    match dtype {
        "fp32" | "f32" | "fp16" | "f16" => Ok(1e-2),
        "int8" | "i8" => Ok(5e-2),
        "int4" | "i4" => Ok(1.5e-1),
        // ggml k-quants: q6_k is ~int8 quality, q5_k between, q4_k / the q4_k_m mix ~int4
        // (imatrix-free GGUFs of a different export can sit near the int4 budget's edge).
        "q6_k" => Ok(5e-2),
        "q5_k" => Ok(1e-1),
        "q4_k" | "q4_k_m" | "q4_k_s" => Ok(1.5e-1),
        other => Err(Error::new(
            "unsupported_dtype",
            format!("unknown model dtype {other:?} (expected fp32, fp16, int8, int4, q4_k, q4_k_m, q5_k or q6_k)"),
        )),
    }
}

/// Relative-error tolerance of quantized kernels against a fake-quant reference (the same
/// weights dequantized to f32, run with f32 kernels, with or without the kernels' int8
/// activation quantization). Weight quantization loss is excluded. Not 1e-2: with int8
/// activations, last-ulp differences (accumulation order) flip quantized values by a whole
/// step, and the 12-layer Whisper encoder chain amplifies that from 2e-4 at layer 0 to
/// ~1.3e-2 at the output (measured, lapid-cpu int8/int4) — still ~3x below int8 quality loss.
pub const KERNEL_TOLERANCE: f32 = 2e-2;

/// Row-wise argmax agreement of two `[rows, cols]` buffers.
pub fn argmax_rows(ours: &[f32], reference: &[f32], cols: usize) -> Vec<bool> {
    if cols == 0 {
        return Vec::new();
    }
    ours.chunks_exact(cols)
        .zip(reference.chunks_exact(cols))
        .map(|(a, b)| lapid_nn::argmax(a) == lapid_nn::argmax(b))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_is_zero() {
        let a = [1.0f32, -2.0, 3.0];
        let d = diff(&a, &a).unwrap();
        assert_eq!(d, Diff { max_abs: 0.0, mean_abs: 0.0, rel: 0.0 });
    }

    #[test]
    fn rel_error() {
        let b = [3.0f32, 4.0]; // norm 5
        let a = [3.0f32, 4.5];
        let d = diff(&a, &b).unwrap();
        assert!((d.rel - 0.1).abs() < 1e-6);
        assert!((d.max_abs - 0.5).abs() < 1e-6);
        assert!((d.mean_abs - 0.25).abs() < 1e-6);
    }

    #[test]
    fn nan_fails() {
        let d = diff(&[f32::NAN, 1.0], &[1.0, 1.0]).unwrap();
        assert!(!LayerReport::new("x", d, 1.0).ok);
    }

    #[test]
    fn size_mismatch() {
        assert_eq!(diff(&[1.0], &[1.0, 2.0]).unwrap_err().code, "shape_mismatch");
    }

    #[test]
    fn tolerances() {
        assert_eq!(tolerance("int8").unwrap(), 5e-2);
        assert_eq!(tolerance("q4_k_m").unwrap(), 1.5e-1);
        assert_eq!(tolerance("q6_k").unwrap(), 5e-2);
        assert!(tolerance("q3").is_err());
    }

    #[test]
    fn argmax_agreement() {
        let a = [0.0, 1.0, 5.0, 0.0];
        let b = [0.0, 2.0, 0.0, 5.0];
        assert_eq!(argmax_rows(&a, &b, 2), vec![true, false]);
    }
}
