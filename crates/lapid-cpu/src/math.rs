//! Scalar math kernels written to auto-vectorize (branch-free, no libm calls).
//!
//! * `exp`: Cody–Waite range reduction + degree-7 polynomial, ~1-2 ulp, exact 0 for
//!   x <= -104 (so `exp(-inf) == 0`), +inf above the f32 range, NaN propagates.
//! * `erf`: piecewise cubic Hermite interpolation of a table computed once (in f64, via the
//!   everywhere-convergent series erf(x) = 2/√π·e^{-x²}·Σ 2^n x^{2n+1}/(2n+1)!!) on a 1/64
//!   grid over [0, 4]. Interpolation error ≤ h⁴/384·max|erf⁽⁴⁾| ≈ 1.7e-10; total error is
//!   dominated by f32 rounding (< 3e-7 absolute). For |x| ≥ 4, erf = ±1 (true error < 1.6e-8).

use std::sync::OnceLock;

const LOG2E: f32 = std::f32::consts::LOG2_E;
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;
/// 1.5 * 2^23: adding and subtracting rounds to the nearest integer (|v| < 2^22).
const ROUND_MAGIC: f32 = 12_582_912.0;

/// Fast vectorizable e^x.
#[inline(always)]
pub(crate) fn exp(x: f32) -> f32 {
    let xc = x.clamp(-104.0, 88.722_84);
    let n = (xc * LOG2E + ROUND_MAGIC) - ROUND_MAGIC;
    let r = (xc - n * LN2_HI) - n * LN2_LO;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0)))))));
    // Split 2^n into two factors so that n in [-150, 128] never leaves the normal range.
    let ni = n as i32;
    let n1 = ni >> 1;
    let n2 = ni - n1;
    let s1 = f32::from_bits(((n1 + 127) as u32) << 23);
    let s2 = f32::from_bits(((n2 + 127) as u32) << 23);
    let y = p * s1 * s2;
    if x > 88.722_84 { f32::INFINITY } else { y }
}

/// e^x for x <= 0 (softmax after max subtraction): same reduction and polynomial as `exp`
/// without the overflow / subnormal handling; results below e^-87 flush to exactly 0
/// (so exp_neg(-inf) == 0). Inputs > 0 are not supported (clamped to 0).
#[inline(always)]
pub(crate) fn exp_neg(x: f32) -> f32 {
    let xc = x.clamp(-87.0, 0.0);
    let n = (xc * LOG2E + ROUND_MAGIC) - ROUND_MAGIC;
    let r = (xc - n * LN2_HI) - n * LN2_LO;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0)))))));
    let y = p * f32::from_bits(((n as i32 + 127) as u32) << 23);
    if x < -87.0 { 0.0 } else { y }
}

const ERF_STEPS: usize = 64; // intervals per unit
const ERF_MAX: f32 = 4.0;
const ERF_INTERVALS: usize = 4 * ERF_STEPS;

/// Per-interval cubic coefficients in the local coordinate t ∈ [0, 1).
pub(crate) struct ErfTable {
    c: Vec<[f32; 4]>,
}

fn erf_series(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    let mut n = 0u32;
    loop {
        n += 1;
        term *= 2.0 * x2 / (2 * n + 1) as f64;
        sum += term;
        if term < 1e-18 * sum || n > 1000 {
            break;
        }
    }
    std::f64::consts::FRAC_2_SQRT_PI * (-x2).exp() * sum
}

fn build_erf_table() -> ErfTable {
    let h = 1.0 / ERF_STEPS as f64;
    let node = |i: usize| -> (f64, f64) {
        let x = i as f64 * h;
        let y = erf_series(x);
        let dy = std::f64::consts::FRAC_2_SQRT_PI * (-x * x).exp();
        (y, dy * h)
    };
    let mut c = Vec::with_capacity(ERF_INTERVALS);
    let mut prev = node(0);
    for i in 0..ERF_INTERVALS {
        let next = node(i + 1);
        let (y0, s0) = prev;
        let (y1, s1) = next;
        let c2 = 3.0 * (y1 - y0) - 2.0 * s0 - s1;
        let c3 = 2.0 * (y0 - y1) + s0 + s1;
        c.push([y0 as f32, s0 as f32, c2 as f32, c3 as f32]);
        prev = next;
    }
    ErfTable { c }
}

pub(crate) fn erf_table() -> &'static ErfTable {
    static TABLE: OnceLock<ErfTable> = OnceLock::new();
    TABLE.get_or_init(build_erf_table)
}

impl ErfTable {
    #[inline(always)]
    pub(crate) fn erf(&self, x: f32) -> f32 {
        let ax = x.abs();
        if ax >= ERF_MAX {
            return 1.0f32.copysign(x);
        }
        let u = ax * ERF_STEPS as f32;
        let i = (u as usize).min(ERF_INTERVALS - 1);
        let t = u - i as f32;
        let c = &self.c[i];
        let p = ((c[3] * t + c[2]) * t + c[1]) * t + c[0];
        p.copysign(x)
    }
}

/// Convenience scalar erf (initializes the table on first use).
pub(crate) fn erf(x: f32) -> f32 {
    erf_table().erf(x)
}

/// Round to nearest (ties to even) for |v| < 2^22; vectorizes on every target.
#[inline(always)]
pub(crate) fn round_fast(v: f32) -> f32 {
    (v + ROUND_MAGIC) - ROUND_MAGIC
}

/// FMA where it is a single instruction (aarch64), plain mul+add elsewhere (x86 baseline has
/// no FMA, and `mul_add` would become a libm call).
#[inline(always)]
pub(crate) fn fmadd(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a * b + c
    }
}

/// Sum with 8 independent lanes (vectorizes; better conditioned than a serial sum).
#[inline]
pub(crate) fn sum8(v: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let mut chunks = v.chunks_exact(8);
    for c in &mut chunks {
        for l in 0..8 {
            acc[l] += c[l];
        }
    }
    for (l, &x) in chunks.remainder().iter().enumerate() {
        acc[l] += x;
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))
}

/// Max with 8 lanes; -inf for empty input. NaNs are ignored (f32::max semantics).
#[inline]
pub(crate) fn max8(v: &[f32]) -> f32 {
    let mut acc = [f32::NEG_INFINITY; 8];
    let mut chunks = v.chunks_exact(8);
    for c in &mut chunks {
        for l in 0..8 {
            acc[l] = acc[l].max(c[l]);
        }
    }
    for (l, &x) in chunks.remainder().iter().enumerate() {
        acc[l] = acc[l].max(x);
    }
    acc.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_neg_matches_exp() {
        let mut x = 0.0f32;
        while x > -90.0 {
            let want = (x as f64).exp();
            let got = exp_neg(x) as f64;
            if x >= -87.0 {
                assert!(((got - want) / want).abs() < 4e-7, "exp_neg({x}) = {got}, want {want}");
            } else {
                assert_eq!(got, 0.0);
            }
            x -= 0.0137;
        }
        assert_eq!(exp_neg(f32::NEG_INFINITY), 0.0);
        assert_eq!(exp_neg(0.0), 1.0);
    }

    #[test]
    fn erf_accuracy() {
        let mut max_err = 0f64;
        let mut x = -6.0f64;
        while x <= 6.0 {
            let want = if x < 0.0 { -erf_series(-x) } else { erf_series(x) };
            let got = erf(x as f32) as f64;
            max_err = max_err.max((got - want).abs());
            x += 0.000_37;
        }
        assert!(max_err <= 1e-6, "max erf error {max_err}");
        assert_eq!(erf(0.0), 0.0);
        assert_eq!(erf(10.0), 1.0);
        assert_eq!(erf(-10.0), -1.0);
        assert!(erf(f32::NAN).is_nan());
        // Known values.
        assert!((erf(0.5) - 0.520_499_9).abs() < 3e-7);
        assert!((erf(1.0) - 0.842_700_8).abs() < 3e-7);
    }

    #[test]
    fn exp_accuracy() {
        let mut x = -100.0f32;
        while x < 88.0 {
            let want = (x as f64).exp();
            let got = exp(x) as f64;
            let rel = ((got - want) / want).abs();
            assert!(rel < 4e-7 || want < 1e-37, "exp({x}) = {got} want {want}");
            x += 0.0137;
        }
        assert_eq!(exp(f32::NEG_INFINITY), 0.0);
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(f32::INFINITY), f32::INFINITY);
        assert!(exp(f32::NAN).is_nan());
    }

    #[test]
    fn reductions() {
        let v: Vec<f32> = (0..1003).map(|i| (i % 17) as f32 - 3.0).collect();
        let s: f32 = v.iter().sum();
        assert!((sum8(&v) - s).abs() < 1e-3);
        assert_eq!(max8(&v), 13.0);
        assert_eq!(max8(&[]), f32::NEG_INFINITY);
    }
}
