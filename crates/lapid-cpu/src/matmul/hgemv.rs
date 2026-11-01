//! f16-weight dot products for the GEMV path: R f32 activation rows x 4 f16 weight rows,
//! converted in registers (NEON `fcvtl`), so weights are read exactly once with no f32
//! staging copy.

use half::f16;

use crate::Isa;
use crate::typed::f16_bits_to_f32;

/// `out[r][c] = dot(a[r], b[c])` over `k = b[c].len()` elements.
///
/// # Safety
/// Every `a[r]` and `b[c]` holds at least `k` elements; the kernel's ISA is available
/// (guaranteed by `select`).
pub(crate) type HDot<const R: usize> = unsafe fn([&[f32]; R], [&[f16]; 4], usize) -> [[f32; 4]; R];

#[derive(Clone, Copy)]
pub(crate) struct HKernels {
    pub d1: HDot<1>,
    pub d2: HDot<2>,
}

pub(crate) fn select(isa: &Isa) -> Option<HKernels> {
    #[cfg(target_arch = "aarch64")]
    if isa.neon && isa.fp16 {
        return Some(HKernels { d1: dot_neon::<1>, d2: dot_neon::<2> });
    }
    let _ = isa;
    None
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn dot_neon<const R: usize>(a: [&[f32]; R], b: [&[f16]; 4], k: usize) -> [[f32; 4]; R] {
    use std::arch::aarch64::*;
    // SAFETY: lengths guaranteed by the caller (see `HDot`); loads stay below k.
    unsafe {
        let mut acc = [[[vdupq_n_f32(0.0); 2]; 4]; R];
        let full = k / 8 * 8;
        let mut p = 0;
        while p < full {
            let mut av = [[vdupq_n_f32(0.0); 2]; R];
            for r in 0..R {
                let ap = a[r].as_ptr().add(p);
                av[r] = [vld1q_f32(ap), vld1q_f32(ap.add(4))];
            }
            for c in 0..4 {
                let h = vreinterpretq_f16_u16(vld1q_u16(b[c].as_ptr().add(p) as *const u16));
                let lo = vcvt_f32_f16(vget_low_f16(h));
                let hi = vcvt_high_f32_f16(h);
                for r in 0..R {
                    acc[r][c][0] = vfmaq_f32(acc[r][c][0], av[r][0], lo);
                    acc[r][c][1] = vfmaq_f32(acc[r][c][1], av[r][1], hi);
                }
            }
            p += 8;
        }
        let mut out = [[0f32; 4]; R];
        for r in 0..R {
            for c in 0..4 {
                let mut s = vaddvq_f32(vaddq_f32(acc[r][c][0], acc[r][c][1]));
                for q in full..k {
                    s += a[r][q] * f16_bits_to_f32(b[c][q].to_bits());
                }
                out[r][c] = s;
            }
        }
        out
    }
}
