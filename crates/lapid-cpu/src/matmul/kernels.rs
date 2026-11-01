//! f32 GEMM microkernels: `MR x NR` output tile over packed panels.
//!
//! Packed A panel: `a[p * MR + r]` (k-major, MR rows). Packed B panel: `b[p * NR + j]`
//! (k-major, NR columns). The kernel overwrites `c[r * ldc + j]` for the full tile.
//! 6x16 suits both NEON (24 accumulators of 4 lanes + 4 B + 2 A registers out of 32) and
//! AVX2+FMA (12 accumulators of 8 lanes + 2 B + 1 broadcast out of 16).

use crate::Isa;

pub(crate) const MR: usize = 6;
pub(crate) const NR: usize = 16;

/// # Safety
/// `a` must be valid for `k * MR` reads, `b` for `k * NR` reads and `c` for writes of the
/// `MR x NR` tile with row stride `ldc >= NR`.
pub(crate) type MicroFn = unsafe fn(k: usize, a: *const f32, b: *const f32, c: *mut f32, ldc: usize);

pub(crate) fn select(isa: &Isa) -> MicroFn {
    #[cfg(target_arch = "aarch64")]
    if isa.neon {
        return micro_neon;
    }
    #[cfg(target_arch = "x86_64")]
    if isa.avx2 && isa.fma {
        return micro_avx2;
    }
    let _ = isa;
    micro_scalar
}

/// Portable kernel (auto-vectorized).
pub(crate) unsafe fn micro_scalar(k: usize, a: *const f32, b: *const f32, c: *mut f32, ldc: usize) {
    let mut acc = [[0f32; NR]; MR];
    // SAFETY: caller guarantees the panel extents.
    let (ap, bp) = unsafe { (std::slice::from_raw_parts(a, k * MR), std::slice::from_raw_parts(b, k * NR)) };
    for (ar, br) in ap.as_chunks::<MR>().0.iter().zip(bp.as_chunks::<NR>().0) {
        for r in 0..MR {
            let av = ar[r];
            for j in 0..NR {
                acc[r][j] = crate::math::fmadd(av, br[j], acc[r][j]);
            }
        }
    }
    for (r, row) in acc.iter().enumerate() {
        // SAFETY: caller guarantees the tile is writable.
        unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), c.add(r * ldc), NR) };
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn micro_neon(k: usize, a: *const f32, b: *const f32, c: *mut f32, ldc: usize) {
    use std::arch::aarch64::*;
    macro_rules! row {
        ($acc:ident, $f:ident, $av:expr, $lane:literal, $b0:expr, $b1:expr, $b2:expr, $b3:expr) => {
            $acc[0] = $f::<$lane>($acc[0], $b0, $av);
            $acc[1] = $f::<$lane>($acc[1], $b1, $av);
            $acc[2] = $f::<$lane>($acc[2], $b2, $av);
            $acc[3] = $f::<$lane>($acc[3], $b3, $av);
        };
    }
    // SAFETY: caller guarantees the panel extents and the writable tile.
    unsafe {
        let z = vdupq_n_f32(0.0);
        let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5) = ([z; 4], [z; 4], [z; 4], [z; 4], [z; 4], [z; 4]);
        let mut ap = a;
        let mut bp = b;
        for _ in 0..k {
            let b0 = vld1q_f32(bp);
            let b1 = vld1q_f32(bp.add(4));
            let b2 = vld1q_f32(bp.add(8));
            let b3 = vld1q_f32(bp.add(12));
            let a0 = vld1q_f32(ap);
            let a1 = vld1_f32(ap.add(4));
            row!(c0, vfmaq_laneq_f32, a0, 0, b0, b1, b2, b3);
            row!(c1, vfmaq_laneq_f32, a0, 1, b0, b1, b2, b3);
            row!(c2, vfmaq_laneq_f32, a0, 2, b0, b1, b2, b3);
            row!(c3, vfmaq_laneq_f32, a0, 3, b0, b1, b2, b3);
            row!(c4, vfmaq_lane_f32, a1, 0, b0, b1, b2, b3);
            row!(c5, vfmaq_lane_f32, a1, 1, b0, b1, b2, b3);
            ap = ap.add(MR);
            bp = bp.add(NR);
        }
        for (r, acc) in [c0, c1, c2, c3, c4, c5].iter().enumerate() {
            let dst = c.add(r * ldc);
            vst1q_f32(dst, acc[0]);
            vst1q_f32(dst.add(4), acc[1]);
            vst1q_f32(dst.add(8), acc[2]);
            vst1q_f32(dst.add(12), acc[3]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_avx2(k: usize, a: *const f32, b: *const f32, c: *mut f32, ldc: usize) {
    use std::arch::x86_64::*;
    // SAFETY: caller guarantees the panel extents and the writable tile; features checked
    // by `select`.
    unsafe {
        let mut acc = [[_mm256_setzero_ps(); 2]; MR];
        let mut ap = a;
        let mut bp = b;
        for _ in 0..k {
            let b0 = _mm256_loadu_ps(bp);
            let b1 = _mm256_loadu_ps(bp.add(8));
            for (r, row) in acc.iter_mut().enumerate() {
                let av = _mm256_set1_ps(*ap.add(r));
                row[0] = _mm256_fmadd_ps(av, b0, row[0]);
                row[1] = _mm256_fmadd_ps(av, b1, row[1]);
            }
            ap = ap.add(MR);
            bp = bp.add(NR);
        }
        for (r, row) in acc.iter().enumerate() {
            _mm256_storeu_ps(c.add(r * ldc), row[0]);
            _mm256_storeu_ps(c.add(r * ldc + 8), row[1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn micro_matches_naive() {
        let k = 37;
        let a: Vec<f32> = (0..k * MR).map(|i| ((i * 7) % 13) as f32 * 0.1 - 0.6).collect();
        let b: Vec<f32> = (0..k * NR).map(|i| ((i * 5) % 11) as f32 * 0.2 - 1.0).collect();
        let ldc = NR + 3;
        let mut want = vec![0f32; MR * ldc];
        for r in 0..MR {
            for j in 0..NR {
                want[r * ldc + j] = (0..k).map(|p| a[p * MR + r] * b[p * NR + j]).sum();
            }
        }
        let isa = crate::Isa::detect();
        for f in [micro_scalar as MicroFn, select(&isa)] {
            let mut c = vec![0f32; MR * ldc];
            unsafe { f(k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), ldc) };
            for r in 0..MR {
                for j in 0..NR {
                    assert!((c[r * ldc + j] - want[r * ldc + j]).abs() < 1e-4);
                }
            }
        }
    }
}
