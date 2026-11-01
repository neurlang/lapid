//! int8 x int8/int4 dot-product tiles: R rows of quantized A x 4 columns of quantized B.
//!
//! Every kernel returns `out[r][c] = sum_g da[r,g] * db[c,g] * dot(qa[r,g,:], qb[c,g,:])`.
//! I4 weights are unpacked to signed values (q - 8) so a single signed dot serves both
//! formats. Kernels: aarch64 `sdot` (dotprod), x86_64 AVX2 (`maddubs` with the sign trick),
//! portable scalar.

use lapid_quant::{Format, GROUP};

use crate::Isa;
use crate::qmat::QMat;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use crate::typed::f16_bits_to_f32;

/// Quantized activations: row r, group g at `q[r * k + g * 32 ..]`, scale `s[r * k / 32 + g]`.
pub(crate) struct QA<'a> {
    pub q: &'a [i8],
    pub s: &'a [f32],
    pub k: usize,
}

/// # Safety
/// Row indices must be < rows of `QA`, column indices < rows of `QMat`, `QA.k == QMat.cols`,
/// and the kernel's ISA must be available (guaranteed by `select`).
pub(crate) type Tile1 = unsafe fn(&QA, [usize; 1], &QMat, [usize; 4]) -> [[f32; 4]; 1];
pub(crate) type Tile4 = unsafe fn(&QA, [usize; 4], &QMat, [usize; 4]) -> [[f32; 4]; 4];

#[derive(Clone, Copy)]
pub(crate) struct QKernels {
    pub t1: Tile1,
    pub t4: Tile4,
}

pub(crate) fn select(isa: &Isa, fmt: Format) -> QKernels {
    let i4 = fmt == Format::Int4;
    #[cfg(target_arch = "aarch64")]
    if isa.dotprod {
        return if i4 {
            QKernels { t1: tile_sdot::<1, true>, t4: tile_sdot::<4, true> }
        } else {
            QKernels { t1: tile_sdot::<1, false>, t4: tile_sdot::<4, false> }
        };
    }
    #[cfg(target_arch = "x86_64")]
    if isa.avx2 && isa.fma {
        return if i4 {
            QKernels { t1: tile_avx2::<1, true>, t4: tile_avx2::<4, true> }
        } else {
            QKernels { t1: tile_avx2::<1, false>, t4: tile_avx2::<4, false> }
        };
    }
    let _ = isa;
    let _ = i4;
    QKernels { t1: tile_scalar::<1>, t4: tile_scalar::<4> }
}

/// Portable kernel.
pub(crate) unsafe fn tile_scalar<const R: usize>(a: &QA, rows: [usize; R], b: &QMat, cols: [usize; 4]) -> [[f32; 4]; R] {
    let kg = a.k / GROUP;
    let mut acc = [[0f32; 4]; R];
    let mut bq = [[0i8; GROUP]; 4];
    for g in 0..kg {
        let mut db = [0f32; 4];
        for c in 0..4 {
            b.group_i8(cols[c], g, &mut bq[c]);
            db[c] = b.scale(cols[c], g);
        }
        for r in 0..R {
            let o = rows[r] * a.k + g * GROUP;
            let qa = &a.q[o..o + GROUP];
            let da = a.s[rows[r] * kg + g];
            for c in 0..4 {
                let mut s = 0i32;
                for t in 0..GROUP {
                    s += qa[t] as i32 * bq[c][t] as i32;
                }
                acc[r][c] += s as f32 * (da * db[c]);
            }
        }
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn tile_sdot<const R: usize, const I4: bool>(a: &QA, rows: [usize; R], b: &QMat, cols: [usize; 4]) -> [[f32; 4]; R] {
    use std::arch::aarch64::*;
    let k = a.k;
    let kg = k / GROUP;
    // SAFETY: indices validated by the caller (see `Tile1`); every load stays inside the
    // quants / scales planes.
    unsafe {
        let zero = vdupq_n_s32(0);
        let m0f = vdupq_n_u8(0x0f);
        let eight = vdupq_n_s8(8);
        let mut acc = [[vdupq_n_f32(0.0); 4]; R];
        let aq = a.q.as_ptr();
        let asc = a.s.as_ptr();
        let qs = b.qs.as_ptr();
        let bsc = b.scales.as_ptr();
        for g in 0..kg {
            let mut av = [[vdupq_n_s8(0); 2]; R];
            let mut da = [0f32; R];
            for r in 0..R {
                let p = aq.add(rows[r] * k + g * GROUP);
                av[r][0] = vld1q_s8(p);
                av[r][1] = vld1q_s8(p.add(16));
                da[r] = *asc.add(rows[r] * kg + g);
            }
            for c in 0..4 {
                let (b0, b1) = if I4 {
                    let v = vld1q_u8(qs.add((cols[c] * k + g * GROUP) / 2));
                    let lo = vreinterpretq_s8_u8(vandq_u8(v, m0f));
                    let hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(v));
                    (vsubq_s8(lo, eight), vsubq_s8(hi, eight))
                } else {
                    let p = qs.add(cols[c] * k + g * GROUP) as *const i8;
                    (vld1q_s8(p), vld1q_s8(p.add(16)))
                };
                let db = f16_bits_to_f32(*bsc.add(cols[c] * kg + g));
                for r in 0..R {
                    let isum = vdotq_s32(vdotq_s32(zero, av[r][0], b0), av[r][1], b1);
                    acc[r][c] = vfmaq_n_f32(acc[r][c], vcvtq_f32_s32(isum), da[r] * db);
                }
            }
        }
        let mut out = [[0f32; 4]; R];
        for r in 0..R {
            for c in 0..4 {
                out[r][c] = vaddvq_f32(acc[r][c]);
            }
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_avx2<const R: usize, const I4: bool>(a: &QA, rows: [usize; R], b: &QMat, cols: [usize; 4]) -> [[f32; 4]; R] {
    use std::arch::x86_64::*;
    let k = a.k;
    let kg = k / GROUP;
    // SAFETY: indices validated by the caller (see `Tile1`); features checked by `select`.
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let m0f = _mm_set1_epi8(0x0f);
        let eight = _mm256_set1_epi8(8);
        let mut acc = [[_mm256_setzero_ps(); 4]; R];
        let aq = a.q.as_ptr();
        let asc = a.s.as_ptr();
        let qs = b.qs.as_ptr();
        let bsc = b.scales.as_ptr();
        for g in 0..kg {
            let mut av = [_mm256_setzero_si256(); R];
            let mut da = [0f32; R];
            for r in 0..R {
                av[r] = _mm256_loadu_si256(aq.add(rows[r] * k + g * GROUP) as *const __m256i);
                da[r] = *asc.add(rows[r] * kg + g);
            }
            for c in 0..4 {
                let bv = if I4 {
                    let v = _mm_loadu_si128(qs.add((cols[c] * k + g * GROUP) / 2) as *const __m128i);
                    let lo = _mm_and_si128(v, m0f);
                    let hi = _mm_and_si128(_mm_srli_epi16::<4>(v), m0f);
                    _mm256_sub_epi8(_mm256_set_m128i(hi, lo), eight)
                } else {
                    _mm256_loadu_si256(qs.add(cols[c] * k + g * GROUP) as *const __m256i)
                };
                let db = f16_bits_to_f32(*bsc.add(cols[c] * kg + g));
                // |b| (unsigned) x sign(a, b) (signed) == a * b, pairwise in i16, no overflow
                // since |q| <= 127.
                let ax = _mm256_sign_epi8(bv, bv);
                for r in 0..R {
                    let sy = _mm256_sign_epi8(av[r], bv);
                    let d32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sy), ones);
                    acc[r][c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d32), _mm256_set1_ps(da[r] * db), acc[r][c]);
                }
            }
        }
        let mut out = [[0f32; 4]; R];
        for r in 0..R {
            for c in 0..4 {
                let v = acc[r][c];
                let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
                let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
                let s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
                out[r][c] = _mm_cvtss_f32(s);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_b(fmt: Format, n: usize, k: usize) -> Vec<u8> {
        let l = lapid_quant::layout(fmt, n, k);
        let mut bytes = vec![0u8; l.total];
        for (i, b) in bytes[..l.qs_bytes].iter_mut().enumerate() {
            *b = ((i * 73 + 11) % 251) as u8;
            if fmt == Format::Int8 && *b == 0x80 {
                *b = 0x81; // keep |q| <= 127
            }
        }
        for g in 0..n * k / GROUP {
            let h = half::f16::from_f32(0.01 + (g % 7) as f32 * 0.003).to_bits().to_le_bytes();
            bytes[l.scales_offset + 2 * g] = h[0];
            bytes[l.scales_offset + 2 * g + 1] = h[1];
        }
        bytes
    }

    #[test]
    fn simd_tiles_match_scalar() {
        let (rows, n, k) = (5, 7, 96);
        let q: Vec<i8> = (0..rows * k).map(|i| ((i * 31 % 255) as i32 - 127) as i8).collect();
        let s: Vec<f32> = (0..rows * k / GROUP).map(|i| 0.02 + i as f32 * 0.001).collect();
        let a = QA { q: &q, s: &s, k };
        let isa = Isa::detect();
        for fmt in [Format::Int8, Format::Int4] {
            let bytes = make_b(fmt, n, k);
            let b = QMat::new(fmt, &bytes, n, k).unwrap();
            let fast = select(&isa, fmt);
            let cols = [0, 3, 6, 6];
            unsafe {
                let want4 = tile_scalar::<4>(&a, [0, 1, 2, 4], &b, cols);
                let got4 = (fast.t4)(&a, [0, 1, 2, 4], &b, cols);
                let want1 = tile_scalar::<1>(&a, [3], &b, cols);
                let got1 = (fast.t1)(&a, [3], &b, cols);
                for c in 0..4 {
                    for r in 0..4 {
                        assert!((want4[r][c] - got4[r][c]).abs() <= 1e-3 * want4[r][c].abs().max(1.0));
                    }
                    assert!((want1[0][c] - got1[0][c]).abs() <= 1e-3 * want1[0][c].abs().max(1.0));
                }
            }
            // Scalar tile vs dequantized reference.
            let mut brow = vec![0f32; k];
            b.dequant_row(3, &mut brow);
            let want: f32 = (0..k).map(|t| q[2 * k + t] as f32 * s[2 * k / GROUP + t / GROUP] * brow[t]).sum();
            let got = unsafe { tile_scalar::<1>(&a, [2], &b, [3, 3, 3, 3]) }[0][0];
            assert!((want - got).abs() <= 1e-3 * want.abs().max(1.0));
        }
    }
}
