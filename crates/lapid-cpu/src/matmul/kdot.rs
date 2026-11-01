//! aarch64 `sdot` tiles for ggml K-quant weights (q4_K, q5_K, q6_K) x int8 activations
//! quantized per 32-group (`KA`): R activation rows x 4 weight rows.
//!
//! Per weight row (one at a time, so its blocks stream sequentially) and per 256-block:
//! the eight 32-group integer dots are reduced across lanes with pairwise adds into two
//! vectors of four group sums, then scaled in float by `(d*sc[g]) * da[g]` for four groups
//! per FMA; the q4_K / q5_K min term `dmin*m[g] * bs[g]` is folded into the same lanes. Three
//! independent accumulators per row keep the FMA latency off the critical path. The block
//! scales are decoded with vector ops (ggml's 6-bit scale shuffle), once per block for all R
//! rows.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

use crate::kquant::{KA, KMat, QK};
use crate::typed::f16_bits_to_f32;

#[inline(always)]
unsafe fn f16_at(p: *const u8) -> f32 {
    // SAFETY: caller passes a pointer to two readable bytes.
    f16_bits_to_f32(unsafe { (p as *const u16).read_unaligned() })
}

/// q4_K / q5_K block scales: `d*sc[0..4], d*sc[4..8], dmin*m[0..4], dmin*m[4..8]`.
#[inline(always)]
unsafe fn k4_scales(p: *const u8) -> [float32x4_t; 4] {
    // SAFETY: `p` points at a full q4_K / q5_K block (header 16 bytes).
    unsafe {
        let (d, dmin) = (f16_at(p), f16_at(p.add(2)));
        let s = p.add(4) as *const u32;
        let (u0, u1, u2) = (s.read_unaligned(), s.add(1).read_unaligned(), s.add(2).read_unaligned());
        const K1: u32 = 0x3f3f_3f3f;
        const K2: u32 = 0x0f0f_0f0f;
        const K3: u32 = 0x0303_0303;
        let sc_lo = u0 & K1;
        let sc_hi = (u2 & K2) | (((u0 >> 6) & K3) << 4);
        let m_lo = u1 & K1;
        let m_hi = ((u2 >> 4) & K2) | (((u1 >> 6) & K3) << 4);
        let sc = vreinterpret_u8_u32(vcreate_u32(sc_lo as u64 | (sc_hi as u64) << 32));
        let mn = vreinterpret_u8_u32(vcreate_u32(m_lo as u64 | (m_hi as u64) << 32));
        let sc16 = vmovl_u8(sc);
        let mn16 = vmovl_u8(mn);
        let f = |v: uint16x4_t, x: f32| vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(v)), x);
        [f(vget_low_u16(sc16), d), f(vget_high_u16(sc16), d), f(vget_low_u16(mn16), dmin), f(vget_high_u16(mn16), dmin)]
    }
}

/// Unpacked signed quants of one block: 8 groups x 2 vectors of 16.
#[inline(always)]
pub(crate) unsafe fn unpack<const F: u8>(p: *const u8) -> [[int8x16_t; 2]; 8] {
    // SAFETY: `p` points at a full block of format F.
    unsafe {
        let m0f = vdupq_n_u8(0x0f);
        let mut w = [[vdupq_n_s8(0); 2]; 8];
        if F == 6 {
            let m3 = vdupq_n_u8(3);
            let m32 = vdupq_n_s8(32);
            for n in 0..2 {
                let ql = p.add(n * 64);
                let qh = p.add(128 + n * 32);
                let (l0, l1, l2, l3) = (vld1q_u8(ql), vld1q_u8(ql.add(16)), vld1q_u8(ql.add(32)), vld1q_u8(ql.add(48)));
                let (h0, h1) = (vld1q_u8(qh), vld1q_u8(qh.add(16)));
                let mk = |lo: uint8x16_t, h: uint8x16_t| -> int8x16_t {
                    vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo, vshlq_n_u8::<4>(vandq_u8(h, m3)))), m32)
                };
                w[n * 4] = [mk(vandq_u8(l0, m0f), h0), mk(vandq_u8(l1, m0f), h1)];
                w[n * 4 + 1] = [mk(vandq_u8(l2, m0f), vshrq_n_u8::<2>(h0)), mk(vandq_u8(l3, m0f), vshrq_n_u8::<2>(h1))];
                w[n * 4 + 2] = [mk(vshrq_n_u8::<4>(l0), vshrq_n_u8::<4>(h0)), mk(vshrq_n_u8::<4>(l1), vshrq_n_u8::<4>(h1))];
                w[n * 4 + 3] = [mk(vshrq_n_u8::<4>(l2), vshrq_n_u8::<6>(h0)), mk(vshrq_n_u8::<4>(l3), vshrq_n_u8::<6>(h1))];
            }
        } else {
            let qs = if F == 5 { p.add(48) } else { p.add(16) };
            let (mut h0, mut h1) = if F == 5 { (vld1q_u8(p.add(16)), vld1q_u8(p.add(32))) } else { (vdupq_n_u8(0), vdupq_n_u8(0)) };
            let one = vdupq_n_u8(1);
            for j in 0..4 {
                let (v0, v1) = (vld1q_u8(qs.add(j * 32)), vld1q_u8(qs.add(j * 32 + 16)));
                let (lo, hi) = if F == 5 {
                    // Bit 2j of qh -> +16 for the low nibbles (shift-left-insert keeps the
                    // nibble), bit 2j+1 for the high ones (shift-right-insert of the nibble).
                    let two = vdupq_n_u8(2);
                    let r = (
                        [vsliq_n_u8::<4>(v0, vandq_u8(h0, one)), vsliq_n_u8::<4>(v1, vandq_u8(h1, one))],
                        [vsriq_n_u8::<4>(vshlq_n_u8::<3>(vandq_u8(h0, two)), v0), vsriq_n_u8::<4>(vshlq_n_u8::<3>(vandq_u8(h1, two)), v1)],
                    );
                    h0 = vshrq_n_u8::<2>(h0);
                    h1 = vshrq_n_u8::<2>(h1);
                    r
                } else {
                    ([vandq_u8(v0, m0f), vandq_u8(v1, m0f)], [vshrq_n_u8::<4>(v0), vshrq_n_u8::<4>(v1)])
                };
                w[2 * j] = [vreinterpretq_s8_u8(lo[0]), vreinterpretq_s8_u8(lo[1])];
                w[2 * j + 1] = [vreinterpretq_s8_u8(hi[0]), vreinterpretq_s8_u8(hi[1])];
            }
        }
        w
    }
}

/// `F` = 4 (q4_K), 5 (q5_K) or 6 (q6_K).
///
/// # Safety
/// Same contract as `kquant::KTile1`; requires dotprod.
#[target_feature(enable = "neon,dotprod")]
pub(crate) unsafe fn tile<const R: usize, const F: u8>(a: &KA, rows: [usize; R], b: &KMat, cols: [usize; 4]) -> [[f32; 4]; R] {
    let k = a.k;
    let kg = k / 32;
    let nblk = k / QK;
    let bb = b.fmt.block_bytes();
    let mut out = [[0f32; 4]; R];
    // SAFETY: indices validated by the caller (see `KTile1`); every load stays inside a
    // block of row `cols[c]` or inside an activation row / its scale rows.
    unsafe {
        let zero = vdupq_n_s32(0);
        let aq = a.q.as_ptr();
        let asc = a.s.as_ptr();
        let abs = a.bs.as_ptr();
        for c in 0..4 {
            if c > 0 && cols[c] == cols[c - 1] {
                for o in out.iter_mut() {
                    o[c] = o[c - 1];
                }
                continue;
            }
            let row = b.bytes.as_ptr().add(cols[c] * nblk * bb);
            let mut acc = [[vdupq_n_f32(0.0); 3]; R];
            for blk in 0..nblk {
                let p = row.add(blk * bb);
                let w = unpack::<F>(p);
                if F == 6 {
                    let d = f16_at(p.add(208));
                    // 16 sub-block scales as 4 x i32x4 (sub-blocks 4i .. 4i+4).
                    let sc8 = vld1q_s8(p.add(192) as *const i8);
                    let (sl, sh) = (vmovl_s8(vget_low_s8(sc8)), vmovl_high_s8(sc8));
                    let sc = [vmovl_s16(vget_low_s16(sl)), vmovl_high_s16(sl), vmovl_s16(vget_low_s16(sh)), vmovl_high_s16(sh)];
                    for r in 0..R {
                        let ap = aq.add(rows[r] * k + blk * QK);
                        let so = rows[r] * kg + blk * 8;
                        // 16-element sub-block dots, reduced to 4 x (4 sub-blocks), scaled,
                        // then adjacent sub-blocks summed into the 8 groups.
                        let mut h = [zero; 16];
                        for g in 0..8 {
                            h[2 * g] = vdotq_s32(zero, vld1q_s8(ap.add(g * 32)), w[g][0]);
                            h[2 * g + 1] = vdotq_s32(zero, vld1q_s8(ap.add(g * 32 + 16)), w[g][1]);
                        }
                        let q4: [int32x4_t; 4] = std::array::from_fn(|i| {
                            let v = vpaddq_s32(vpaddq_s32(h[4 * i], h[4 * i + 1]), vpaddq_s32(h[4 * i + 2], h[4 * i + 3]));
                            vmulq_s32(v, sc[i])
                        });
                        let s_lo = vpaddq_s32(q4[0], q4[1]);
                        let s_hi = vpaddq_s32(q4[2], q4[3]);
                        let da0 = vmulq_n_f32(vld1q_f32(asc.add(so)), d);
                        let da1 = vmulq_n_f32(vld1q_f32(asc.add(so + 4)), d);
                        acc[r][0] = vfmaq_f32(acc[r][0], vcvtq_f32_s32(s_lo), da0);
                        acc[r][1] = vfmaq_f32(acc[r][1], vcvtq_f32_s32(s_hi), da1);
                    }
                } else {
                    let [s0, s1, m0, m1] = k4_scales(p);
                    for r in 0..R {
                        let ap = aq.add(rows[r] * k + blk * QK);
                        let so = rows[r] * kg + blk * 8;
                        let mut dg = [zero; 8];
                        for (g, dgg) in dg.iter_mut().enumerate() {
                            *dgg = vdotq_s32(vdotq_s32(zero, vld1q_s8(ap.add(g * 32)), w[g][0]), vld1q_s8(ap.add(g * 32 + 16)), w[g][1]);
                        }
                        let s_lo = vpaddq_s32(vpaddq_s32(dg[0], dg[1]), vpaddq_s32(dg[2], dg[3]));
                        let s_hi = vpaddq_s32(vpaddq_s32(dg[4], dg[5]), vpaddq_s32(dg[6], dg[7]));
                        let da0 = vld1q_f32(asc.add(so));
                        let da1 = vld1q_f32(asc.add(so + 4));
                        acc[r][0] = vfmaq_f32(acc[r][0], vcvtq_f32_s32(s_lo), vmulq_f32(s0, da0));
                        acc[r][1] = vfmaq_f32(acc[r][1], vcvtq_f32_s32(s_hi), vmulq_f32(s1, da1));
                        let mm = vfmaq_f32(vmulq_f32(m0, vld1q_f32(abs.add(so))), m1, vld1q_f32(abs.add(so + 4)));
                        acc[r][2] = vaddq_f32(acc[r][2], mm);
                    }
                }
            }
            for r in 0..R {
                out[r][c] = vaddvq_f32(vsubq_f32(vaddq_f32(acc[r][0], acc[r][1]), acc[r][2]));
            }
        }
    }
    out
}
