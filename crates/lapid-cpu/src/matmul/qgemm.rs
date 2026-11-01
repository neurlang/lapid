//! Packed int8 GEMM with i8mm (`smmla`) for quantized B and larger m (encoder, cross K/V).
//!
//! `smmla` multiplies a 2x8 int8 block by an 8x2 block into a 2x2 int32 tile, so both
//! operands are re-laid out per 32-group as pairs of rows / columns interleaved in 8-byte
//! chunks:
//! * A (quantized per launch): row pair `rp`, group `g` → 64 bytes at `(rp*kg + g)*64`,
//!   chunk `ch` = [row0[8ch..8ch+8], row1[8ch..8ch+8]]; scales `[da0, da0, da1, da1]`.
//! * B (int8 / int4 weights): column pair `cp` likewise; scales `[db0, db1, db0, db1]`
//!   (f16 → f32 once). Packing B costs O(n·k), negligible against O(m·n·k) for m ≥ 64.
//!
//! The 4x8 microkernel keeps 8 f32 2x2 tiles; per group it issues 32 `smmla` and one
//! `cvt + fma` per tile with the scale outer product `sa * sb`.

use lapid_compiler::MatMulDims;
use lapid_quant::GROUP;

use super::quant::quantize_row;
use super::{AOp, Epilogue, Workspace};
use crate::Ctx;
use crate::par::{self, PAR_MIN_WORK, SyncPtr, block_size, grow};
use crate::qmat::QMat;

/// Rows per task row block (multiple of 4).
const ROW_BLOCK: usize = 64;
/// m from which the packed path is used.
pub(crate) const MIN_M: usize = 16;

pub(crate) fn available(ctx: &Ctx) -> bool {
    cfg!(target_arch = "aarch64") && ctx.isa.i8mm && ctx.isa.dotprod
}

pub(crate) fn matmul(ctx: &Ctx, ws: &mut Workspace, a: &AOp, b: &QMat, d: &MatMulDims, epi: &Epilogue) {
    #[cfg(target_arch = "aarch64")]
    imp(ctx, ws, a, b, d, epi);
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (ctx, ws, a, b, d, epi);
        unreachable!("qgemm requires aarch64 i8mm");
    }
}

#[cfg(target_arch = "aarch64")]
fn imp(ctx: &Ctx, ws: &mut Workspace, a: &AOp, b: &QMat, d: &MatMulDims, epi: &Epilogue) {
    let (batch, m, n, k) = (d.batch, d.m, d.n, d.k);
    let rows = batch * m;
    let kg = k / GROUP;
    let parallel = rows * n * k >= PAR_MIN_WORK && ctx.threads > 1;
    let rpairs = rows.div_ceil(4) * 2; // row pairs, padded to a multiple of 4 rows
    let cpairs = n.div_ceil(8) * 4; // column pairs, padded to a multiple of 8 columns

    let Workspace { pa, pb, psa, psb, .. } = ws;
    let pa = grow(pa, rpairs * kg * 64);
    let psa = grow(psa, rpairs * kg * 4);
    let pb = grow(pb, cpairs * kg * 64);
    let psb = grow(psb, cpairs * kg * 4);

    // 1. Quantize + pack A, one row pair per unit.
    {
        let (qp, sp) = (SyncPtr(pa.as_mut_ptr()), SyncPtr(psa.as_mut_ptr()));
        let init = || (vec![0f32; k], vec![0i8; 2 * k], vec![0f32; 2 * kg]);
        par::for_each(parallel, rpairs, init, |(tmp, q, s), rp| {
            for r in 0..2 {
                let row = rp * 2 + r;
                let (qr, sr) = (&mut q[r * k..(r + 1) * k], &mut s[r * kg..(r + 1) * kg]);
                if row < rows {
                    let (bt, i) = (row / m, row % m);
                    a.src.load_contig(a.row(bt, i), tmp);
                    quantize_row(tmp, qr, sr);
                } else {
                    qr.fill(0);
                    sr.fill(0.0);
                }
            }
            let (qp, sp) = (qp, sp);
            // SAFETY: row pair rp owns bytes / scales of its kg groups.
            let (dq, ds) = unsafe {
                (
                    std::slice::from_raw_parts_mut(qp.0.add(rp * kg * 64), kg * 64),
                    std::slice::from_raw_parts_mut(sp.0.add(rp * kg * 4), kg * 4),
                )
            };
            for g in 0..kg {
                for ch in 0..4 {
                    for r in 0..2 {
                        let src = &q[r * k + g * GROUP + ch * 8..][..8];
                        dq[g * 64 + ch * 16 + r * 8..][..8].copy_from_slice(src);
                    }
                }
                ds[g * 4..g * 4 + 4].copy_from_slice(&[s[g], s[g], s[kg + g], s[kg + g]]);
            }
        });
    }

    // 2. Pack B, one column pair per unit.
    {
        let (qp, sp) = (SyncPtr(pb.as_mut_ptr()), SyncPtr(psb.as_mut_ptr()));
        par::for_each(parallel, cpairs, || [0i8; GROUP], |gq, cp| {
            let (qp, sp) = (qp, sp);
            // SAFETY: column pair cp owns bytes / scales of its kg groups.
            let (dq, ds) = unsafe {
                (
                    std::slice::from_raw_parts_mut(qp.0.add(cp * kg * 64), kg * 64),
                    std::slice::from_raw_parts_mut(sp.0.add(cp * kg * 4), kg * 4),
                )
            };
            for g in 0..kg {
                let mut sc = [0f32; 2];
                for c in 0..2 {
                    let col = cp * 2 + c;
                    if col < n {
                        b.group_i8(col, g, gq);
                        sc[c] = b.scale(col, g);
                    } else {
                        gq.fill(0);
                    }
                    for ch in 0..4 {
                        dq[g * 64 + ch * 16 + c * 8..][..8].copy_from_slice(&gq[ch * 8..ch * 8 + 8]);
                    }
                }
                ds[g * 4..g * 4 + 4].copy_from_slice(&[sc[0], sc[1], sc[0], sc[1]]);
            }
        });
    }
    let (pa, psa, pb, psb): (&[i8], &[f32], &[i8], &[f32]) = (pa, psa, pb, psb);

    // 3. Tiles: (row block, column block) tasks; 4x8 microkernel inside.
    let rb = ROW_BLOCK.min(rows.div_ceil(4) * 4);
    let row_blocks = rows.div_ceil(rb);
    let target = if parallel { ctx.threads * 4 } else { 1 };
    let nb = block_size(n, target.div_ceil(row_blocks), 8, 32, 256);
    let col_blocks = n.div_ceil(nb);

    par::for_each(parallel, row_blocks * col_blocks, || (vec![0f32; rb * nb], epi.regs()), |(cbuf, regs), t| {
        let (rbi, cbi) = (t / col_blocks, t % col_blocks);
        let (r0, r1) = (rbi * rb, ((rbi + 1) * rb).min(rows));
        let (j0, j1) = (cbi * nb, ((cbi + 1) * nb).min(n));
        let mut c = j0;
        while c < j1 {
            let cp = c / 2;
            let mut r = r0;
            while r < r1 {
                let rp = r / 2;
                // SAFETY: packed buffers are padded to whole 4x8 tiles; i8mm checked by
                // `available`.
                let tile = unsafe { kernel_4x8(&pa[rp * kg * 64..], &psa[rp * kg * 4..], &pb[cp * kg * 64..], &psb[cp * kg * 4..], kg) };
                let valid_c = (j1 - c).min(8);
                for (q, row) in tile.iter().enumerate() {
                    if r + q < r1 {
                        let o = (r + q - r0) * nb + (c - j0);
                        cbuf[o..o + valid_c].copy_from_slice(&row[..valid_c]);
                    }
                }
                r += 4;
            }
            c += 8;
        }
        for r in r0..r1 {
            let o = (r - r0) * nb;
            // SAFETY: this task exclusively owns rows r0..r1, columns j0..j1.
            unsafe { epi.apply(regs, r / m, r % m, j0, &cbuf[o..o + (j1 - j0)]) };
        }
    });
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn smmla(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut acc = acc;
    // SAFETY: register-only instruction; i8mm enabled on the calling function.
    unsafe {
        std::arch::asm!(
            "smmla {0:v}.4s, {1:v}.16b, {2:v}.16b",
            inout(vreg) acc,
            in(vreg) a,
            in(vreg) b,
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    acc
}

/// 4 rows (2 packed pairs starting at `qa`) x 8 columns (4 packed pairs at `qb`).
///
/// # Safety
/// `qa`/`sa` hold 2 row pairs and `qb`/`sb` 4 column pairs of `kg` groups (pair stride
/// `kg * 64` bytes / `kg * 4` scales); the CPU supports i8mm.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod,i8mm")]
unsafe fn kernel_4x8(qa: &[i8], sa: &[f32], qb: &[i8], sb: &[f32], kg: usize) -> [[f32; 8]; 4] {
    use std::arch::aarch64::*;
    // SAFETY: bounds per the function contract.
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); 4]; 2];
        let (pa, pb) = (qa.as_ptr(), qb.as_ptr());
        let (psa, psb) = (sa.as_ptr(), sb.as_ptr());
        let pstride = kg * 64;
        let sstride = kg * 4;
        for g in 0..kg {
            let mut av = [[vdupq_n_s8(0); 4]; 2];
            let mut sav = [vdupq_n_f32(0.0); 2];
            for pr in 0..2 {
                let p = pa.add(pr * pstride + g * 64);
                av[pr] = [vld1q_s8(p), vld1q_s8(p.add(16)), vld1q_s8(p.add(32)), vld1q_s8(p.add(48))];
                sav[pr] = vld1q_f32(psa.add(pr * sstride + g * 4));
            }
            #[allow(clippy::needless_range_loop)]
            for pc in 0..4 {
                let p = pb.add(pc * pstride + g * 64);
                let bv = [vld1q_s8(p), vld1q_s8(p.add(16)), vld1q_s8(p.add(32)), vld1q_s8(p.add(48))];
                let sbv = vld1q_f32(psb.add(pc * sstride + g * 4));
                for pr in 0..2 {
                    let mut t = vdupq_n_s32(0);
                    for ch in 0..4 {
                        t = smmla(t, av[pr][ch], bv[ch]);
                    }
                    acc[pr][pc] = vfmaq_f32(acc[pr][pc], vcvtq_f32_s32(t), vmulq_f32(sav[pr], sbv));
                }
            }
        }
        let mut out = [[0f32; 8]; 4];
        for pr in 0..2 {
            for pc in 0..4 {
                let mut v = [0f32; 4];
                vst1q_f32(v.as_mut_ptr(), acc[pr][pc]);
                out[2 * pr][2 * pc] = v[0];
                out[2 * pr][2 * pc + 1] = v[1];
                out[2 * pr + 1][2 * pc] = v[2];
                out[2 * pr + 1][2 * pc + 1] = v[3];
            }
        }
        out
    }
}
