//! Quantized-B GEMM for larger m (encoder, prefill) with bf16 `bfmmla` (aarch64 BF16).
//!
//! Activations are split into hi + lo bf16 halves (no int8 activation blocks); int8 / int4
//! weights are exact (their integer quants are bf16 values) with the f16 group scale applied
//! per 32-group in f32, K-quant weights are dequantized and split into hi + lo bf16 too, so
//! results match f32 math on the dequantized weights to ~1e-5 (see `split_mode`). f32 accumulation:
//! `bfmmla` multiplies a 2x4 bf16 block by a 4x2 block into a 2x2 f32 tile, 4x the MACs of
//! an f32 `fmla` per instruction.
//!
//! Layout (both operands, per pair of rows / columns, k in chunks of 4): pair `p`, chunk `c`
//! → 8 bf16 at `(p * kc + c) * 8` = `[row0[4c..4c+4], row1[4c..4c+4]]`, k zero-padded to a
//! multiple of 4, rows to a multiple of 4 and columns to a multiple of 8. B scales: column
//! pair `p`, group `g` → `[s0, s1, s0, s1]` at `(p * kg + g) * 4` (1.0 for K-quants).

use lapid_compiler::MatMulDims;

use super::{AOp, Epilogue, Workspace};
use crate::Ctx;
use crate::par::{self, PAR_MIN_WORK, SyncPtr, block_size, grow};
use crate::qmat::QMat;
use lapid_quant::GROUP;

/// Rows per task row block (multiple of 4).
const ROW_BLOCK: usize = 64;
/// k chunks (of 4) per k block.
const KC_BLOCK: usize = 128;
/// m from which this path is used.
pub(crate) const MIN_M: usize = 16;

pub(crate) fn available(ctx: &Ctx) -> bool {
    cfg!(target_arch = "aarch64") && ctx.isa.bf16 && ctx.isa.neon
}

/// f32 → bf16 bits, round to nearest even (NaN stays NaN).
#[inline(always)]
fn bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

/// Operand splitting into hi + lo bf16 halves: `LAPID_CPU_BF16_SPLIT=0` none (plain bf16
/// rounding, fastest), `a` activations only (int8/int4 exact, K-quant weights rounded to
/// bf16), default both (K-quant weights too; ~f32 exact, ~1.5x the time on K-quants).
fn split_mode() -> (bool, bool) {
    static MODE: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("LAPID_CPU_BF16_SPLIT").as_deref().map(str::trim) {
        Ok("0") => (false, false),
        Ok("a") => (true, false),
        _ => (true, true),
    })
}

/// Residual of `r` after bf16 rounding (in place: `r[i] -= bf16(r[i])`).
fn residual(r: &mut [f32]) {
    for x in r {
        *x -= half::bf16::from_bits(bf16(*x)).to_f32();
    }
}

/// Pack two f32 rows (`None` = zero row) of length `k` into pair layout `dst` (`kc * 8`).
fn pack_pair(r0: Option<&[f32]>, r1: Option<&[f32]>, k: usize, dst: &mut [u16]) {
    for (c, chunk) in dst.chunks_exact_mut(8).enumerate() {
        for (half, row) in [r0, r1].into_iter().enumerate() {
            for j in 0..4 {
                let p = 4 * c + j;
                chunk[half * 4 + j] = match row {
                    Some(r) if p < k => bf16(r[p]),
                    _ => 0,
                };
            }
        }
    }
}

pub(crate) fn matmul(ctx: &Ctx, ws: &mut Workspace, a: &AOp, b: &QMat, d: &MatMulDims, epi: &Epilogue) {
    #[cfg(target_arch = "aarch64")]
    imp(ctx, ws, a, b, d, epi);
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (ctx, ws, a, b, d, epi);
        unreachable!("qbf16 requires aarch64 bf16");
    }
}

#[cfg(target_arch = "aarch64")]
fn imp(ctx: &Ctx, ws: &mut Workspace, a: &AOp, b: &QMat, d: &MatMulDims, epi: &Epilogue) {
    let (batch, m, n, k) = (d.batch, d.m, d.n, d.k);
    let rows = batch * m;
    let kc = k.div_ceil(4);
    let parallel = rows * n * k >= PAR_MIN_WORK && ctx.threads > 1;
    let rpairs = rows.div_ceil(4) * 2;
    let cpairs = n.div_ceil(8) * 4;
    let pair = kc * 8;

    let kg = k / GROUP;
    let (split, split_k) = split_mode();
    // A pair p: hi at `p * apair`, lo (if split) right after it.
    let apair = if split { 2 * pair } else { pair };
    // K-quant weights are not bf16 values: B pair p gets a lo (residual) half too, so the
    // products are a_hi*b_hi + a_lo*b_hi + a_hi*b_lo (~f32 exact).
    let split_b = split_k && b.is_k();
    let bpair = if split_b { 2 * pair } else { pair };
    let Workspace { ha, hb, psb, .. } = ws;
    let ha = grow(ha, rpairs * apair);
    let hb = grow(hb, cpairs * bpair);
    let psb = grow(psb, cpairs * kg * 4);

    // 1. A rows → bf16 pairs.
    {
        let hp = SyncPtr(ha.as_mut_ptr());
        par::for_each(parallel, rpairs, || (vec![0f32; k], vec![0f32; k]), |(t0, t1), rp| {
            // Capture the whole `SyncPtr` (Sync), not its raw-pointer field.
            #[allow(clippy::redundant_locals)]
            let hp = hp;
            let load = |row: usize, t: &mut Vec<f32>| -> bool {
                if row >= rows {
                    return false;
                }
                a.src.load_contig(a.row(row / m, row % m), t);
                true
            };
            let (h0, h1) = (load(rp * 2, t0), load(rp * 2 + 1, t1));
            // SAFETY: row pair rp owns its `pair` elements.
            let dst = unsafe { std::slice::from_raw_parts_mut(hp.0.add(rp * apair), apair) };
            let (hi, lo) = dst.split_at_mut(pair);
            pack_pair(h0.then_some(&t0[..]), h1.then_some(&t1[..]), k, hi);
            if split {
                residual(t0);
                residual(t1);
                pack_pair(h0.then_some(&t0[..]), h1.then_some(&t1[..]), k, lo);
            }
        });
    }
    // 2. B (weight rows = output columns) → bf16 pairs: integer quants + group scales, or
    //    dequantized values for K-quants.
    {
        let (hp, sp) = (SyncPtr(hb.as_mut_ptr()), SyncPtr(psb.as_mut_ptr()));
        par::for_each(parallel, cpairs, || (vec![0f32; k], vec![0f32; k], [0i8; GROUP]), |(t0, t1, q), cp| {
            let (hp, sp) = (hp, sp);
            // SAFETY: column pair cp owns its `pair` elements and `kg * 4` scales.
            let (dst, ds) = unsafe {
                (
                    std::slice::from_raw_parts_mut(hp.0.add(cp * bpair), bpair),
                    std::slice::from_raw_parts_mut(sp.0.add(cp * kg * 4), kg * 4),
                )
            };
            let cols = [cp * 2, cp * 2 + 1];
            let mut sc = vec![[1f32; 2]; kg];
            for (ci, t) in [&mut *t0, &mut *t1].into_iter().enumerate() {
                let col = cols[ci];
                if col >= n {
                    continue;
                }
                if b.is_k() {
                    b.dequant_row(col, t);
                    continue;
                }
                for g in 0..kg {
                    b.group_i8(col, g, q);
                    sc[g][ci] = b.scale(col, g);
                    for (x, &v) in t[g * GROUP..(g + 1) * GROUP].iter_mut().zip(q.iter()) {
                        *x = v as f32;
                    }
                }
            }
            for (g, s) in sc.iter().enumerate() {
                ds[g * 4..g * 4 + 4].copy_from_slice(&[s[0], s[1], s[0], s[1]]);
            }
            let (hi, lo) = dst.split_at_mut(pair);
            pack_pair((cols[0] < n).then_some(&t0[..]), (cols[1] < n).then_some(&t1[..]), k, hi);
            if split_b {
                residual(t0);
                residual(t1);
                pack_pair((cols[0] < n).then_some(&t0[..]), (cols[1] < n).then_some(&t1[..]), k, lo);
            }
        });
    }
    let (ha, hb, psb): (&[u16], &[u16], &[f32]) = (ha, hb, psb);

    // 3. Tiles.
    let rb = ROW_BLOCK.min(rows.div_ceil(4) * 4);
    let row_blocks = rows.div_ceil(rb);
    let target = if parallel { ctx.threads * 4 } else { 1 };
    let nb = block_size(n, target.div_ceil(row_blocks), 8, 32, 256);
    let col_blocks = n.div_ceil(nb);
    par::for_each(parallel, row_blocks * col_blocks, || (vec![0f32; rb * nb], epi.regs()), |(cbuf, regs), t| {
        let (rbi, cbi) = (t / col_blocks, t % col_blocks);
        let (r0, r1) = (rbi * rb, ((rbi + 1) * rb).min(rows));
        let (j0, j1) = (cbi * nb, ((cbi + 1) * nb).min(n));
        cbuf.fill(0.0);
        // k blocks keep the A rows and B columns of a block in L2.
        let mut c0 = 0;
        while c0 < kc {
            let cn = KC_BLOCK.min(kc - c0);
            let mut c = j0;
            while c < j1 {
                let mut r = r0;
                while r < r1 {
                    // SAFETY: packed buffers are padded to whole 4x8 tiles and hold `kc`
                    // chunks per pair; bf16 checked by `available`.
                    let tile = unsafe {
                        kernel_4x8(
                            &ha[(r / 2) * apair + c0 * 8..],
                            &hb[(c / 2) * bpair + c0 * 8..],
                            &psb[((c / 2) * kg + c0 / 8) * 4..],
                            cn / 8,
                            (apair, bpair, kg * 4),
                            (split.then_some(pair), split_b.then_some(pair)),
                        )
                    };
                    let valid_c = (j1 - c).min(8);
                    for (q, row) in tile.iter().enumerate() {
                        if r + q < r1 {
                            let o = (r + q - r0) * nb + (c - j0);
                            for (x, &v) in cbuf[o..o + valid_c].iter_mut().zip(row) {
                                *x += v;
                            }
                        }
                    }
                    r += 4;
                }
                c += 8;
            }
            c0 += cn;
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
unsafe fn bfmmla(
    acc: std::arch::aarch64::float32x4_t,
    a: std::arch::aarch64::uint16x8_t,
    b: std::arch::aarch64::uint16x8_t,
) -> std::arch::aarch64::float32x4_t {
    let mut acc = acc;
    // SAFETY: register-only instruction; bf16 enabled on the calling function.
    unsafe {
        std::arch::asm!(
            "bfmmla {0:v}.4s, {1:v}.8h, {2:v}.8h",
            inout(vreg) acc,
            in(vreg) a,
            in(vreg) b,
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    acc
}

/// 4 rows (2 pairs from `pa`) x 8 columns (4 pairs from `pb`, scales `sb`) over `groups`
/// 32-groups (8 chunks each); A pairs are `pas`, B pairs `ps` elements / `ss` scales apart.
/// `lo`: offsets of each A / B pair's low (residual) half (`a_lo * b_hi`, `a_hi * b_lo`).
///
/// # Safety
/// `pa` holds 2 and `pb` / `sb` 4 pairs of at least `groups` groups; the CPU supports BF16.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,bf16")]
unsafe fn kernel_4x8(
    pa: &[u16],
    pb: &[u16],
    sb: &[f32],
    groups: usize,
    (pas, ps, ss): (usize, usize, usize),
    (lo, blo): (Option<usize>, Option<usize>),
) -> [[f32; 8]; 4] {
    use std::arch::aarch64::*;
    // SAFETY: bounds per the function contract.
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); 4]; 2];
        let (a, b, s) = (pa.as_ptr(), pb.as_ptr(), sb.as_ptr());
        for g in 0..groups {
            let mut t = [[vdupq_n_f32(0.0); 4]; 2];
            for c in g * 8..g * 8 + 8 {
                let av = [vld1q_u16(a.add(c * 8)), vld1q_u16(a.add(pas + c * 8))];
                let bv = [vld1q_u16(b.add(c * 8)), vld1q_u16(b.add(ps + c * 8)), vld1q_u16(b.add(2 * ps + c * 8)), vld1q_u16(b.add(3 * ps + c * 8))];
                for pr in 0..2 {
                    for pc in 0..4 {
                        t[pr][pc] = bfmmla(t[pr][pc], av[pr], bv[pc]);
                    }
                }
                if let Some(lo) = lo {
                    let al = [vld1q_u16(a.add(lo + c * 8)), vld1q_u16(a.add(pas + lo + c * 8))];
                    for pr in 0..2 {
                        for pc in 0..4 {
                            t[pr][pc] = bfmmla(t[pr][pc], al[pr], bv[pc]);
                        }
                    }
                }
                if let Some(lo) = blo {
                    let bl = [
                        vld1q_u16(b.add(lo + c * 8)),
                        vld1q_u16(b.add(ps + lo + c * 8)),
                        vld1q_u16(b.add(2 * ps + lo + c * 8)),
                        vld1q_u16(b.add(3 * ps + lo + c * 8)),
                    ];
                    for pr in 0..2 {
                        for pc in 0..4 {
                            t[pr][pc] = bfmmla(t[pr][pc], av[pr], bl[pc]);
                        }
                    }
                }
            }
            for pc in 0..4 {
                let sv = vld1q_f32(s.add(pc * ss + g * 4));
                for pr in 0..2 {
                    acc[pr][pc] = vfmaq_f32(acc[pr][pc], t[pr][pc], sv);
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
