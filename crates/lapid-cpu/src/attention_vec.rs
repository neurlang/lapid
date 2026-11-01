//! Attention for few queries (decoder steps): no packing, K and V rows streamed once.
//!
//! Work units are (batch, head, key split) and handle all `sq` (<= 4) queries of the head
//! together, so each K / V row is loaded once for all of them (grouped-query decode folds
//! the query heads sharing a KV head into `sq`). Each unit walks its key range in blocks:
//! scores are dot products with contiguous K rows, an online softmax keeps (max, sum) per
//! query and `o += p * v` accumulates V rows. Keys are split across threads until there are
//! about two units per thread; partial `(m, l, o)` triples are combined exactly afterwards.

use std::sync::atomic::{AtomicUsize, Ordering};

use lapid_compiler::{AttentionDims, AttentionSpec, View};

use crate::Ctx;
use crate::attention::AttnInputs;
use crate::math::{self, fmadd, max8, sum8};
use crate::par::{self, PAR_MIN_WORK, SyncPtr};
use crate::typed::{Dst, Src};

/// Keys per block (scores kept in L1).
const KB: usize = 128;
/// Minimum keys per split.
const MIN_SPLIT: usize = 32;
/// Max queries per (batch, head) on this path.
const SQ_MAX: usize = 4;

/// Rows of an attention view: (b, h, s) -> (offset, inner stride).
#[inline]
fn row(v: &View, b: usize, h: usize, s: usize) -> (usize, isize) {
    let st = &v.strides;
    let off = v.offset as isize + b as isize * st[0] + h as isize * st[1] + s as isize * st[2];
    (off as usize, st[3])
}

/// Row `(b, h, s)` of `src` as f32: borrowed when contiguous f32, else converted into `tmp`.
#[inline]
fn load_row<'a: 'r, 'r>(src: &Src<'a>, v: &View, b: usize, h: usize, s: usize, dh: usize, tmp: &'r mut [f32]) -> &'r [f32] {
    let (base, st) = row(v, b, h, s);
    if st == 1
        && let Some(f) = src.as_f32()
    {
        return &f[base..base + dh];
    }
    src.load(base, st, &mut tmp[..dh]);
    &tmp[..dh]
}

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0f32; 16];
    let full = n / 16 * 16;
    for (ca, cb) in a[..full].as_chunks::<16>().0.iter().zip(b[..full].as_chunks::<16>().0) {
        for l in 0..16 {
            acc[l] = fmadd(ca[l], cb[l], acc[l]);
        }
    }
    let mut s = 0f32;
    for p in full..n {
        s = fmadd(a[p], b[p], s);
    }
    let mut h = [0f32; 4];
    for q in 0..4 {
        h[q] = (acc[q] + acc[q + 4]) + (acc[q + 8] + acc[q + 12]);
    }
    ((h[0] + h[1]) + (h[2] + h[3])) + s
}

#[inline(always)]
fn axpy(y: &mut [f32], alpha: f32, x: &[f32]) {
    for (o, &v) in y.iter_mut().zip(x) {
        *o = fmadd(alpha, v, *o);
    }
}

/// Use this path for `sq` queries per (batch, head)?
pub(crate) fn applies(d: &AttentionDims) -> bool {
    d.sq <= SQ_MAX
}

struct Scratch {
    q: Vec<f32>,
    s: Vec<f32>,
    tk: Vec<f32>,
    tv: Vec<f32>,
}

pub(crate) fn run(ctx: &Ctx, spec: &AttentionSpec, d: &AttentionDims, ins: &AttnInputs, out: &Dst) {
    let (bsz, heads, sq, sk, dh) = (d.batch, d.heads, d.sq, d.sk, d.d);
    let bh_n = bsz * heads;
    let rows = bh_n * sq;
    let scale = f32::from_bits(d.scale);
    let parallel = rows * sk.max(1) * dh * 2 >= PAR_MIN_WORK && ctx.threads > 1;
    let target = if parallel { ctx.threads * 2 } else { 1 };
    let nsplit = target.div_ceil(bh_n).min(sk / MIN_SPLIT).max(1);
    let split = sk.div_ceil(nsplit).max(1);
    let nsplit = sk.div_ceil(split).max(1);
    // Partial results per (row, split): [m, l, o[dh]].
    let pw = dh + 2;
    let mut parts = vec![0f32; if nsplit > 1 { rows * nsplit * pw } else { 0 }];
    let pptr = SyncPtr(parts.as_mut_ptr());
    let done: Vec<AtomicUsize> = (0..if nsplit > 1 { bh_n } else { 0 }).map(|_| AtomicUsize::new(0)).collect();

    let init = || Scratch { q: vec![0.0; SQ_MAX * dh], s: vec![0.0; SQ_MAX * KB], tk: vec![0.0; dh], tv: vec![0.0; dh] };
    // Store the normalized row `r` of the output.
    let store = |r: usize, o: &mut [f32], l: f32| {
        let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
        for x in o.iter_mut() {
            *x *= inv;
        }
        // SAFETY: row r of the contiguous [B, H, Sq, D] output is written by one task.
        unsafe { out.store(r * dh, 1, o) };
    };
    par::for_each(parallel, bh_n * nsplit, init, |s, u| {
        let (bh, sp) = (u / nsplit, u % nsplit);
        let (b, h) = (bh / heads, bh % heads);
        let Scratch { q, s: sc, tk, tv } = s;
        for i in 0..sq {
            let (base, st) = row(&d.q, b, h, i);
            let qi = &mut q[i * dh..(i + 1) * dh];
            ins.q.load(base, st, qi);
            for x in qi.iter_mut() {
                *x *= scale;
            }
        }
        // Keys visible to query i: [0, kend(i)).
        let kend = |i: usize| if spec.causal { (i as isize + 1 + sk as isize - sq as isize).clamp(0, sk as isize) as usize } else { sk };
        let k_lo = sp * split;
        let k_hi = ((sp + 1) * split).min(kend(sq - 1));
        let mut m = [f32::NEG_INFINITY; SQ_MAX];
        let mut l = [0f32; SQ_MAX];
        let mut o = [[0f32; 256]; SQ_MAX];
        let mut obig: Vec<f32> = Vec::new();
        if dh > 256 {
            obig = vec![0f32; sq * dh];
        }
        let mut k0 = k_lo;
        while k0 < k_hi {
            let kn = KB.min(k_hi - k0);
            for j in 0..kn {
                let kr = load_row(&ins.k, &d.k, b, h, k0 + j, dh, tk);
                for i in 0..sq {
                    sc[i * KB + j] = if k0 + j < kend(i) { dot(&q[i * dh..(i + 1) * dh], kr) } else { f32::NEG_INFINITY };
                }
            }
            for i in 0..sq {
                let srow = &mut sc[i * KB..i * KB + kn];
                if let (Some(mv), Some(ms)) = (&d.mask, &ins.mask) {
                    let (base, st) = row(mv, b, h, i);
                    for (j, x) in srow.iter_mut().enumerate() {
                        *x += ms.get((base as isize + (k0 + j) as isize * st) as usize);
                    }
                }
                let m_new = m[i].max(max8(srow));
                if m_new == f32::NEG_INFINITY {
                    srow.fill(0.0);
                    continue;
                }
                let alpha = math::exp(m[i] - m_new);
                for x in srow.iter_mut() {
                    *x = math::exp_neg(*x - m_new);
                }
                l[i] = l[i] * alpha + sum8(srow);
                m[i] = m_new;
                if alpha != 1.0 {
                    let oi = if dh > 256 { &mut obig[i * dh..(i + 1) * dh] } else { &mut o[i][..dh] };
                    for x in oi.iter_mut() {
                        *x *= alpha;
                    }
                }
            }
            for j in 0..kn {
                if (0..sq).all(|i| sc[i * KB + j] == 0.0) {
                    continue;
                }
                let vr = load_row(&ins.v, &d.v, b, h, k0 + j, dh, tv);
                for i in 0..sq {
                    let p = sc[i * KB + j];
                    if p != 0.0 {
                        let oi = if dh > 256 { &mut obig[i * dh..(i + 1) * dh] } else { &mut o[i][..dh] };
                        axpy(oi, p, vr);
                    }
                }
            }
            k0 += kn;
        }
        for i in 0..sq {
            let r = bh * sq + i;
            let oi = if dh > 256 { &mut obig[i * dh..(i + 1) * dh] } else { &mut o[i][..dh] };
            if nsplit == 1 {
                store(r, oi, l[i]);
            } else {
                let p = pptr;
                // SAFETY: slot (r, sp) of `parts` belongs to this unit only.
                let slot = unsafe { std::slice::from_raw_parts_mut(p.0.add((r * nsplit + sp) * pw), pw) };
                slot[0] = m[i];
                slot[1] = l[i];
                slot[2..].copy_from_slice(oi);
            }
        }
        // The last split of a (batch, head) to finish combines all of them (exactly).
        if nsplit > 1 && done[bh].fetch_add(1, Ordering::AcqRel) == nsplit - 1 {
            for i in 0..sq {
                let r = bh * sq + i;
                let p = pptr;
                // SAFETY: every split of this (batch, head) has published its slots (the
                // AcqRel counter orders their writes before this read); no one writes them now.
                let ps = unsafe { std::slice::from_raw_parts(p.0.add(r * nsplit * pw), nsplit * pw) };
                let mx = ps.chunks(pw).fold(f32::NEG_INFINITY, |a, p| a.max(p[0]));
                let oi = if dh > 256 { &mut obig[i * dh..(i + 1) * dh] } else { &mut o[i][..dh] };
                oi.fill(0.0);
                let mut lsum = 0f32;
                if mx != f32::NEG_INFINITY {
                    for p in ps.chunks(pw) {
                        if p[0] == f32::NEG_INFINITY {
                            continue;
                        }
                        let w = math::exp(p[0] - mx);
                        lsum += p[1] * w;
                        axpy(oi, w, &p[2..]);
                    }
                }
                store(r, oi, lsum);
            }
        }
    });
}
