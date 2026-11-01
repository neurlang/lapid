//! Attention for few queries (decoder steps): no packing, K and V rows streamed once.
//!
//! Work units are (batch, head, query, key split). Each unit walks its key range in blocks:
//! scores are dot products with contiguous K rows, an online softmax keeps (max, sum), and
//! `o += p * v` accumulates V rows. With long caches (cross attention over 1500 frames) the
//! keys are split across threads so that more than `heads` cores stream K/V; partial
//! `(m, l, o)` triples are combined exactly afterwards.

use lapid_compiler::{AttentionDims, AttentionSpec, View};

use crate::Ctx;
use crate::attention::AttnInputs;
use crate::math::{self, fmadd, max8, sum8};
use crate::par::{self, PAR_MIN_WORK};
use crate::typed::{Dst, Src};

/// Keys per block (scores kept in L1).
const KB: usize = 128;
/// Minimum keys per split.
const MIN_SPLIT: usize = 256;

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
    d.sq <= 4
}

struct Scratch {
    q: Vec<f32>,
    s: Vec<f32>,
    tk: Vec<f32>,
    tv: Vec<f32>,
}

/// Result of one unit: running max, running sum, unnormalized output.
struct Part {
    m: f32,
    l: f32,
    o: Vec<f32>,
}

pub(crate) fn run(ctx: &Ctx, spec: &AttentionSpec, d: &AttentionDims, ins: &AttnInputs, out: &Dst) {
    let (bsz, heads, sq, sk, dh) = (d.batch, d.heads, d.sq, d.sk, d.d);
    let rows = bsz * heads * sq;
    let scale = f32::from_bits(d.scale);
    let parallel = rows * sk.max(1) * dh * 2 >= PAR_MIN_WORK && ctx.threads > 1;
    let target = if parallel { ctx.threads * 2 } else { 1 };
    let nsplit = target.div_ceil(rows).min(sk / MIN_SPLIT).max(1);
    let split = sk.div_ceil(nsplit).max(1);
    let nsplit = sk.div_ceil(split).max(1);

    let init = || Scratch { q: vec![0.0; dh], s: vec![0.0; KB], tk: vec![0.0; dh], tv: vec![0.0; dh] };
    let unit = |s: &mut Scratch, u: usize| -> Part {
        let r = u / nsplit;
        let sp = u % nsplit;
        let (bh, i) = (r / sq, r % sq);
        let (b, h) = (bh / heads, bh % heads);
        let Scratch { q, s: sc, tk, tv } = s;
        {
            let (base, st) = row(&d.q, b, h, i);
            ins.q.load(base, st, &mut q[..dh]);
            for x in q.iter_mut() {
                *x *= scale;
            }
        }
        // Keys visible to query i.
        let kend = if spec.causal { (i as isize + 1 + sk as isize - sq as isize).clamp(0, sk as isize) as usize } else { sk };
        let k_lo = sp * split;
        let k_hi = ((sp + 1) * split).min(kend);
        let mut m = f32::NEG_INFINITY;
        let mut l = 0f32;
        let mut o = vec![0f32; dh];
        let mut k0 = k_lo;
        while k0 < k_hi {
            let kn = KB.min(k_hi - k0);
            let srow = &mut sc[..kn];
            for (j, x) in srow.iter_mut().enumerate() {
                let kr = load_row(&ins.k, &d.k, b, h, k0 + j, dh, tk);
                *x = dot(q, kr);
            }
            if let (Some(mv), Some(ms)) = (&d.mask, &ins.mask) {
                let (base, st) = row(mv, b, h, i);
                for (j, x) in srow.iter_mut().enumerate() {
                    *x += ms.get((base as isize + (k0 + j) as isize * st) as usize);
                }
            }
            let m_new = m.max(max8(srow));
            if m_new == f32::NEG_INFINITY {
                k0 += kn;
                continue;
            }
            let alpha = math::exp(m - m_new);
            for x in srow.iter_mut() {
                *x = math::exp_neg(*x - m_new);
            }
            l = l * alpha + sum8(srow);
            m = m_new;
            if alpha != 1.0 {
                for x in o.iter_mut() {
                    *x *= alpha;
                }
            }
            for (j, &p) in srow.iter().enumerate() {
                if p != 0.0 {
                    let vr = load_row(&ins.v, &d.v, b, h, k0 + j, dh, tv);
                    axpy(&mut o, p, vr);
                }
            }
            k0 += kn;
        }
        Part { m, l, o }
    };

    let store = |r: usize, o: &mut [f32], l: f32| {
        let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
        for x in o.iter_mut() {
            *x *= inv;
        }
        // SAFETY: row r of the contiguous [B, H, Sq, D] output is written by one task.
        unsafe { out.store(r * dh, 1, o) };
    };

    if nsplit == 1 {
        par::for_each(parallel, rows, init, |s, r| {
            let mut p = unit(s, r);
            store(r, &mut p.o, p.l);
        });
        return;
    }
    let parts = par::map(parallel, rows * nsplit, init, unit);
    par::for_each(parallel && rows > 1, rows, || (), |_, r| {
        let ps = &parts[r * nsplit..(r + 1) * nsplit];
        let m = ps.iter().fold(f32::NEG_INFINITY, |a, p| a.max(p.m));
        let mut o = vec![0f32; dh];
        let mut l = 0f32;
        if m != f32::NEG_INFINITY {
            for p in ps {
                if p.m == f32::NEG_INFINITY {
                    continue;
                }
                let w = math::exp(p.m - m);
                l += p.l * w;
                axpy(&mut o, w, &p.o);
            }
        }
        store(r, &mut o, l);
    });
}
