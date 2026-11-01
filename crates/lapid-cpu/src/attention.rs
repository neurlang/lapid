//! Flash-style attention: `out = softmax(scale * q k^T + mask, causal) v` without ever
//! materializing the `Sq x Sk` matrix.
//!
//! Work units are (batch, head, block of `QB` queries), run in parallel. Each unit walks the
//! visible keys in blocks of `KB`: scores `S = Q K^T` and the update `T = P V` are both small
//! GEMMs run through the f32 6x16 microkernel on panels packed per block; an online softmax
//! (running max `m`, running sum `l`, rescale factor `alpha`) keeps the result exact.

use lapid_compiler::{AttentionDims, AttentionSpec, View};
use lapid_tensor::{Error, Result};

use crate::math::{self, max8, sum8};
use crate::matmul::{MR, MicroFn, NR, kernels};
use crate::matmul::Workspace;
use crate::par::{self, PAR_MIN_WORK, SyncPtr, grow};
use crate::typed::{Dst, Src};
use crate::view::check_view;
use crate::Ctx;

/// Queries per unit (multiple of MR).
const QB_MAX: usize = MR * 8;
/// Keys per block (multiple of NR).
const KB: usize = NR * 8;

pub(crate) struct AttnInputs<'a> {
    pub q: Src<'a>,
    pub k: Src<'a>,
    pub v: Src<'a>,
    pub mask: Option<Src<'a>>,
}

#[derive(Default)]
struct Scratch {
    qp: Vec<f32>,
    pp: Vec<f32>,
    s: Vec<f32>,
    t: Vec<f32>,
    o: Vec<f32>,
    m: Vec<f32>,
    l: Vec<f32>,
    tmp: Vec<f32>,
}

/// Element offset of row (b, h, s) of a rank-4 view and its innermost stride.
#[inline]
fn row(v: &View, b: usize, h: usize, s: usize) -> (usize, isize) {
    let st = &v.strides;
    let off = v.offset as isize + b as isize * st[0] + h as isize * st[1] + s as isize * st[2];
    (off as usize, st[3])
}

pub(crate) fn run(ctx: &Ctx, ws: &mut Workspace, spec: &AttentionSpec, d: &AttentionDims, ins: &AttnInputs, out: &Dst) -> Result<()> {
    let (bsz, heads, sq, sk, dh) = (d.batch, d.heads, d.sq, d.sk, d.d);
    let n_out = bsz * heads * sq * dh;
    if n_out == 0 {
        return Ok(());
    }
    if out.len() < n_out {
        return Err(Error::new("out_of_bounds", format!("cpu attention: output holds {} < {n_out} elements", out.len())));
    }
    check_view("attention q", &[bsz, heads, sq, dh], &d.q, ins.q.len())?;
    check_view("attention k", &[bsz, heads, sk, dh], &d.k, ins.k.len())?;
    check_view("attention v", &[bsz, heads, sk, dh], &d.v, ins.v.len())?;
    match (&d.mask, &ins.mask) {
        (Some(mv), Some(ms)) => check_view("attention mask", &[bsz, heads, sq, sk], mv, ms.len())?,
        (None, None) => {}
        _ => return Err(Error::new("invalid_launch", "cpu attention: mask view / mask buffer mismatch")),
    }
    if crate::attention_vec::applies(d) {
        crate::attention_vec::run(ctx, spec, d, ins, out);
        return Ok(());
    }
    let scale = f32::from_bits(d.scale);
    let causal = spec.causal;
    let qb = QB_MAX.min(sq.div_ceil(MR) * MR);
    let qblocks = sq.div_ceil(qb);
    let dp = dh.div_ceil(NR) * NR;
    let micro = kernels::select(&ctx.isa);
    let parallel = bsz * heads * sq * sk.max(1) * dh >= PAR_MIN_WORK && ctx.threads > 1;

    // K and V are packed once per (batch, head) and shared by every query block.
    let kblocks = sk.div_ceil(KB);
    let per_k = kblocks * KB * dh;
    let per_v = kblocks * KB * dp;
    let packed = grow(&mut ws.f, bsz * heads * (per_k + per_v));
    let (kpk, vpk) = packed.split_at_mut(bsz * heads * per_k);
    {
        let (kptr, vptr) = (SyncPtr(kpk.as_mut_ptr()), SyncPtr(vpk.as_mut_ptr()));
        par::for_each(parallel, bsz * heads * kblocks, || vec![0f32; dh], |tmp, u| {
            let (kp, vp) = (kptr, vptr);
            let (bh, bk) = (u / kblocks, u % kblocks);
            let (b, h) = (bh / heads, bh % heads);
            let k0 = bk * KB;
            let kn = KB.min(sk - k0);
            // SAFETY: every (bh, bk) unit owns its KB-sized block of both packed buffers.
            let (kb, vb) = unsafe {
                (
                    std::slice::from_raw_parts_mut(kp.0.add(bh * per_k + bk * KB * dh), KB * dh),
                    std::slice::from_raw_parts_mut(vp.0.add(bh * per_v + bk * KB * dp), KB * dp),
                )
            };
            pack_kv(d, ins, b, h, k0, kn, dh, dp, tmp, kb, vb);
        });
    }
    let (kpk, vpk): (&[f32], &[f32]) = (kpk, vpk);

    let init = || Scratch {
        qp: vec![0.0; qb * dh],
        pp: vec![0.0; qb * KB],
        s: vec![0.0; qb * KB],
        t: vec![0.0; qb * dp],
        o: vec![0.0; qb * dp],
        m: vec![0.0; qb],
        l: vec![0.0; qb],
        tmp: vec![0.0; dh.max(KB)],
    };
    let geo = Geo { sq, sk, dh, dp, qb, scale, causal };
    par::for_each(parallel, bsz * heads * qblocks, init, |s, u| {
        let bh = u / qblocks;
        let (b, h) = (bh / heads, bh % heads);
        let q0 = (u % qblocks) * qb;
        let packs = (&kpk[bh * per_k..(bh + 1) * per_k], &vpk[bh * per_v..(bh + 1) * per_v]);
        unit(&geo, micro, d, ins, out, s, packs, b, h, q0);
    });
    Ok(())
}

/// Pack keys `k0 .. k0 + kn` of (b, h): K into k-major panels of NR keys over D
/// (`kb[p*NR*dh + dd*NR + c] = K[k0 + p*NR + c, dd]`), V into panels of NR columns over the
/// keys (`vb[q*kn*NR + j*NR + c] = V[k0 + j, q*NR + c]`); zero padded.
#[allow(clippy::too_many_arguments)]
fn pack_kv(
    d: &AttentionDims,
    ins: &AttnInputs,
    b: usize,
    h: usize,
    k0: usize,
    kn: usize,
    dh: usize,
    dp: usize,
    tmp: &mut [f32],
    kb: &mut [f32],
    vb: &mut [f32],
) {
    let kpan = kn.div_ceil(NR);
    kb[..kpan * NR * dh].fill(0.0);
    for j in 0..kn {
        let (base, st) = row(&d.k, b, h, k0 + j);
        ins.k.load(base, st, &mut tmp[..dh]);
        let panel = &mut kb[(j / NR) * NR * dh..(j / NR + 1) * NR * dh];
        let c = j % NR;
        for (dd, &v) in tmp[..dh].iter().enumerate() {
            panel[dd * NR + c] = v;
        }
    }
    let vpan = dp / NR;
    for j in 0..kn {
        let (base, st) = row(&d.v, b, h, k0 + j);
        ins.v.load(base, st, &mut tmp[..dh]);
        for q in 0..vpan {
            let dst = &mut vb[q * kn * NR + j * NR..q * kn * NR + (j + 1) * NR];
            let c0 = q * NR;
            let valid = NR.min(dh - c0);
            dst[..valid].copy_from_slice(&tmp[c0..c0 + valid]);
            dst[valid..].fill(0.0);
        }
    }
}

struct Geo {
    sq: usize,
    sk: usize,
    dh: usize,
    dp: usize,
    qb: usize,
    scale: f32,
    causal: bool,
}

#[allow(clippy::too_many_arguments)]
fn unit(
    g: &Geo,
    micro: MicroFn,
    d: &AttentionDims,
    ins: &AttnInputs,
    out: &Dst,
    s: &mut Scratch,
    (kpk, vpk): (&[f32], &[f32]),
    b: usize,
    h: usize,
    q0: usize,
) {
    let (sq, sk, dh, dp, qb) = (g.sq, g.sk, g.dh, g.dp, g.qb);
    let qn = qb.min(sq - q0);
    let qpan = qn.div_ceil(MR);
    let Scratch { qp, pp, s: sc, t, o, m, l, tmp } = s;

    // Pack Q (scaled) into k-major panels of MR rows over D.
    qp.fill(0.0);
    for i in 0..qn {
        let (base, st) = row(&d.q, b, h, q0 + i);
        ins.q.load(base, st, &mut tmp[..dh]);
        let panel = &mut qp[(i / MR) * MR * dh..(i / MR + 1) * MR * dh];
        let r = i % MR;
        for (dd, &v) in tmp[..dh].iter().enumerate() {
            panel[dd * MR + r] = v * g.scale;
        }
    }
    o[..qb * dp].fill(0.0);
    m[..qb].fill(f32::NEG_INFINITY);
    l[..qb].fill(0.0);

    // Keys visible to the last query of the block.
    let kend = if g.causal {
        (q0 as isize + qn as isize + sk as isize - sq as isize).clamp(0, sk as isize) as usize
    } else {
        sk
    };
    let mut k0 = 0;
    while k0 < kend {
        let kn = KB.min(kend - k0);
        let kpan = kn.div_ceil(NR);
        // V panels of this block were packed with the full block length (`pack_kv`), which
        // exceeds `kn` when the causal limit cuts the block short.
        let vstride = KB.min(sk - k0);

        let kp = &kpk[k0 * dh..];
        let vp = &vpk[k0 * dp..];
        // S = Q K^T  (qpan x kpan tiles of MR x NR, row stride KB).
        for ip in 0..qpan {
            for kq in 0..kpan {
                let a = qp[ip * MR * dh..(ip + 1) * MR * dh].as_ptr();
                let bp = kp[kq * NR * dh..(kq + 1) * NR * dh].as_ptr();
                let off = ip * MR * KB + kq * NR;
                let c = sc[off..off + (MR - 1) * KB + NR].as_mut_ptr();
                // SAFETY: panels hold dh*MR / dh*NR floats, tile slice covers MR x NR at KB.
                unsafe { micro(dh, a, bp, c, KB) };
            }
        }
        // Mask, causal limit, online softmax. P overwrites S.
        for i in 0..qn {
            let srow = &mut sc[i * KB..i * KB + kn];
            if let (Some(mv), Some(ms)) = (&d.mask, &ins.mask) {
                let (base, st) = row(mv, b, h, q0 + i);
                ms.load((base as isize + k0 as isize * st) as usize, st, &mut tmp[..kn]);
                for (x, &mk) in srow.iter_mut().zip(&tmp[..kn]) {
                    *x += mk;
                }
            }
            if g.causal {
                // Key j visible iff k0 + j <= q0 + i + sk - sq.
                let lim = q0 as isize + i as isize + sk as isize - sq as isize - k0 as isize;
                let first_hidden = (lim + 1).clamp(0, kn as isize) as usize;
                srow[first_hidden..].fill(f32::NEG_INFINITY);
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
                for x in &mut o[i * dp..i * dp + dh] {
                    *x *= alpha;
                }
            }
        }
        // Pack P into k-major panels of MR rows over the kn keys (rows >= qn are zero).
        for ip in 0..qpan {
            let panel = &mut pp[ip * MR * kn..(ip + 1) * MR * kn];
            for r in 0..MR {
                let i = ip * MR + r;
                if i < qn {
                    let srow = &sc[i * KB..i * KB + kn];
                    for (j, &p) in srow.iter().enumerate() {
                        panel[j * MR + r] = p;
                    }
                } else {
                    for j in 0..kn {
                        panel[j * MR + r] = 0.0;
                    }
                }
            }
        }
        let vpan = dp / NR;
        // T = P V, then O += T.
        for ip in 0..qpan {
            for q in 0..vpan {
                let a = pp[ip * MR * kn..(ip + 1) * MR * kn].as_ptr();
                let bp = vp[q * vstride * NR..q * vstride * NR + kn * NR].as_ptr();
                let off = ip * MR * dp + q * NR;
                let c = t[off..off + (MR - 1) * dp + NR].as_mut_ptr();
                // SAFETY: panels hold kn*MR / kn*NR floats, tile slice covers MR x NR at dp.
                unsafe { micro(kn, a, bp, c, dp) };
            }
        }
        for i in 0..qn {
            for (x, &y) in o[i * dp..i * dp + dh].iter_mut().zip(&t[i * dp..i * dp + dh]) {
                *x += y;
            }
        }
        k0 += kn;
    }

    // Normalize and store rows (b, h, q0 + i, :) of the contiguous output.
    for i in 0..qn {
        let orow = &mut o[i * dp..i * dp + dh];
        let inv = if l[i] > 0.0 { 1.0 / l[i] } else { 0.0 };
        for x in orow.iter_mut() {
            *x *= inv;
        }
        let base = ((b * d.heads + h) * sq + q0 + i) * dh;
        // SAFETY: each unit owns its (b, h, query block) rows of the output.
        unsafe { out.store(base, 1, orow) };
    }
}

#[cfg(test)]
#[path = "attention_tests.rs"]
mod attention_tests;

#[cfg(test)]
pub(crate) fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    (bsz, heads, sq, sk, dh): (usize, usize, usize, usize, usize),
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut out = vec![0f32; bsz * heads * sq * dh];
    for bh in 0..bsz * heads {
        for i in 0..sq {
            let mut sc = vec![f64::NEG_INFINITY; sk];
            for j in 0..sk {
                if causal && j as isize > i as isize + sk as isize - sq as isize {
                    continue;
                }
                let mut dot = 0f64;
                for dd in 0..dh {
                    dot += q[(bh * sq + i) * dh + dd] as f64 * k[(bh * sk + j) * dh + dd] as f64;
                }
                sc[j] = dot * scale as f64 + mask.map_or(0.0, |mk| mk[(bh * sq + i) * sk + j] as f64);
            }
            let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if mx == f64::NEG_INFINITY {
                continue;
            }
            let e: Vec<f64> = sc.iter().map(|&x| (x - mx).exp()).collect();
            let sum: f64 = e.iter().sum();
            for dd in 0..dh {
                let mut acc = 0f64;
                for j in 0..sk {
                    acc += e[j] * v[(bh * sk + j) * dh + dd] as f64;
                }
                out[(bh * sq + i) * dh + dd] = (acc / sum) as f32;
            }
        }
    }
    out
}
