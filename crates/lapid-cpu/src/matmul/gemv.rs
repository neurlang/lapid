//! Float-B matmul for small m (decoder steps): no packing, B streamed exactly once.
//!
//! Transposed B (weights `[n, k]`): every output column is a dot product with a contiguous B
//! row; up to 4 A rows share each B load. Non-transposed B (`[k, n]`): rank-1 updates of a
//! column block. Parallel over (batch, column block).

use lapid_compiler::MatMulDims;

use super::{AOp, BOp, Epilogue, Workspace};
use crate::math::fmadd;
use crate::par::{self, PAR_MIN_WORK, block_size, grow};
use crate::Ctx;
use crate::typed::Src;

use super::hgemv;

const LANES: usize = 16;

/// Dot products of R rows of `a` with `b` (all of length `b.len()`), 16 lanes per row.
#[inline(always)]
fn dot_rows<const R: usize>(a: [&[f32]; R], b: &[f32]) -> [f32; R] {
    let k = b.len();
    let mut acc = [[0f32; LANES]; R];
    let full = k / LANES * LANES;
    for (ci, bc) in b[..full].as_chunks::<LANES>().0.iter().enumerate() {
        let p = ci * LANES;
        for r in 0..R {
            let ar = &a[r][p..p + LANES];
            for l in 0..LANES {
                acc[r][l] = fmadd(ar[l], bc[l], acc[r][l]);
            }
        }
    }
    let mut out = [0f32; R];
    for r in 0..R {
        let mut s = 0f32;
        for p in full..k {
            s = fmadd(a[r][p], b[p], s);
        }
        let l = &acc[r];
        let mut h = [0f32; 4];
        for q in 0..4 {
            h[q] = (l[q] + l[q + 4]) + (l[q + 8] + l[q + 12]);
        }
        out[r] = ((h[0] + h[1]) + (h[2] + h[3])) + s;
    }
    out
}

/// y += alpha * x
#[inline(always)]
fn axpy(y: &mut [f32], alpha: f32, x: &[f32]) {
    for (o, &v) in y.iter_mut().zip(x) {
        *o = fmadd(alpha, v, *o);
    }
}

pub(crate) fn matmul(ctx: &Ctx, ws: &mut Workspace, a: &AOp, b: &BOp, d: &MatMulDims, epi: &Epilogue) {
    let (batch, m, n, k) = (d.batch, d.m, d.n, d.k);
    let parallel = batch * m * n * k >= PAR_MIN_WORK && ctx.threads > 1;

    // A rows as contiguous f32 (converted once; tiny for small m).
    let a_rows = grow(&mut ws.f, batch * m * k);
    for bt in 0..batch {
        for i in 0..m {
            let r = bt * m + i;
            a.src.load_contig(a.row(bt, i), &mut a_rows[r * k..(r + 1) * k]);
        }
    }
    let a_rows: &[f32] = a_rows;
    let arow = |bt: usize, i: usize| &a_rows[(bt * m + i) * k..(bt * m + i + 1) * k];

    let target = if parallel { ctx.threads * 4 } else { 1 };
    let per_batch = target.div_ceil(batch).max(1);
    let nb = block_size(n, per_batch, LANES, 32, 1024).min(n.div_ceil(LANES) * LANES);
    let blocks = n.div_ceil(nb);
    let b_f32 = b.src.as_f32();
    let hk = match b.src {
        Src::F16(h) if b.transposed => hgemv::select(&ctx.isa).map(|k| (k, h)),
        _ => None,
    };

    par::for_each(
        parallel,
        batch * blocks,
        || (vec![0f32; m * nb], vec![0f32; if b.transposed { k } else { nb }], epi.regs()),
        |(acc, tmp, regs), t| {
            let bt = t / blocks;
            let j0 = (t % blocks) * nb;
            let j1 = (j0 + nb).min(n);
            let w = j1 - j0;
            let bbase = b.offset + bt * b.bs;
            if let Some((hk, bh)) = hk {
                let mut j = j0;
                while j < j1 {
                    let valid = (j1 - j).min(4);
                    let brow = |q: usize| {
                        let start = bbase + (j + q.min(valid - 1)) * b.rs;
                        &bh[start..start + k]
                    };
                    let bs = [brow(0), brow(1), brow(2), brow(3)];
                    let mut i = 0;
                    while i + 2 <= m {
                        // SAFETY: rows hold k elements (sliced above); ISA checked by select.
                        let r = unsafe { (hk.d2)([arow(bt, i), arow(bt, i + 1)], bs, k) };
                        for q in 0..2 {
                            acc[(i + q) * nb + (j - j0)..][..valid].copy_from_slice(&r[q][..valid]);
                        }
                        i += 2;
                    }
                    if i < m {
                        // SAFETY: as above.
                        let r = unsafe { (hk.d1)([arow(bt, i)], bs, k) };
                        acc[i * nb + (j - j0)..][..valid].copy_from_slice(&r[0][..valid]);
                    }
                    j += 4;
                }
            } else if b.transposed {
                for j in j0..j1 {
                    let start = bbase + j * b.rs;
                    let brow: &[f32] = match b_f32 {
                        Some(s) => &s[start..start + k],
                        None => {
                            b.src.load_contig(start, &mut tmp[..k]);
                            &tmp[..k]
                        }
                    };
                    let mut i = 0;
                    while i + 4 <= m {
                        let r = dot_rows([arow(bt, i), arow(bt, i + 1), arow(bt, i + 2), arow(bt, i + 3)], brow);
                        for q in 0..4 {
                            acc[(i + q) * nb + (j - j0)] = r[q];
                        }
                        i += 4;
                    }
                    while i + 2 <= m {
                        let r = dot_rows([arow(bt, i), arow(bt, i + 1)], brow);
                        acc[i * nb + (j - j0)] = r[0];
                        acc[(i + 1) * nb + (j - j0)] = r[1];
                        i += 2;
                    }
                    if i < m {
                        acc[i * nb + (j - j0)] = dot_rows([arow(bt, i)], brow)[0];
                    }
                }
            } else {
                for i in 0..m {
                    acc[i * nb..i * nb + w].fill(0.0);
                }
                for p in 0..k {
                    let start = bbase + p * b.rs + j0;
                    let seg: &[f32] = match b_f32 {
                        Some(s) => &s[start..start + w],
                        None => {
                            b.src.load_contig(start, &mut tmp[..w]);
                            &tmp[..w]
                        }
                    };
                    for i in 0..m {
                        axpy(&mut acc[i * nb..i * nb + w], arow(bt, i)[p], seg);
                    }
                }
            }
            for i in 0..m {
                // SAFETY: this task exclusively owns columns j0..j1 of batch bt.
                unsafe { epi.apply(regs, bt, i, j0, &acc[i * nb..i * nb + w]) };
            }
        },
    );
}
