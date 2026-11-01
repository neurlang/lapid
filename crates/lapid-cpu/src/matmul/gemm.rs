//! Blocked f32 GEMM for float B (f32 / f16 / bf16 storage) and larger m.
//!
//! A and B are packed once per launch into k-major micro-panels (`MR` rows / `NR` columns,
//! converted to f32, zero padded) in parallel. Tasks then cover (batch, row block, column
//! block) tiles; inside a task every B panel stays hot in L1 while the row block's A panels
//! stream from L2. Each 6x16 tile accumulates the full k in registers; finished tile rows go
//! through the epilogue.

use lapid_compiler::MatMulDims;

use super::kernels::{self, MR, NR};
use super::{AOp, BOp, Epilogue, Workspace};
use crate::par::{self, PAR_MIN_WORK, grow};
use crate::typed::Src;
use crate::Ctx;

/// `dst[p * w + r]` = element p of row r (k contiguous elements at `base(r)`), or 0.
pub(crate) fn pack_k_major(
    src: &Src,
    w: usize,
    base: impl Fn(usize) -> Option<usize>,
    k: usize,
    tmp: &mut [f32],
    dst: &mut [f32],
) {
    let dst = &mut dst[..k * w];
    for r in 0..w {
        match base(r) {
            Some(b) => {
                let row: &[f32] = match src.as_f32() {
                    Some(s) => &s[b..b + k],
                    None => {
                        src.load_contig(b, &mut tmp[..k]);
                        &tmp[..k]
                    }
                };
                for (d, &v) in dst[r..].iter_mut().step_by(w).zip(row) {
                    *d = v;
                }
            }
            None => {
                for d in dst[r..].iter_mut().step_by(w) {
                    *d = 0.0;
                }
            }
        }
    }
}

/// Non-transposed B panel: `dst[p * NR + c] = B[p, j0 + c]` (row p at `base + p * rs`),
/// zero padded beyond `valid` columns.
fn pack_b_rows(src: &Src, base: usize, rs: usize, k: usize, valid: usize, dst: &mut [f32]) {
    for (p, d) in dst[..k * NR].as_chunks_mut::<NR>().0.iter_mut().enumerate() {
        src.load_contig(base + p * rs, &mut d[..valid]);
        d[valid..].fill(0.0);
    }
}

pub(crate) fn matmul(ctx: &Ctx, ws: &mut Workspace, a: &AOp, b: &BOp, d: &MatMulDims, epi: &Epilogue) {
    let (batch, m, n, k) = (d.batch, d.m, d.n, d.k);
    let mp = m.div_ceil(MR);
    let np = n.div_ceil(NR);
    let a_batches = if a.bs == 0 { 1 } else { batch };
    let b_batches = if b.bs == 0 { 1 } else { batch };
    let a_len = a_batches * mp * k * MR;
    let b_len = b_batches * np * k * NR;
    let parallel = batch * m * n * k >= PAR_MIN_WORK && ctx.threads > 1;

    let buf = grow(&mut ws.f, a_len + b_len);
    let (apack, bpack) = buf.split_at_mut(a_len);
    par::chunks_mut(parallel, apack, k * MR, || vec![0f32; k], |tmp, u, dst| {
        let bt = u / mp;
        let i0 = (u % mp) * MR;
        pack_k_major(&a.src, MR, |r| (i0 + r < m).then(|| a.row(bt, i0 + r)), k, tmp, dst);
    });
    par::chunks_mut(parallel, bpack, k * NR, || vec![0f32; k], |tmp, u, dst| {
        let bt = u / np;
        let j0 = (u % np) * NR;
        let base = b.offset + bt * b.bs;
        if b.transposed {
            pack_k_major(&b.src, NR, |c| (j0 + c < n).then(|| base + (j0 + c) * b.rs), k, tmp, dst);
        } else {
            pack_b_rows(&b.src, base + j0, b.rs, k, NR.min(n - j0), dst);
        }
    });
    let apack: &[f32] = apack;
    let bpack: &[f32] = bpack;

    // Tile sizes: start large for cache reuse, shrink until there are enough tasks to
    // balance the (heterogeneous) cores.
    let target = if parallel { ctx.threads * 4 } else { 1 };
    let tasks = |mc: usize, nc: usize| batch * m.div_ceil(mc) * n.div_ceil(nc);
    let (mut mc, mut nc) = (MR * 16, NR * 16);
    while tasks(mc, nc) < target && nc > NR * 2 {
        nc /= 2;
    }
    while tasks(mc, nc) < target && mc > MR * 2 {
        mc /= 2;
    }
    let mt = m.div_ceil(mc);
    let nt = n.div_ceil(nc);
    let micro = kernels::select(&ctx.isa);

    par::for_each(parallel, batch * mt * nt, || (vec![0f32; mc * nc], epi.regs()), |(cbuf, regs), t| {
        let bt = t / (mt * nt);
        let rem = t % (mt * nt);
        let (ti, tj) = (rem / nt, rem % nt);
        let (i0, j0) = (ti * mc, tj * nc);
        let (i1, j1) = ((i0 + mc).min(m), (j0 + nc).min(n));
        let abt = if a.bs == 0 { 0 } else { bt };
        let bbt = if b.bs == 0 { 0 } else { bt };
        let (ip0, ip1) = (i0 / MR, i1.div_ceil(MR));
        let (jp0, jp1) = (j0 / NR, j1.div_ceil(NR));
        let ldc = nc;
        for jp in jp0..jp1 {
            let bp = bpack[(bbt * np + jp) * k * NR..][..k * NR].as_ptr();
            for ip in ip0..ip1 {
                let ap = apack[(abt * mp + ip) * k * MR..][..k * MR].as_ptr();
                let off = (ip - ip0) * MR * ldc + (jp - jp0) * NR;
                let tile = cbuf[off..off + (MR - 1) * ldc + NR].as_mut_ptr();
                // SAFETY: panels hold k*MR / k*NR floats (sliced above); the tile slice
                // covers MR rows of NR columns at stride ldc.
                unsafe { micro(k, ap, bp, tile, ldc) };
            }
        }
        for i in i0..i1 {
            let r = i - i0;
            // SAFETY: this task exclusively owns output rows i0..i1, columns j0..j1.
            unsafe { epi.apply(regs, bt, i, j0, &cbuf[r * ldc..r * ldc + (j1 - j0)]) };
        }
    });
}
