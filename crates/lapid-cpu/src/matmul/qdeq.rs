//! Quantized-B matmul for larger m (encoder, prefill): f32 activations x dequantized weights.
//!
//! Quantizing activations to int8 (the `quant` / `qgemm` paths) costs accuracy that adds up
//! over deep encoders; with many rows the weight dequantization is amortized, so here:
//! 1. A rows (flattened over batch) are packed once into k-major `MR`-row panels (f32);
//! 2. B is dequantized column chunk by column chunk (bounded scratch) into k-major `NR`-column
//!    f32 panels, in parallel, once per launch — O(n·k) against O(m·n·k) of compute;
//! 3. tiles of (row block, column block) run the f32 6x16 microkernel, rows go through the
//!    epilogue. Same numerics as a float GEMM on the dequantized weights.

use lapid_compiler::MatMulDims;
use lapid_quant::GROUP;

use super::gemm::pack_k_major;
use super::kernels::{self, MR, NR};
use super::{AOp, Epilogue, Workspace};
use crate::Ctx;
use crate::par::{self, PAR_MIN_WORK, grow};
use crate::qmat::QMat;

/// Max dequantized B floats held at once (16 MiB).
const B_BUDGET: usize = 1 << 22;

/// Dequantize columns `j0 .. j0 + NR` of `B = W^T` (rows of `w`) into a k-major panel
/// `dst[p * NR + c] = W[j0 + c, p]`, zero padded beyond `n`.
fn pack_panel(w: &QMat, j0: usize, dst: &mut [f32]) {
    let (n, k) = (w.rows, w.cols);
    let dst = &mut dst[..k * NR];
    let mut q = [0i8; GROUP];
    let mut row = if w.is_k() { vec![0f32; k] } else { Vec::new() };
    for c in 0..NR {
        let j = j0 + c;
        if j >= n {
            for x in dst[c..].iter_mut().step_by(NR) {
                *x = 0.0;
            }
            continue;
        }
        if w.is_k() {
            w.dequant_row(j, &mut row);
            for (x, &v) in dst[c..].iter_mut().step_by(NR).zip(&row) {
                *x = v;
            }
            continue;
        }
        for g in 0..w.groups() {
            let s = w.scale(j, g);
            w.group_i8(j, g, &mut q);
            let base = g * GROUP * NR + c;
            for (t, &v) in q.iter().enumerate() {
                dst[base + t * NR] = v as f32 * s;
            }
        }
    }
}

pub(crate) fn matmul(ctx: &Ctx, ws: &mut Workspace, a: &AOp, w: &QMat, d: &MatMulDims, epi: &Epilogue) {
    let (batch, m, n, k) = (d.batch, d.m, d.n, d.k);
    let rows = batch * m;
    let mp = rows.div_ceil(MR);
    let parallel = rows * n * k >= PAR_MIN_WORK && ctx.threads > 1;
    // Columns per chunk: whole panels within the budget.
    let chunk = ((B_BUDGET / k.max(1)) / NR).max(1) * NR;
    let chunk = chunk.min(n.div_ceil(NR) * NR);
    let a_len = mp * k * MR;
    let b_len = chunk * k;

    let buf = grow(&mut ws.f, a_len + b_len);
    let (apack, bpack) = buf.split_at_mut(a_len);
    par::chunks_mut(parallel, apack, k * MR, || vec![0f32; k], |tmp, u, dst| {
        let r0 = u * MR;
        pack_k_major(&a.src, MR, |r| (r0 + r < rows).then(|| a.row((r0 + r) / m, (r0 + r) % m)), k, tmp, dst);
    });
    let apack: &[f32] = apack;

    // Tile sizes as in `gemm`: large for reuse, shrunk until the cores are balanced.
    let target = if parallel { ctx.threads * 4 } else { 1 };
    let micro = kernels::select(&ctx.isa);
    let mut c0 = 0;
    while c0 < n {
        let cn = chunk.min(n - c0);
        let np = cn.div_ceil(NR);
        let bpk = &mut bpack[..np * k * NR];
        par::chunks_mut(parallel, bpk, k * NR, || (), |_, u, dst| pack_panel(w, c0 + u * NR, dst));
        let bpk: &[f32] = bpk;

        let tasks = |mc: usize, nc: usize| rows.div_ceil(mc) * cn.div_ceil(nc);
        let (mut mc, mut nc) = (MR * 16, NR * 16);
        while tasks(mc, nc) < target && nc > NR * 2 {
            nc /= 2;
        }
        while tasks(mc, nc) < target && mc > MR * 2 {
            mc /= 2;
        }
        let (mt, nt) = (rows.div_ceil(mc), cn.div_ceil(nc));
        par::for_each(parallel, mt * nt, || (vec![0f32; mc * nc], epi.regs()), |(cbuf, regs), t| {
            let (ti, tj) = (t / nt, t % nt);
            let (i0, j0) = (ti * mc, tj * nc);
            let (i1, j1) = ((i0 + mc).min(rows), (j0 + nc).min(cn));
            let (ip0, ip1) = (i0 / MR, i1.div_ceil(MR));
            let (jp0, jp1) = (j0 / NR, j1.div_ceil(NR));
            let ldc = nc;
            for jp in jp0..jp1 {
                let bp = bpk[jp * k * NR..][..k * NR].as_ptr();
                for ip in ip0..ip1 {
                    let ap = apack[ip * k * MR..][..k * MR].as_ptr();
                    let off = (ip - ip0) * MR * ldc + (jp - jp0) * NR;
                    let tile = cbuf[off..off + (MR - 1) * ldc + NR].as_mut_ptr();
                    // SAFETY: panels hold k*MR / k*NR floats (sliced above); the tile slice
                    // covers MR rows of NR columns at stride ldc.
                    unsafe { micro(k, ap, bp, tile, ldc) };
                }
            }
            for r in i0..i1 {
                let o = (r - i0) * ldc;
                // SAFETY: this task exclusively owns flattened rows i0..i1, columns
                // c0+j0..c0+j1 of the output.
                unsafe { epi.apply(regs, r / m, r % m, c0 + j0, &cbuf[o..o + (j1 - j0)]) };
            }
        });
        c0 += cn;
    }
}
