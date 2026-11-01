//! Quantized-B matmul driver: int8 activations x int8/int4 weights, f32 accumulation.
//!
//! 1. Every A row is quantized once per launch to int8 per 32-block (`d = max|x| / 127`).
//! 2. Tasks cover (row block, column block). Inside, 4 output columns at a time share the
//!    B group loads with up to 4 A rows (4x4 register tile); the A row block stays in L2,
//!    the 4 B rows in L1.
//! 3. Per group: `acc += f32(dot_i32(qa, qb)) * (da * db)`; lanes are reduced at the end.

use lapid_compiler::MatMulDims;
use lapid_quant::GROUP;

use super::qkernels::{self, QA};
use super::{AOp, Epilogue, Workspace};
use crate::math::round_fast;
use crate::par::{self, PAR_MIN_WORK, SyncPtr, block_size, grow};
use crate::kquant::{self, KA};
use crate::qmat::QMat;
use crate::Ctx;

/// Rows per row block (A block of 64 x k int8 stays in L2).
const ROW_BLOCK: usize = 64;

/// Quantize one f32 row to int8 per 32-element group. `x.len() == q.len() == 32 * s.len()`.
pub(crate) fn quantize_row(x: &[f32], q: &mut [i8], s: &mut [f32]) {
    for ((xg, qg), sg) in x.as_chunks::<GROUP>().0.iter().zip(q.as_chunks_mut::<GROUP>().0).zip(s.iter_mut()) {
        let mut mx = [0f32; 8];
        for c in xg.as_chunks::<8>().0 {
            for l in 0..8 {
                mx[l] = mx[l].max(c[l].abs());
            }
        }
        let amax = mx.iter().fold(0f32, |m, &v| m.max(v));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        *sg = d;
        for (o, &v) in qg.iter_mut().zip(xg) {
            *o = (round_fast(v * id) as i32).clamp(-127, 127) as i8;
        }
    }
}

pub(crate) fn matmul(ctx: &Ctx, ws: &mut Workspace, a: &AOp, b: &QMat, d: &MatMulDims, epi: &Epilogue) {
    let (batch, m, n, k) = (d.batch, d.m, d.n, d.k);
    let rows = batch * m;
    let kg = k / GROUP;
    let parallel = rows * n * k >= PAR_MIN_WORK && ctx.threads > 1;

    // 1. Quantize activations.
    let qa = grow(&mut ws.q, rows * k);
    let sa = grow(&mut ws.s, rows * kg);
    let quant = |tmp: &mut Vec<f32>, r: usize, q: &mut [i8], s: &mut [f32]| {
        let (bt, i) = (r / m, r % m);
        a.src.load_contig(a.row(bt, i), &mut tmp[..k]);
        quantize_row(&tmp[..k], q, s);
    };
    {
        let sp = SyncPtr(sa.as_mut_ptr());
        par::chunks_mut(parallel && rows > 1, qa, k, || vec![0f32; k], |tmp, r, q| {
            let p = sp;
            // SAFETY: row r owns scales r*kg..(r+1)*kg exclusively.
            let s = unsafe { std::slice::from_raw_parts_mut(p.0.add(r * kg), kg) };
            quant(tmp, r, q, s);
        });
    }
    // K formats: per-group `da * sum(q)` for the min terms.
    let bsum = if b.is_k() {
        let bs = grow(&mut ws.pa_f, rows * kg);
        for (g, o) in bs.iter_mut().enumerate() {
            *o = sa[g] * qa[g * GROUP..(g + 1) * GROUP].iter().map(|&v| v as i32).sum::<i32>() as f32;
        }
        Some(&*bs)
    } else {
        None
    };
    let qa = QA { q: qa, s: sa, k };

    // 2. Tiles.
    let rb = ROW_BLOCK.min(rows);
    let row_blocks = rows.div_ceil(rb);
    let target = if parallel { ctx.threads * 4 } else { 1 };
    let nb = block_size(n, target.div_ceil(row_blocks), 4, 16, 512);
    let col_blocks = n.div_ceil(nb);
    let kern = qkernels::select(&ctx.isa, b.fmt);
    let kkern = kquant::select(&ctx.isa, b.fmt);
    let (ka, km) = (KA { q: qa.q, s: qa.s, bs: bsum.unwrap_or(&[]), k }, b.kmat());
    let is_k = b.is_k();
    // SAFETY (both): forwarded contract of `Tile4` / `KTile4` (see the call sites).
    let t4 = |r: [usize; 4], c: [usize; 4]| unsafe { if is_k { (kkern.t4)(&ka, r, &km, c) } else { (kern.t4)(&qa, r, b, c) } };
    let t1 = |r: [usize; 1], c: [usize; 4]| unsafe { if is_k { (kkern.t1)(&ka, r, &km, c) } else { (kern.t1)(&qa, r, b, c) } };

    par::for_each(parallel, row_blocks * col_blocks, || (vec![0f32; rb * nb], epi.regs()), |(cbuf, regs), t| {
        let (rbi, cbi) = (t / col_blocks, t % col_blocks);
        let (r0, r1) = (rbi * rb, ((rbi + 1) * rb).min(rows));
        let (j0, j1) = (cbi * nb, ((cbi + 1) * nb).min(n));
        let ldc = nb;
        let mut c = j0;
        while c < j1 {
            let valid = (j1 - c).min(4);
            // Tail columns repeat the last valid column; their results are discarded.
            let last = valid - 1;
            let cols = [c, c + last.min(1), c + last.min(2), c + last];
            let mut r = r0;
            while r + 4 <= r1 {
                // rows < `rows`, cols < n, buffers sized/validated by the caller; the
                // kernel's ISA was detected by `select`.
                let out = t4([r, r + 1, r + 2, r + 3], cols);
                for (q, row) in out.iter().enumerate() {
                    let o = (r + q - r0) * ldc + (c - j0);
                    cbuf[o..o + valid].copy_from_slice(&row[..valid]);
                }
                r += 4;
            }
            while r < r1 {
                let out = t1([r], cols);
                let o = (r - r0) * ldc + (c - j0);
                cbuf[o..o + valid].copy_from_slice(&out[0][..valid]);
                r += 1;
            }
            c += 4;
        }
        for r in r0..r1 {
            let o = (r - r0) * ldc;
            // SAFETY: this task exclusively owns rows r0..r1, columns j0..j1.
            unsafe { epi.apply(regs, r / m, r % m, j0, &cbuf[o..o + (j1 - j0)]) };
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_row_roundtrip() {
        let x: Vec<f32> = (0..64).map(|i| ((i * 37) % 29) as f32 * 0.37 - 5.0).collect();
        let mut q = vec![0i8; 64];
        let mut s = vec![0f32; 2];
        quantize_row(&x, &mut q, &mut s);
        for g in 0..2 {
            for t in 0..32 {
                let v = q[g * 32 + t] as f32 * s[g];
                assert!((v - x[g * 32 + t]).abs() <= s[g] * 0.5 + 1e-6);
            }
        }
        let z = vec![0f32; 32];
        let mut qz = vec![1i8; 32];
        let mut sz = vec![1f32; 1];
        quantize_row(&z, &mut qz, &mut sz);
        assert!(qz.iter().all(|&v| v == 0) && sz[0] == 0.0);
    }
}
