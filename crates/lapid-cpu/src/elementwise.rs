//! Elementwise kernel: fused expression over arbitrary strided views.
//!
//! The iteration space is collapsed first (size-1 dims dropped, linear dims merged). When the
//! innermost dimension is long, work units are (row, chunk) pairs and every view is accessed
//! as a contiguous / broadcast / constant-stride run. Otherwise units are linear chunks and
//! per-element offsets come from an incremental nd counter. Plain copies move whole runs.
//!
//! Small launches (most of a decode step) stay on the calling thread: a launch goes
//! parallel only when its estimated cost (elements x expression cost) pays for a fork-join,
//! and the register file is reused across launches per thread.

use std::cell::RefCell;

use lapid_compiler::{EwDims, EwSpec};
use lapid_tensor::{Error, Result};

use crate::expr::{CHUNK, Regs};
use crate::kcache::Compiled;
use crate::par::{self, PAR_MIN_COST};
use crate::typed::{self, Dst, Src};
use crate::view::{check_view, collapse, fill_offsets, offset_of};
use crate::Ctx;

/// Inner extent from which row-wise processing is used.
const ROW_MIN: usize = 48;
/// Elements per unit of a plain copy.
const COPY_PIECE: usize = 1 << 14;

thread_local! {
    /// Register file kept per thread between launches (no allocation / zeroing per launch).
    static REGS: RefCell<Regs> = RefCell::new(Regs::default());
}

#[derive(Default)]
struct Scratch {
    regs: Regs,
    idx: Vec<usize>,
    cur: Vec<isize>,
    offs: Vec<Vec<usize>>,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let regs = std::mem::take(&mut self.regs);
        let _ = REGS.try_with(|r| {
            if let Ok(mut r) = r.try_borrow_mut() {
                *r = regs;
            }
        });
    }
}

pub(crate) fn run(ctx: &Ctx, spec: &EwSpec, dims: &EwDims, ins: &[Src], out: &Dst, c: &Compiled) -> Result<()> {
    let shape = &dims.shape;
    if shape.len() != spec.rank || dims.in_views.len() != ins.len() || ins.len() != spec.in_dtypes.len() {
        return Err(Error::new(
            "invalid_launch",
            format!(
                "cpu elementwise: rank {} / shape {:?}, {} views, {} inputs, {} dtypes",
                spec.rank,
                shape,
                dims.in_views.len(),
                ins.len(),
                spec.in_dtypes.len()
            ),
        ));
    }
    let numel: usize = shape.iter().product();
    if numel == 0 {
        return Ok(());
    }
    for (i, (v, s)) in dims.in_views.iter().zip(ins).enumerate() {
        check_view(&format!("elementwise input {i}"), shape, v, s.len())?;
    }
    check_view("elementwise output", shape, &dims.out_view, out.len())?;
    let prog = &*c.prog;
    if prog.n_inputs() > ins.len() {
        return Err(Error::new(
            "invalid_launch",
            format!("cpu elementwise: expression loads {} inputs, launch has {}", prog.n_inputs(), ins.len()),
        ));
    }

    // View 0 is the output, views 1.. are the inputs.
    let mut strides: Vec<&[isize]> = Vec::with_capacity(1 + ins.len());
    strides.push(&dims.out_view.strides);
    strides.extend(dims.in_views.iter().map(|v| v.strides.as_slice()));
    let mut bases = Vec::with_capacity(1 + ins.len());
    bases.push(dims.out_view.offset);
    bases.extend(dims.in_views.iter().map(|v| v.offset));
    let mut c2 = collapse(shape, &strides);
    let rank = c2.shape.len();
    // A short innermost dim (e.g. [6144, 2] state shifts) would take the per-element offset
    // path; every element is independent, so iterate the longest dim innermost instead (its
    // constant-stride runs go through the fast row path).
    if c2.shape[rank - 1] < ROW_MIN {
        let best = (0..rank).max_by_key(|&d| c2.shape[d]).unwrap_or(rank - 1);
        if c2.shape[best] >= ROW_MIN {
            c2.shape.swap(best, rank - 1);
            for s in &mut c2.strides {
                s.swap(best, rank - 1);
            }
        }
    }
    let c2 = c2;
    let inner = c2.shape[rank - 1];
    let nviews = bases.len();
    let copy_from = prog.identity().filter(|&i| ins[i].dtype() == out.dtype());
    // Strided (non-contiguous, non-broadcast) innermost accesses cost about one more unit.
    let strided = c2.strides.iter().filter(|s| !matches!(s[rank - 1], 0 | 1)).count();
    let cost = numel * (c.cost + strided);
    let parallel = cost >= PAR_MIN_COST && ctx.threads > 1;
    let init = || {
        let regs = REGS.try_with(|r| r.try_borrow_mut().map(|mut r| std::mem::take(&mut *r)).unwrap_or_default());
        let mut s = Scratch { regs: regs.unwrap_or_default(), idx: Vec::new(), cur: Vec::new(), offs: Vec::new() };
        prog.prepare(&mut s.regs);
        s
    };
    // Groups of `per` consecutive units per task: ~4 tasks per thread.
    let group = |units: usize| if parallel { units.div_ceil(ctx.threads * 4).max(1) } else { units };

    if inner >= ROW_MIN {
        let rows = numel / inner;
        let outer = &c2.shape[..rank - 1];
        let start = |v: usize, row: usize, c0: usize| -> (usize, isize) {
            let st = &c2.strides[v];
            let inner_s = st[rank - 1];
            let off = bases[v] as isize + offset_of(row, outer, &st[..rank - 1]) + c0 as isize * inner_s;
            (off as usize, inner_s)
        };
        if let Some(i) = copy_from {
            let piece = if parallel { numel.div_ceil(ctx.threads * 4).clamp(CHUNK, COPY_PIECE) } else { COPY_PIECE };
            let ppr = inner.div_ceil(piece);
            let units = rows * ppr;
            let per = group(units);
            par::for_each(parallel, units.div_ceil(per), || (), |_, t| {
                for u in t * per..((t + 1) * per).min(units) {
                    let (row, c0) = (u / ppr, (u % ppr) * piece);
                    let len = piece.min(inner - c0);
                    let (ob, os) = start(0, row, c0);
                    let (ib, is) = start(1 + i, row, c0);
                    // SAFETY: units write disjoint output runs; buffers validated above.
                    unsafe { typed::copy_run(&ins[i], ib, is, out, ob, os, len) };
                }
            });
            return Ok(());
        }
        if inner < CHUNK && rows > 1 {
            // Short rows: pack `rpu` whole rows into one program run.
            let rpu = (CHUNK / inner).min(rows);
            let units = rows.div_ceil(rpu);
            let per = group(units);
            par::for_each(parallel, units.div_ceil(per), init, |s, t| {
                for u in t * per..((t + 1) * per).min(units) {
                    let r0 = u * rpu;
                    let nr = rpu.min(rows - r0);
                    for (i, src) in ins.iter().enumerate() {
                        if let Some(reg) = prog.input(&mut s.regs, i) {
                            for q in 0..nr {
                                let (ib, is) = start(1 + i, r0 + q, 0);
                                src.load(ib, is, &mut reg[q * inner..(q + 1) * inner]);
                            }
                        }
                    }
                    let res = prog.run(&mut s.regs, nr * inner);
                    for q in 0..nr {
                        let (ob, os) = start(0, r0 + q, 0);
                        // SAFETY: units write disjoint output rows; buffers validated above.
                        unsafe { out.store(ob, os, &res[q * inner..(q + 1) * inner]) };
                    }
                }
            });
            return Ok(());
        }
        let cpr = inner.div_ceil(CHUNK);
        let units = rows * cpr;
        let per = group(units);
        par::for_each(parallel, units.div_ceil(per), init, |s, t| {
            for u in t * per..((t + 1) * per).min(units) {
                let row = u / cpr;
                let c0 = (u % cpr) * CHUNK;
                let len = CHUNK.min(inner - c0);
                let (ob, os) = start(0, row, c0);
                for (i, src) in ins.iter().enumerate() {
                    if let Some(reg) = prog.input(&mut s.regs, i) {
                        let (ib, is) = start(1 + i, row, c0);
                        src.load(ib, is, &mut reg[..len]);
                    }
                }
                let res = prog.run(&mut s.regs, len);
                // SAFETY: units write disjoint output runs; buffers validated above.
                unsafe { out.store(ob, os, res) };
            }
        });
    } else {
        let units = numel.div_ceil(CHUNK);
        let per = group(units);
        par::for_each(parallel, units.div_ceil(per), init, |s, t| {
            for u in t * per..((t + 1) * per).min(units) {
                let start = u * CHUNK;
                let len = CHUNK.min(numel - start);
                let Scratch { regs, idx, cur, offs } = &mut *s;
                if offs.len() != nviews {
                    *offs = vec![Vec::with_capacity(CHUNK); nviews];
                }
                fill_offsets(start, len, &c2.shape, &bases, &c2.strides, idx, cur, offs);
                if let Some(i) = copy_from {
                    // SAFETY: units write disjoint output elements; offsets validated above.
                    unsafe { typed::copy_offsets(&ins[i], &offs[1 + i], out, &offs[0]) };
                    continue;
                }
                for (i, src) in ins.iter().enumerate() {
                    if let Some(reg) = prog.input(regs, i) {
                        src.gather(&offs[1 + i], &mut reg[..len]);
                    }
                }
                let res = prog.run(regs, len);
                // SAFETY: as above.
                unsafe { out.scatter(&offs[0], res) };
            }
        });
    }
    Ok(())
}
