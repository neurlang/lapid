//! Elementwise kernel: fused expression over arbitrary strided views.
//!
//! The iteration space is collapsed first (size-1 dims dropped, linear dims merged). When the
//! innermost dimension is long, work units are (row, chunk) pairs and every view is accessed
//! as a contiguous / broadcast / constant-stride run. Otherwise units are linear chunks and
//! per-element offsets come from an incremental nd counter.

use lapid_compiler::{EwDims, EwSpec};
use lapid_tensor::{Error, Result};

use crate::expr::{CHUNK, Regs};
use crate::par::{self, PAR_MIN_ELEMS};
use crate::typed::{self, Dst, Src};
use crate::view::{check_view, collapse, fill_offsets, offset_of};
use crate::Ctx;

/// Inner extent from which row-wise processing is used.
const ROW_MIN: usize = 48;

#[derive(Default)]
struct Scratch {
    regs: Regs,
    idx: Vec<usize>,
    cur: Vec<isize>,
    offs: Vec<Vec<usize>>,
}

pub(crate) fn run(ctx: &Ctx, spec: &EwSpec, dims: &EwDims, ins: &[Src], out: &Dst, raw: &[bool]) -> Result<()> {
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
    let prog = ctx.program_typed(&spec.expr, &spec.in_dtypes, raw);
    if prog.n_inputs() > ins.len() {
        return Err(Error::new(
            "invalid_launch",
            format!("cpu elementwise: expression loads {} inputs, launch has {}", prog.n_inputs(), ins.len()),
        ));
    }

    // View 0 is the output, views 1.. are the inputs.
    let mut strides: Vec<&[isize]> = vec![&dims.out_view.strides];
    strides.extend(dims.in_views.iter().map(|v| v.strides.as_slice()));
    let mut bases = vec![dims.out_view.offset];
    bases.extend(dims.in_views.iter().map(|v| v.offset));
    let c = collapse(shape, &strides);
    let rank = c.shape.len();
    let inner = c.shape[rank - 1];
    let nviews = bases.len();
    let copy_from = prog.identity().filter(|&i| ins[i].dtype() == out.dtype());
    let parallel = numel >= PAR_MIN_ELEMS && ctx.threads > 1;
    let init = || {
        let mut s = Scratch::default();
        prog.prepare(&mut s.regs);
        s.offs = vec![Vec::with_capacity(CHUNK); nviews];
        s
    };

    if inner >= ROW_MIN {
        let cpr = inner.div_ceil(CHUNK);
        let rows = numel / inner;
        let outer = &c.shape[..rank - 1];
        par::for_each(parallel, rows * cpr, init, |s, u| {
            let row = u / cpr;
            let c0 = (u % cpr) * CHUNK;
            let len = CHUNK.min(inner - c0);
            let start = |v: usize| -> (usize, isize) {
                let st = &c.strides[v];
                let inner_s = st[rank - 1];
                let off = bases[v] as isize + offset_of(row, outer, &st[..rank - 1]) + c0 as isize * inner_s;
                (off as usize, inner_s)
            };
            let (ob, os) = start(0);
            if let Some(i) = copy_from {
                let (ib, is) = start(1 + i);
                // SAFETY: units write disjoint output runs; buffers validated above.
                unsafe { typed::copy_run(&ins[i], ib, is, out, ob, os, len) };
                return;
            }
            for (i, src) in ins.iter().enumerate() {
                if let Some(reg) = prog.input(&mut s.regs, i) {
                    let (ib, is) = start(1 + i);
                    src.load(ib, is, &mut reg[..len]);
                }
            }
            let res = prog.run(&mut s.regs, len);
            // SAFETY: as above.
            unsafe { out.store(ob, os, res) };
        });
    } else {
        let units = numel.div_ceil(CHUNK);
        par::for_each(parallel, units, init, |s, u| {
            let start = u * CHUNK;
            let len = CHUNK.min(numel - start);
            let Scratch { regs, idx, cur, offs } = s;
            fill_offsets(start, len, &c.shape, &bases, &c.strides, idx, cur, offs);
            if let Some(i) = copy_from {
                // SAFETY: units write disjoint output elements; offsets validated above.
                unsafe { typed::copy_offsets(&ins[i], &offs[1 + i], out, &offs[0]) };
                return;
            }
            for (i, src) in ins.iter().enumerate() {
                if let Some(reg) = prog.input(regs, i) {
                    src.gather(&offs[1 + i], &mut reg[..len]);
                }
            }
            let res = prog.run(regs, len);
            // SAFETY: as above.
            unsafe { out.scatter(&offs[0], res) };
        });
    }
    Ok(())
}
