//! Reduce kernel: `out[o] = reduce_j expr(inputs at (o, j))`, reduced axis last.
//!
//! The outer (kept) dims are collapsed across all views first. Two strategies:
//! * short rows (`len <= COL_MAX`, e.g. a depthwise conv window `[6144, 3]`): blocks of up to
//!   `CHUNK` rows are evaluated column by column, one vectorized program run per reduced
//!   index over the whole block, accumulating per row;
//! * long rows: rows (or, when there are fewer rows than threads, row segments whose
//!   partials are combined afterwards) are walked in `CHUNK` pieces with 8-lane partial
//!   sums. Short-but-many rows are grouped so a task touches at least `TASK_ELEMS` elements.
//!
//! Work is split across threads only when the estimated cost (elements x expression cost)
//! pays for a fork-join.

use lapid_compiler::{ReduceDims, ReduceSpec};
use lapid_tensor::{Error, ReduceOp, Result};

use crate::expr::{CHUNK, Regs};
use crate::kcache::Compiled;
use crate::math::{max8, sum8};
use crate::par::{self, PAR_MIN_COST};
use crate::typed::{Dst, Src};
use crate::view::{check_view, collapse, fill_offsets, offset_of};
use crate::Ctx;

/// Minimum segment length when splitting a row across threads.
const MIN_SEGMENT: usize = 8 * CHUNK;
/// Rows up to this length use the column-wise strategy.
const COL_MAX: usize = 32;
/// Elements per task for the long-row path.
const TASK_ELEMS: usize = 4096;

#[derive(Default)]
struct Scratch {
    regs: Regs,
    acc: Vec<f32>,
    idx: Vec<usize>,
    cur: Vec<isize>,
    offs: Vec<Vec<usize>>,
    bases: Vec<isize>,
}

pub(crate) fn run(ctx: &Ctx, spec: &ReduceSpec, dims: &ReduceDims, ins: &[Src], out: &Dst, c: &Compiled) -> Result<()> {
    let shape = &dims.shape;
    let rank = shape.len();
    if rank == 0 || rank != spec.rank || dims.in_views.len() != ins.len() || ins.len() != spec.in_dtypes.len() {
        return Err(Error::new(
            "invalid_launch",
            format!("cpu reduce: rank {} / shape {:?}, {} views, {} inputs", spec.rank, shape, dims.in_views.len(), ins.len()),
        ));
    }
    let outer = &shape[..rank - 1];
    let rows: usize = outer.iter().product();
    let len = shape[rank - 1];
    if rows == 0 {
        return Ok(());
    }
    if out.len() < rows {
        return Err(Error::new("out_of_bounds", format!("cpu reduce: output holds {} < {rows} elements", out.len())));
    }
    let op = spec.op;
    let identity = match op {
        ReduceOp::Sum => 0.0f32,
        ReduceOp::Max => f32::NEG_INFINITY,
    };
    if len == 0 {
        // SAFETY: single-threaded store of `rows` validated elements.
        unsafe { out.store(0, 1, &vec![identity; rows]) };
        return Ok(());
    }
    for (i, (v, s)) in dims.in_views.iter().zip(ins).enumerate() {
        check_view(&format!("reduce input {i}"), shape, v, s.len())?;
    }
    let prog = &c.prog;
    if prog.n_inputs() > ins.len() {
        return Err(Error::new(
            "invalid_launch",
            format!("cpu reduce: expression loads {} inputs, launch has {}", prog.n_inputs(), ins.len()),
        ));
    }

    // Outer dims collapsed across the inputs (the output is contiguous over them).
    let outer_strides: Vec<&[isize]> = dims.in_views.iter().map(|v| &v.strides[..rank - 1]).collect();
    let co = collapse(outer, &outer_strides);
    let bases: Vec<usize> = dims.in_views.iter().map(|v| v.offset).collect();
    let inner: Vec<isize> = dims.in_views.iter().map(|v| v.strides[rank - 1]).collect();
    let cost = rows * len * c.cost;
    let parallel = cost >= PAR_MIN_COST && ctx.threads > 1;
    let nin = ins.len();

    if len <= COL_MAX {
        let blocks = rows.div_ceil(CHUNK);
        let linear = co.shape.len() == 1;
        let init = || {
            let mut s = Scratch::default();
            prog.prepare(&mut s.regs);
            s.acc = vec![0.0; CHUNK];
            s.offs = vec![Vec::with_capacity(CHUNK); nin];
            s
        };
        par::for_each(parallel, blocks, init, |s, blk| {
            let r0 = blk * CHUNK;
            let nb = CHUNK.min(rows - r0);
            let Scratch { regs, acc, idx, cur, offs, .. } = s;
            let acc = &mut acc[..nb];
            acc.fill(identity);
            if !linear {
                fill_offsets(r0, nb, &co.shape, &bases, &co.strides, idx, cur, offs);
            }
            for j in 0..len {
                for (i, src) in ins.iter().enumerate() {
                    if let Some(reg) = prog.input(regs, i) {
                        let jo = j as isize * inner[i];
                        if linear {
                            let os = co.strides[i][0];
                            src.load((bases[i] as isize + r0 as isize * os + jo) as usize, os, &mut reg[..nb]);
                        } else {
                            for (o, r) in offs[i].iter().zip(reg[..nb].iter_mut()) {
                                *r = src.get((*o as isize + jo) as usize);
                            }
                        }
                    }
                }
                let v = prog.run(regs, nb);
                match op {
                    ReduceOp::Sum => acc.iter_mut().zip(v).for_each(|(a, &x)| *a += x),
                    ReduceOp::Max => acc.iter_mut().zip(v).for_each(|(a, &x)| *a = a.max(x)),
                }
            }
            // SAFETY: blocks write disjoint output rows.
            unsafe { out.store(r0, 1, acc) };
        });
        return Ok(());
    }

    let nseg = if parallel && rows < ctx.threads * 2 && len >= 2 * MIN_SEGMENT {
        (ctx.threads * 2).div_ceil(rows).min(len / MIN_SEGMENT).max(1)
    } else {
        1
    };
    let seg_len = len.div_ceil(nseg).div_ceil(CHUNK) * CHUNK;
    let nseg = len.div_ceil(seg_len);
    // Rows per task (segmented rows are one task per segment).
    let per = if nseg > 1 { 1 } else { (TASK_ELEMS / len).max(1) };
    let tasks = if nseg > 1 { rows * nseg } else { rows.div_ceil(per) };
    let init = || {
        let mut s = Scratch::default();
        prog.prepare(&mut s.regs);
        s.bases = vec![0; nin];
        s
    };
    let row_sum = |s: &mut Scratch, row: usize, j0: usize, j1: usize| -> f32 {
        let Scratch { regs, bases: rb, .. } = s;
        for (i, b) in rb.iter_mut().enumerate() {
            *b = bases[i] as isize + offset_of(row, &co.shape, &co.strides[i]);
        }
        let mut acc = identity;
        let mut j = j0;
        while j < j1 {
            let n = CHUNK.min(j1 - j);
            for (i, src) in ins.iter().enumerate() {
                if let Some(reg) = prog.input(regs, i) {
                    src.load((rb[i] + j as isize * inner[i]) as usize, inner[i], &mut reg[..n]);
                }
            }
            let v = prog.run(regs, n);
            acc = match op {
                ReduceOp::Sum => acc + sum8(v),
                ReduceOp::Max => acc.max(max8(v)),
            };
            j += n;
        }
        acc
    };
    if nseg == 1 {
        // Rows of at most half a chunk are packed `rpu` per program run.
        let rpu = if 2 * len <= CHUNK { CHUNK / len } else { 1 };
        par::for_each(parallel, tasks, init, |s, t| {
            let r0 = t * per;
            let r1 = (r0 + per).min(rows);
            let mut res = [0f32; 64];
            let mut r = r0;
            while r < r1 {
                let n = (r1 - r).min(if rpu > 1 { rpu.min(64) } else { 64 });
                if rpu > 1 {
                    let regs = &mut s.regs;
                    for (i, src) in ins.iter().enumerate() {
                        if let Some(reg) = prog.input(regs, i) {
                            for q in 0..n {
                                let b = bases[i] as isize + offset_of(r + q, &co.shape, &co.strides[i]);
                                src.load(b as usize, inner[i], &mut reg[q * len..(q + 1) * len]);
                            }
                        }
                    }
                    let v = prog.run(regs, n * len);
                    for q in 0..n {
                        let row = &v[q * len..(q + 1) * len];
                        res[q] = match op {
                            ReduceOp::Sum => identity + sum8(row),
                            ReduceOp::Max => identity.max(max8(row)),
                        };
                    }
                } else {
                    for k in 0..n {
                        res[k] = row_sum(s, r + k, 0, len);
                    }
                }
                // SAFETY: tasks write disjoint output rows.
                unsafe { out.store(r, 1, &res[..n]) };
                r += n;
            }
        });
        return Ok(());
    }
    let partials = par::map(parallel, tasks, init, |s, u| {
        let (row, seg) = (u / nseg, u % nseg);
        let j0 = seg * seg_len;
        row_sum(s, row, j0, (j0 + seg_len).min(len))
    });
    let result: Vec<f32> = partials
        .chunks(nseg)
        .map(|p| match op {
            ReduceOp::Sum => p.iter().sum(),
            ReduceOp::Max => p.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x)),
        })
        .collect();
    // SAFETY: single-threaded store of `rows` validated elements.
    unsafe { out.store(0, 1, &result) };
    Ok(())
}
