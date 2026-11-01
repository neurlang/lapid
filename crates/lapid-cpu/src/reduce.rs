//! Reduce kernel: `out[o] = reduce_j expr(inputs at (o, j))`, reduced axis last.
//!
//! Rows are processed in parallel. When there are fewer rows than threads and rows are long,
//! each row is split into segments whose partial results are combined afterwards. Sums use
//! 8-lane partials per chunk accumulated blockwise in f32.

use lapid_compiler::{ReduceDims, ReduceSpec};
use lapid_tensor::{Error, ReduceOp, Result};

use crate::expr::{CHUNK, Regs};
use crate::math::{max8, sum8};
use crate::par::{self, PAR_MIN_ELEMS};
use crate::typed::{Dst, Src};
use crate::view::{check_view, offset_of};
use crate::Ctx;

/// Minimum segment length when splitting a row across threads.
const MIN_SEGMENT: usize = 8 * CHUNK;

pub(crate) fn run(ctx: &Ctx, spec: &ReduceSpec, dims: &ReduceDims, ins: &[Src], out: &Dst, raw: &[bool]) -> Result<()> {
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
    let identity = match spec.op {
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
    let prog = ctx.program_typed(&spec.expr, &spec.in_dtypes, raw);
    if prog.n_inputs() > ins.len() {
        return Err(Error::new(
            "invalid_launch",
            format!("cpu reduce: expression loads {} inputs, launch has {}", prog.n_inputs(), ins.len()),
        ));
    }

    let total = rows * len;
    let parallel = total >= PAR_MIN_ELEMS && ctx.threads > 1;
    let nseg = if parallel && rows < ctx.threads * 2 && len >= 2 * MIN_SEGMENT {
        (ctx.threads * 2).div_ceil(rows).min(len / MIN_SEGMENT).max(1)
    } else {
        1
    };
    let seg_len = len.div_ceil(nseg).div_ceil(CHUNK) * CHUNK;
    let nseg = len.div_ceil(seg_len);
    let op = spec.op;

    let partials = par::map(
        parallel,
        rows * nseg,
        || {
            let mut r = Regs::default();
            prog.prepare(&mut r);
            r
        },
        |regs, u| {
            let row = u / nseg;
            let seg = u % nseg;
            let j0 = seg * seg_len;
            let j1 = (j0 + seg_len).min(len);
            let bases: Vec<(isize, isize)> = dims
                .in_views
                .iter()
                .map(|v| {
                    let inner = v.strides[rank - 1];
                    (v.offset as isize + offset_of(row, outer, &v.strides[..rank - 1]), inner)
                })
                .collect();
            let mut acc = identity;
            let mut j = j0;
            while j < j1 {
                let n = CHUNK.min(j1 - j);
                for (i, src) in ins.iter().enumerate() {
                    if let Some(reg) = prog.input(regs, i) {
                        let (b, s) = bases[i];
                        src.load((b + j as isize * s) as usize, s, &mut reg[..n]);
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
        },
    );

    let result: Vec<f32> = if nseg == 1 {
        partials
    } else {
        partials
            .chunks(nseg)
            .map(|p| match op {
                ReduceOp::Sum => p.iter().sum(),
                ReduceOp::Max => p.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x)),
            })
            .collect()
    };
    // SAFETY: single-threaded store of `rows` validated elements.
    unsafe { out.store(0, 1, &result) };
    Ok(())
}
