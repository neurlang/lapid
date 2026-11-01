//! Indexing kernels: take_along_axis (`GatherAxis`) and row top-k (`TopK`).

use lapid_compiler::{GatherAxisDims, TopKDims, View};
use lapid_tensor::{Error, Result};

use crate::Ctx;
use crate::par::{self, PAR_MIN_ELEMS};
use crate::typed::{self, Dst, Src};

fn at(v: &View, a: usize, b: usize, c: usize) -> isize {
    v.offset as isize + a as isize * v.strides[0] + b as isize * v.strides[1] + c as isize * v.strides[2]
}

pub(crate) fn gather_axis(ctx: &Ctx, d: &GatherAxisDims, x: &Src, idx: &Src, out: &Dst) -> Result<()> {
    let (outer, n, inner) = (d.outer, d.n, d.inner);
    let total = outer * n * inner;
    if total == 0 {
        return Ok(());
    }
    if d.x.strides.len() != 3 || d.idx.strides.len() != 3 {
        return Err(Error::new("invalid_launch", "cpu gather_axis: views must be rank 3"));
    }
    let Src::I32(ids) = idx else {
        return Err(Error::new("unsupported_dtype", "cpu gather_axis: indices must be i32"));
    };
    if out.len() < total || x.dtype() != out.dtype() {
        return Err(Error::new("invalid_launch", "cpu gather_axis: output too small or dtype mismatch"));
    }
    // Validate every index read and the x elements it selects before writing anything.
    let in_buf = |o: isize, len: usize| o >= 0 && (o as usize) < len;
    for o in 0..outer {
        for j in 0..n {
            for i in 0..inner {
                let io = at(&d.idx, o, j, i);
                if !in_buf(io, ids.len()) {
                    return Err(Error::new("out_of_bounds", "cpu gather_axis: index view outside its buffer"));
                }
                let v = ids[io as usize];
                if v < 0 || v as usize >= d.axis_len {
                    return Err(Error::new(
                        "index_out_of_range",
                        format!("cpu gather_axis: index {v} out of range for axis of {}", d.axis_len),
                    ));
                }
                if !in_buf(at(&d.x, o, v as usize, i), x.len()) {
                    return Err(Error::new("out_of_bounds", "cpu gather_axis: input view outside its buffer"));
                }
            }
        }
    }
    let rows = outer * n;
    par::for_each(total >= PAR_MIN_ELEMS && ctx.threads > 1, rows, || (Vec::new(), Vec::new()), |(so, dof), r| {
        let (o, j) = (r / n, r % n);
        let so: &mut Vec<usize> = so;
        let dof: &mut Vec<usize> = dof;
        so.clear();
        dof.clear();
        for i in 0..inner {
            let v = ids[at(&d.idx, o, j, i) as usize] as usize;
            so.push(at(&d.x, o, v, i) as usize);
            dof.push(r * inner + i);
        }
        // SAFETY: rows write disjoint output ranges; offsets validated above.
        unsafe { typed::copy_offsets(x, so, out, dof) };
    });
    Ok(())
}

pub(crate) fn topk(ctx: &Ctx, d: &TopKDims, x: &Src, out: &Dst) -> Result<()> {
    let (rows, cols, k) = (d.rows, d.cols, d.k);
    if rows == 0 {
        return Ok(());
    }
    if k == 0 || k > cols {
        return Err(Error::new("invalid_launch", format!("cpu topk: k = {k} with {cols} columns")));
    }
    if x.len() < rows * cols || out.len() < rows * k {
        return Err(Error::new("out_of_bounds", "cpu topk: buffer too small"));
    }
    let parallel = rows * cols >= PAR_MIN_ELEMS && ctx.threads > 1 && rows > 1;
    par::for_each(parallel, rows, || (vec![0f32; cols], Vec::<u32>::new(), vec![0f32; k]), |(vals, order, res), r| {
        x.load_contig(r * cols, vals);
        // Descending by value, ties by index; NaN sorts last.
        let key = |a: &u32, b: &u32| {
            let (va, vb) = (vals[*a as usize], vals[*b as usize]);
            vb.partial_cmp(&va).unwrap_or_else(|| va.is_nan().cmp(&vb.is_nan())).then(a.cmp(b))
        };
        order.clear();
        if k == 1 {
            let mut best = 0u32;
            for c in 1..cols as u32 {
                if key(&c, &best).is_lt() {
                    best = c;
                }
            }
            order.push(best);
        } else {
            order.extend(0..cols as u32);
            if k < cols {
                order.select_nth_unstable_by(k - 1, key);
                order.truncate(k);
            }
            order.sort_unstable_by(key);
        }
        for (o, &c) in res.iter_mut().zip(order.iter()) {
            *o = f32::from_bits(c);
        }
        // Stored as raw words through an f32 view of the i32 output (see lib.rs dispatch).
        // SAFETY: each row writes its own k outputs.
        unsafe { out.store(r * k, 1, res) };
    });
    Ok(())
}
