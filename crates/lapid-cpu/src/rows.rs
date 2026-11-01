//! Row kernels: softmax, layer norm and gather (embedding lookup / dequant rows).

use lapid_compiler::{GatherDims, GatherSpec, LayerNormDims, LayerNormSpec, RowDims, SoftmaxSpec};
use lapid_tensor::{Error, Result};

use crate::math::{self, max8, sum8};
use crate::par::{self, PAR_MIN_ELEMS};
use crate::qmat::QMat;
use crate::typed::{self, Dst, Src};
use crate::Ctx;

/// Rows per task so that a task touches roughly this many elements.
const TASK_ELEMS: usize = 8192;

fn need(what: &str, have: usize, want: usize) -> Result<()> {
    if have < want {
        return Err(Error::new("out_of_bounds", format!("cpu {what}: buffer holds {have} < {want} elements")));
    }
    Ok(())
}

fn row_tasks(rows: usize, cols: usize) -> (usize, usize) {
    let per = (TASK_ELEMS / cols.max(1)).max(1);
    (per, rows.div_ceil(per))
}

/// Numerically stable softmax of one row, in place. Rows that are entirely -inf (fully
/// masked) produce zeros instead of NaN.
pub(crate) fn softmax_row(v: &mut [f32]) {
    let m = max8(v);
    if m == f32::NEG_INFINITY {
        v.fill(0.0);
        return;
    }
    for x in v.iter_mut() {
        *x = math::exp_neg(*x - m);
    }
    let inv = 1.0 / sum8(v);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

pub(crate) fn softmax(ctx: &Ctx, _spec: &SoftmaxSpec, dims: &RowDims, x: &Src, out: &Dst) -> Result<()> {
    let (rows, cols) = (dims.rows, dims.cols);
    let n = rows * cols;
    if n == 0 {
        return Ok(());
    }
    need("softmax input", x.len(), n)?;
    need("softmax output", out.len(), n)?;
    let (per, tasks) = row_tasks(rows, cols);
    par::for_each(n >= PAR_MIN_ELEMS && ctx.threads > 1, tasks, || vec![0f32; cols], |buf, t| {
        for r in t * per..((t + 1) * per).min(rows) {
            x.load_contig(r * cols, buf);
            softmax_row(buf);
            // SAFETY: each task writes its own rows.
            unsafe { out.store(r * cols, 1, buf) };
        }
    });
    Ok(())
}

pub(crate) fn layer_norm(
    ctx: &Ctx,
    _spec: &LayerNormSpec,
    dims: &LayerNormDims,
    x: &Src,
    w: &Src,
    b: &Src,
    out: &Dst,
) -> Result<()> {
    let (rows, cols) = (dims.rows, dims.cols);
    let n = rows * cols;
    if n == 0 {
        return Ok(());
    }
    need("layer_norm input", x.len(), n)?;
    need("layer_norm weight", w.len(), cols)?;
    need("layer_norm bias", b.len(), cols)?;
    need("layer_norm output", out.len(), n)?;
    let eps = f32::from_bits(dims.eps);
    let mut wf = vec![0f32; cols];
    let mut bf = vec![0f32; cols];
    w.load_contig(0, &mut wf);
    b.load_contig(0, &mut bf);
    let inv_n = 1.0 / cols as f32;
    let (per, tasks) = row_tasks(rows, cols);
    par::for_each(n >= PAR_MIN_ELEMS && ctx.threads > 1, tasks, || vec![0f32; cols], |buf, t| {
        for r in t * per..((t + 1) * per).min(rows) {
            x.load_contig(r * cols, buf);
            let mean = sum8(buf) * inv_n;
            // Second pass over centered values: exact two-pass variance.
            let mut acc = [0f32; 8];
            let mut chunks = buf.chunks_exact(8);
            for c in &mut chunks {
                for l in 0..8 {
                    let d = c[l] - mean;
                    acc[l] = math::fmadd(d, d, acc[l]);
                }
            }
            for (l, &v) in chunks.remainder().iter().enumerate() {
                let d = v - mean;
                acc[l] += d * d;
            }
            let var = acc.iter().sum::<f32>() * inv_n;
            let rstd = 1.0 / (var + eps).sqrt();
            for ((v, &g), &bb) in buf.iter_mut().zip(&wf).zip(&bf) {
                *v = math::fmadd((*v - mean) * rstd, g, bb);
            }
            // SAFETY: each task writes its own rows.
            unsafe { out.store(r * cols, 1, buf) };
        }
    });
    Ok(())
}

/// x * rsqrt(mean(x^2) + eps) * w per row.
pub(crate) fn rms_norm(ctx: &Ctx, dims: &LayerNormDims, x: &Src, w: &Src, out: &Dst) -> Result<()> {
    let (rows, cols) = (dims.rows, dims.cols);
    let n = rows * cols;
    if n == 0 {
        return Ok(());
    }
    need("rms_norm input", x.len(), n)?;
    need("rms_norm weight", w.len(), cols)?;
    need("rms_norm output", out.len(), n)?;
    let eps = f32::from_bits(dims.eps);
    let mut wf = vec![0f32; cols];
    w.load_contig(0, &mut wf);
    let inv_n = 1.0 / cols as f32;
    let (per, tasks) = row_tasks(rows, cols);
    par::for_each(n >= PAR_MIN_ELEMS && ctx.threads > 1, tasks, || vec![0f32; cols], |buf, t| {
        for r in t * per..((t + 1) * per).min(rows) {
            x.load_contig(r * cols, buf);
            let mut acc = [0f32; 8];
            let mut chunks = buf.chunks_exact(8);
            for c in &mut chunks {
                for l in 0..8 {
                    acc[l] = math::fmadd(c[l], c[l], acc[l]);
                }
            }
            for (l, &v) in chunks.remainder().iter().enumerate() {
                acc[l] += v * v;
            }
            let ms = acc.iter().sum::<f32>() * inv_n;
            let rstd = 1.0 / (ms + eps).sqrt();
            for (v, &g) in buf.iter_mut().zip(&wf) {
                *v = *v * rstd * g;
            }
            // SAFETY: each task writes its own rows.
            unsafe { out.store(r * cols, 1, buf) };
        }
    });
    Ok(())
}

/// Gather table: plain typed rows or a quantized matrix.
pub(crate) enum Table<'a> {
    Plain(Src<'a>),
    Quant(QMat<'a>),
}

pub(crate) fn gather(
    ctx: &Ctx,
    _spec: &GatherSpec,
    dims: &GatherDims,
    table: &Table,
    idx: &Src,
    out: &Dst,
) -> Result<()> {
    let (rows, cols, count) = (dims.rows, dims.cols, dims.count);
    if count == 0 || cols == 0 {
        return Ok(());
    }
    let Src::I32(ids) = idx else {
        return Err(Error::new(
            "unsupported_dtype",
            format!("cpu gather: indices must be i32, got {}", idx.dtype().name()),
        ));
    };
    need("gather indices", ids.len(), count)?;
    need("gather output", out.len(), count * cols)?;
    match table {
        Table::Plain(s) => need("gather table", s.len(), rows * cols)?,
        Table::Quant(q) => {
            if q.rows != rows || q.cols != cols {
                return Err(Error::new("invalid_launch", "cpu gather: quantized table shape mismatch"));
            }
        }
    }
    let ids = &ids[..count];
    if let Some((i, &bad)) = ids.iter().enumerate().find(|&(_, &v)| v < 0 || v as usize >= rows) {
        return Err(Error::new(
            "index_out_of_range",
            format!("cpu gather: index {bad} at position {i} out of range for {rows} rows"),
        ));
    }
    let (per, tasks) = row_tasks(count, cols);
    par::for_each(count * cols >= PAR_MIN_ELEMS && ctx.threads > 1, tasks, || vec![0f32; cols], |buf, t| {
        for i in t * per..((t + 1) * per).min(count) {
            let r = ids[i] as usize;
            // SAFETY: each task writes its own output rows.
            unsafe {
                match table {
                    Table::Plain(s) => {
                        if !typed::copy_run(s, r * cols, 1, out, i * cols, 1, cols) {
                            s.load_contig(r * cols, buf);
                            out.store(i * cols, 1, buf);
                        }
                    }
                    Table::Quant(q) => {
                        q.dequant_row(r, buf);
                        out.store(i * cols, 1, buf);
                    }
                }
            }
        }
    });
    Ok(())
}
