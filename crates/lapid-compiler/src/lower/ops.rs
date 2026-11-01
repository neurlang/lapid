//! Lowering of the buffer-producing primitives: Reduce, MatMul, Gather, Pad, Cat, Contiguous.

use std::sync::Arc;

use lapid_tensor::{DType, ReduceOp, Result, Tensor};

use super::lexpr::{LExpr, Leaf, Loads, Src};
use super::{
    Lowerer, MatMulPlan, PendKind, PendState, Pending, internal, op_name, shape_mismatch, src_of, unsupported, view,
};
use crate::ir::{BufId, GatherDims, GatherSpec, Kernel, MatMulDims, MatView, View};

impl Lowerer {
    // ---------------------------------------------------------------- reduce

    pub(super) fn lower_reduce(&mut self, t: &Tensor, op: ReduceOp, axis: usize) -> Result<LExpr> {
        let s = src_of(t, 0)?;
        let r = s.rank();
        if axis >= r {
            return Err(shape_mismatch(format!("reduce axis {axis} of rank-{r} node #{}", s.id())));
        }
        if s.dtype().is_quant() {
            return Err(unsupported(format!("reduce of quantized node #{}", s.id())));
        }
        let e = self.val(s)?;
        // Iteration space: input dims with the reduced axis moved last (the prologue is the
        // fused expression of the input, its loads permuted accordingly).
        let perm: Vec<usize> = (0..r).filter(|&d| d != axis).chain(std::iter::once(axis)).collect();
        let pe = e.map_views(&mut |v| view::permute(v, &perm)).ok_or_else(|| internal("reduce permutation"))?;
        let shape: Vec<usize> = perm.iter().map(|&p| s.shape()[p]).collect();
        let out = self.alloc_node(t)?;
        self.emit_reduce(op, &pe, &shape, out, t.dtype())?;
        Ok(LExpr::buf(out, t.dtype(), View::contiguous(t.shape())))
    }

    // ---------------------------------------------------------------- matmul

    pub(super) fn lower_matmul(&mut self, t: &Tensor, b_transposed: bool) -> Result<LExpr> {
        let sa = src_of(t, 0)?;
        let sb = src_of(t, 1)?;
        let (ra, rb, r) = (sa.rank(), sb.rank(), t.rank());
        if ra < 2 || rb < 2 || r < 2 {
            return Err(shape_mismatch(format!(
                "matmul node #{} needs rank >= 2 operands (a {:?}, b {:?})",
                t.id(),
                sa.shape(),
                sb.shape()
            )));
        }
        let (m, k) = (sa.shape()[ra - 2], sa.shape()[ra - 1]);
        let n = t.shape()[r - 1];
        let (bk, bn) = if b_transposed {
            (sb.shape()[rb - 1], sb.shape()[rb - 2])
        } else {
            (sb.shape()[rb - 2], sb.shape()[rb - 1])
        };
        if bk != k || bn != n || t.shape()[r - 2] != m {
            return Err(shape_mismatch(format!(
                "matmul node #{}: a {:?}, b {:?} (transposed: {b_transposed}), out {:?}",
                t.id(),
                sa.shape(),
                sb.shape(),
                t.shape()
            )));
        }
        if sa.dtype().is_quant() {
            return Err(unsupported(format!("quantized matmul lhs (node #{})", sa.id())));
        }
        let out_batch = t.shape()[..r - 2].to_vec();
        let batch = view::numel(&out_batch);
        let (a, a_dtype, av) = self.matmul_a(sa, &out_batch, m, k)?;
        let (b, b_dtype, bv, bt) = self.matmul_b(sb, &out_batch, k, n, b_transposed)?;
        let plan = MatMulPlan {
            a,
            b,
            a_dtype,
            b_dtype,
            b_transposed: bt,
            dims: MatMulDims { batch, m, n, k, a: av, b: bv, extra_views: Vec::new(), out: MatView::contiguous(m, n) },
        };
        if self.opts.fuse && self.users_of(t) == 1 && !self.is_output(t) {
            // Defer: the consumer may become the epilogue (see emit::try_epilogue).
            let id = self.pendings.len();
            self.pendings.push(Pending {
                kind: PendKind::MatMul(plan),
                out_shape: t.shape().to_vec(),
                out_dtype: t.dtype(),
                state: PendState::Waiting,
            });
            return Ok(LExpr::Leaf(Leaf { src: Src::Pending(id), dtype: t.dtype(), view: View::contiguous(t.shape()) }));
        }
        let out = self.alloc_node(t)?;
        self.emit_matmul(&plan, out, t.dtype(), None, Loads::new(1, None));
        Ok(LExpr::buf(out, t.dtype(), View::contiguous(t.shape())))
    }

    /// A operand: unit stride along k, one batch stride. Otherwise a dense copy.
    fn matmul_a(&mut self, sa: &Tensor, out_batch: &[usize], m: usize, k: usize) -> Result<(BufId, DType, MatView)> {
        let e = self.val(sa)?;
        let (buf, dt, v) = self.to_buffer(&e, sa.shape(), sa.dtype())?;
        let mut full = out_batch.to_vec();
        full.extend_from_slice(&[m, k]);
        let fr = full.len();
        if dt.is_float()
            && let Some(av) = view::broadcast_to(&v, sa.shape(), &full)
                && (k <= 1 || av.strides[fr - 1] == 1) {
                    let bs = view::flatten_batch(out_batch, &av.strides[..fr - 2]);
                    let row = usize::try_from(av.strides[fr - 2]).ok();
                    if let (Some(batch_stride), Some(row_stride)) = (bs, row) {
                        return Ok((buf, dt, MatView { offset: av.offset, batch_stride, row_stride }));
                    }
                }
        let (tmp, cdt) = self.dense_operand(buf, dt, &v, sa, &full)?;
        Ok((tmp, cdt, MatView { offset: 0, batch_stride: m * k, row_stride: k }))
    }

    /// B operand. Picks the transposed / plain form from the strides (a permuted weight
    /// needs no copy), handles batch broadcast (stride 0) and quantized weights.
    fn matmul_b(
        &mut self,
        sb: &Tensor,
        out_batch: &[usize],
        k: usize,
        n: usize,
        transposed: bool,
    ) -> Result<(BufId, DType, MatView, bool)> {
        let e = self.val(sb)?;
        if sb.dtype().is_quant() {
            if !transposed || sb.rank() != 2 {
                return Err(unsupported(format!(
                    "quantized matmul weight (node #{}) must be 2-D and used as a @ b^T",
                    sb.id()
                )));
            }
            return match e.as_buf() {
                Some((b, dt, v)) if dt.is_quant() && view::is_contiguous(v, sb.shape()) => {
                    Ok((b, dt, MatView { offset: 0, batch_stride: 0, row_stride: k }, true))
                }
                _ => Err(unsupported(format!(
                    "quantized matmul weight (node #{}) must be a host/device leaf",
                    sb.id()
                ))),
            };
        }
        let (buf, dt, v) = self.to_buffer(&e, sb.shape(), sb.dtype())?;
        let rb = sb.rank();
        let mut full = out_batch.to_vec();
        full.extend_from_slice(&sb.shape()[rb - 2..]);
        let fr = full.len();
        if dt.is_float()
            && let Some(bv) = view::broadcast_to(&v, sb.shape(), &full) {
                let (sk, sn) = if transposed {
                    (bv.strides[fr - 1], bv.strides[fr - 2])
                } else {
                    (bv.strides[fr - 2], bv.strides[fr - 1])
                };
                if let Some(batch_stride) = view::flatten_batch(out_batch, &bv.strides[..fr - 2]) {
                    let k_unit = k <= 1 || sk == 1;
                    let n_unit = n <= 1 || sn == 1;
                    // Prefer the requested form; fall back to the other one if it fits.
                    let form = match (transposed, k_unit, n_unit) {
                        (true, true, _) => Some(true),
                        (true, false, true) => Some(false),
                        (false, _, true) => Some(false),
                        (false, true, false) => Some(true),
                        _ => None,
                    };
                    if let Some(tr) = form {
                        let row = if tr { sn } else { sk };
                        if let Ok(row_stride) = usize::try_from(row) {
                            return Ok((buf, dt, MatView { offset: bv.offset, batch_stride, row_stride }, tr));
                        }
                    }
                }
            }
        let (tmp, cdt) = self.dense_operand(buf, dt, &v, sb, &full)?;
        let (r0, r1) = (full[fr - 2], full[fr - 1]);
        Ok((tmp, cdt, MatView { offset: 0, batch_stride: r0 * r1, row_stride: r1 }, transposed))
    }

    /// Row-major copy of an operand broadcast to `full` (= out batch dims + its 2 matrix
    /// dims), converted to f32 if its storage is not float.
    fn dense_operand(&mut self, buf: BufId, dt: DType, v: &View, s: &Tensor, full: &[usize]) -> Result<(BufId, DType)> {
        let bv = view::broadcast_to(v, s.shape(), full).ok_or_else(|| {
            shape_mismatch(format!("matmul operand {:?} does not broadcast to {:?}", s.shape(), full))
        })?;
        let cdt = if dt.is_float() { dt } else { DType::F32 };
        let tmp = self.new_temp(cdt, full)?;
        self.write_dense(&LExpr::buf(buf, dt, bv), full, cdt, tmp, &View::contiguous(full))?;
        Ok((tmp, cdt))
    }

    // ---------------------------------------------------------------- gather

    pub(super) fn lower_gather(&mut self, t: &Tensor) -> Result<LExpr> {
        let tab = src_of(t, 0)?;
        let idx = src_of(t, 1)?;
        if tab.rank() != 2 {
            return Err(unsupported(format!("gather table must be 2-D, got {:?}", tab.shape())));
        }
        let (rows, cols) = (tab.shape()[0], tab.shape()[1]);
        let tv = self.val(tab)?;
        let (tb, tdt) = if tab.dtype().is_quant() {
            match tv.as_buf() {
                Some((b, dt, v)) if view::is_contiguous(v, tab.shape()) => (b, dt),
                _ => {
                    return Err(unsupported(format!(
                        "quantized gather table (node #{}) must be a host/device leaf",
                        tab.id()
                    )));
                }
            }
        } else {
            self.to_contig(&tv, tab.shape(), tab.dtype())?
        };
        let iv = self.val(idx)?;
        let ib = match iv.as_buf() {
            Some((b, DType::I32, v)) if view::is_contiguous(v, idx.shape()) => b,
            _ => self.materialize_contig(&iv, idx.shape(), DType::I32)?,
        };
        let count = idx.numel();
        let out = self.alloc_node(t)?;
        if count * cols > 0 {
            let spec = GatherSpec { table_dtype: tdt, out_dtype: t.dtype() };
            self.push(Kernel::Gather(Arc::new(spec), GatherDims { rows, cols, count }), vec![tb, ib], out);
        }
        Ok(LExpr::buf(out, t.dtype(), View::contiguous(t.shape())))
    }

    // ---------------------------------------------------------------- pad / cat / contiguous

    pub(super) fn lower_pad(
        &mut self,
        t: &Tensor,
        axis: usize,
        before: usize,
        after: usize,
        value: f64,
    ) -> Result<LExpr> {
        let s = src_of(t, 0)?;
        let shape = t.shape().to_vec();
        if axis >= shape.len() || s.rank() != shape.len() || shape[axis] != s.shape()[axis] + before + after {
            return Err(shape_mismatch(format!(
                "pad axis {axis} ({before}, {after}) of {:?} -> {:?}",
                s.shape(),
                shape
            )));
        }
        if s.dtype().is_quant() {
            return Err(unsupported(format!("pad of quantized node #{}", s.id())));
        }
        let e = self.val(s)?;
        let out = self.alloc_node(t)?;
        let cv = View::contiguous(&shape);
        let st = cv.strides[axis] as usize;
        let fill = LExpr::Const(value as f32);
        let mut region = shape.clone();
        if before > 0 {
            region[axis] = before;
            self.emit_ew(&fill, &region, out, t.dtype(), cv.clone())?;
        }
        if after > 0 {
            region[axis] = after;
            let off = (before + s.shape()[axis]) * st;
            self.emit_ew(&fill, &region, out, t.dtype(), View { offset: off, strides: cv.strides.clone() })?;
        }
        // Interior: the source expression (fused) written through a strided output view.
        let interior = View { offset: before * st, strides: cv.strides.clone() };
        self.emit_ew(&e, s.shape(), out, t.dtype(), interior)?;
        Ok(LExpr::buf(out, t.dtype(), cv))
    }

    pub(super) fn lower_cat(&mut self, t: &Tensor, axis: usize) -> Result<LExpr> {
        let shape = t.shape().to_vec();
        if axis >= shape.len() {
            return Err(shape_mismatch(format!("cat axis {axis} of rank-{} output", shape.len())));
        }
        let mut total = 0usize;
        for s in t.src() {
            let ok = s.rank() == shape.len()
                && (0..shape.len()).all(|d| d == axis || s.shape()[d] == shape[d]);
            if !ok {
                return Err(shape_mismatch(format!("cat piece {:?} vs output {:?} (axis {axis})", s.shape(), shape)));
            }
            if s.dtype().is_quant() {
                return Err(unsupported(format!("cat of quantized node #{}", s.id())));
            }
            total += s.shape()[axis];
        }
        if total != shape[axis] {
            return Err(shape_mismatch(format!("cat pieces sum to {total} along axis {axis}, output has {}", shape[axis])));
        }
        if let Some(v) = self.cat_alias(t, axis)? {
            return Ok(v);
        }
        if self.opts.fuse && self.users_of(t) == 1 && !self.is_output(t) {
            // Defer: the consumer may write the pieces straight into its own region.
            let pieces = t.src().iter().map(|s| Ok((self.val(s)?, s.shape().to_vec()))).collect::<Result<Vec<_>>>()?;
            let id = self.pendings.len();
            self.pendings.push(Pending {
                kind: PendKind::Cat { axis, pieces },
                out_shape: shape.clone(),
                out_dtype: t.dtype(),
                state: PendState::Waiting,
            });
            return Ok(LExpr::Leaf(Leaf { src: Src::Pending(id), dtype: t.dtype(), view: View::contiguous(&shape) }));
        }
        let out = self.alloc_node(t)?;
        let cv = View::contiguous(&shape);
        let st = cv.strides[axis] as usize;
        let mut at = 0usize;
        for s in t.src() {
            let e = self.val(s)?;
            let piece = View { offset: at * st, strides: cv.strides.clone() };
            self.emit_ew(&e, s.shape(), out, t.dtype(), piece)?;
            at += s.shape()[axis];
        }
        Ok(LExpr::buf(out, t.dtype(), cv))
    }

    /// A `Cat` whose pieces are adjacent views of one buffer (e.g. the taps of a depthwise
    /// conv sliced from one window) is itself a view of that buffer: no copy.
    fn cat_alias(&self, t: &Tensor, axis: usize) -> Result<Option<LExpr>> {
        let mut pieces = Vec::with_capacity(t.src().len());
        for s in t.src() {
            let e = self.val(s)?;
            match e.as_buf() {
                Some((b, dt, v)) if dt == t.dtype() && v.strides.len() == t.rank() => pieces.push((b, v.clone(), s.shape()[axis])),
                _ => return Ok(None),
            }
        }
        let Some((b0, v0, _)) = pieces.first().cloned() else { return Ok(None) };
        // Stride along `axis`: from a piece longer than 1, else from the first offset step.
        let st = match pieces.iter().find(|p| p.2 > 1) {
            Some(p) => p.1.strides[axis],
            None if pieces.len() > 1 => pieces[1].1.offset as isize - v0.offset as isize,
            None => return Ok(None),
        };
        let mut at = 0isize;
        for (b, v, len) in &pieces {
            let same_strides =
                v.strides.iter().zip(&v0.strides).enumerate().all(|(d, (x, y))| d == axis || t.shape()[d] <= 1 || x == y);
            let at_ok = v.offset as isize == v0.offset as isize + at * st;
            if *b != b0 || !same_strides || !at_ok || (*len > 1 && v.strides[axis] != st) {
                return Ok(None);
            }
            at += *len as isize;
        }
        if st < 0 {
            return Ok(None);
        }
        let mut strides = v0.strides.clone();
        strides[axis] = st;
        Ok(Some(LExpr::buf(b0, t.dtype(), View { offset: v0.offset, strides })))
    }

    pub(super) fn lower_contiguous(&mut self, t: &Tensor) -> Result<LExpr> {
        let s = src_of(t, 0)?;
        let e = self.val(s)?;
        let already = matches!(e, LExpr::Const(_))
            || e.as_buf().is_some_and(|(_, _, v)| view::is_contiguous(v, t.shape()));
        if already {
            return Ok(e);
        }
        if s.dtype().is_quant() {
            return Err(unsupported(format!("{} of a quantized view (node #{})", op_name(t.op()), s.id())));
        }
        let out = self.alloc_node(t)?;
        let cv = View::contiguous(t.shape());
        self.write_dense(&e, t.shape(), t.dtype(), out, &cv)?;
        Ok(LExpr::buf(out, t.dtype(), cv))
    }
}
