//! Lowering of the indexing ops (GatherAxis, TopK) and the in-place slice write.

use std::sync::Arc;

use lapid_tensor::{DType, Op, Result, Tensor};

use super::lexpr::LExpr;
use super::{BufKind, Lowerer, bytes_of, internal, shape_mismatch, src_of, unsupported, view};
use crate::ir::{BufId, GatherAxisDims, GatherAxisSpec, Kernel, TopKDims, TopKSpec, View};

/// `[outer, len, inner]` split of `shape` around `axis`.
fn split3(shape: &[usize], axis: usize) -> [usize; 3] {
    [view::numel(&shape[..axis]), shape[axis], view::numel(&shape[axis + 1..])]
}

impl Lowerer {
    // ---------------------------------------------------------------- gather_axis

    pub(super) fn lower_gather_axis(&mut self, t: &Tensor, axis: usize) -> Result<LExpr> {
        let x = src_of(t, 0)?;
        let idx = src_of(t, 1)?;
        let r = x.rank();
        let fits = axis < r
            && idx.rank() == r
            && idx.dtype() == DType::I32
            && t.shape() == idx.shape()
            && (0..r).all(|d| d == axis || x.shape()[d] == idx.shape()[d]);
        if !fits {
            return Err(shape_mismatch(format!(
                "gather_axis node #{}: x {:?}, indices {:?} ({}), axis {axis}",
                t.id(),
                x.shape(),
                idx.shape(),
                idx.dtype().name()
            )));
        }
        if x.dtype().is_quant() {
            return Err(unsupported(format!("gather_axis of quantized node #{}", x.id())));
        }
        let [outer, axis_len, inner] = split3(x.shape(), axis);
        let n = idx.shape()[axis];
        let (xb, xv) = self.view3(x, [outer, axis_len, inner])?;
        let (ib, iv) = self.view3(idx, [outer, n, inner])?;
        let out = self.alloc_node(t)?;
        if outer * n * inner > 0 {
            let spec = GatherAxisSpec { dtype: t.dtype() };
            let dims = GatherAxisDims { outer, n, inner, axis_len, x: xv, idx: iv };
            self.push(Kernel::GatherAxis(Arc::new(spec), dims), vec![xb, ib], out);
        }
        Ok(LExpr::buf(out, t.dtype(), View::contiguous(t.shape())))
    }

    /// Buffer + rank-3 view of node `s` (stored in its own dtype) over `dims3`; copies to
    /// row-major when its view cannot be merged that way.
    fn view3(&mut self, s: &Tensor, dims3: [usize; 3]) -> Result<(BufId, View)> {
        let e = self.val(s)?;
        let (b, dt, v) = self.to_buffer(&e, s.shape(), s.dtype())?;
        if dt == s.dtype()
            && let Some(v3) = view::reshape(&v, s.shape(), &dims3) {
                return Ok((b, v3));
            }
        let src = LExpr::buf(b, dt, v);
        let c = self.materialize_contig(&src, s.shape(), s.dtype())?;
        Ok((c, View::contiguous(&dims3)))
    }

    // ---------------------------------------------------------------- topk

    pub(super) fn lower_topk(&mut self, t: &Tensor, k: usize, axis: usize) -> Result<LExpr> {
        let x = src_of(t, 0)?;
        let r = x.rank();
        if axis >= r || t.rank() != r || t.shape()[axis] != k || k == 0 || k > x.shape()[axis] {
            return Err(shape_mismatch(format!("topk node #{}: k {k} axis {axis} of {:?}", t.id(), x.shape())));
        }
        if x.dtype().is_quant() {
            return Err(unsupported(format!("topk of quantized node #{}", x.id())));
        }
        // Move `axis` last and read row-major [rows, cols].
        let perm: Vec<usize> = (0..r).filter(|&d| d != axis).chain(std::iter::once(axis)).collect();
        let pshape: Vec<usize> = perm.iter().map(|&p| x.shape()[p]).collect();
        let e = self.val(x)?;
        let pe = e.map_views(&mut |v| view::permute(v, &perm)).ok_or_else(|| internal("topk permutation"))?;
        let (xb, xdt) = self.to_contig(&pe, &pshape, x.dtype())?;
        let cols = x.shape()[axis];
        let rows = view::numel(&pshape[..r - 1]);
        let mut oshape = pshape.clone();
        oshape[r - 1] = k;
        let last = axis == r - 1;
        let out = if last { self.alloc_node(t)? } else { self.new_temp(DType::I32, &oshape)? };
        if rows > 0 {
            let spec = TopKSpec { in_dtype: xdt };
            self.push(Kernel::TopK(Arc::new(spec), TopKDims { rows, cols, k }), vec![xb], out);
        }
        let cv = View::contiguous(&oshape);
        if last {
            return Ok(LExpr::buf(out, DType::I32, cv));
        }
        // Inverse permutation back to the node's dim order.
        let mut inv = vec![0usize; r];
        for (i, &p) in perm.iter().enumerate() {
            inv[p] = i;
        }
        let v = view::permute(&cv, &inv).ok_or_else(|| internal("topk inverse permutation"))?;
        Ok(LExpr::buf(out, DType::I32, v))
    }

    // ---------------------------------------------------------------- write_slice

    pub(super) fn lower_write_slice(&mut self, t: &Tensor, axis: usize, start: usize) -> Result<LExpr> {
        let dest = src_of(t, 0)?;
        let src = src_of(t, 1)?;
        let r = dest.rank();
        let fits = axis < r
            && src.rank() == r
            && t.shape() == dest.shape()
            && src.dtype() == dest.dtype()
            && start + src.shape()[axis] <= dest.shape()[axis]
            && (0..r).all(|d| d == axis || src.shape()[d] == dest.shape()[d]);
        if !fits {
            return Err(shape_mismatch(format!(
                "write_slice node #{}: dest {:?}, src {:?}, axis {axis}, start {start}",
                t.id(),
                dest.shape(),
                src.shape()
            )));
        }
        if dest.dtype().is_quant() {
            return Err(unsupported(format!("write_slice into quantized node #{}", dest.id())));
        }
        let dtype = t.dtype();
        let shape = t.shape().to_vec();
        let cv = View::contiguous(&shape);
        let de = self.val(dest)?;
        let se = self.val(src)?;
        let region = View { offset: start * cv.strides[axis] as usize, strides: cv.strides.clone() };

        if let Some(b) = self.in_place_target(dest, &de)? {
            if let Some(i) = self.claimable(t) {
                // The updated dest is itself the requested output: hand out its buffer. A temp
                // (an earlier fresh write_slice in this graph) becomes an output buffer, or
                // the caller would get pooled memory that later evals reuse.
                if let Some(buf) = self.bufs.get_mut(b) {
                    if matches!(buf.kind, BufKind::Temp) {
                        buf.kind = BufKind::Output;
                    }
                }
                if let Some(slot) = self.out_bufs.get_mut(i) {
                    *slot = Some(b);
                }
            }
            self.write_region(&se, src.shape(), dtype, b, &region)?;
            return Ok(LExpr::buf(b, dtype, cv));
        }

        // Functional form: copy the untouched parts of dest, then write src.
        let out = self.alloc_node(t)?;
        let n = src.shape()[axis];
        let mut part = shape.clone();
        if start > 0 {
            part[axis] = start;
            self.emit_ew(&de, &part, out, dtype, cv.clone())?;
        }
        let tail = shape[axis] - start - n;
        if tail > 0 {
            part[axis] = tail;
            let pe = de
                .map_views(&mut |v| view::slice(v, axis, start + n))
                .ok_or_else(|| internal("write_slice tail view"))?;
            let tv = View { offset: (start + n) * cv.strides[axis] as usize, strides: cv.strides.clone() };
            self.emit_ew(&pe, &part, out, dtype, tv)?;
        }
        self.write_region(&se, src.shape(), dtype, out, &region)?;
        Ok(LExpr::buf(out, dtype, cv))
    }

    /// The buffer `dest` may be overwritten in place: a realized device leaf (or an earlier
    /// write into one) that nothing else in the graph reads and that is not an output.
    fn in_place_target(&self, dest: &Tensor, de: &LExpr) -> Result<Option<BufId>> {
        if !matches!(dest.op(), Op::Device(_) | Op::WriteSlice { .. }) || self.users_of(dest) != 1 || self.is_output(dest)
        {
            return Ok(None);
        }
        let Some((b, dt, v)) = de.as_buf() else { return Ok(None) };
        if dt != dest.dtype() || v.offset != 0 || !view::is_contiguous(v, dest.shape()) {
            return Ok(None);
        }
        let Some(buf) = self.bufs.get(b) else { return Err(internal("write_slice dest buffer missing")) };
        if buf.bytes < bytes_of(dest.dtype(), dest.shape())? {
            return Ok(None);
        }
        let exclusive = match buf.kind {
            BufKind::Device(_) => self.dev_nodes.get(&b).copied() == Some(1),
            // A buffer an earlier in-place / fresh write_slice produced.
            BufKind::Temp | BufKind::Output => matches!(dest.op(), Op::WriteSlice { .. }),
            BufKind::Host(_) => false,
        };
        Ok(exclusive.then_some(b))
    }
}
