//! Kernel emission and materialization: turning lazy values into launches.

use std::sync::Arc;

use lapid_tensor::{DType, ReduceOp, Result, Tensor};

use super::lexpr::{self, LExpr, Leaf, Loads, Src};
use super::{Lowerer, MatMulPlan, PendKind, PendState, Pending, internal, unsupported, view};
use crate::ir::{
    BufId, EwDims, EwSpec, Expr, ExprType, Kernel, Launch, MatMulSpec, MatView, ReduceDims, ReduceSpec, View,
};

impl Lowerer {
    pub(super) fn push(&mut self, kernel: Kernel, inputs: Vec<BufId>, output: BufId) {
        self.launches.push(Launch { kernel, inputs, output });
    }

    // ---------------------------------------------------------------- raw kernels

    /// `out[out_view(idx)] = e(idx)` for every `idx` in `shape` (pending leaves are flushed).
    pub(super) fn emit_ew(
        &mut self,
        e: &LExpr,
        shape: &[usize],
        out: BufId,
        out_dtype: DType,
        out_view: View,
    ) -> Result<()> {
        if view::numel(shape) == 0 {
            return Ok(());
        }
        if out_dtype.is_quant() {
            return Err(internal("elementwise kernel with a quantized output"));
        }
        let e = self.flush_cow(e)?;
        // Integer words are stored raw into i32 outputs only; anything else stores f32.
        let e = if e.ty() == ExprType::Int && out_dtype != DType::I32 {
            std::borrow::Cow::Owned(lexpr::as_f32(e.into_owned()))
        } else {
            e
        };
        let mut loads = Loads::new(0, None);
        let expr = loads.expr(&e)?;
        let rank = shape.len();
        if out_view.strides.len() != rank || loads.views.iter().any(|v| v.strides.len() != rank) {
            return Err(internal(format!("elementwise view rank mismatch over shape {shape:?}")));
        }
        let mut strides: Vec<Vec<isize>> = loads.views.iter().map(|v| v.strides.clone()).collect();
        strides.push(out_view.strides.clone());
        let (cshape, mut cstrides) = view::coalesce(shape, &strides);
        let out_strides = cstrides.pop().ok_or_else(|| internal("coalesce lost the output view"))?;
        let in_views: Vec<View> = loads
            .views
            .iter()
            .zip(cstrides)
            .map(|(v, s)| View { offset: v.offset, strides: s })
            .collect();
        let spec = EwSpec { rank: cshape.len(), in_dtypes: loads.dtypes, out_dtype, expr };
        let dims = EwDims { shape: cshape, in_views, out_view: View { offset: out_view.offset, strides: out_strides } };
        self.push(Kernel::Elementwise(Arc::new(spec), dims), loads.bufs, out);
        Ok(())
    }

    /// Reduce `e` over the last dim of `shape` into contiguous `out` (numel = outer dims).
    pub(super) fn emit_reduce(
        &mut self,
        op: ReduceOp,
        e: &LExpr,
        shape: &[usize],
        out: BufId,
        out_dtype: DType,
    ) -> Result<()> {
        let r = shape.len();
        if r == 0 {
            return Err(internal("reduce over a rank-0 iteration space"));
        }
        let outer = &shape[..r - 1];
        if view::numel(outer) == 0 {
            return Ok(());
        }
        let e = lexpr::as_f32(self.flush_cow(e)?.into_owned());
        let mut loads = Loads::new(0, None);
        let expr = loads.expr(&e)?;
        if loads.views.iter().any(|v| v.strides.len() != r) {
            return Err(internal(format!("reduce view rank mismatch over shape {shape:?}")));
        }
        // The output is row-major over the outer dims, so only input views constrain merging.
        let outer_strides: Vec<Vec<isize>> = loads.views.iter().map(|v| v.strides[..r - 1].to_vec()).collect();
        let (mut cshape, cstrides) = view::coalesce(outer, &outer_strides);
        cshape.push(shape[r - 1]);
        let in_views: Vec<View> = loads
            .views
            .iter()
            .zip(cstrides)
            .map(|(v, mut s)| {
                s.push(v.strides[r - 1]);
                View { offset: v.offset, strides: s }
            })
            .collect();
        let spec = ReduceSpec { op, rank: cshape.len(), in_dtypes: loads.dtypes, out_dtype, expr };
        self.push(Kernel::Reduce(Arc::new(spec), ReduceDims { shape: cshape, in_views }), loads.bufs, out);
        Ok(())
    }

    /// Matmul into contiguous `out`; `extras` are the epilogue's `Load(1..)` inputs.
    pub(super) fn emit_matmul(
        &mut self,
        plan: &MatMulPlan,
        out: BufId,
        out_dtype: DType,
        epilogue: Option<Expr>,
        extras: Loads,
    ) {
        let d = &plan.dims;
        if d.batch * d.m * d.n == 0 {
            return;
        }
        let mut inputs = vec![plan.a, plan.b];
        inputs.extend_from_slice(&extras.bufs);
        let spec = MatMulSpec {
            a_dtype: plan.a_dtype,
            b_dtype: plan.b_dtype,
            b_transposed: plan.b_transposed,
            extra_dtypes: extras.dtypes,
            out_dtype,
            epilogue,
        };
        let mut dims = d.clone();
        dims.extra_views = extras.views;
        self.push(Kernel::MatMul(Arc::new(spec), dims), inputs, out);
    }

    // ---------------------------------------------------------------- pending matmuls

    /// Replace every pending-matmul leaf by a view of its (now emitted) result buffer.
    pub(super) fn flush(&mut self, e: &LExpr) -> Result<LExpr> {
        if !e.has_pending() {
            return Ok(e.clone());
        }
        self.flush_pendings(e)
    }

    /// `flush` without the clone when nothing is pending.
    pub(super) fn flush_cow<'e>(&mut self, e: &'e LExpr) -> Result<std::borrow::Cow<'e, LExpr>> {
        if !e.has_pending() {
            return Ok(std::borrow::Cow::Borrowed(e));
        }
        self.flush_pendings(e).map(std::borrow::Cow::Owned)
    }

    fn flush_pendings(&mut self, e: &LExpr) -> Result<LExpr> {
        e.map_leaves(&mut |l| match l.src {
            Src::Pending(p) => {
                let b = self.flush_pending(p)?;
                Ok(Leaf { src: Src::Buf(b), dtype: l.dtype, view: l.view.clone() })
            }
            Src::Buf(_) => Ok(l.clone()),
        })
    }

    fn flush_pending(&mut self, p: usize) -> Result<BufId> {
        let pd = self.pendings.get(p).ok_or_else(|| internal(format!("unknown pending matmul #{p}")))?;
        if let PendState::Flushed(b) = pd.state {
            return Ok(b);
        }
        // Waiting, or Consumed by an epilogue while a cheap duplicated expression still
        // references the raw product: emit (again) without epilogue.
        let (shape, dtype) = (pd.out_shape.clone(), pd.out_dtype);
        let kind = match &pd.kind {
            PendKind::MatMul(plan) => PendKind::MatMul(plan.clone()),
            PendKind::Cat { axis, pieces } => PendKind::Cat { axis: *axis, pieces: pieces.clone() },
        };
        let out = self.new_temp(dtype, &shape)?;
        match kind {
            PendKind::MatMul(mut plan) => {
                plan.dims.out = MatView::contiguous(plan.dims.m, plan.dims.n);
                self.emit_matmul(&plan, out, dtype, None, Loads::new(1, None));
            }
            PendKind::Cat { axis, pieces } => {
                let cv = View::contiguous(&shape);
                self.write_pieces(&pieces, axis, dtype, out, &cv)?;
            }
        }
        if let Some(pd) = self.pendings.get_mut(p) {
            pd.state = PendState::Flushed(out);
        }
        Ok(out)
    }

    /// If `e` (over `shape`, written row-major into `out`) is "pending matmul ∘ elementwise",
    /// emit one matmul with that expression as epilogue. Returns false if not applicable.
    fn try_epilogue(&mut self, e: &LExpr, shape: &[usize], dtype: DType, out: BufId) -> Result<bool> {
        let cv = View::contiguous(shape);
        self.try_epilogue_into(e, shape, dtype, out, &cv)
    }

    /// `try_epilogue` writing through `out_view` (over `shape`), which must map onto the
    /// matmul's `[batch, m, n]` with unit stride along n (e.g. a slice of a KV cache).
    pub(super) fn try_epilogue_into(
        &mut self,
        e: &LExpr,
        shape: &[usize],
        dtype: DType,
        out: BufId,
        out_view: &View,
    ) -> Result<bool> {
        if !self.opts.fuse || e.ty() == ExprType::Int || dtype.is_quant() {
            return Ok(false);
        }
        let ids = e.pending_ids();
        if ids.len() != 1 {
            return Ok(false);
        }
        let p = ids[0];
        let (plan, numel) = match self.pendings.get(p) {
            Some(Pending { kind: PendKind::MatMul(plan), state: PendState::Waiting, out_shape, .. }) => {
                (plan.clone(), view::numel(out_shape))
            }
            Some(_) => return Ok(false),
            None => return Err(internal(format!("unknown pending matmul #{p}"))),
        };
        if numel != view::numel(shape) {
            return Ok(false);
        }
        // The product must be read element-for-element in the order it is written.
        let mut aligned = true;
        e.visit_leaves(&mut |l| {
            if l.src == Src::Pending(p) && !view::is_contiguous(&l.view, shape) {
                aligned = false;
            }
        });
        if !aligned {
            return Ok(false);
        }
        let mn = [plan.dims.batch, plan.dims.m, plan.dims.n];
        let Some(me) = e.map_views(&mut |v| view::reshape(v, shape, &mn)) else {
            return Ok(false);
        };
        let Some(ov) = view::reshape(out_view, shape, &mn) else {
            return Ok(false);
        };
        let (Ok(batch_stride), Ok(row_stride)) = (usize::try_from(ov.strides[0]), usize::try_from(ov.strides[1])) else {
            return Ok(false);
        };
        if plan.dims.n > 1 && ov.strides[2] != 1 {
            return Ok(false);
        }
        let mut plan = plan;
        plan.dims.out = MatView { offset: ov.offset, batch_stride, row_stride };
        let mut loads = Loads::new(1, Some(p));
        let expr = loads.expr(&me)?;
        let epilogue = if expr == Expr::Load(0) { None } else { Some(expr) };
        if let Some(pd) = self.pendings.get_mut(p) {
            pd.state = PendState::Consumed;
        }
        self.emit_matmul(&plan, out, dtype, epilogue, loads);
        Ok(true)
    }

    // ---------------------------------------------------------------- materialization

    /// Write value `e` (over `shape`) into `out`, laid out by the dense view `out_view`
    /// (row-major or a permutation of it). Uses a matmul epilogue when possible.
    pub(super) fn write_dense(
        &mut self,
        e: &LExpr,
        shape: &[usize],
        dtype: DType,
        out: BufId,
        out_view: &View,
    ) -> Result<()> {
        if view::numel(shape) == 0 {
            return Ok(());
        }
        if self.try_cat_into(e, shape, dtype, out, out_view)? {
            return Ok(());
        }
        if out_view.offset == 0 && view::is_dense(out_view, shape) {
            // Row-major target: no permutation needed.
            if self.try_epilogue(e, shape, dtype, out)? {
                return Ok(());
            }
            return self.emit_ew(e, shape, out, dtype, out_view.clone());
        }
        let perm = view::dense_perm(out_view, shape)
            .ok_or_else(|| internal(format!("write_dense: {out_view:?} is not dense over {shape:?}")))?;
        let pshape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        let pe = e
            .map_views(&mut |v| view::permute(v, &perm))
            .ok_or_else(|| internal("write_dense: permutation rank mismatch"))?;
        if self.try_epilogue(&pe, &pshape, dtype, out)? {
            return Ok(());
        }
        let cv = View::contiguous(&pshape);
        self.emit_ew(&pe, &pshape, out, dtype, cv)
    }

    /// Write `e` (over `shape`) into the region `out_view` of `out` (any strided view, e.g. a
    /// slice of a larger buffer). A pending matmul writes there directly when the region is
    /// row-compatible with its output; otherwise one elementwise copy.
    pub(super) fn write_region(
        &mut self,
        e: &LExpr,
        shape: &[usize],
        dtype: DType,
        out: BufId,
        out_view: &View,
    ) -> Result<()> {
        if view::numel(shape) == 0 {
            return Ok(());
        }
        if self.try_cat_into(e, shape, dtype, out, out_view)? {
            return Ok(());
        }
        if self.try_epilogue_into(e, shape, dtype, out, out_view)? {
            return Ok(());
        }
        // A permuted pending product (e.g. `[B, n, H, D]` read as `[B, H, n, D]`) may match
        // the region in another dim order.
        if let Some(perm) = self.pending_perm(e, shape) {
            let pshape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
            if let (Some(pe), Some(pv)) =
                (e.map_views(&mut |v| view::permute(v, &perm)), view::permute(out_view, &perm))
                && self.try_epilogue_into(&pe, &pshape, dtype, out, &pv)? {
                    return Ok(());
                }
        }
        self.emit_ew(e, shape, out, dtype, out_view.clone())
    }

    /// Write cat `pieces` (along `axis`) through `view` (over the cat's shape) of `out`.
    fn write_pieces(&mut self, pieces: &[(LExpr, Vec<usize>)], axis: usize, dtype: DType, out: BufId, view: &View) -> Result<()> {
        let mut at = 0usize;
        for (pe, ps) in pieces {
            let pv = view::slice(view, axis, at).ok_or_else(|| internal("cat piece view"))?;
            self.write_region(pe, ps, dtype, out, &pv)?;
            at += ps[axis];
        }
        Ok(())
    }

    /// If `e` is exactly a (permuted) read of a pending `Cat`, write its pieces straight into
    /// `out_view` (over `shape`) — e.g. RoPE halves into a KV-cache slice — instead of
    /// materializing the concatenation and copying it.
    fn try_cat_into(&mut self, e: &LExpr, shape: &[usize], dtype: DType, out: BufId, out_view: &View) -> Result<bool> {
        let LExpr::Leaf(l) = e else { return Ok(false) };
        let Src::Pending(p) = l.src else { return Ok(false) };
        let (axis, pieces, cshape) = match self.pendings.get(p) {
            Some(Pending { kind: PendKind::Cat { axis, pieces }, state: PendState::Waiting, out_shape, out_dtype })
                if *out_dtype == dtype && l.dtype == dtype =>
            {
                (*axis, pieces.clone(), out_shape.clone())
            }
            _ => return Ok(false),
        };
        let r = shape.len();
        if l.view.offset != 0 || cshape.len() != r || out_view.strides.len() != r || l.view.strides.len() != r {
            return Ok(false);
        }
        // Match every non-1 dim of the read to the cat dim it walks (a permutation of the
        // cat's row-major layout); the target view is then re-indexed by cat dims.
        let cst = View::contiguous(&cshape).strides;
        let mut tstr = vec![0isize; r];
        let mut used = vec![false; r];
        #[allow(clippy::needless_range_loop)]
        for i in 0..r {
            if shape[i] == 1 {
                continue;
            }
            let Some(j) = (0..r).find(|&j| !used[j] && cshape[j] == shape[i] && cst[j] == l.view.strides[i]) else {
                return Ok(false);
            };
            used[j] = true;
            tstr[j] = out_view.strides[i];
        }
        if (0..r).any(|j| !used[j] && cshape[j] != 1) {
            return Ok(false);
        }
        if let Some(pd) = self.pendings.get_mut(p) {
            pd.state = PendState::Consumed;
        }
        let tv = View { offset: out_view.offset, strides: tstr };
        self.write_pieces(&pieces, axis, dtype, out, &tv)?;
        Ok(true)
    }

    /// Dim order in which the single pending leaf of `e` is stored row-major.
    fn pending_perm(&self, e: &LExpr, shape: &[usize]) -> Option<Vec<usize>> {
        let mut found = None;
        e.visit_leaves(&mut |l| {
            if found.is_none() && matches!(l.src, Src::Pending(_)) {
                found = view::dense_perm(&l.view, shape);
            }
        });
        found.filter(|p| p.iter().enumerate().any(|(i, &x)| i != x))
    }

    /// Layout for materializing `e` into a temp: the storage order of its pending matmul
    /// leaf when that is a dense permutation (so `matmul → reshape → permute → +bias`
    /// becomes one epilogue matmul and the consumer gets a strided view), else row-major.
    fn pick_layout(&self, e: &LExpr, shape: &[usize]) -> View {
        let mut found: Option<View> = None;
        e.visit_leaves(&mut |l| {
            if found.is_none() && matches!(l.src, Src::Pending(_)) && view::dense_perm(&l.view, shape).is_some() {
                found = Some(View { offset: 0, strides: l.view.strides.clone() });
            }
        });
        found.unwrap_or_else(|| View::contiguous(shape))
    }

    /// Materialize `e` into a fresh temp in the best layout; returns the value as a leaf.
    pub(super) fn materialize_tmp(&mut self, e: &LExpr, shape: &[usize], dtype: DType) -> Result<LExpr> {
        let layout = self.pick_layout(e, shape);
        let out = self.new_temp(dtype, shape)?;
        self.write_dense(e, shape, dtype, out, &layout)?;
        Ok(LExpr::buf(out, dtype, layout))
    }

    /// Materialize `e` row-major into a fresh temp.
    pub(super) fn materialize_contig(&mut self, e: &LExpr, shape: &[usize], dtype: DType) -> Result<BufId> {
        let out = self.new_temp(dtype, shape)?;
        self.write_dense(e, shape, dtype, out, &View::contiguous(shape))?;
        Ok(out)
    }

    /// Materialize the value of node `t` (into its Output buffer when it is one).
    pub(super) fn materialize_node(&mut self, t: &Tensor, e: &LExpr) -> Result<LExpr> {
        if self.claimable(t).is_some() {
            let out = self.alloc_node(t)?;
            let cv = View::contiguous(t.shape());
            self.write_dense(e, t.shape(), t.dtype(), out, &cv)?;
            return Ok(LExpr::buf(out, t.dtype(), cv));
        }
        self.materialize_tmp(e, t.shape(), t.dtype())
    }

    /// Any strided buffer view holding `e` (no copy when `e` already is one). Constants
    /// become a 1-element buffer read with all-zero strides.
    pub(super) fn to_buffer(&mut self, e: &LExpr, shape: &[usize], dtype: DType) -> Result<(BufId, DType, View)> {
        match e {
            LExpr::Leaf(l) => {
                if l.dtype.is_quant() {
                    return Err(unsupported(format!("quantized ({}) operand here", l.dtype.name())));
                }
                let b = match l.src {
                    Src::Buf(b) => b,
                    Src::Pending(p) => self.flush_pending(p)?,
                };
                Ok((b, l.dtype, l.view.clone()))
            }
            LExpr::Const(_) | LExpr::ConstInt(_) => {
                let b = self.new_temp(dtype, &[1])?;
                self.emit_ew(e, &[1], b, dtype, View { offset: 0, strides: vec![1] })?;
                Ok((b, dtype, View { offset: 0, strides: vec![0; shape.len()] }))
            }
            _ => match self.materialize_tmp(e, shape, dtype)? {
                LExpr::Leaf(Leaf { src: Src::Buf(b), dtype, view }) => Ok((b, dtype, view)),
                _ => Err(internal("materialize_tmp did not return a buffer")),
            },
        }
    }

    /// A row-major buffer starting at element 0 holding `e`.
    pub(super) fn to_contig(&mut self, e: &LExpr, shape: &[usize], dtype: DType) -> Result<(BufId, DType)> {
        if let LExpr::Leaf(l) = e {
            if l.dtype.is_quant() {
                return Err(unsupported(format!("quantized ({}) operand here", l.dtype.name())));
            }
            if let Src::Buf(b) = l.src
                && view::is_contiguous(&l.view, shape) {
                    return Ok((b, l.dtype));
                }
        }
        let b = self.materialize_contig(e, shape, dtype)?;
        Ok((b, dtype))
    }

    /// `to_contig`, converting non-float storage to f32.
    pub(super) fn float_contig(&mut self, e: &LExpr, shape: &[usize], dtype: DType) -> Result<(BufId, DType)> {
        let (b, dt) = self.to_contig(e, shape, dtype)?;
        if dt.is_float() {
            return Ok((b, dt));
        }
        let src = LExpr::buf(b, dt, View::contiguous(shape));
        Ok((self.materialize_contig(&src, shape, DType::F32)?, DType::F32))
    }

    /// `to_buffer`, converting non-float storage to f32.
    pub(super) fn float_buffer(&mut self, e: &LExpr, shape: &[usize], dtype: DType) -> Result<(BufId, DType, View)> {
        let (b, dt, v) = self.to_buffer(e, shape, dtype)?;
        if dt.is_float() {
            return Ok((b, dt, v));
        }
        let src = LExpr::buf(b, dt, v);
        Ok((self.materialize_contig(&src, shape, DType::F32)?, DType::F32, View::contiguous(shape)))
    }
}
