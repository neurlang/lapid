//! Lowering of the round-3 kernels: `MatMulId` (mixture-of-experts matmul) and
//! `GatedDeltaNet` (gated delta rule). Both read contiguous inputs.

use std::sync::Arc;

use lapid_tensor::{DType, Result, Tensor};

use super::lexpr::LExpr;
use super::{Lowerer, shape_mismatch, src_of, unsupported, view};
use crate::ir::{GatedDeltaNetDims, GatedDeltaNetSpec, Kernel, MatMulIdDims, MatMulIdSpec, View};

impl Lowerer {
    pub(super) fn native_matmul_id(&mut self, t: &Tensor) -> Result<Option<LExpr>> {
        let x = src_of(t, 0)?;
        let w = src_of(t, 1)?;
        let ids = src_of(t, 2)?;
        if x.rank() != 2 || w.rank() != 3 || ids.rank() != 2 || ids.dtype() != DType::I32 {
            return Err(shape_mismatch(format!(
                "matmul_id node #{}: x {:?}, experts {:?}, ids {:?} ({})",
                t.id(),
                x.shape(),
                w.shape(),
                ids.shape(),
                ids.dtype().name()
            )));
        }
        let (tt, k) = (x.shape()[0], x.shape()[1]);
        let (e, n) = (w.shape()[0], w.shape()[1]);
        let r = ids.shape()[1];
        if w.shape()[2] != k || ids.shape()[0] != tt || t.shape() != [tt, r, n] {
            return Err(shape_mismatch(format!(
                "matmul_id node #{}: x {:?}, experts {:?}, ids {:?}, out {:?}",
                t.id(),
                x.shape(),
                w.shape(),
                ids.shape(),
                t.shape()
            )));
        }
        let xe = self.val(x)?;
        let (xb, xdt) = self.float_contig(&xe, x.shape(), x.dtype())?;
        let we = self.val(w)?;
        let (wb, wdt) = if w.dtype().is_quant() {
            match we.as_buf() {
                Some((b, dt, v)) if dt.is_quant() && view::is_contiguous(v, w.shape()) => (b, dt),
                _ => {
                    return Err(unsupported(format!(
                        "quantized matmul_id experts (node #{}) must be a host/device leaf",
                        w.id()
                    )));
                }
            }
        } else {
            self.float_contig(&we, w.shape(), w.dtype())?
        };
        let ie = self.val(ids)?;
        let ib = match ie.as_buf() {
            Some((b, DType::I32, v)) if view::is_contiguous(v, ids.shape()) => b,
            _ => self.materialize_contig(&ie, ids.shape(), DType::I32)?,
        };
        let out = self.alloc_node(t)?;
        if tt * r * n > 0 {
            let spec = MatMulIdSpec { x_dtype: xdt, w_dtype: wdt, out_dtype: t.dtype() };
            let dims = MatMulIdDims { t: tt, r, e, n, k };
            self.push(Kernel::MatMulId(Arc::new(spec), dims), vec![xb, wb, ib], out);
        }
        Ok(Some(LExpr::buf(out, t.dtype(), View::contiguous(t.shape()))))
    }

    pub(super) fn native_gated_delta_net(&mut self, t: &Tensor, scale: f32) -> Result<Option<LExpr>> {
        let srcs: Vec<&Tensor> = (0..6).map(|i| src_of(t, i)).collect::<Result<_>>()?;
        let (q, v) = (srcs[0], srcs[2]);
        if q.rank() != 4 || v.rank() != 4 {
            return Ok(None);
        }
        let (b, tt, h, dk) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
        let dv = v.shape()[3];
        let shapes: [&[usize]; 6] = [&[b, tt, h, dk], &[b, tt, h, dk], &[b, tt, h, dv], &[b, tt, h], &[b, tt, h], &[b, h, dk, dv]];
        let fits = srcs.iter().zip(shapes).all(|(s, want)| s.shape() == want && s.dtype() == DType::F32)
            && t.shape() == [b, h, tt + dk, dv];
        if !fits {
            return Ok(None);
        }
        let mut inputs = Vec::with_capacity(6);
        for s in &srcs {
            let e = self.val(s)?;
            let (buf, _) = self.to_contig(&e, s.shape(), DType::F32)?;
            inputs.push(buf);
        }
        // `to_contig` keeps a buffer's own dtype; every input must be f32 storage.
        let out = self.alloc_node(t)?;
        if b * h * dv * (tt + dk) > 0 {
            let spec = GatedDeltaNetSpec { out_dtype: t.dtype() };
            let dims = GatedDeltaNetDims { b, t: tt, h, dk, dv, scale: scale.to_bits() };
            self.push(Kernel::GatedDeltaNet(Arc::new(spec), dims), inputs, out);
        }
        Ok(Some(LExpr::buf(out, t.dtype(), View::contiguous(t.shape()))))
    }
}
