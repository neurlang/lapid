//! Builders for indexing (take_along_axis, top-k, argmax), integer bit ops and the in-place
//! slice write used by KV caches. Semantics: docs/spec/ops.md.

use crate::bail;
use crate::dtype::DType;
use crate::error::Result;
use crate::op::{BinaryOp, Op};
use crate::tensor::{Shape, Tensor};
use smallvec::smallvec;
use crate::tensor::check;

impl Tensor {
    // ---------- in-place update ----------
    /// `self` (the destination) with `[start, start + src.shape[axis])` along `axis`
    /// replaced by `src`. Returns the updated tensor (same shape and dtype as `self`).
    ///
    /// When `self` is a realized device tensor (`Op::Device`) used by nothing else in the
    /// evaluated graph, the write happens in place: the result aliases the same device
    /// buffer and only the slice is written (the old handle observes the new contents after
    /// eval). Otherwise the result is a fresh buffer (functional copy).
    pub fn write_slice(&self, axis: isize, start: usize, src: &Tensor) -> Result<Tensor> {
        check::not_quant(self, "write_slice")?;
        check::not_quant(src, "write_slice")?;
        let axis = self.axis(axis)?;
        if src.dtype() != self.dtype() {
            bail!(
                "unsupported_dtype",
                "write_slice: src dtype {} != dest dtype {} (cast first)",
                src.dtype().name(),
                self.dtype().name()
            );
        }
        let same = src.rank() == self.rank()
            && src.shape().iter().zip(self.shape()).enumerate().all(|(d, (a, b))| d == axis || a == b);
        if !same {
            bail!("shape_mismatch", "write_slice: src {:?} does not fit dest {:?} along axis {axis}", src.shape(), self.shape());
        }
        let n = src.shape()[axis];
        if start + n > self.shape()[axis] {
            bail!(
                "invalid_argument",
                "write_slice: [{start}, {}) out of range for axis {axis} of {:?}",
                start + n,
                self.shape()
            );
        }
        if n == 0 {
            return Ok(self.clone());
        }
        Ok(Tensor::from_op(
            Op::WriteSlice { axis, start },
            vec![self.clone(), src.clone()],
            self.shape(),
            self.dtype(),
        ))
    }

    // ---------- indexing ----------
    /// take_along_axis: `indices` (i32) has the rank of `self` and the same dims except
    /// along `axis`; `out[.., j, ..] = self[.., indices[.., j, ..], ..]`. Indices outside
    /// `[0, self.shape[axis])` make eval fail with "index_out_of_range".
    pub fn gather_axis(&self, axis: isize, indices: &Tensor) -> Result<Tensor> {
        check::not_quant(self, "gather_axis")?;
        let axis = self.axis(axis)?;
        if indices.dtype() != DType::I32 {
            bail!("unsupported_dtype", "gather_axis indices must be i32, got {}", indices.dtype().name());
        }
        let same = indices.rank() == self.rank()
            && indices.shape().iter().zip(self.shape()).enumerate().all(|(d, (a, b))| d == axis || a == b);
        if !same {
            bail!("shape_mismatch", "gather_axis: indices {:?} vs input {:?} (axis {axis})", indices.shape(), self.shape());
        }
        if self.shape()[axis] == 0 && indices.shape()[axis] > 0 {
            bail!("invalid_argument", "gather_axis from an empty axis");
        }
        Ok(Tensor::from_op(
            Op::GatherAxis { axis },
            vec![self.clone(), indices.clone()],
            indices.shape(),
            self.dtype(),
        ))
    }

    /// Indices (i32) of the `k` largest values along `axis`, sorted descending (ties: lower
    /// index first). Shape: `self`'s with `axis` set to `k`.
    pub fn topk_indices(&self, k: usize, axis: isize) -> Result<Tensor> {
        check::not_quant(self, "topk")?;
        let axis = self.axis(axis)?;
        let n = self.shape()[axis];
        if k == 0 || k > n {
            bail!("invalid_argument", "topk: k = {k} must be in 1..={n} (axis {axis} of {:?})", self.shape());
        }
        let mut shape = Shape::from_slice(self.shape());
        shape[axis] = k;
        Ok(Tensor::from_op(Op::TopK { k, axis }, smallvec![self.clone()], shape, DType::I32))
    }

    /// `(values, indices)` of the `k` largest values along `axis` (descending; values have
    /// `self`'s dtype, indices are i32). Values are `gather_axis(axis, indices)`.
    pub fn topk(&self, k: usize, axis: isize) -> Result<(Tensor, Tensor)> {
        let idx = self.topk_indices(k, axis)?;
        let vals = self.gather_axis(axis, &idx)?;
        Ok((vals, idx))
    }

    /// Index (i32) of the largest value along `axis` (first one on ties); the axis is removed.
    pub fn argmax(&self, axis: isize) -> Result<Tensor> {
        let axis = self.axis(axis)?;
        self.topk_indices(1, axis as isize)?.squeeze(axis as isize)
    }

    // ---------- integer ops (i32 only) ----------
    pub fn rem(&self, rhs: &Tensor) -> Result<Tensor> {
        self.int_binary(BinaryOp::Rem, rhs)
    }
    pub fn bitand(&self, rhs: &Tensor) -> Result<Tensor> {
        self.int_binary(BinaryOp::BitAnd, rhs)
    }
    pub fn bitor(&self, rhs: &Tensor) -> Result<Tensor> {
        self.int_binary(BinaryOp::BitOr, rhs)
    }
    pub fn bitxor(&self, rhs: &Tensor) -> Result<Tensor> {
        self.int_binary(BinaryOp::BitXor, rhs)
    }
    pub fn shl(&self, rhs: &Tensor) -> Result<Tensor> {
        self.int_binary(BinaryOp::Shl, rhs)
    }
    pub fn shr(&self, rhs: &Tensor) -> Result<Tensor> {
        self.int_binary(BinaryOp::Shr, rhs)
    }

    fn int_binary(&self, op: BinaryOp, rhs: &Tensor) -> Result<Tensor> {
        if self.dtype() != DType::I32 || rhs.dtype() != DType::I32 {
            bail!("unsupported_dtype", "{op:?} needs i32 operands, got {} and {}", self.dtype().name(), rhs.dtype().name());
        }
        self.binary(op, rhs)
    }
}

/// Builder-side check used by `Tensor::binary` / `binary_scalar`.
pub(crate) fn check_int_only(op: BinaryOp, dt: DType) -> Result<()> {
    if op.is_int_only() && dt != DType::I32 {
        bail!("unsupported_dtype", "{op:?} is only defined for i32 tensors, got {}", dt.name());
    }
    Ok(())
}
