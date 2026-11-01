//! Shape / dtype validation shared by the tensor builders.

use crate::bail;
use crate::dtype::DType;
use crate::error::Result;
use crate::tensor::{Shape, Tensor};

/// Resolve `axis` (negative counts from the end) against `rank`.
pub(crate) fn resolve_axis(axis: isize, rank: usize) -> Result<usize> {
    let r = rank as isize;
    let a = if axis < 0 { axis + r } else { axis };
    if a < 0 || a >= r {
        bail!("invalid_argument", "axis {axis} out of range for rank {rank}");
    }
    Ok(a as usize)
}

/// Numpy broadcast of two shapes (aligned at the trailing dim).
pub(crate) fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Shape> {
    if a == b {
        return Ok(Shape::from_slice(a));
    }
    let rank = a.len().max(b.len());
    let mut out = Shape::from_elem(0, rank);
    for i in 0..rank {
        let da = if i < rank - a.len() { 1 } else { a[i - (rank - a.len())] };
        let db = if i < rank - b.len() { 1 } else { b[i - (rank - b.len())] };
        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => bail!("shape_mismatch", "shapes {a:?} and {b:?} are not broadcastable"),
        };
    }
    Ok(out)
}

/// Resolve a reshape target with at most one `usize::MAX` ("infer") dim.
pub(crate) fn infer_shape(numel: usize, from: &[usize], to: &[usize]) -> Result<Shape> {
    let mut shape = Shape::from_slice(to);
    let mut inferred = (0..shape.len()).filter(|&i| shape[i] == usize::MAX);
    let first = inferred.next();
    if inferred.next().is_some() {
        bail!("invalid_argument", "reshape {to:?}: at most one dim may be inferred");
    }
    let known: usize = shape.iter().filter(|&&d| d != usize::MAX).product();
    if let Some(i) = first {
        if known == 0 || !numel.is_multiple_of(known) {
            bail!("shape_mismatch", "reshape {from:?} -> {to:?}: cannot infer dim");
        }
        shape[i] = numel / known;
    } else if known != numel {
        bail!("shape_mismatch", "reshape {from:?} ({numel} elements) -> {to:?} ({known} elements)");
    }
    Ok(shape)
}

pub(crate) fn numel_matches(len: usize, shape: &[usize]) -> Result<()> {
    let n: usize = shape.iter().product();
    if n != len {
        bail!("shape_mismatch", "{len} values do not fill shape {shape:?} ({n} elements)");
    }
    Ok(())
}

/// Byte size of `shape` in `dtype`, validating the quantized-matrix constraints first so
/// `DType::storage_bytes` never panics.
pub(crate) fn storage_bytes(shape: &[usize], dtype: DType) -> Result<usize> {
    if dtype.is_quant() {
        if shape.len() != 2 && shape.len() != 3 {
            bail!("shape_mismatch", "quantized {} tensors must be 2-D (or 3-D expert stacks), got {shape:?}", dtype.name());
        }
        let block = dtype.quant_format().map_or(DType::QUANT_GROUP, |f| f.block_elems());
        if !shape[shape.len() - 1].is_multiple_of(block) {
            bail!("shape_mismatch", "quantized {} cols must be a multiple of {block}, got {shape:?}", dtype.name());
        }
    }
    Ok(dtype.storage_bytes(shape))
}

/// Quantized tensors cannot pass through general ops.
pub(crate) fn not_quant(t: &Tensor, what: &str) -> Result<()> {
    if t.dtype().is_quant() {
        bail!("unsupported_dtype", "{what} is not supported on quantized {} tensors", t.dtype().name());
    }
    Ok(())
}

/// Elementwise / reduction operands: any plain (non-quantized) dtype.
pub(crate) fn elementwise_dtype(t: &Tensor, what: &str) -> Result<()> {
    not_quant(t, what)
}

pub(crate) fn float_dtype(t: &Tensor, what: &str) -> Result<()> {
    if !t.dtype().is_float() {
        bail!("unsupported_dtype", "{what} requires a float tensor, got {}", t.dtype().name());
    }
    Ok(())
}
