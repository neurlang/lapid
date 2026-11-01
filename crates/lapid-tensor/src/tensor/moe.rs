//! Builders for the round-3 ops: mixture-of-experts matmul (`MatMulId`) and the gated
//! delta rule (`GatedDeltaNet`). Semantics: `op.rs` doc comments and docs/spec/ops.md.

use crate::bail;
use crate::dtype::DType;
use crate::error::Result;
use crate::op::Op;
use crate::tensor::{Srcs, Tensor};
use smallvec::smallvec;
use crate::tensor::check;

impl Tensor {
    /// Mixture-of-experts matmul (ggml `MUL_MAT_ID`): `self` x `[T, K]` (float), `w` the
    /// expert stack `[E, N, K]` (float or quantized, each expert in `matmul_t` form), `ids`
    /// i32 `[T, R]` -> `[T, R, N]` (dtype of `self`) with `out[t, r] = w[ids[t, r]] @ x[t]`.
    /// An id outside `[0, E)` makes eval fail with "index_out_of_range".
    pub fn matmul_id(&self, w: &Tensor, ids: &Tensor) -> Result<Tensor> {
        check::float_dtype(self, "matmul_id input")?;
        if !(w.dtype().is_float() || w.dtype().is_quant()) {
            bail!("unsupported_dtype", "matmul_id experts must be float or quantized, got {}", w.dtype().name());
        }
        if ids.dtype() != DType::I32 {
            bail!("unsupported_dtype", "matmul_id ids must be i32, got {}", ids.dtype().name());
        }
        if self.rank() != 2 || w.rank() != 3 || ids.rank() != 2 {
            bail!(
                "shape_mismatch",
                "matmul_id needs x [T, K], experts [E, N, K], ids [T, R]; got {:?}, {:?}, {:?}",
                self.shape(),
                w.shape(),
                ids.shape()
            );
        }
        let (t, k) = (self.shape()[0], self.shape()[1]);
        let (e, n, wk) = (w.shape()[0], w.shape()[1], w.shape()[2]);
        if wk != k || ids.shape()[0] != t {
            bail!(
                "shape_mismatch",
                "matmul_id: x {:?}, experts {:?}, ids {:?} (need x [T, K], experts [E, N, K], ids [T, R])",
                self.shape(),
                w.shape(),
                ids.shape()
            );
        }
        let r = ids.shape()[1];
        if e == 0 && t * r > 0 {
            bail!("invalid_argument", "matmul_id with an empty expert stack");
        }
        let srcs: Srcs = smallvec![self.clone(), w.clone(), ids.clone()];
        Ok(Tensor::from_op(Op::MatMulId, srcs, vec![t, r, n], self.dtype()))
    }

    /// Gated delta rule (ggml `GATED_DELTA_NET`): q, k `[B, T, H, Dk]`, v `[B, T, H, Dv]`,
    /// g (log decay) and beta `[B, T, H]`, state `[B, H, Dk, Dv]`, all f32. Per (b, h),
    /// sequentially over t: `S = exp(g) S; S += k ⊗ (beta (v - Sᵀk)); o = Sᵀ (q scale)`.
    /// Returns `[B, H, T + Dk, Dv]`: rows `[0, T)` are the outputs, rows `[T, T + Dk)` the
    /// final state.
    pub fn gated_delta_net(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        state: &Tensor,
        scale: f32,
    ) -> Result<Tensor> {
        for (name, x) in [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta), ("state", state)] {
            if x.dtype() != DType::F32 {
                bail!("unsupported_dtype", "gated_delta_net {name} must be f32, got {}", x.dtype().name());
            }
        }
        if q.rank() != 4 || v.rank() != 4 {
            bail!("shape_mismatch", "gated_delta_net q / v must be [B, T, H, D], got {:?} / {:?}", q.shape(), v.shape());
        }
        let (b, t, h, dk) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
        let dv = v.shape()[3];
        let ok = k.shape() == [b, t, h, dk]
            && v.shape() == [b, t, h, dv]
            && g.shape() == [b, t, h]
            && beta.shape() == [b, t, h]
            && state.shape() == [b, h, dk, dv];
        if !ok {
            bail!(
                "shape_mismatch",
                "gated_delta_net: q {:?} k {:?} v {:?} g {:?} beta {:?} state {:?} (need q/k [B,T,H,Dk], v [B,T,H,Dv], g/beta [B,T,H], state [B,H,Dk,Dv])",
                q.shape(),
                k.shape(),
                v.shape(),
                g.shape(),
                beta.shape(),
                state.shape()
            );
        }
        if !scale.is_finite() {
            bail!("invalid_argument", "gated_delta_net scale must be finite, got {scale}");
        }
        let srcs: Srcs = smallvec![q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), state.clone()];
        Ok(Tensor::from_op(Op::GatedDeltaNet { scale }, srcs, vec![b, h, t + dk, dv], DType::F32))
    }
}
