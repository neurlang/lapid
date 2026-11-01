//! Round-3 kernel: `MatMulId` (mixture-of-experts matmul). `GatedDeltaNet` lives in `delta`.
//!
//! MatMulId groups the (token, slot) pairs by expert and runs one ordinary matmul per used
//! expert (`matmul::run`: GEMV / int8-dot / K-quant / GEMM paths by row count) on the gathered
//! token rows, then scatters the result rows.

use lapid_compiler::{MatMulDims, MatMulIdDims, MatMulIdSpec, MatMulSpec, MatView};
use lapid_tensor::{DType, Error, Result};

use crate::Ctx;
use crate::buffer::CpuBuffer;
use crate::matmul::{self, BInput, Workspace};
use crate::qmat::{self, QMat};
use crate::typed::{Dst, Src};

fn err(code: &'static str, msg: String) -> Error {
    Error::new(code, msg)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_id(
    ctx: &Ctx,
    ws: &mut Workspace,
    spec: &MatMulIdSpec,
    d: &MatMulIdDims,
    x: &Src,
    w: &CpuBuffer,
    ids: &Src,
    out: &Dst,
) -> Result<()> {
    let (t, r, e, n, k) = (d.t, d.r, d.e, d.n, d.k);
    if t * r * n == 0 {
        return Ok(());
    }
    let Src::I32(ids) = ids else {
        return Err(err("unsupported_dtype", "cpu matmul_id: ids must be i32".into()));
    };
    if ids.len() < t * r || x.len() < t * k || out.len() < t * r * n {
        return Err(err("out_of_bounds", format!("cpu matmul_id: buffers too small for t={t} r={r} n={n} k={k}")));
    }
    // Expert -> list of flat (token, slot) indices.
    let mut by_expert: Vec<Vec<usize>> = vec![Vec::new(); e];
    for (i, &id) in ids[..t * r].iter().enumerate() {
        if id < 0 || id as usize >= e {
            return Err(err(
                "index_out_of_range",
                format!("cpu matmul_id: expert id {id} at position {i} out of range for {e} experts"),
            ));
        }
        by_expert[id as usize].push(i);
    }
    let quant = qmat::format_of(spec.w_dtype);
    // SAFETY: launches run sequentially and `w` is an input (never the output buffer).
    let wbytes = unsafe { w.bytes() };
    let one = match quant {
        Some(f) => lapid_quant::layout(f, n, k).total,
        None => n * k,
    };
    if quant.is_some() && wbytes.len() < e * one {
        return Err(err("out_of_bounds", format!("cpu matmul_id: expert stack holds {} < {} bytes", wbytes.len(), e * one)));
    }
    // SAFETY: as above.
    let wsrc = if quant.is_none() { Some(unsafe { w.src(spec.w_dtype)? }) } else { None };
    if let Some(s) = &wsrc
        && s.len() < e * n * k
    {
        return Err(err("out_of_bounds", format!("cpu matmul_id: expert stack holds {} < {} elements", s.len(), e * n * k)));
    }

    let mm_spec = MatMulSpec {
        a_dtype: DType::F32,
        b_dtype: spec.w_dtype,
        b_transposed: true,
        extra_dtypes: Vec::new(),
        out_dtype: DType::F32,
        epilogue: None,
    };
    let mut a_rows: Vec<f32> = Vec::new();
    let mut res: Vec<f32> = Vec::new();
    for (ex, list) in by_expert.iter().enumerate() {
        let m = list.len();
        if m == 0 {
            continue;
        }
        a_rows.resize(m * k, 0.0);
        res.resize(m * n, 0.0);
        for (row, &i) in list.iter().enumerate() {
            x.load_contig((i / r) * k, &mut a_rows[row * k..(row + 1) * k]);
        }
        let (b, b_off) = match (quant, &wsrc) {
            (Some(f), _) => (BInput::Quant(QMat::new(f, &wbytes[ex * one..(ex + 1) * one], n, k)?), 0),
            (None, Some(s)) => (BInput::Plain(*s), ex * n * k),
            (None, None) => return Err(err("internal", "cpu matmul_id: no expert source".into())),
        };
        let dims = MatMulDims {
            batch: 1,
            m,
            n,
            k,
            a: MatView::contiguous(m, k),
            b: MatView { offset: b_off, batch_stride: 0, row_stride: k },
            extra_views: Vec::new(),
            out: MatView::contiguous(m, n),
        };
        let a = Src::F32(&a_rows);
        let dst = Dst::new(res.as_mut_ptr() as *mut u8, res.len(), DType::F32);
        // Prefill-sized launches keep float activations for every expert (as dense
        // matmuls with many rows do), however few rows an expert got: one numeric path.
        ws.float_act = t > 8;
        let r = matmul::run(ctx, ws, &mm_spec, &dims, &a, &b, &[], &dst);
        ws.float_act = false;
        r?;
        for (row, &i) in list.iter().enumerate() {
            // SAFETY: every (token, slot) index appears in exactly one expert list, so the
            // output rows written here are disjoint; single-threaded.
            unsafe { out.store(i * n, 1, &res[row * n..(row + 1) * n]) };
        }
    }
    Ok(())
}
