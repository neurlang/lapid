//! Row kernels: softmax, layer norm (threadgroup per row) and gather (thread per element).
//! Templates: `kernels/softmax.metal`, `kernels/layer_norm.metal`, `kernels/gather.metal`.

use std::sync::Arc;

use lapid_compiler::{GatherDims, GatherSpec, LayerNormDims, LayerNormSpec, RowDims, SoftmaxSpec};
use lapid_tensor::{DType, Result};

use super::msl::bindings;
use super::template::{self, GATHER, LAYER_NORM, SOFTMAX};
use super::index::STATUS_DECL;
use super::{Bind, Key, Params, Stage, bad, check_plain, grid_items, grid_threads, io_binds};

/// Threads for a row of `cols`: ~4 elements per thread, 32..=256.
fn row_threads(cols: usize) -> u32 {
    (cols.div_ceil(4).div_ceil(32) * 32).clamp(32, 256) as u32
}

pub(super) fn plan_softmax(s: &Arc<SoftmaxSpec>, d: &RowDims) -> Result<Stage> {
    check_plain(s.in_dtype, "softmax input")?;
    check_plain(s.out_dtype, "softmax output")?;
    let mut p = Params::default();
    p.u(d.rows)?.u(d.cols)?;
    let groups = if d.cols == 0 { [0; 3] } else { grid_items(d.rows) };
    Ok(Stage { key: Key::Softmax(s.clone()), params: p.0, groups, threads: row_threads(d.cols), binds: io_binds(1) })
}

pub(super) fn source_softmax(s: &SoftmaxSpec) -> String {
    let binds = bindings(&[("x".into(), s.in_dtype)], Some(s.out_dtype), &[]);
    template::module(SOFTMAX, &[("BINDINGS", &binds)], &[])
}

pub(super) fn plan_layer_norm(s: &Arc<LayerNormSpec>, d: &LayerNormDims) -> Result<Stage> {
    check_plain(s.in_dtype, "layer_norm input")?;
    check_plain(s.param_dtype, "layer_norm params")?;
    check_plain(s.out_dtype, "layer_norm output")?;
    let mut p = Params::default();
    p.u(d.rows)?.u(d.cols)?.bits(d.eps);
    let groups = if d.cols == 0 { [0; 3] } else { grid_items(d.rows) };
    Ok(Stage { key: Key::LayerNorm(s.clone()), params: p.0, groups, threads: row_threads(d.cols), binds: io_binds(3) })
}

pub(super) fn source_layer_norm(s: &LayerNormSpec) -> String {
    let binds = bindings(
        &[("x".into(), s.in_dtype), ("w".into(), s.param_dtype), ("b".into(), s.param_dtype)],
        Some(s.out_dtype),
        &[],
    );
    template::module(LAYER_NORM, &[("BINDINGS", &binds)], &[])
}

const GATHER_THREADS: u32 = 256;

pub(super) fn plan_gather(s: &Arc<GatherSpec>, d: &GatherDims) -> Result<Stage> {
    check_plain(s.out_dtype, "gather output")?;
    let mut p = Params::default();
    p.u(d.rows)?.u(d.cols)?.u(d.count)?;
    let scales = match s.table_dtype.quant_format() {
        Some(f) => {
            if !d.cols.is_multiple_of(f.block_elems()) {
                return Err(bad(format!("gather: quantized table cols {} not a multiple of {}", d.cols, f.block_elems())));
            }
            if f.is_aos() { 0 } else { lapid_quant::layout(f, d.rows, d.cols).scales_offset / 2 }
        }
        None => 0,
    };
    p.u(scales)?;
    let n = d.count.checked_mul(d.cols).ok_or_else(|| bad("gather: size overflow"))?;
    p.u(n)?;
    Ok(Stage {
        key: Key::Gather(s.clone()),
        params: p.0,
        groups: grid_threads(n, GATHER_THREADS),
        threads: GATHER_THREADS,
        binds: vec![Bind::Input(0), Bind::Input(1), Bind::Output, Bind::Status],
    })
}

pub(super) fn source_gather(s: &GatherSpec) -> String {
    let load = match s.table_dtype {
        DType::Int8 => "ld_int8(table, P[3], uint(idx) * cols + c)",
        DType::Int4 => "ld_int4(table, P[3], uint(idx) * cols + c)",
        DType::Q4K => "ld_q4k(table, uint(idx) * cols + c)",
        DType::Q5K => "ld_q5k(table, uint(idx) * cols + c)",
        DType::Q6K => "ld_q6k(table, uint(idx) * cols + c)",
        _ => "ldv(table, uint(idx) * cols + c)",
    };
    let binds = bindings(&[("table".into(), s.table_dtype), ("indices".into(), DType::I32)], Some(s.out_dtype), &[STATUS_DECL]);
    let slots = [("BINDINGS", binds.as_str()), ("LOAD", load)];
    if matches!(s.table_dtype, DType::Q4K | DType::Q5K | DType::Q6K) {
        template::module_kq(GATHER, &slots, &[])
    } else {
        template::module(GATHER, &slots, &[])
    }
}
