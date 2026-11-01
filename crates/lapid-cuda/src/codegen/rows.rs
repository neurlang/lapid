//! Row kernels: softmax and layer norm (one block per row), and gather.

use std::sync::Arc;

use lapid_compiler::{GatherDims, GatherSpec, LayerNormDims, LayerNormSpec, RowDims, SoftmaxSpec};
use lapid_tensor::Result;

use super::ptx::{self, Gen, Red, block_reduce};
use super::{Key, Params, Plan, bad, check_plain, grid_threads, grid1, scales_offset, template};
use crate::context::Caps;

const SOFTMAX: &str = include_str!("../kernels/softmax.ptx");
const LAYER_NORM: &str = include_str!("../kernels/layer_norm.ptx");
const GATHER: &str = include_str!("../kernels/gather.ptx");

pub(super) fn plan_softmax(spec: &Arc<SoftmaxSpec>, d: &RowDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.in_dtype, "softmax input")?;
    check_plain(spec.out_dtype, "softmax output")?;
    let mut p = Params::default();
    p.u(d.rows)?.u(d.cols)?;
    let grid = if d.cols == 0 {
        [0, 0, 0]
    } else {
        grid1(d.rows as u64, caps)?
    };
    Ok(Plan::new(Key::Softmax(spec.clone()), p.0, grid, 256))
}

pub(super) fn plan_layer_norm(
    spec: &Arc<LayerNormSpec>,
    d: &LayerNormDims,
    caps: &Caps,
    rms: bool,
) -> Result<Plan> {
    check_plain(spec.in_dtype, "layer_norm input")?;
    check_plain(spec.param_dtype, "layer_norm params")?;
    check_plain(spec.out_dtype, "layer_norm output")?;
    let mut p = Params::default();
    p.u(d.rows)?.u(d.cols)?.bits(d.eps);
    let grid = if d.cols == 0 {
        [0, 0, 0]
    } else {
        grid1(d.rows as u64, caps)?
    };
    Ok(Plan::new(Key::LayerNorm(spec.clone(), rms), p.0, grid, 256))
}

pub(super) fn plan_gather(spec: &Arc<GatherSpec>, d: &GatherDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.out_dtype, "gather output")?;
    let qb = super::quant_block(spec.table_dtype);
    if spec.table_dtype.is_quant() && !d.cols.is_multiple_of(qb) {
        return Err(bad(format!("gather: quantized table needs cols % {qb} == 0")));
    }
    let so = if spec.table_dtype.is_quant() {
        scales_offset(spec.table_dtype, d.rows, d.cols)
    } else {
        0
    };
    let total = d
        .count
        .checked_mul(d.cols)
        .ok_or_else(|| bad("gather: size overflow"))?;
    let mut p = Params::default();
    p.u(d.rows)?.u(d.cols)?.u(d.count)?.u(so)?.u(total)?;
    let grid = if d.rows == 0 {
        [0, 0, 0]
    } else {
        grid_threads(total as u64, caps)?
    };
    Ok(Plan::new(Key::Gather(spec.clone()), p.0, grid, 256))
}

fn snippet(f: impl FnOnce(&mut Gen)) -> String {
    let mut g = Gen::new();
    f(&mut g);
    g.finish()
}

pub(super) fn source_softmax(spec: &SoftmaxSpec) -> String {
    let (args, ptrs) = ptx::args(&["x", "out"]);
    let load = snippet(|g| ptx::load(g, spec.in_dtype, "%g_x", "%idx", "%x", true));
    let store = snippet(|g| ptx::store(g, spec.out_dtype, "%g_out", "%idx", "%e"));
    template::fill(
        SOFTMAX,
        &[
            ("ARGS", &args),
            ("PTRS", &ptrs),
            ("LOAD_X", &load),
            ("REDUCE_MAX", &block_reduce(Red::Max, "%m")),
            ("REDUCE_SUM", &block_reduce(Red::Sum, "%sum")),
            ("STORE", &store),
        ],
        &[],
    )
}

pub(super) fn source_layer_norm(spec: &LayerNormSpec, rms: bool) -> String {
    let names: &[&str] = if rms { &["x", "w", "out"] } else { &["x", "w", "b", "out"] };
    let (args, ptrs) = ptx::args(names);
    let load = snippet(|g| ptx::load(g, spec.in_dtype, "%g_x", "%idx", "%x", true));
    let load_wb = snippet(|g| {
        ptx::load(g, spec.param_dtype, "%g_w", "%j", "%w", true);
        if !rms {
            ptx::load(g, spec.param_dtype, "%g_b", "%j", "%b", true);
        }
    });
    let store = snippet(|g| ptx::store(g, spec.out_dtype, "%g_out", "%idx", "%y"));
    template::fill(
        LAYER_NORM,
        &[
            ("ARGS", &args),
            ("PTRS", &ptrs),
            ("LOAD_X", &load),
            ("LOAD_WB", &load_wb),
            ("REDUCE_SUM", &block_reduce(Red::Sum, "%sum")),
            ("REDUCE_SQ", &block_reduce(Red::Sum, "%sq")),
            ("STORE", &store),
        ],
        if rms { &["RMS"] } else { &[] },
    )
}

pub(super) fn source_gather(spec: &GatherSpec) -> String {
    let (args, ptrs) = ptx::args(&["tab", "idx", "out", "flag"]);
    let load = snippet(|g| {
        if spec.table_dtype.is_quant() {
            ptx::load_quant(g, spec.table_dtype, "%g_tab", "%so", "%src", "%v");
        } else {
            ptx::load(g, spec.table_dtype, "%g_tab", "%src", "%v", true);
        }
    });
    let store = snippet(|g| ptx::store(g, spec.out_dtype, "%g_out", "%e", "%v"));
    template::fill(
        GATHER,
        &[
            ("ARGS", &args),
            ("PTRS", &ptrs),
            ("LOAD", &load),
            ("STORE", &store),
        ],
        &[],
    )
}
