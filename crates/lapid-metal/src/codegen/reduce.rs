//! Reductions over the last axis of the iteration space, with a fused prologue expression.
//! Templates: `kernels/reduce_row.metal` (threadgroup per row), `kernels/reduce_thread.metal`.

use std::fmt::Write;
use std::sync::Arc;

use lapid_compiler::{ExprType, ReduceDims, ReduceSpec};
use lapid_tensor::{ReduceOp, Result};

use super::msl::{self, bindings, decompose};
use super::template::{self, REDUCE_ROW, REDUCE_THREAD};
use super::{Key, Params, Stage, bad, check_plain, grid_items, grid_threads, io_binds};

pub(super) fn plan(s: &Arc<ReduceSpec>, d: &ReduceDims) -> Result<Stage> {
    let rank = s.rank;
    if rank == 0 || d.shape.len() != rank || d.in_views.len() != s.in_dtypes.len() {
        return Err(bad(format!(
            "reduce: rank {rank} / {} inputs vs dims rank {} / {} views",
            s.in_dtypes.len(),
            d.shape.len(),
            d.in_views.len()
        )));
    }
    for &dt in &s.in_dtypes {
        check_plain(dt, "reduce input")?;
    }
    check_plain(s.out_dtype, "reduce output")?;
    if s.expr.ty() != Some(ExprType::F32) {
        return Err(bad("reduce: the prologue expression must be f32"));
    }
    let len = d.shape[rank - 1];
    let rows: usize = d.shape[..rank - 1].iter().product();
    let mut p = Params::default();
    p.u(rows)?.u(len)?;
    for &x in &d.shape[..rank - 1] {
        p.u(x)?;
    }
    for v in &d.in_views {
        p.view(v, rank)?;
    }
    // Long rows (or too few rows to fill the GPU with one thread each): a threadgroup per row.
    let row = len >= 64 || rows < 4096;
    let (groups, threads) = if row {
        let threads = (len.div_ceil(4).next_power_of_two().clamp(32, 256)) as u32;
        (grid_items(rows), threads)
    } else {
        (grid_threads(rows, 256), 256)
    };
    Ok(Stage { key: Key::Reduce(s.clone(), row), params: p.0, groups, threads, binds: io_binds(s.in_dtypes.len()) })
}

pub(super) fn source(s: &ReduceSpec, row: bool) -> String {
    let rank = s.rank;
    let outer = rank - 1;
    let view_base = |i: usize| 2 + outer + i * (1 + rank);
    let indent = "        ";
    let (mut outer_code, coords) = decompose("o", 2, outer, indent);
    let mut loads = String::new();
    for i in 0..s.in_dtypes.len() {
        if !msl::uses_any(&s.expr, i) {
            continue;
        }
        let base = view_base(i);
        let mut off = format!("P[{base}]");
        for (dd, c) in coords.iter().enumerate() {
            let _ = write!(off, " + {c} * P[{}]", base + 1 + dd);
        }
        let _ = writeln!(outer_code, "{indent}const uint r{i} = {off};");
        let index = format!("r{i} + j * P[{}]", base + rank);
        loads += &msl::load_lines(&s.expr, i, &format!("in{i}"), &index, "            ");
    }
    let expr = msl::render(&s.expr);
    let (ident, combine, tg_reduce) = match s.op {
        ReduceOp::Sum => ("0.0f", "acc + v", "tg_sum"),
        ReduceOp::Max => ("-INFINITY", "max(acc, v)", "tg_max"),
    };
    let ins: Vec<(String, _)> = s.in_dtypes.iter().enumerate().map(|(i, &d)| (format!("in{i}"), d)).collect();
    let binds = bindings(&ins, Some(s.out_dtype), &[]);
    template::module(
        if row { REDUCE_ROW } else { REDUCE_THREAD },
        &[
            ("BINDINGS", &binds),
            ("OUTER", &outer_code),
            ("LOADS", &loads),
            ("EXPR", &expr),
            ("IDENT", ident),
            ("COMBINE", combine),
            ("TG_REDUCE", tg_reduce),
        ],
        &[],
    )
}
