//! Reduction over the last axis of the iteration space with a fused prologue expression
//! (template `kernels/reduce.ptx`). Row variant: one block per output (warp shuffles +
//! shared memory). Thread variant: one thread per output (short reduced axis).

use std::sync::Arc;

use lapid_compiler::{ReduceDims, ReduceSpec};
use lapid_tensor::{ReduceOp, Result};

use super::expr::{check_root, render, uses, uses_as};
use super::ptx::{self, Gen, Red, block_reduce, decompose, fimm, strided_index};
use super::{Key, Params, Plan, bad, check_plain, grid_threads, grid1, template};
use crate::context::Caps;

const KERNEL: &str = include_str!("../kernels/reduce.ptx");

pub(super) fn plan(spec: &Arc<ReduceSpec>, d: &ReduceDims, caps: &Caps) -> Result<Plan> {
    let r = spec.rank;
    if r == 0 || d.shape.len() != r {
        return Err(bad(format!("reduce: rank {r} vs shape {:?}", d.shape)));
    }
    if d.in_views.len() != spec.in_dtypes.len() {
        return Err(bad("reduce: in_views / in_dtypes length mismatch"));
    }
    for &t in &spec.in_dtypes {
        check_plain(t, "reduce input")?;
    }
    check_plain(spec.out_dtype, "reduce output")?;
    if spec.expr.num_loads() > spec.in_dtypes.len() {
        return Err(bad("reduce: expression loads more inputs than bound"));
    }
    check_root(&spec.expr, false, "reduce")?;
    let inner = d.shape[r - 1];
    let outer: usize = d.shape[..r - 1].iter().product();
    let mut p = Params::default();
    p.u(outer)?.u(inner)?;
    for &s in &d.shape[..r - 1] {
        p.u(s)?;
    }
    for v in &d.in_views {
        p.view(v, r)?;
    }
    let row = inner >= 64 || outer < 256;
    let grid = if row {
        grid1(outer as u64, caps)?
    } else {
        grid_threads(outer as u64, caps)?
    };
    Ok(Plan::new(Key::Reduce(spec.clone(), row), p.0, grid, 256))
}

pub(super) fn source(spec: &ReduceSpec, row: bool) -> String {
    let r = spec.rank;
    let n = spec.in_dtypes.len();
    let names = ptx::io_names(n);
    let (args, ptrs) = ptx::args(&names.iter().map(String::as_str).collect::<Vec<_>>());
    // params: [outer, inner, shape[0..r-1], in_i(off, strides[r])...]
    let in_base = |i: usize| 2 + (r - 1) + i * (1 + r);
    let (ident, op) = match spec.op {
        ReduceOp::Sum => (0.0f32, Red::Sum),
        ReduceOp::Max => (f32::NEG_INFINITY, Red::Max),
    };
    // setup (function scope): per-input row base and step along the reduced axis
    let mut s = Gen::named("s_");
    let coords = decompose(&mut s, "%o", 2, r - 1);
    let mut bases = vec![(String::new(), String::new()); n];
    for (i, b) in bases.iter_mut().enumerate() {
        if uses(&spec.expr, i) {
            let base = strided_index(&mut s, in_base(i), &coords);
            let step = s.param(in_base(i) + r);
            *b = (base, step);
        }
    }
    // loop body: load, prologue expression, combine
    let mut g = Gen::new();
    let mut xs = vec![(String::new(), String::new()); n];
    for (i, x) in xs.iter_mut().enumerate() {
        if uses(&spec.expr, i) {
            let idx = g.reg("u32");
            g.op(format!(
                "mad.lo.u32 {idx}, %j, {}, {}",
                bases[i].1, bases[i].0
            ));
            let (wf, wb) = (uses_as(&spec.expr, i, false), uses_as(&spec.expr, i, true));
            *x = ptx::load_input(&mut g, spec.in_dtypes[i], &format!("%g_in{i}"), &idx, wf, wb, false);
        }
    }
    let v = render(&mut g, &spec.expr, &|i, bits| {
        if bits { xs[i].1.clone() } else { xs[i].0.clone() }
    });
    g.op(format!("{} %acc, %acc, {v}", op.instr()));
    let mut st = Gen::new();
    ptx::store(&mut st, spec.out_dtype, "%g_out", "%o", "%acc");
    let br = if row {
        block_reduce(op, "%acc")
    } else {
        String::new()
    };
    template::fill(
        KERNEL,
        &[
            ("ARGS", &args),
            ("PTRS", &ptrs),
            ("SETUP", &s.finish_flat()),
            ("IDENTITY", &fimm(ident)),
            ("BODY", &g.finish()),
            ("BLOCK_REDUCE", &br),
            ("STORE", &st.finish()),
        ],
        if row { &["ROW"] } else { &[] },
    )
}
