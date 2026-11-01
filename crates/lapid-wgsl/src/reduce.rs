//! Reduction over the last axis of the iteration space with a fused prologue expression.
//! Row variant: one workgroup per output (subgroup / shared-memory reduction).
//! Thread variant: one thread per output (short reduced axis).

use std::fmt::Write;
use std::sync::Arc;

use lapid_compiler::{ExprType, ReduceDims, ReduceSpec};
use lapid_tensor::{ReduceOp, Result};

use super::common::{self, NEG_INF, decompose, strided_index};
use super::template;
use super::expr::{self, render, uses, uses_bits, uses_load};
use super::{Key, Params, Plan, bad, check_plain, grid, grid_threads};
use crate::Caps;

const KERNEL: &str = include_str!("kernels/reduce.wgsl");

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
    expr::check(&spec.expr, &spec.in_dtypes, Some(ExprType::F32), "reduce")?;
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
    let groups = if row { grid(outer as u64, caps) } else { grid_threads(outer as u64, caps) };
    Ok(Plan { scratch: 0, next: None, key: Key::Reduce(spec.clone(), row), params: p.0, groups })
}

pub(super) fn source(spec: &ReduceSpec, row: bool, caps: &Caps) -> String {
    let r = spec.rank;
    let n = spec.in_dtypes.len();
    let mut bindings = String::new();
    for (i, &t) in spec.in_dtypes.iter().enumerate() {
        bindings.push_str(&common::input(1 + i, &format!("in{i}"), t, caps));
        if uses_bits(&spec.expr, i) {
            bindings.push_str(&common::input_bits(&format!("in{i}"), t));
        }
    }
    bindings.push_str(&common::output(1 + n, spec.out_dtype, caps));
    if row {
        bindings.push_str(template::WG_REDUCE);
    }
    // params: [outer, inner, shape[0..r-1], in_i(off, strides[r])...]
    let in_base = |i: usize| 2 + (r - 1) + i * (1 + r);
    let (ident, combine, wg_fn) = match spec.op {
        ReduceOp::Sum => ("0.0".to_string(), "acc + v", "wg_sum"),
        ReduceOp::Max => (NEG_INF.to_string(), "max(acc, v)", "wg_max"),
    };
    // shape of outer dims sits at P[2..2 + r - 1]; decompose expects dim d at base + d.
    let (coords_code, coords) = decompose("o", 2, r - 1, "        ");
    let mut bases = String::new();
    let mut loads = String::new();
    for i in 0..n {
        if !uses(&spec.expr, i) {
            continue;
        }
        let base = in_base(i);
        let _ = writeln!(bases, "        let base{i} = {};", strided_index(base, &coords));
        let _ = writeln!(bases, "        let step{i} = P[{}u];", base + r);
        if uses_load(&spec.expr, i) {
            let _ = writeln!(loads, "            let x{i} = ld_in{i}(base{i} + j * step{i});");
        }
        if uses_bits(&spec.expr, i) {
            let _ = writeln!(loads, "            let xb{i} = ldb_in{i}(base{i} + j * step{i});");
        }
    }
    let value = render(&spec.expr, &|i, bits| if bits { format!("xb{i}") } else { format!("x{i}") });
    template::module(
        caps,
        KERNEL,
        &[
            ("BINDINGS", &bindings),
            ("COORDS", &coords_code),
            ("BASES", &bases),
            ("IDENTITY", &ident),
            ("LOADS", &loads),
            ("VALUE", &value),
            ("COMBINE", combine),
            ("WG_REDUCE", wg_fn),
        ],
        if row { &["ROW"] } else { &[] },
        256,
        row,
    )
}
