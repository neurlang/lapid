//! Elementwise kernels (fused chains, copies, fills). Template: `kernels/elementwise.metal`.

use std::sync::Arc;

use lapid_compiler::{EwDims, EwSpec, ExprType, View};
use lapid_tensor::{DType, Result};

use super::msl::{self, bindings, decompose, strided_index};
use super::template::{self, ELEMENTWISE};
use super::{Idx, Key, Params, Stage, bad, check_plain, grid_threads, io_binds};

const THREADS: u32 = 256;

fn int_root(s: &EwSpec) -> bool {
    s.out_dtype == DType::I32 && s.expr.ty() == Some(ExprType::Int)
}

/// Index mode of `v` over `shape`.
pub(crate) fn mode(v: &View, shape: &[usize]) -> Idx {
    if shape.iter().zip(&v.strides).all(|(&n, &s)| n == 1 || s == 0) {
        return Idx::Scalar;
    }
    let mut expect = 1isize;
    let mut contiguous = true;
    for d in (0..shape.len()).rev() {
        if shape[d] != 1 && v.strides[d] != expect {
            contiguous = false;
            break;
        }
        expect *= shape[d] as isize;
    }
    if contiguous { Idx::Flat } else { Idx::Strided }
}

pub(super) fn plan(s: &Arc<EwSpec>, d: &EwDims) -> Result<Stage> {
    let rank = s.rank;
    if d.shape.len() != rank || d.in_views.len() != s.in_dtypes.len() {
        return Err(bad(format!(
            "elementwise: rank {rank} / {} inputs vs dims rank {} / {} views",
            s.in_dtypes.len(),
            d.shape.len(),
            d.in_views.len()
        )));
    }
    for &dt in &s.in_dtypes {
        check_plain(dt, "elementwise input")?;
    }
    check_plain(s.out_dtype, "elementwise output")?;
    match s.expr.ty() {
        None => return Err(bad("elementwise: ill-typed expression")),
        Some(ExprType::Int) if s.out_dtype != DType::I32 => {
            return Err(bad("elementwise: integer expression root needs an i32 output"));
        }
        _ => {}
    }
    let n: usize = d.shape.iter().product();
    let mut p = Params::default();
    p.u(n)?;
    for &x in &d.shape {
        p.u(x)?;
    }
    for v in &d.in_views {
        p.view(v, rank)?;
    }
    p.view(&d.out_view, rank)?;
    let modes = d.in_views.iter().map(|v| mode(v, &d.shape)).collect();
    let out_mode = mode(&d.out_view, &d.shape);
    Ok(Stage {
        key: Key::Ew(s.clone(), modes, out_mode),
        params: p.0,
        groups: grid_threads(n, THREADS),
        threads: THREADS,
        binds: io_binds(s.in_dtypes.len()),
    })
}

pub(super) fn source(s: &EwSpec, modes: &[Idx], out_mode: Idx) -> String {
    let rank = s.rank;
    let nin = s.in_dtypes.len();
    let view_base = |i: usize| 1 + rank + i * (1 + rank);
    let any_strided = modes.iter().chain(std::iter::once(&out_mode)).any(|m| *m == Idx::Strided);
    let (index, coords) = if any_strided { decompose("i", 1, rank, "        ") } else { (String::new(), Vec::new()) };
    let index_of = |mode: Idx, base: usize| match mode {
        Idx::Flat => format!("P[{base}] + i"),
        Idx::Scalar => format!("P[{base}]"),
        Idx::Strided => strided_index(base, &coords),
    };
    let mut loads = String::new();
    for (i, &m) in modes.iter().enumerate() {
        loads += &msl::load_lines(&s.expr, i, &format!("in{i}"), &index_of(m, view_base(i)), "        ");
    }
    let expr = msl::render(&s.expr);
    // An integer root stores the raw word (i32 output); f32 roots convert at the store.
    let (rty, store) = if int_root(s) { ("uint", "stw") } else { ("float", "stv") };
    let out_index = index_of(out_mode, view_base(nin));
    let ins: Vec<(String, _)> = s.in_dtypes.iter().enumerate().map(|(i, &d)| (format!("in{i}"), d)).collect();
    let binds = bindings(&ins, Some(s.out_dtype), &[]);
    template::module(
        ELEMENTWISE,
        &[
            ("BINDINGS", &binds),
            ("INDEX", &index),
            ("LOADS", &loads),
            ("RTY", rty),
            ("EXPR", &expr),
            ("STORE", store),
            ("OUT_INDEX", &out_index),
        ],
        &[],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes() {
        let shape = [2, 3, 4];
        assert_eq!(mode(&View::contiguous(&shape), &shape), Idx::Flat);
        assert_eq!(mode(&View { offset: 5, strides: vec![0, 0, 0] }, &shape), Idx::Scalar);
        assert_eq!(mode(&View { offset: 0, strides: vec![1, 2, 6] }, &shape), Idx::Strided);
        // size-1 dims never matter
        let shape1 = [2, 1, 4];
        assert_eq!(mode(&View { offset: 3, strides: vec![4, 99, 1] }, &shape1), Idx::Flat);
    }
}
