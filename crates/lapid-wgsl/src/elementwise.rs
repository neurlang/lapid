//! Elementwise kernel: grid-stride loop over the iteration space; strided views decoded
//! from the linear index, with structural fast paths for contiguous and scalar views.

use std::fmt::Write;
use std::sync::Arc;

use lapid_compiler::{EwDims, EwSpec, View};

use super::common::{self, decompose, strided_index};
use super::template;
use super::expr::{self, render, uses, uses_bits, uses_load};
use super::{Key, Params, Plan, bad, check_plain, grid_threads};
use crate::Caps;
use lapid_compiler::ExprType;
use lapid_tensor::{DType, Result};

const KERNEL: &str = include_str!("kernels/elementwise.wgsl");

/// How a view is indexed (compile-time structure, chosen per launch).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum IdxMode {
    /// Contiguous row-major over the iteration space: `offset + lin`.
    Flat,
    /// Every element is the same: `offset`.
    Scalar,
    /// General: `offset + sum(c[d] * stride[d])`.
    Strided,
}

pub fn mode_of(view: &View, shape: &[usize]) -> IdxMode {
    if view.strides.len() != shape.len() {
        return IdxMode::Strided; // rejected later by `Params::view`
    }
    let contiguous = View::contiguous(shape);
    let mut flat = true;
    let mut scalar = true;
    for (d, &len) in shape.iter().enumerate() {
        if len <= 1 {
            continue;
        }
        if view.strides[d] != contiguous.strides[d] {
            flat = false;
        }
        if view.strides[d] != 0 {
            scalar = false;
        }
    }
    if flat {
        IdxMode::Flat
    } else if scalar {
        IdxMode::Scalar
    } else {
        IdxMode::Strided
    }
}

pub(super) fn plan(spec: &Arc<EwSpec>, d: &EwDims, caps: &Caps) -> Result<Plan> {
    let r = spec.rank;
    if d.shape.len() != r || d.out_view.strides.len() != r {
        return Err(bad(format!("elementwise: rank {r} vs shape {:?}", d.shape)));
    }
    if d.in_views.len() != spec.in_dtypes.len() {
        return Err(bad("elementwise: in_views / in_dtypes length mismatch"));
    }
    for &t in &spec.in_dtypes {
        check_plain(t, "elementwise input")?;
    }
    check_plain(spec.out_dtype, "elementwise output")?;
    // an integer root is stored as a raw word (I32 outputs only)
    let root = if spec.out_dtype == DType::I32 { None } else { Some(ExprType::F32) };
    expr::check(&spec.expr, &spec.in_dtypes, root, "elementwise")?;
    let numel: usize = d.shape.iter().product();
    let modes: Vec<IdxMode> = d.in_views.iter().map(|v| mode_of(v, &d.shape)).collect();
    let mut out_mode = mode_of(&d.out_view, &d.shape);
    if out_mode == IdxMode::Scalar && numel > 1 {
        // writing many elements to one address: keep the general path (last writer wins)
        out_mode = IdxMode::Strided;
    }
    let mut p = Params::default();
    p.u(numel)?;
    for &s in &d.shape {
        p.u(s)?;
    }
    p.view(&d.out_view, r)?;
    for v in &d.in_views {
        p.view(v, r)?;
    }
    Ok(Plan { scratch: 0, next: None,
        key: Key::Ew(spec.clone(), modes, out_mode),
        params: p.0,
        groups: grid_threads(numel as u64, caps),
    })
}

fn index(mode: IdxMode, base: usize, coords: &[String]) -> String {
    match mode {
        IdxMode::Flat => format!("P[{base}u] + lin"),
        IdxMode::Scalar => format!("P[{base}u]"),
        IdxMode::Strided => strided_index(base, coords),
    }
}

pub(super) fn source(spec: &EwSpec, modes: &[IdxMode], out_mode: IdxMode, caps: &Caps) -> String {
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

    // params layout: [numel, shape[r], out(off, strides[r]), in_i(off, strides[r])...]
    let out_base = 1 + r;
    let in_base = |i: usize| 1 + r + (1 + r) * (1 + i);
    let need_coords = out_mode == IdxMode::Strided
        || modes.iter().enumerate().any(|(i, &m)| m == IdxMode::Strided && uses(&spec.expr, i));
    let (coords_code, coords) = if need_coords { decompose("lin", 1, r, "        ") } else { (String::new(), Vec::new()) };
    let mut loads = String::new();
    for (i, &mode) in modes.iter().enumerate().take(n) {
        if uses_load(&spec.expr, i) {
            let _ = writeln!(loads, "        let x{i} = ld_in{i}({});", index(mode, in_base(i), &coords));
        }
        if uses_bits(&spec.expr, i) {
            let _ = writeln!(loads, "        let xb{i} = ldb_in{i}({});", index(mode, in_base(i), &coords));
        }
    }
    let value = render(&spec.expr, &|i, bits| if bits { format!("xb{i}") } else { format!("x{i}") });
    let store = if spec.out_dtype == DType::I32 && spec.expr.ty() == Some(ExprType::Int) { "st_bits" } else { "st" };
    let out_index = index(out_mode, out_base, &coords);
    template::module(
        caps,
        KERNEL,
        &[
            ("BINDINGS", &bindings),
            ("COORDS", &coords_code),
            ("LOADS", &loads),
            ("STORE", store),
            ("OUT_INDEX", &out_index),
            ("VALUE", &value),
        ],
        &[],
        256,
        false,
    )
}
