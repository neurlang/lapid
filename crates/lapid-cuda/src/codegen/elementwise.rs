//! Elementwise kernel (template `kernels/elementwise.ptx`): one thread per element; strided
//! views decoded from the linear index, with structural fast paths for contiguous and
//! scalar (broadcast) views.

use std::sync::Arc;

use lapid_compiler::{EwDims, EwSpec, View};
use lapid_tensor::Result;

use lapid_compiler::{Expr, ExprType};
use lapid_tensor::DType;

use super::expr::{check_root, render_typed, uses, uses_as};
use super::ptx::{self, Gen, decompose, strided_index};
use super::{Key, Params, Plan, bad, check_plain, grid_threads, template};
use crate::context::Caps;

const KERNEL: &str = include_str!("../kernels/elementwise.ptx");

/// How a view is indexed (compile-time structure, chosen per launch).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum IdxMode {
    /// Contiguous row-major over the iteration space: `offset + lin`.
    Flat,
    /// Every element is the same: `offset`.
    Scalar,
    /// General: `offset + sum(c[d] * stride[d])`.
    Strided,
}

pub(crate) fn mode_of(view: &View, shape: &[usize]) -> IdxMode {
    if view.strides.len() != shape.len() {
        return IdxMode::Strided; // rejected later by `Params::view`
    }
    let contiguous = View::contiguous(shape);
    let mut flat = true;
    let mut scalar = true;
    for ((&n, &s), &c) in shape.iter().zip(&view.strides).zip(&contiguous.strides) {
        if n <= 1 {
            continue;
        }
        if s != c {
            flat = false;
        }
        if s != 0 {
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
    if spec.expr.num_loads() > spec.in_dtypes.len() {
        return Err(bad("elementwise: expression loads more inputs than bound"));
    }
    check_root(&spec.expr, spec.out_dtype == DType::I32, "elementwise")?;
    let numel: usize = d.shape.iter().product();
    let modes: Vec<IdxMode> = d.in_views.iter().map(|v| mode_of(v, &d.shape)).collect();
    let mut out_mode = mode_of(&d.out_view, &d.shape);
    if out_mode == IdxMode::Scalar && numel > 1 {
        // many elements to one address: keep the general path (last writer wins)
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
    let grid = grid_threads(numel as u64, caps)?;
    Ok(Plan::new(
        Key::Ew(spec.clone(), modes, out_mode),
        p.0,
        grid,
        256,
    ))
}

fn index(g: &mut Gen, mode: IdxMode, base: usize, coords: &[String]) -> String {
    match mode {
        IdxMode::Flat => {
            let r = g.param(base);
            g.op(format!("add.u32 {r}, {r}, %lin"));
            r
        }
        IdxMode::Scalar => g.param(base),
        IdxMode::Strided => strided_index(g, base, coords),
    }
}

pub(super) fn source(spec: &EwSpec, modes: &[IdxMode], out_mode: IdxMode) -> String {
    let r = spec.rank;
    let n = spec.in_dtypes.len();
    let names = ptx::io_names(n);
    let (args, ptrs) = ptx::args(&names.iter().map(String::as_str).collect::<Vec<_>>());
    // params layout: [numel, shape[r], out(off, strides[r]), in_i(off, strides[r])...]
    let out_base = 1 + r;
    let in_base = |i: usize| 1 + r + (1 + r) * (1 + i);
    let need_coords = out_mode == IdxMode::Strided
        || modes
            .iter()
            .enumerate()
            .any(|(i, &m)| m == IdxMode::Strided && uses(&spec.expr, i));
    let mut g = Gen::new();
    let coords = if need_coords {
        decompose(&mut g, "%lin", 1, r)
    } else {
        Vec::new()
    };
    let copy = match spec.expr {
        Expr::Load(i) if spec.in_dtypes.get(i) == Some(&spec.out_dtype) => Some(i),
        _ => None,
    };
    if let Some(i) = copy {
        // exact element copy (no f32 round trip: keeps every i32 bit)
        let si = index(&mut g, modes[i], in_base(i), &coords);
        let oi = index(&mut g, out_mode, out_base, &coords);
        ptx::copy_raw(&mut g, spec.out_dtype, &format!("%g_in{i}"), &si, "%g_out", &oi, false);
    } else {
        let mut xs = vec![(String::new(), String::new()); n];
        for (i, x) in xs.iter_mut().enumerate() {
            let (wf, wb) = (uses_as(&spec.expr, i, false), uses_as(&spec.expr, i, true));
            if wf || wb {
                let idx = index(&mut g, modes[i], in_base(i), &coords);
                *x = ptx::load_input(&mut g, spec.in_dtypes[i], &format!("%g_in{i}"), &idx, wf, wb, false);
            }
        }
        let v = render_typed(&mut g, &spec.expr, &|i, bits| {
            if bits { xs[i].1.clone() } else { xs[i].0.clone() }
        });
        let oi = index(&mut g, out_mode, out_base, &coords);
        if v.ty == ExprType::Int {
            ptx::store_bits(&mut g, "%g_out", &oi, &v.reg);
        } else {
            ptx::store(&mut g, spec.out_dtype, "%g_out", &oi, &v.reg);
        }
    }
    template::fill(
        KERNEL,
        &[("ARGS", &args), ("PTRS", &ptrs), ("BODY", &g.finish())],
        &[],
    )
}
