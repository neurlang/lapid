//! Top-k indices per row (`kernels/topk.wgsl`): a dedicated argmax variant for k = 1 and a
//! radix-select + bitonic-sort variant for 1 < k <= `MAXK`.

use std::sync::Arc;

use lapid_compiler::{TopKDims, TopKSpec};
use lapid_tensor::{Error, Result};

use super::{Key, Params, Plan, bad, check_plain, common, grid, template};
use crate::Caps;

const KERNEL: &str = include_str!("kernels/topk.wgsl");
/// Largest k of the sorting variant (shared memory: 8 bytes per entry).
pub const MAXK: usize = 1024;

pub(super) fn plan(spec: &Arc<TopKSpec>, d: &TopKDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.in_dtype, "topk input")?;
    if d.rows > 0 && (d.k == 0 || d.k > d.cols) {
        return Err(bad(format!("topk: k = {} with {} columns", d.k, d.cols)));
    }
    if d.k > MAXK {
        return Err(Error::new("unsupported", format!("topk: k = {} > {MAXK} on vulkan", d.k)));
    }
    let mut p = Params::default();
    p.u(d.rows)?.u(d.cols)?.u(d.k)?;
    let groups = if d.k == 0 { [0, 0, 0] } else { grid(d.rows as u64, caps) };
    if d.k > 1 && d.k <= 16 && d.cols <= 1024 && caps.subgroup {
        return Ok(Plan { scratch: 0, next: None, key: Key::TopKSmall(spec.clone()), params: p.0, groups });
    }
    Ok(Plan { scratch: 0, next: None, key: Key::TopK(spec.clone(), d.k == 1), params: p.0, groups })
}

pub(super) fn source_small(spec: &TopKSpec, caps: &Caps) -> String {
    let bindings = common::input(1, "x", spec.in_dtype, caps);
    template::module(caps, KERNEL, &[("BINDINGS", &bindings), ("MAXK", &MAXK.to_string())], &["SMALLK"], 256, true)
}

pub(super) fn source(spec: &TopKSpec, argmax: bool, caps: &Caps) -> String {
    let bindings = common::input(1, "x", spec.in_dtype, caps);
    let maxk = MAXK.to_string();
    let flags: &[&str] = if argmax { &["ARGMAX"] } else { &[] };
    template::module(caps, KERNEL, &[("BINDINGS", &bindings), ("MAXK", &maxk)], flags, 256, argmax)
}
