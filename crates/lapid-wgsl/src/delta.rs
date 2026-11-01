//! `Kernel::GatedDeltaNet` (gated delta rule, ggml GATED_DELTA_NET): template
//! `kernels/gated_delta.wgsl`. Inputs are contiguous f32.

use std::sync::Arc;

use lapid_compiler::{GatedDeltaNetDims, GatedDeltaNetSpec};
use lapid_tensor::Result;

use super::gemv::gemv_cols;
use super::{Key, Params, Plan, bad, check_plain, common, grid, template};
use crate::Caps;

const KERNEL: &str = include_str!("kernels/gated_delta.wgsl");
/// Registers per lane for the state column (template `MAXP`).
const MAXP: usize = 16;

pub(super) fn plan(spec: &Arc<GatedDeltaNetSpec>, d: &GatedDeltaNetDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.out_dtype, "gated_delta_net output")?;
    let lanes = if caps.subgroup { caps.subgroup_size.max(1) as usize } else { 32 };
    if d.dk > MAXP * lanes {
        return Err(bad(format!("gated_delta_net: dk {} > {} on this device", d.dk, MAXP * lanes)));
    }
    let ncb = d.dv.div_ceil(gemv_cols(caps));
    let mut p = Params::default();
    p.u(d.b)?.u(d.t)?.u(d.h)?.u(d.dk)?.u(d.dv)?.bits(d.scale).u(ncb)?;
    p.u(d.b * d.h * (d.t + d.dk) * d.dv)?;
    let items = (d.b * d.h * ncb) as u64;
    Ok(Plan { scratch: 0, next: None, key: Key::GatedDeltaNet(spec.clone()), params: p.0, groups: grid(items, caps) })
}

pub(super) fn source(spec: &GatedDeltaNetSpec, caps: &Caps) -> String {
    let bindings = common::output(7, spec.out_dtype, caps);
    template::module(caps, KERNEL, &[("BINDINGS", &bindings)], &[], 256, true)
}
