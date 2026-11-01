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
    // decode / short chunks (state I/O dominates): 8 lanes per column on 32-lane full
    // subgroups, every column of a layer resident at once. Long sequences keep 32 lanes per
    // column (measured on GB10, t = 512: 32 lanes 1.2 ms, 8 lanes 1.9 ms, 16 lanes 3.6 ms).
    let lpc = 8;
    if d.t <= 16 && caps.subgroup && caps.subgroup_size == 32 && caps.full_subgroups && d.dk.is_multiple_of(lpc) && d.dk > 0 && d.dk / lpc <= 32 {
        let ncb = d.dv.div_ceil(8 * 32 / lpc);
        let mut p = Params::default();
        p.u(d.b)?.u(d.t)?.u(d.h)?.u(d.dk)?.u(d.dv)?.bits(d.scale).u(ncb)?;
        p.u(d.b * d.h * (d.t + d.dk) * d.dv)?;
        let items = (d.b * d.h * ncb) as u64;
        let key = Key::GatedDeltaNet8(spec.clone(), (d.dk / lpc) as u32, lpc as u32);
        return Ok(Plan { scratch: 0, next: None, key, params: p.0, groups: grid(items, caps) });
    }
    let ncb = d.dv.div_ceil(gemv_cols(caps));
    let mut p = Params::default();
    p.u(d.b)?.u(d.t)?.u(d.h)?.u(d.dk)?.u(d.dv)?.bits(d.scale).u(ncb)?;
    p.u(d.b * d.h * (d.t + d.dk) * d.dv)?;
    let items = (d.b * d.h * ncb) as u64;
    // rows per lane baked in: constant trip counts keep the state column in registers
    let np = d.dk.div_ceil(lanes).max(1) as u32;
    Ok(Plan { scratch: 0, next: None, key: Key::GatedDeltaNet(spec.clone(), np), params: p.0, groups: grid(items, caps) })
}

const KERNEL8: &str = include_str!("kernels/gated_delta8.wgsl");

pub(super) fn source8(spec: &GatedDeltaNetSpec, np: u32, lpc: u32, caps: &Caps) -> String {
    let bindings = common::output(7, spec.out_dtype, caps);
    let (np, cps, lpc) = (np.to_string(), (32 / lpc).to_string(), lpc.to_string());
    template::module(caps, KERNEL8, &[("BINDINGS", &bindings), ("NP", &np), ("LPC", &lpc), ("CPS", &cps)], &[], 256, true)
}

pub(super) fn source(spec: &GatedDeltaNetSpec, np: u32, caps: &Caps) -> String {
    let bindings = common::output(7, spec.out_dtype, caps);
    template::module(caps, KERNEL, &[("BINDINGS", &bindings), ("NP", &np.to_string())], &[], 256, true)
}
