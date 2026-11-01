//! Row-wise composites: softmax, layer norm and RMS norm, one workgroup per row.

use std::sync::Arc;

use lapid_compiler::{LayerNormDims, LayerNormSpec, RowDims, SoftmaxSpec};
use lapid_tensor::Result;

use super::{common, template};
use super::{Key, Params, Plan, check_plain, grid};
use crate::Caps;

const SOFTMAX: &str = include_str!("kernels/softmax.wgsl");
const LAYER_NORM: &str = include_str!("kernels/layer_norm.wgsl");
const RMS_NORM: &str = include_str!("kernels/rms_norm.wgsl");

pub(super) fn plan_softmax(spec: &Arc<SoftmaxSpec>, d: &RowDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.in_dtype, "softmax input")?;
    check_plain(spec.out_dtype, "softmax output")?;
    let mut p = Params::default();
    p.u(d.rows)?.u(d.cols)?;
    let groups = if d.cols == 0 { [0, 0, 0] } else { grid(d.rows as u64, caps) };
    Ok(Plan { scratch: 0, next: None, key: Key::Softmax(spec.clone()), params: p.0, groups })
}

pub(super) fn plan_layer_norm(spec: &Arc<LayerNormSpec>, d: &LayerNormDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.in_dtype, "layer_norm input")?;
    check_plain(spec.param_dtype, "layer_norm params")?;
    check_plain(spec.out_dtype, "layer_norm output")?;
    let mut p = Params::default();
    p.u(d.rows)?.u(d.cols)?.bits(d.eps);
    let groups = if d.cols == 0 { [0, 0, 0] } else { grid(d.rows as u64, caps) };
    Ok(Plan { scratch: 0, next: None, key: Key::LayerNorm(spec.clone()), params: p.0, groups })
}

pub(super) fn plan_rms_norm(spec: &Arc<LayerNormSpec>, d: &LayerNormDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.in_dtype, "rms_norm input")?;
    check_plain(spec.param_dtype, "rms_norm params")?;
    check_plain(spec.out_dtype, "rms_norm output")?;
    let mut p = Params::default();
    p.u(d.rows)?.u(d.cols)?.bits(d.eps);
    let groups = if d.cols == 0 { [0, 0, 0] } else { grid(d.rows as u64, caps) };
    Ok(Plan { scratch: 0, next: None, key: Key::RmsNorm(spec.clone()), params: p.0, groups })
}

pub(super) fn source_rms_norm(spec: &LayerNormSpec, caps: &Caps) -> String {
    let mut b = common::input(1, "x", spec.in_dtype, caps);
    b.push_str(&common::input(2, "w", spec.param_dtype, caps));
    b.push_str(&common::output(3, spec.out_dtype, caps));
    b.push_str(template::WG_REDUCE);
    template::module(caps, RMS_NORM, &[("BINDINGS", &b)], &[], 256, true)
}

pub(super) fn source_softmax(spec: &SoftmaxSpec, caps: &Caps) -> String {
    let mut b = common::input(1, "x", spec.in_dtype, caps);
    b.push_str(&common::output(2, spec.out_dtype, caps));
    b.push_str(template::WG_REDUCE);
    template::module(caps, SOFTMAX, &[("BINDINGS", &b)], &[], 256, true)
}

pub(super) fn source_layer_norm(spec: &LayerNormSpec, caps: &Caps) -> String {
    let mut b = common::input(1, "x", spec.in_dtype, caps);
    b.push_str(&common::input(2, "w", spec.param_dtype, caps));
    b.push_str(&common::input(3, "bias", spec.param_dtype, caps));
    b.push_str(&common::output(4, spec.out_dtype, caps));
    b.push_str(template::WG_REDUCE);
    template::module(caps, LAYER_NORM, &[("BINDINGS", &b)], &[], 256, true)
}
