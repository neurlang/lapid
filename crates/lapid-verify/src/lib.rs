//! lapid-verify — model-independent verification and benchmarking engine.
//!
//! * [`reference::Activations`]: named reference activations (safetensors, written by a
//!   reference run of the original model).
//! * [`compare::Comparison`]: compare named activations with error metrics and
//!   tolerances, collecting per-tensor rows and notes.
//! * [`timing`]: timed stages, medians, run statistics.
//! * [`report`]: metrics (`diff`), report rows, dtype tolerances.
//! * [`harness`]: the per-model [`Descriptor`], reference directories, tap naming and the
//!   fake-quant kernel comparison behind the generic `lapid verify` / `lapid bench`.
//!
//! Model wiring (which tensors, prompts, end-to-end checks) lives with the model in
//! `lapid-hub`. Reports are JSON (`serde_json::Value`) for the CLI and for agents.

pub mod compare;
pub mod harness;
pub mod reference;
pub mod report;
pub mod timing;

use lapid::{Device, Result};
use serde_json::Value;

pub use compare::Comparison;
pub use harness::{Descriptor, Reference};
pub use reference::Activations;
pub use report::{Diff, KERNEL_TOLERANCE, LayerReport, argmax_rows, diff, tolerance};
pub use timing::Stages;

/// `None` → `Device::best()`, otherwise `Device::from_name` ("cpu", "vulkan", "vulkan:1").
pub fn open_device(name: Option<&str>) -> Result<Device> {
    match name {
        Some(n) => Device::from_name(n),
        None => Ok(Device::best()),
    }
}

/// Merge the keys of object `extra` into object `into` (existing keys win).
pub fn merge_object(into: &mut Value, extra: Value) {
    if let (Value::Object(map), Value::Object(extra)) = (into, extra) {
        for (k, v) in extra {
            map.entry(k).or_insert(v);
        }
    }
}
