//! WGSL code generation lives in `lapid-wgsl` (shared with the WebGPU backend); this module
//! adapts it to the Vulkan device capabilities.

use lapid_compiler::Kernel;
use lapid_tensor::Result;

pub(crate) use lapid_wgsl::{Extra, IdxMode, Key, Plan};

use crate::context::Caps;

impl Caps {
    /// The code-generation view of the device capabilities.
    pub fn wgsl(&self) -> lapid_wgsl::Caps {
        lapid_wgsl::Caps {
            f16: self.f16,
            subgroup: self.subgroup,
            subgroup_size: self.subgroup_size,
            full_subgroups: self.full_subgroups,
            max_groups: self.max_groups,
            max_shared: self.max_shared,
            coopmat: self.coopmat,
            webgpu: false,
        }
    }
}

pub(crate) fn plan(kernel: &Kernel, caps: &Caps) -> Result<Plan> {
    lapid_wgsl::plan(kernel, &caps.wgsl())
}

/// Generate the WGSL source for `key`.
pub(crate) fn source(key: &Key, caps: &Caps) -> String {
    lapid_wgsl::source(key, &caps.wgsl())
}
