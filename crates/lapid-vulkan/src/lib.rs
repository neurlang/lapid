//! lapid-vulkan: Vulkan compute backend.
//!
//! Kernels are WGSL templates (`src/kernels/*.wgsl`) whose slots are filled per kernel spec,
//! compiled with naga to SPIR-V and cached as compute pipelines; sizes travel in a params storage buffer so one pipeline serves every
//! shape. Memory is sub-allocated from large blocks, preferring device-local + host-visible
//! memory (unified memory on GB10 / ReBAR) so uploads and downloads are plain memcpys.
//!
//! Environment:
//! - `LAPID_VK_VALIDATION=1` enables the Khronos validation layer when installed.
//! - `LAPID_DEBUG=1` prints generated WGSL (with line numbers) on shader compile errors and
//!   writes every compiled kernel to `$LAPID_KERNEL_DIR` (default `<tmp>/lapid-kernels`).
//! - `LAPID_PROFILE=1` records GPU timestamps around every launch and prints per-kernel
//!   totals when the backend is dropped (`VulkanBackend::profile_report`).
//! - Compiled kernels (SPIR-V) and the driver's pipeline cache persist under
//!   `<lapid cache>/kernels/vulkan/` (`LAPID_NO_KERNEL_CACHE=1` disables; see `disk_cache`).
//! - `LAPID_VK_NO_REPLAY=1` disables command-buffer replay of repeated launch sequences;
//!   `LAPID_VK_REPLAY_STATS=1` prints replay counters when the backend drops.
//! - `LAPID_VK_NO_F16`, `LAPID_VK_NO_SUBGROUPS`, `LAPID_VK_NO_INT_DOT` force the fallback
//!   code paths (testing).

mod buffer;
mod codegen;
mod context;
mod disk_cache;
mod execute;
mod memory;
mod pipeline;
mod profile;
mod replay;
#[cfg(test)]
mod bench_tests;
#[cfg(test)]
mod golden_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_ops;
#[cfg(test)]
mod tests_attn;
#[cfg(test)]
mod tests_kquant;
#[cfg(test)]
mod tests_split;

use std::sync::{Arc, Mutex};

use lapid_compiler::{Backend, Launch};
use lapid_tensor::{DeviceData, Error, Result};

pub use buffer::VkBuffer;

use crate::buffer::as_vk;
use crate::context::Shared;
use crate::execute::{Exec, Timing};
use crate::memory::MemUsage;
use crate::pipeline::Pipelines;

/// A Vulkan physical device usable for compute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterInfo {
    pub index: usize,
    pub name: String,
    /// "discrete" | "integrated" | "virtual" | "cpu" | "other"
    pub kind: String,
    pub is_software: bool,
}

/// List usable adapters. Loads libvulkan at runtime; `Err("vulkan_unavailable")` when the
/// loader or a driver is missing. Never panics.
pub fn adapters() -> Result<Vec<AdapterInfo>> {
    match std::panic::catch_unwind(context::enumerate_adapters) {
        Ok(r) => r,
        Err(_) => Err(Error::new("vulkan_unavailable", "vulkan enumeration panicked")),
    }
}

/// The Vulkan backend for one adapter.
pub struct VulkanBackend {
    shared: Arc<Shared>,
    pipelines: Pipelines,
    exec: Mutex<Exec>,
}

impl VulkanBackend {
    /// Lower the storage binding limit below the device's (tests: forces split launches).
    #[doc(hidden)]
    pub fn set_binding_limit(&self, limit: Option<u64>) -> Result<()> {
        self.lock_exec()?.binding_limit = limit;
        Ok(())
    }

    /// Open adapter `index` (index into `adapters()`).
    pub fn new(index: usize) -> Result<VulkanBackend> {
        let shared = context::shared_device(index)?;
        let exec = Exec::new(&shared)?;
        let pipelines = Pipelines::new(&shared);
        Ok(VulkanBackend { shared, pipelines, exec: Mutex::new(exec) })
    }

    /// Short capability summary, e.g. "f16, int8-dot, subgroups(32)".
    pub fn features(&self) -> String {
        let c = &self.shared.caps;
        let mut v = Vec::new();
        if c.f16 {
            v.push("f16".to_string());
        }
        if c.int_dot {
            v.push("int8-dot".to_string());
        }
        if c.subgroup {
            v.push(format!("subgroups({})", c.subgroup_size));
        }
        if c.coopmat {
            v.push("coopmat".to_string());
        }
        if c.push_descriptor {
            v.push("push-descriptors".to_string());
        }
        v.join(", ")
    }

    /// Write the final WGSL of every kernel compiled so far to `dir` (one file per
    /// pipeline, `<kind>-<hash>.wgsl`); returns the number of files. `LAPID_DEBUG=1` also
    /// writes each kernel as it is compiled (to `$LAPID_KERNEL_DIR`, default
    /// `<tmp>/lapid-kernels`).
    pub fn dump_kernels(&self, dir: &std::path::Path) -> Result<usize> {
        self.pipelines.dump(dir)
    }

    /// Number of compiled pipelines (diagnostics).
    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    /// Run `launches` once and return the GPU time (ms) of each launch, measured with
    /// timestamps (no host overhead; a launch includes the barrier wait in front of it).
    /// Empty launches (nothing to do) are skipped in the result.
    /// `Err("unsupported")` when the queue has no timestamps.
    pub fn time_launches(&self, launches: &[Launch], buffers: &[DeviceData]) -> Result<Vec<f64>> {
        if self.shared.caps.timestamp_period <= 0.0 {
            return Err(Error::new("unsupported", "vulkan queue has no timestamp support"));
        }
        let mut out = Vec::new();
        self.lock_exec()?.run(&self.shared, &self.pipelines, launches, buffers, Timing::Collect(&mut out))?;
        Ok(out)
    }

    /// Per-kernel GPU time totals collected so far (`LAPID_PROFILE=1`), or `None` when
    /// profiling is off.
    pub fn profile_report(&self) -> Option<String> {
        let e = self.exec.lock().ok()?;
        if !e.profiling {
            return None;
        }
        e.prof.as_ref().map(|p| p.report())
    }

    /// Command-buffer replay counters (diagnostics).
    pub fn replay_stats(&self) -> Option<String> {
        let e = self.exec.lock().ok()?;
        let [full, partial, chunks, recorded] = e.replay.stats;
        Some(format!("replay: {full} full hits, {partial} partial hits ({chunks} chunks re-recorded), {recorded} sequences recorded"))
    }

    /// Clear the profile totals.
    pub fn profile_reset(&self) {
        if let Ok(mut e) = self.exec.lock()
            && let Some(p) = e.prof.as_mut() {
                p.reset();
            }
    }

    fn lock_exec(&self) -> Result<std::sync::MutexGuard<'_, Exec>> {
        self.exec.lock().map_err(|_| Error::new("vulkan", "execution state poisoned"))
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        // SAFETY: wait for all work before destroying pipelines and exec objects. Buffers
        // keep `Shared` (device + allocator) alive on their own.
        self.shared.wait_idle();
        if let Some(r) = self.profile_report() {
            eprintln!("[lapid-vulkan] GPU profile ({}):\n{r}", self.shared.name);
        }
        if std::env::var("LAPID_VK_REPLAY_STATS").is_ok_and(|v| !v.is_empty() && v != "0")
            && let Some(r) = self.replay_stats() {
                eprintln!("[lapid-vulkan] {r}");
            }
        self.pipelines.save(&self.shared);
        self.pipelines.destroy(&self.shared.device);
        if let Ok(mut e) = self.exec.lock() {
            e.destroy(&self.shared.device);
        }
    }
}

impl Backend for VulkanBackend {
    fn name(&self) -> &'static str {
        "vulkan"
    }

    fn device_name(&self) -> String {
        self.shared.name.clone()
    }

    fn alloc(&self, bytes: usize) -> Result<DeviceData> {
        Ok(Arc::new(VkBuffer::create(&self.shared, bytes, MemUsage::Device)?))
    }

    fn upload(&self, bytes: &[u8]) -> Result<DeviceData> {
        let buf = VkBuffer::create(&self.shared, bytes.len(), MemUsage::Device)?;
        if bytes.is_empty() {
            return Ok(Arc::new(buf));
        }
        if buf.mapped().is_some() {
            buf.write_mapped(0, bytes)?;
        } else {
            let staging = VkBuffer::create(&self.shared, bytes.len(), MemUsage::Upload)?;
            staging.write_mapped(0, bytes)?;
            self.lock_exec()?.copy(&self.shared, &staging, &buf, bytes.len() as u64)?;
        }
        Ok(Arc::new(buf))
    }

    fn download(&self, buf: &DeviceData) -> Result<Vec<u8>> {
        let b = as_vk(buf, &self.shared)?;
        let n = b.len_bytes();
        let mut out = vec![0u8; n];
        if n == 0 {
            return Ok(out);
        }
        if b.mapped().is_some() {
            // Serialize with any in-flight execute on another thread.
            let _guard = self.lock_exec()?;
            b.read_mapped(0, &mut out)?;
        } else {
            let staging = VkBuffer::create(&self.shared, n, MemUsage::Readback)?;
            self.lock_exec()?.copy(&self.shared, b, &staging, n as u64)?;
            staging.read_mapped(0, &mut out)?;
        }
        Ok(out)
    }

    fn prepare(&self, launches: &[Launch]) -> Result<()> {
        for l in launches {
            let plan = codegen::plan(&l.kernel, &self.shared.caps)?;
            self.pipelines.get(&self.shared, &plan.key)?;
        }
        self.pipelines.save(&self.shared);
        Ok(())
    }

    fn execute(&self, launches: &[Launch], buffers: &[DeviceData]) -> Result<()> {
        if launches.is_empty() {
            return Ok(());
        }
        let r = self.lock_exec()?.run(&self.shared, &self.pipelines, launches, buffers, Timing::None);
        // persist the driver cache whenever this execute created pipelines (first runs only)
        self.pipelines.save(&self.shared);
        r
    }
}
