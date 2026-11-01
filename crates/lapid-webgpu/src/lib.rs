//! lapid-webgpu: WebGPU compute backend on `wgpu`.
//!
//! Natively wgpu runs on Vulkan / Metal / DX12 (drivers loaded at runtime, pure Rust); on
//! `wasm32` it drives the browser's `navigator.gpu`. Kernels are the shared `lapid-wgsl`
//! templates rendered with `Caps { webgpu: true, .. }` (no cooperative matrices; `f16` and
//! subgroups when the adapter has them). Sizes travel in a params storage buffer, so one
//! pipeline serves every shape of a kernel spec.
//!
//! Downloads need a buffer map, which is asynchronous on the web: natively the backend
//! blocks on `device.poll`; on wasm `download` repeatedly calls the hook registered with
//! [`set_wait_hook`] (lapid-web installs a JSPI-suspending import that yields to the browser
//! event loop) until the map completes.
//!
//! Environment (native): `LAPID_WGPU_VALIDATION=1` enables the wgpu/driver validation layers.

mod buffer;
mod context;
mod execute;
mod pipeline;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

use std::sync::{Arc, Mutex};

use lapid_compiler::{Backend, Launch};
use lapid_tensor::{DeviceData, Error, Result};

pub use buffer::GpuBuffer;

use crate::context::Ctx;
use crate::execute::Exec;
use crate::pipeline::Pipelines;

/// A WebGPU adapter usable for compute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterInfo {
    pub index: usize,
    pub name: String,
    /// "vulkan" | "metal" | "dx12" | "gl" | "browser" | "other"
    pub backend: String,
    /// "discrete" | "integrated" | "virtual" | "cpu" | "other"
    pub kind: String,
}

/// List the native adapters (`webgpu:N` indexes this list). Never panics.
#[cfg(not(target_arch = "wasm32"))]
pub fn adapters() -> Result<Vec<AdapterInfo>> {
    match std::panic::catch_unwind(context::enumerate) {
        Ok(r) => r,
        Err(_) => Err(Error::new("webgpu_unavailable", "wgpu adapter enumeration panicked")),
    }
}

static WAIT_HOOK: Mutex<Option<fn()>> = Mutex::new(None);

/// wasm32: called repeatedly while `download` waits for a buffer map (lapid-web registers a
/// JSPI-suspending import that yields to the browser event loop). Without a hook, download
/// on wasm returns `Err("would_block")`. Natively unused (downloads block on the device).
pub fn set_wait_hook(hook: fn()) {
    if let Ok(mut h) = WAIT_HOOK.lock() {
        *h = Some(hook);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub(crate) fn wait_hook() -> Option<fn()> {
    WAIT_HOOK.lock().ok().and_then(|h| *h)
}

/// The WebGPU backend for one adapter.
pub struct WebGpuBackend {
    ctx: Arc<Ctx>,
    pipelines: Pipelines,
    exec: Mutex<Exec>,
}

impl WebGpuBackend {
    /// Open adapter `index`. Natively it indexes [`adapters`]; in the browser only index 0
    /// exists (the high-performance `navigator.gpu` adapter).
    pub async fn new_async(index: usize) -> Result<WebGpuBackend> {
        let ctx = Arc::new(Ctx::open(index).await?);
        Ok(WebGpuBackend { pipelines: Pipelines::default(), exec: Mutex::new(Exec::default()), ctx })
    }

    /// Blocking [`WebGpuBackend::new_async`] (native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(index: usize) -> Result<WebGpuBackend> {
        match std::panic::catch_unwind(|| pollster::block_on(WebGpuBackend::new_async(index))) {
            Ok(r) => r,
            Err(_) => Err(Error::new("webgpu_unavailable", "wgpu device creation panicked")),
        }
    }

    /// Short capability summary, e.g. "f16, subgroups(32)".
    pub fn features(&self) -> String {
        let c = &self.ctx.caps;
        let mut v = Vec::new();
        if c.f16 {
            v.push("f16".to_string());
        }
        if c.subgroup {
            v.push(format!("subgroups({})", c.subgroup_size));
        }
        v.push(format!("max binding {} MB", self.ctx.max_binding >> 20));
        v.join(", ")
    }

    /// The WGSL of every kernel compiled so far, as (file name, source).
    pub fn kernel_sources(&self) -> Vec<(String, String)> {
        self.pipelines.sources()
    }

    /// Number of compiled pipelines (diagnostics).
    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    /// Lower the storage binding limit below the adapter's (tests: forces split launches).
    #[doc(hidden)]
    pub fn set_binding_limit(&self, limit: Option<u64>) -> Result<()> {
        self.lock_exec()?.binding_limit = limit;
        Ok(())
    }

    fn lock_exec(&self) -> Result<std::sync::MutexGuard<'_, Exec>> {
        self.exec.lock().map_err(|_| Error::new("webgpu", "execution state poisoned"))
    }
}

impl Backend for WebGpuBackend {
    fn name(&self) -> &'static str {
        "webgpu"
    }

    fn device_name(&self) -> String {
        format!("{} ({})", self.ctx.name, self.ctx.backend)
    }

    fn alloc(&self, bytes: usize) -> Result<DeviceData> {
        Ok(Arc::new(GpuBuffer::create(&self.ctx, bytes, None)?))
    }

    fn upload(&self, bytes: &[u8]) -> Result<DeviceData> {
        Ok(Arc::new(GpuBuffer::create(&self.ctx, bytes.len(), Some(bytes))?))
    }

    fn download(&self, buf: &DeviceData) -> Result<Vec<u8>> {
        let b = buffer::as_gpu(buf, &self.ctx)?;
        let mut exec = self.lock_exec()?;
        exec.download(&self.ctx, b)
    }

    fn prepare(&self, launches: &[Launch]) -> Result<()> {
        for l in launches {
            let mut plan = Some(lapid_wgsl::plan(&l.kernel, &self.ctx.caps)?);
            while let Some(p) = plan {
                if !p.is_empty() {
                    self.pipelines.get(&self.ctx, &p.key)?;
                }
                plan = p.next.map(|n| *n);
            }
        }
        Ok(())
    }

    fn execute(&self, launches: &[Launch], buffers: &[DeviceData]) -> Result<()> {
        if launches.is_empty() {
            return Ok(());
        }
        self.lock_exec()?.run(&self.ctx, &self.pipelines, launches, buffers)
    }
}
