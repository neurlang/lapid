//! The Metal backend (Apple targets only).

mod buffer;
mod device;
mod execute;
mod pipeline;

use std::sync::{Mutex, MutexGuard};

use lapid_compiler::{Backend, Launch};
use lapid_tensor::{DeviceData, Result};

pub use buffer::MetalBuffer;
pub use device::devices;

use crate::MetalDeviceInfo;
use crate::codegen;
use device::Ctx;
use execute::Exec;
use pipeline::Pipelines;

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
}

/// The Metal backend for one device.
pub struct MetalBackend {
    ctx: Ctx,
    pipelines: Pipelines,
    exec: Mutex<Exec>,
}

impl MetalBackend {
    /// Open device `index` (index into `devices()`).
    pub fn new(index: usize) -> Result<MetalBackend> {
        let ctx = Ctx::open(index)?;
        let per_cb = std::env::var("LAPID_METAL_PROFILE").is_ok_and(|v| v.trim() == "cb");
        let exec = Exec::new(env_flag("LAPID_PROFILE"), per_cb);
        Ok(MetalBackend { ctx, pipelines: Pipelines::default(), exec: Mutex::new(exec) })
    }

    pub fn info(&self) -> &MetalDeviceInfo {
        &self.ctx.info
    }

    /// Number of compiled pipelines (diagnostics).
    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    /// Run `launches` once, each in its own command buffer, and return the GPU time (ms)
    /// of every launch (0 for launches with nothing to do).
    pub fn time_launches(&self, launches: &[Launch], buffers: &[DeviceData]) -> Result<Vec<f64>> {
        let mut out = Vec::with_capacity(launches.len());
        self.lock_exec().run(&self.ctx, &self.pipelines, launches, buffers, Some(&mut out))?;
        Ok(out)
    }

    /// Compile the pipeline of `key` (tests: every variant must compile).
    #[cfg(test)]
    pub(crate) fn compile_key(&self, key: &codegen::Key) -> Result<()> {
        self.pipelines.get(&self.ctx, key).map(|_| ())
    }

    fn lock_exec(&self) -> MutexGuard<'_, Exec> {
        self.exec.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn device_name(&self) -> String {
        self.ctx.info.name.clone()
    }

    fn alloc(&self, bytes: usize) -> Result<DeviceData> {
        Ok(buffer::wrap(buffer::alloc(&self.ctx, bytes)?))
    }

    fn upload(&self, bytes: &[u8]) -> Result<DeviceData> {
        Ok(buffer::wrap(buffer::upload(&self.ctx, bytes)?))
    }

    fn download(&self, buf: &DeviceData) -> Result<Vec<u8>> {
        // execute() waits for completion, so there is never pending GPU work here; the lock
        // orders the copy after any execute running on another thread.
        let _guard = self.lock_exec();
        Ok(buffer::download(buffer::as_metal(buf, &self.ctx)?))
    }

    fn prepare(&self, launches: &[Launch]) -> Result<()> {
        for l in launches {
            for stage in codegen::plan(&l.kernel)?.stages {
                if !stage.is_empty() {
                    self.pipelines.get(&self.ctx, &stage.key)?;
                }
            }
        }
        Ok(())
    }

    fn execute(&self, launches: &[Launch], buffers: &[DeviceData]) -> Result<()> {
        if launches.is_empty() {
            return Ok(());
        }
        self.lock_exec().run(&self.ctx, &self.pipelines, launches, buffers, None)
    }

    fn take_profile(&self) -> Vec<(&'static str, usize, f64)> {
        self.lock_exec().profile.as_mut().map(|p| p.take()).unwrap_or_default()
    }
}

impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalBackend({})", self.ctx.info.name)
    }
}

#[allow(dead_code)]
fn _assert_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<MetalBackend>();
}
