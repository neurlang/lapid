//! `GpuBuffer`: a wgpu storage buffer, plus mapped readback.

use std::any::Any;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::Arc;

use lapid_tensor::{DeviceBuffer, DeviceData, Error, Result};

use crate::context::Ctx;

/// A WebGPU storage buffer (usage STORAGE | COPY_SRC | COPY_DST). Freed on drop.
pub struct GpuBuffer {
    pub(crate) buffer: wgpu::Buffer,
    len: usize,
    /// Buffer size (>= len, multiple of 16, >= 16).
    pub(crate) size: u64,
    /// Process-unique serial (aliasing checks).
    pub(crate) id: u64,
    ctx_id: u64,
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuBuffer({} bytes)", self.len)
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

impl GpuBuffer {
    /// Create a buffer of `bytes` bytes (zero-filled), optionally initialized with `init`.
    pub(crate) fn create(ctx: &Ctx, bytes: usize, init: Option<&[u8]>) -> Result<GpuBuffer> {
        let size = (bytes as u64).div_ceil(16).max(1) * 16;
        if size > ctx.max_buffer {
            return Err(Error::new("too_large", format!("webgpu: buffer of {bytes} bytes exceeds maxBufferSize ({})", ctx.max_buffer)));
        }
        let init = init.filter(|b| !b.is_empty());
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: init.is_some(),
        });
        ctx.take_error()?;
        if let Some(data) = init {
            {
                let mut view = buffer
                    .get_mapped_range_mut(..)
                    .map_err(|e| Error::new("webgpu", format!("map at creation: {e}")))?;
                view.slice(..data.len()).copy_from_slice(data);
            }
            buffer.unmap();
        }
        Ok(GpuBuffer { buffer, len: bytes, size, id: NEXT_ID.fetch_add(1, Ordering::Relaxed), ctx_id: ctx.id })
    }

    pub fn len_bytes(&self) -> usize {
        self.len
    }
}

impl DeviceBuffer for GpuBuffer {
    fn backend(&self) -> &'static str {
        "webgpu"
    }
    fn len(&self) -> usize {
        self.len
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Downcast a device buffer of this backend (and this device).
pub(crate) fn as_gpu<'a>(d: &'a DeviceData, ctx: &Ctx) -> Result<&'a GpuBuffer> {
    let b = d
        .as_any()
        .downcast_ref::<GpuBuffer>()
        .ok_or_else(|| Error::new("wrong_backend", format!("buffer belongs to backend '{}', not webgpu", d.backend())))?;
    if b.ctx_id != ctx.id {
        return Err(Error::new("wrong_backend", "buffer belongs to another webgpu device"));
    }
    Ok(b)
}

const PENDING: u8 = 0;
const MAPPED: u8 = 1;
const FAILED: u8 = 2;

/// Copy `regions` (buffer, bytes) into one staging buffer, map it and return each region's
/// bytes. Waits for all previously submitted work. Native: blocks on the device; wasm:
/// spins the wait hook (see `set_wait_hook`).
pub(crate) fn read_back(ctx: &Ctx, regions: &[(&wgpu::Buffer, u64)]) -> Result<Vec<Vec<u8>>> {
    let mut offs = Vec::with_capacity(regions.len());
    let mut total = 0u64;
    for &(_, n) in regions {
        offs.push(total);
        total += n.div_ceil(16) * 16;
    }
    if total == 0 {
        return Ok(regions.iter().map(|_| Vec::new()).collect());
    }
    #[cfg(target_arch = "wasm32")]
    let hook = crate::wait_hook().ok_or_else(|| {
        Error::new("would_block", "webgpu: reading GPU memory in the browser needs a wait hook (lapid-web installs one; requires JSPI)")
    })?;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lapid readback"),
        size: total,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    for (&(buf, n), &off) in regions.iter().zip(&offs) {
        let n = n.div_ceil(16) * 16;
        if n > 0 {
            enc.copy_buffer_to_buffer(buf, 0, &staging, off, n);
        }
    }
    ctx.queue.submit([enc.finish()]);
    let state = Arc::new(AtomicU8::new(PENDING));
    let s2 = state.clone();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        s2.store(if r.is_ok() { MAPPED } else { FAILED }, Ordering::Release);
    });
    #[cfg(not(target_arch = "wasm32"))]
    ctx.wait_idle()?;
    #[cfg(target_arch = "wasm32")]
    while state.load(Ordering::Acquire) == PENDING {
        hook();
    }
    ctx.take_error()?;
    if state.load(Ordering::Acquire) != MAPPED {
        return Err(Error::new("webgpu", "mapping the readback buffer failed"));
    }
    let out = {
        let view = staging.get_mapped_range(..).map_err(|e| Error::new("webgpu", format!("readback: {e}")))?;
        regions.iter().zip(&offs).map(|(&(_, n), &off)| view[off as usize..(off + n) as usize].to_vec()).collect()
    };
    staging.unmap();
    Ok(out)
}
