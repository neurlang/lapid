//! Device buffers: `MTLStorageModeShared` (unified memory, CPU-visible), untracked hazards
//! (the executor places barriers itself).

use std::any::Any;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use lapid_tensor::{DeviceBuffer, DeviceData, Error, Result};

use super::device::Ctx;

pub(crate) type RawBuffer = ProtocolObject<dyn MTLBuffer>;

/// A Metal buffer owned by a `MetalBackend` (backend name "metal").
pub struct MetalBuffer {
    raw: Retained<RawBuffer>,
    len: usize,
    device: u64,
}

// SAFETY: MTLBuffer objects are thread-safe to retain/release and to bind from any thread;
// CPU access to the contents only happens while no GPU work on the buffer is in flight
// (execute waits for completion before returning).
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl DeviceBuffer for MetalBuffer {
    fn backend(&self) -> &'static str {
        "metal"
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl MetalBuffer {
    pub(crate) fn raw(&self) -> &RawBuffer {
        &self.raw
    }

    fn contents(&self) -> *mut u8 {
        self.raw.contents().as_ptr().cast::<u8>()
    }
}

/// A raw buffer owned by the backend itself (scratch memory).
pub(crate) struct OwnedRaw(pub Retained<RawBuffer>);

// SAFETY: as for `MetalBuffer`; only used under the backend's execution lock.
unsafe impl Send for OwnedRaw {}
unsafe impl Sync for OwnedRaw {}

const OPTIONS: MTLResourceOptions =
    MTLResourceOptions::StorageModeShared.union(MTLResourceOptions::HazardTrackingModeUntracked);

/// Raw shared buffer of at least `bytes` (never zero-sized).
pub(crate) fn new_raw(ctx: &Ctx, bytes: usize) -> Result<Retained<RawBuffer>> {
    ctx.device.newBufferWithLength_options(bytes.max(4), OPTIONS).ok_or_else(|| {
        Error::new("out_of_memory", format!("metal: cannot allocate a buffer of {bytes} bytes"))
    })
}

pub(crate) fn alloc(ctx: &Ctx, bytes: usize) -> Result<MetalBuffer> {
    Ok(MetalBuffer { raw: new_raw(ctx, bytes)?, len: bytes, device: ctx.id })
}

pub(crate) fn upload(ctx: &Ctx, bytes: &[u8]) -> Result<MetalBuffer> {
    let b = alloc(ctx, bytes.len())?;
    // SAFETY: the buffer is fresh (no GPU work), CPU-visible and at least `bytes.len()` long.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents(), bytes.len()) };
    Ok(b)
}

pub(crate) fn download(b: &MetalBuffer) -> Vec<u8> {
    let mut out = vec![0u8; b.len];
    // SAFETY: execute() waits for completion, so no GPU write is in flight; the buffer is
    // CPU-visible and holds `len` bytes.
    unsafe { std::ptr::copy_nonoverlapping(b.contents(), out.as_mut_ptr(), b.len) };
    out
}

/// Downcast a device buffer and check that it belongs to `ctx`'s device.
pub(crate) fn as_metal<'a>(d: &'a DeviceData, ctx: &Ctx) -> Result<&'a MetalBuffer> {
    let b = d
        .as_any()
        .downcast_ref::<MetalBuffer>()
        .ok_or_else(|| Error::new("wrong_backend", format!("expected a metal buffer, got a '{}' buffer", d.backend())))?;
    if b.device != ctx.id {
        return Err(Error::new("wrong_backend", "metal buffer belongs to another Metal device"));
    }
    Ok(b)
}

pub(crate) fn wrap(b: MetalBuffer) -> DeviceData {
    Arc::new(b)
}
