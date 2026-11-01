//! `VkBuffer`: a storage buffer bound to sub-allocated device memory.

use std::any::Any;
use std::sync::Arc;

use ash::vk;
use lapid_tensor::{DeviceBuffer, DeviceData, Error, Result};

use crate::context::{Shared, vk_err};
use crate::memory::{Allocation, MemUsage};

/// A Vulkan storage buffer. Freed (memory returned to the allocator) on drop.
pub struct VkBuffer {
    shared: Arc<Shared>,
    pub(crate) buffer: vk::Buffer,
    alloc: Allocation,
    len: usize,
    /// Size of the VkBuffer (>= len, multiple of 16, >= 16).
    pub(crate) size: u64,
    /// Process-unique serial (handles may be reused after destruction; recorded command
    /// buffers are matched by serial).
    pub(crate) id: u64,
}

impl std::fmt::Debug for VkBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VkBuffer({} bytes, mapped: {})", self.len, self.alloc.mapped.is_some())
    }
}

impl VkBuffer {
    pub(crate) fn create(shared: &Arc<Shared>, bytes: usize, usage: MemUsage) -> Result<VkBuffer> {
        let size = (bytes as u64).div_ceil(16).max(1) * 16;
        let dev = &shared.device;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid device + create info.
        let buffer = unsafe { dev.create_buffer(&info, None) }.map_err(|e| vk_err("vkCreateBuffer", e))?;
        // SAFETY: buffer was just created.
        let req = unsafe { dev.get_buffer_memory_requirements(buffer) };
        let alloc = {
            let mut a = shared.allocator.lock().map_err(|_| Error::new("vulkan", "allocator poisoned"))?;
            match a.alloc(dev, req, usage) {
                Ok(x) => x,
                Err(e) => {
                    // SAFETY: unbound buffer, never used.
                    unsafe { dev.destroy_buffer(buffer, None) };
                    return Err(e);
                }
            }
        };
        // SAFETY: memory range satisfies the buffer's requirements.
        if let Err(e) = unsafe { dev.bind_buffer_memory(buffer, alloc.memory, alloc.offset) } {
            // SAFETY: cleanup of objects that were never used by the GPU.
            unsafe { dev.destroy_buffer(buffer, None) };
            if let Ok(mut a) = shared.allocator.lock() {
                a.free(dev, &alloc);
            }
            return Err(vk_err("vkBindBufferMemory", e));
        }
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(VkBuffer { shared: shared.clone(), buffer, alloc, len: bytes, size, id })
    }

    /// Host pointer if the buffer lives in host-visible coherent memory.
    pub(crate) fn mapped(&self) -> Option<*mut u8> {
        self.alloc.mapped.map(|p| p.0.as_ptr())
    }

    /// Copy `data` into the mapped buffer at byte `offset`. Caller guarantees the GPU is not
    /// using the range.
    pub(crate) fn write_mapped(&self, offset: usize, data: &[u8]) -> Result<()> {
        let Some(p) = self.mapped() else {
            return Err(Error::new("vulkan", "buffer is not host visible"));
        };
        if offset + data.len() > self.size as usize {
            return Err(Error::new("out_of_bounds", format!("write of {} bytes at {offset} into {} byte buffer", data.len(), self.size)));
        }
        // SAFETY: bounds checked above; mapping is valid for the buffer's lifetime.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), p.add(offset), data.len()) };
        Ok(())
    }

    /// Copy `out.len()` bytes from byte `offset` of the mapped buffer.
    pub(crate) fn read_mapped(&self, offset: usize, out: &mut [u8]) -> Result<()> {
        let Some(p) = self.mapped() else {
            return Err(Error::new("vulkan", "buffer is not host visible"));
        };
        if offset + out.len() > self.size as usize {
            return Err(Error::new("out_of_bounds", "read past end of buffer"));
        }
        // SAFETY: bounds checked above; GPU work has completed (execute is synchronous).
        unsafe { std::ptr::copy_nonoverlapping(p.add(offset), out.as_mut_ptr(), out.len()) };
        Ok(())
    }

    /// Requested size in bytes.
    pub(crate) fn len_bytes(&self) -> usize {
        self.len
    }
}

impl Drop for VkBuffer {
    fn drop(&mut self) {
        // SAFETY: execute() waits for completion before returning, so the GPU no longer
        // references this buffer when the last handle is dropped.
        unsafe { self.shared.device.destroy_buffer(self.buffer, None) };
        if let Ok(mut a) = self.shared.allocator.lock() {
            a.free(&self.shared.device, &self.alloc);
        }
    }
}

impl DeviceBuffer for VkBuffer {
    fn backend(&self) -> &'static str {
        "vulkan"
    }
    fn len(&self) -> usize {
        self.len
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Downcast a `DeviceData` to a Vulkan buffer created on the device `shared`.
pub(crate) fn as_vk<'a>(d: &'a DeviceData, shared: &Arc<Shared>) -> Result<&'a VkBuffer> {
    let b = d.as_any().downcast_ref::<VkBuffer>().ok_or_else(|| {
        Error::new("wrong_backend", format!("buffer belongs to backend '{}', not 'vulkan'", d.backend()))
    })?;
    if !Arc::ptr_eq(&b.shared, shared) {
        return Err(Error::new("wrong_backend", "buffer belongs to a different vulkan device"));
    }
    Ok(b)
}
