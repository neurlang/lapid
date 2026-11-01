//! Data held by leaf nodes. CONTRACT FILE — owned by the coordinator.

use std::any::Any;
use std::sync::Arc;

/// Bytes living in host memory (a Vec, or an mmap'd region of a .lapid file).
/// Cloning is cheap. Backends cache device copies keyed by `HostData::key()`.
#[derive(Clone)]
pub struct HostData {
    bytes: Arc<dyn AsRef<[u8]> + Send + Sync>,
}

impl HostData {
    pub fn new(bytes: Arc<dyn AsRef<[u8]> + Send + Sync>) -> Self {
        HostData { bytes }
    }

    pub fn from_vec(v: Vec<u8>) -> Self {
        HostData { bytes: Arc::new(v) }
    }

    pub fn bytes(&self) -> &[u8] {
        (*self.bytes).as_ref()
    }

    /// Stable identity of the underlying allocation (pointer + len), used as upload cache key.
    pub fn key(&self) -> (usize, usize) {
        let b = self.bytes();
        (b.as_ptr() as usize, b.len())
    }
}

impl std::fmt::Debug for HostData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HostData({} bytes)", self.bytes().len())
    }
}

/// A buffer resident on a device. Implemented by each backend; downcast with `as_any`.
pub trait DeviceBuffer: Send + Sync + 'static {
    /// Backend name that owns it ("cpu", "vulkan").
    fn backend(&self) -> &'static str;
    /// Size in bytes.
    fn len(&self) -> usize;
    fn as_any(&self) -> &dyn Any;
}

pub type DeviceData = Arc<dyn DeviceBuffer>;

impl std::fmt::Debug for dyn DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceBuffer({}, {} bytes)", self.backend(), self.len())
    }
}
