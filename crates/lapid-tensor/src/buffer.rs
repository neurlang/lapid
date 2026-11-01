//! Data held by leaf nodes. CONTRACT FILE — owned by the coordinator.

use std::any::Any;
use std::sync::Arc;

/// Bytes living in host memory (a Vec, or an mmap'd region of a .lapid file).
/// Cloning is cheap. Backends cache device copies keyed by `HostData::key()`.
#[derive(Clone)]
pub struct HostData {
    bytes: Arc<dyn AsRef<[u8]> + Send + Sync>,
    /// `key()`, cached: identity checks of graph leaves (every eval, every weight) need no
    /// trip through the shared allocation.
    key: (usize, usize),
}

impl HostData {
    pub fn new(bytes: Arc<dyn AsRef<[u8]> + Send + Sync>) -> Self {
        let b = (*bytes).as_ref();
        let key = (b.as_ptr() as usize, b.len());
        HostData { bytes, key }
    }

    pub fn from_vec(v: Vec<u8>) -> Self {
        HostData::new(Arc::new(v))
    }

    pub fn bytes(&self) -> &[u8] {
        (*self.bytes).as_ref()
    }

    /// Stable identity of the underlying allocation (pointer + len), used as upload cache key.
    pub fn key(&self) -> (usize, usize) {
        self.key
    }

    /// True when this handle is the only one to the data (no other `HostData` clone exists).
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.bytes) == 1
    }

    /// Byte length (no trip through the shared allocation).
    pub fn len(&self) -> usize {
        self.key.1
    }

    pub fn is_empty(&self) -> bool {
        self.key.1 == 0
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
