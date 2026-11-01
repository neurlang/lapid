//! Non-Apple targets: the same public API, always unavailable.

use std::convert::Infallible;

use lapid_compiler::{Backend, Launch};
use lapid_tensor::{DeviceData, Error, Result};

use crate::MetalDeviceInfo;

fn unavailable() -> Error {
    Error::new("metal_unavailable", "Metal is only available on Apple platforms")
}

/// Always `Err("metal_unavailable")` on this platform.
pub fn devices() -> Result<Vec<MetalDeviceInfo>> {
    Err(unavailable())
}

/// Uninhabited on this platform: `new` always fails.
pub struct MetalBackend {
    never: Infallible,
}

impl MetalBackend {
    pub fn new(_index: usize) -> Result<MetalBackend> {
        Err(unavailable())
    }

    pub fn info(&self) -> &MetalDeviceInfo {
        match self.never {}
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn device_name(&self) -> String {
        match self.never {}
    }

    fn alloc(&self, _bytes: usize) -> Result<DeviceData> {
        match self.never {}
    }

    fn upload(&self, _bytes: &[u8]) -> Result<DeviceData> {
        match self.never {}
    }

    fn download(&self, _buf: &DeviceData) -> Result<Vec<u8>> {
        match self.never {}
    }

    fn execute(&self, _launches: &[Launch], _buffers: &[DeviceData]) -> Result<()> {
        match self.never {}
    }
}
