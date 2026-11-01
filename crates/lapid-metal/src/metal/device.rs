//! Device discovery and the shared per-device context (device + command queue).

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLCopyAllDevices, MTLCreateSystemDefaultDevice, MTLDevice};

use lapid_tensor::{Error, Result};

use crate::MetalDeviceInfo;

pub(crate) type RawDevice = ProtocolObject<dyn MTLDevice>;
pub(crate) type RawQueue = ProtocolObject<dyn MTLCommandQueue>;

/// Every Metal device, in system order (falls back to the default device).
fn all_devices() -> Vec<Retained<RawDevice>> {
    let mut list = MTLCopyAllDevices().to_vec();
    if list.is_empty()
        && let Some(d) = MTLCreateSystemDefaultDevice()
    {
        list.push(d);
    }
    list
}

fn info(index: usize, d: &RawDevice) -> MetalDeviceInfo {
    MetalDeviceInfo {
        index,
        name: d.name().to_string(),
        unified_memory: d.hasUnifiedMemory(),
        low_power: d.isLowPower(),
        working_set_bytes: d.recommendedMaxWorkingSetSize(),
    }
}

/// List Metal devices. `Err("metal_unavailable")` when there is none. Never panics.
pub fn devices() -> Result<Vec<MetalDeviceInfo>> {
    let r = std::panic::catch_unwind(|| {
        autoreleasepool(|_| all_devices().iter().enumerate().map(|(i, d)| info(i, d)).collect::<Vec<_>>())
    });
    match r {
        Ok(list) if !list.is_empty() => Ok(list),
        Ok(_) => Err(Error::new("metal_unavailable", "no Metal device found")),
        Err(_) => Err(Error::new("metal_unavailable", "Metal device enumeration panicked")),
    }
}

/// Device + queue shared by the backend, its buffers and pipelines.
pub(crate) struct Ctx {
    pub device: Retained<RawDevice>,
    pub queue: Retained<RawQueue>,
    pub info: MetalDeviceInfo,
    /// `registryID`, identifies the device of a buffer.
    pub id: u64,
}

impl Ctx {
    pub(crate) fn open(index: usize) -> Result<Ctx> {
        autoreleasepool(|_| {
            let mut list = all_devices();
            let n = list.len();
            if index >= n {
                return Err(Error::new(
                    if n == 0 { "metal_unavailable" } else { "device_unavailable" },
                    format!("metal:{index}: {n} Metal device(s) found"),
                ));
            }
            let device = list.swap_remove(index);
            let queue = device
                .newCommandQueue()
                .ok_or_else(|| Error::new("metal", "newCommandQueue returned nil"))?;
            let info = info(index, &device);
            let id = device.registryID();
            Ok(Ctx { device, queue, info, id })
        })
    }
}
