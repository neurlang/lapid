//! lapid-metal: Metal compute backend for Apple GPUs.
//!
//! Kernels are MSL templates (`src/kernels/*.metal`) with small substitution slots, filled
//! per kernel spec, compiled in-process with `newLibraryWithSource` and cached as compute
//! pipelines; sizes travel in a small params block (`setBytes`) so one pipeline serves
//! every shape. Buffers use `MTLStorageModeShared` (unified memory): uploads and downloads
//! are memcpys. Each `execute` encodes all launches into one command buffer with a single
//! concurrent compute encoder and buffer memory barriers between dependent launches.
//!
//! Hot paths: simdgroup-matrix GEMM (f32 accumulate, int8/int4 weights dequantized into
//! threadgroup memory per 32-group), GEMV for m <= 4 (decoder), flash attention on
//! simdgroup matrices (encoder) and a split-key vector attention (decoder).
//!
//! Environment:
//! - `LAPID_DEBUG=1` prints the numbered MSL source on shader compile errors (the error
//!   itself always carries it).
//! - `LAPID_PROFILE=1` runs every launch in its own command buffer and accumulates GPU time
//!   per kernel kind (`Backend::take_profile`). With `LAPID_METAL_PROFILE=cb` execution stays
//!   batched and the GPU time of each whole command buffer is reported instead.
//!
//! On non-Apple targets the crate compiles to a stub: `devices()` returns
//! `Err("metal_unavailable")` and `MetalBackend::new` always fails.

#![cfg_attr(not(target_vendor = "apple"), allow(dead_code))]

mod codegen;

#[cfg(target_vendor = "apple")]
mod metal;
#[cfg(target_vendor = "apple")]
pub use metal::{MetalBackend, MetalBuffer, devices};

#[cfg(not(target_vendor = "apple"))]
mod stub;
#[cfg(not(target_vendor = "apple"))]
pub use stub::{MetalBackend, devices};

#[cfg(test)]
mod tests;

/// A Metal device usable for compute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalDeviceInfo {
    /// Index for `MetalBackend::new` / `metal:N`.
    pub index: usize,
    pub name: String,
    /// CPU and GPU share memory (Apple silicon).
    pub unified_memory: bool,
    pub low_power: bool,
    /// Recommended maximum working set in bytes.
    pub working_set_bytes: u64,
}
