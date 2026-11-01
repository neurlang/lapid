//! Tests. `golden` (shader text, every key renders) runs on every platform; the GPU tests
//! compile every kernel variant with Metal and compare every kernel against naive CPU
//! references (skipped when no Metal device is available).

#[cfg(target_vendor = "apple")]
mod bench;
mod golden;
mod keys;

#[cfg(target_vendor = "apple")]
mod gpu_attention;
#[cfg(target_vendor = "apple")]
mod gpu_attention_sweep;
#[cfg(target_vendor = "apple")]
mod gpu_basic;
#[cfg(target_vendor = "apple")]
mod gpu_index;
#[cfg(target_vendor = "apple")]
mod gpu_int;
#[cfg(target_vendor = "apple")]
mod gpu_kquant;
#[cfg(target_vendor = "apple")]
mod gpu_moe;
#[cfg(target_vendor = "apple")]
mod gpu_matmul;
#[cfg(target_vendor = "apple")]
mod util;
