//! Devices and running graphs: open/list devices, eval, upload, copy to host, debug dumps.

use std::ffi::c_char;

use engine::{Device, Error, Tensor};

use crate::error::{LAPID_LOG_DEBUG, LAPID_LOG_INFO, LapidStatus, alloc_string, guard, invalid, log};
use crate::handle::{LapidDevice, LapidTensor, get, put, put_value, release, retain, str_arg, tensor, tensors};

/// JSON array of usable devices in discovery order (CPU first):
/// `[{"id": "cpu", "description": "20 threads (neon, dotprod)"}, {"id": "cuda:0", ...}]`.
/// `id` is accepted by `lapid_device_open`. Free `*out_json` with `lapid_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_devices_json(out_json: *mut *mut c_char) -> LapidStatus {
    guard(|| {
        let list: Vec<serde_json::Value> = Device::list()
            .iter()
            .map(|line| {
                let (id, desc) = line.split_once(": ").unwrap_or((line.as_str(), ""));
                serde_json::json!({ "id": id, "description": desc })
            })
            .collect();
        put_value(out_json, alloc_string(&serde_json::Value::Array(list).to_string())?)
    })
}

/// The best available device: env `LAPID_DEVICE` wins, else Metal, CUDA, a Vulkan GPU, then
/// CPU. Never fails for lack of a GPU (falls back to CPU). Free with `lapid_device_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_device_best(out: *mut *mut LapidDevice) -> LapidStatus {
    guard(|| {
        let d = Device::best();
        log(LAPID_LOG_INFO, || format!("selected {}", d.description()));
        put(out, LapidDevice(d))
    })
}

/// Open a device by name: "cpu", "gpu" (best GPU), "metal", "metal:N", "cuda", "cuda:N",
/// "vulkan", "vulkan:N". `LAPID_ERR_DEVICE_UNAVAILABLE` if it does not exist or this build
/// lacks the backend. Handles to the same physical device share weights and caches.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_device_open(name: *const c_char, out: *mut *mut LapidDevice) -> LapidStatus {
    guard(|| {
        let name = unsafe { str_arg(name, "device name")? };
        let d = Device::from_name(name)?;
        log(LAPID_LOG_INFO, || format!("opened {}", d.description()));
        put(out, LapidDevice(d))
    })
}

/// Add a reference to a device handle (the same pointer stays valid until every reference
/// is released with `lapid_device_free`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_device_retain(device: *mut LapidDevice) {
    unsafe { retain(device) }
}

/// Drop one reference. NULL is a no-op. Tensors produced on the device stay valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_device_free(device: *mut LapidDevice) {
    unsafe { release(device) }
}

/// Backend name: "cpu", "cuda", "vulkan", "metal" or "webgpu". Static, never free it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_device_backend(device: *const LapidDevice, out: *mut *const c_char) -> LapidStatus {
    guard(|| {
        let d = unsafe { get(device, "device")? };
        let s: &'static str = match d.0.name() {
            "cpu" => "cpu\0",
            "cuda" => "cuda\0",
            "vulkan" => "vulkan\0",
            "metal" => "metal\0",
            "webgpu" => "webgpu\0",
            _ => "unknown\0",
        };
        put_value(out, s.as_ptr().cast())
    })
}

/// Human description, e.g. "cuda: NVIDIA GB10". Free `*out` with `lapid_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_device_description(device: *const LapidDevice, out: *mut *mut c_char) -> LapidStatus {
    guard(|| {
        let d = unsafe { get(device, "device")? };
        put_value(out, alloc_string(&d.0.description())?)
    })
}

/// When on, the engine prints each compiled schedule and per-launch timings to stderr
/// (also enabled by env `LAPID_DEBUG=1`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_device_set_debug(device: *const LapidDevice, on: bool) -> LapidStatus {
    guard(|| {
        unsafe { get(device, "device")? }.0.set_debug(on);
        Ok(())
    })
}

/// Compile and run the graphs of `outputs[0..n]` together on `device`. Writes `n` new
/// handles to `results[0..n]`: realized tensors with the same shapes and dtypes, which can
/// be fed into later graphs without copies (e.g. KV caches). Host leaves (weights) are
/// uploaded once and cached for the device's lifetime. On error nothing is written.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_eval(
    device: *const LapidDevice,
    outputs: *const *const LapidTensor,
    n: usize,
    results: *mut *mut LapidTensor,
) -> LapidStatus {
    guard(|| {
        let d = unsafe { get(device, "device")? };
        let outs = unsafe { tensors(outputs, n, "outputs")? };
        if n > 0 && results.is_null() {
            return Err(invalid("results is NULL"));
        }
        let realized = d.0.eval(&outs)?;
        if realized.len() != n {
            return Err(Error::new("internal", format!("eval returned {} tensors for {n} outputs", realized.len())));
        }
        log(LAPID_LOG_DEBUG, || format!("eval: {n} output(s) on {}", d.0.description()));
        for (i, t) in realized.into_iter().enumerate() {
            // SAFETY: `results` has room for `n` handles (caller contract).
            put(unsafe { results.add(i) }, LapidTensor(t))?;
        }
        Ok(())
    })
}

/// Upload a host tensor now (e.g. all weights at load time). Returns a realized device
/// tensor; the host data is cached so later graphs reuse the upload.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_upload(
    device: *const LapidDevice,
    t: *const LapidTensor,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    guard(|| {
        let d = unsafe { get(device, "device")? };
        let t = unsafe { tensor(t)? };
        put(out, LapidTensor(d.0.upload(t)?))
    })
}

fn check_room(t: &Tensor, count: usize) -> engine::Result<()> {
    if count < t.numel() {
        return Err(invalid(format!("destination holds {count} elements, tensor {:?} has {}", t.shape(), t.numel())));
    }
    Ok(())
}

/// Evaluate `t` (if needed) on `device` and copy it to `dst` as f32 (f16/bf16/i32 are
/// converted). `count` is the capacity of `dst` in elements and must be at least
/// `lapid_tensor_numel(t)`; exactly numel elements are written.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_to_host_f32(
    device: *const LapidDevice,
    t: *const LapidTensor,
    dst: *mut f32,
    count: usize,
) -> LapidStatus {
    guard(|| {
        let d = unsafe { get(device, "device")? };
        let t = unsafe { tensor(t)? };
        check_room(t, count)?;
        let v = d.0.to_vec_f32(t)?;
        copy_out(&v, dst)
    })
}

/// `lapid_to_host_f32` for i32 output (float tensors are truncated toward zero).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_to_host_i32(
    device: *const LapidDevice,
    t: *const LapidTensor,
    dst: *mut i32,
    count: usize,
) -> LapidStatus {
    guard(|| {
        let d = unsafe { get(device, "device")? };
        let t = unsafe { tensor(t)? };
        check_room(t, count)?;
        let v = d.0.to_vec_i32(t)?;
        copy_out(&v, dst)
    })
}

fn copy_out<T: Copy>(v: &[T], dst: *mut T) -> engine::Result<()> {
    if v.is_empty() {
        return Ok(());
    }
    if dst.is_null() {
        return Err(invalid("destination is NULL"));
    }
    // SAFETY: `dst` has room for at least `v.len()` elements (checked against `count`).
    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), dst, v.len()) };
    Ok(())
}

/// Text dump of the kernel schedule the compiler builds for `outputs[0..n]` (one line per
/// launch, default compile options; backends may differ in details). Free with `lapid_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_schedule_dump(
    outputs: *const *const LapidTensor,
    n: usize,
    out: *mut *mut c_char,
) -> LapidStatus {
    guard(|| {
        let outs: Vec<Tensor> = unsafe { tensors(outputs, n, "outputs")? }.into_iter().cloned().collect();
        let schedule = engine::compiler::compile(&outs, &engine::compiler::CompileOptions::default())?;
        put_value(out, alloc_string(&engine::compiler::dump(&schedule))?)
    })
}
