//! Opaque refcounted handles. A handle is an `Arc<T>` turned into a raw pointer: `*_retain`
//! adds a reference, `*_free` drops one, and the object dies with its last reference.

use std::ffi::{CStr, c_char};
use std::sync::Arc;

use engine::{Result, Tensor};

use crate::error::invalid;

/// A device (CPU or GPU backend). Opaque.
pub struct LapidDevice(pub(crate) engine::Device);

/// An opened .lapid file. Opaque.
pub struct LapidFile(pub(crate) engine::LapidFile);

/// A lazy tensor: a node of the compute graph. Opaque, immutable.
pub struct LapidTensor(pub(crate) Tensor);

// Handles are shared across threads by C callers; keep that sound.
const _: () = {
    const fn shareable<T: Send + Sync>() {}
    shareable::<LapidDevice>();
    shareable::<LapidFile>();
    shareable::<LapidTensor>();
};

pub(crate) fn into_raw<T>(v: T) -> *mut T {
    Arc::into_raw(Arc::new(v)).cast_mut()
}

/// Borrow the object behind a handle (error on NULL).
pub(crate) unsafe fn get<'a, T>(p: *const T, what: &str) -> Result<&'a T> {
    // SAFETY: non-null handles were produced by `into_raw` and are alive (caller contract).
    unsafe { p.as_ref() }.ok_or_else(|| invalid(format!("{what} is NULL")))
}

pub(crate) unsafe fn retain<T>(p: *const T) {
    if !p.is_null() {
        // SAFETY: `p` came from `Arc::into_raw` and has at least one live reference.
        unsafe { Arc::increment_strong_count(p) };
    }
}

pub(crate) unsafe fn release<T>(p: *const T) {
    if !p.is_null() {
        // SAFETY: as in `retain`; this gives back one reference.
        unsafe { Arc::decrement_strong_count(p) };
    }
}

pub(crate) unsafe fn tensor<'a>(p: *const LapidTensor) -> Result<&'a Tensor> {
    // SAFETY: forwarded caller contract.
    Ok(unsafe { &get(p, "tensor")?.0 })
}

/// Write a new tensor handle to `out`.
pub(crate) fn put_tensor(out: *mut *mut LapidTensor, t: Tensor) -> Result<()> {
    put(out, LapidTensor(t))
}

/// Write a new handle to `out` (error on NULL `out`).
pub(crate) fn put<T>(out: *mut *mut T, v: T) -> Result<()> {
    if out.is_null() {
        return Err(invalid("output pointer is NULL"));
    }
    // SAFETY: `out` is non-null and points to writable storage (caller contract).
    unsafe { out.write(into_raw(v)) };
    Ok(())
}

/// Write a plain value to a non-NULL out pointer.
pub(crate) fn put_value<T>(out: *mut T, v: T) -> Result<()> {
    if out.is_null() {
        return Err(invalid("output pointer is NULL"));
    }
    // SAFETY: as in `put`.
    unsafe { out.write(v) };
    Ok(())
}

/// `n` elements at `p` (NULL allowed when `n == 0`).
pub(crate) unsafe fn slice<'a, T>(p: *const T, n: usize, what: &str) -> Result<&'a [T]> {
    if n == 0 {
        return Ok(&[]);
    }
    if p.is_null() {
        return Err(invalid(format!("{what} is NULL but its length is {n}")));
    }
    // SAFETY: caller contract: `p` points to `n` readable elements.
    Ok(unsafe { std::slice::from_raw_parts(p, n) })
}

/// Borrow `n` tensor handles.
pub(crate) unsafe fn tensors<'a>(p: *const *const LapidTensor, n: usize, what: &str) -> Result<Vec<&'a Tensor>> {
    // SAFETY: forwarded caller contract.
    let handles = unsafe { slice(p, n, what)? };
    handles.iter().map(|&h| unsafe { tensor(h) }).collect()
}

/// A NUL-terminated UTF-8 string argument.
pub(crate) unsafe fn str_arg<'a>(p: *const c_char, what: &str) -> Result<&'a str> {
    if p.is_null() {
        return Err(invalid(format!("{what} is NULL")));
    }
    // SAFETY: caller contract: NUL-terminated.
    unsafe { CStr::from_ptr(p) }.to_str().map_err(|_| invalid(format!("{what} is not valid UTF-8")))
}

/// An axis argument (`int64_t`, negative counts from the end).
pub(crate) fn axis_arg(a: i64) -> Result<isize> {
    isize::try_from(a).map_err(|_| invalid(format!("axis {a} out of range")))
}
