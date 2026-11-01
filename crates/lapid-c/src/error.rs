//! Status codes, the thread-local last error, panic guard, log callback and the allocator
//! behind `lapid_free`.

use std::alloc::{Layout, alloc, dealloc};
use std::cell::RefCell;
use std::ffi::{CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::RwLock;
use std::sync::atomic::{AtomicI32, Ordering};

use engine::{Error, Result};

/// Return value of every fallible function: `LAPID_OK` (0) or a `LAPID_ERR_*` code.
/// Details (`{"status": n, "error": "<engine code>", "message": "..."}`) are in
/// `lapid_last_error()`.
pub type LapidStatus = i32;
pub const LAPID_OK: LapidStatus = 0;
/// Bad argument: NULL pointer, unknown code, invalid UTF-8, output buffer too small.
pub const LAPID_ERR_INVALID_ARGUMENT: LapidStatus = 1;
pub const LAPID_ERR_SHAPE_MISMATCH: LapidStatus = 2;
pub const LAPID_ERR_UNSUPPORTED_DTYPE: LapidStatus = 3;
/// Missing tensor, blob or metadata.
pub const LAPID_ERR_NOT_FOUND: LapidStatus = 4;
pub const LAPID_ERR_IO: LapidStatus = 5;
/// Malformed or unsupported file contents.
pub const LAPID_ERR_FORMAT: LapidStatus = 6;
pub const LAPID_ERR_DEVICE_UNAVAILABLE: LapidStatus = 7;
/// Backend (driver, shader, launch) failure.
pub const LAPID_ERR_BACKEND: LapidStatus = 8;
pub const LAPID_ERR_OUT_OF_MEMORY: LapidStatus = 9;
pub const LAPID_ERR_INDEX_OUT_OF_RANGE: LapidStatus = 10;
pub const LAPID_ERR_ABI_MISMATCH: LapidStatus = 11;
/// A bug inside lapid (a caught panic). The library stays usable.
pub const LAPID_ERR_PANIC: LapidStatus = 12;
/// Any other engine error; see the "error" field of `lapid_last_error()`.
pub const LAPID_ERR_INTERNAL: LapidStatus = 13;

/// Log levels for `lapid_set_log_callback`.
pub type LapidLogLevel = i32;
pub const LAPID_LOG_ERROR: LapidLogLevel = 0;
pub const LAPID_LOG_WARN: LapidLogLevel = 1;
pub const LAPID_LOG_INFO: LapidLogLevel = 2;
pub const LAPID_LOG_DEBUG: LapidLogLevel = 3;

/// Log sink: `level`, a NUL-terminated message valid only during the call, and the
/// `user` pointer given to `lapid_set_log_callback`. May be called from any thread.
pub type LapidLogFn = Option<unsafe extern "C" fn(level: LapidLogLevel, message: *const c_char, user: *mut c_void)>;

fn status_of(code: &str) -> LapidStatus {
    match code {
        "invalid_argument" => LAPID_ERR_INVALID_ARGUMENT,
        "shape_mismatch" => LAPID_ERR_SHAPE_MISMATCH,
        "unsupported_dtype" | "unsupported" | "unsupported_feature" => LAPID_ERR_UNSUPPORTED_DTYPE,
        "missing_tensor" | "missing_metadata" | "not_found" => LAPID_ERR_NOT_FOUND,
        "io" => LAPID_ERR_IO,
        "format" | "json" | "hash_mismatch" | "unsupported_version" | "unsupported_section" => LAPID_ERR_FORMAT,
        "device_unavailable" | "vulkan_unavailable" | "cuda_unavailable" => LAPID_ERR_DEVICE_UNAVAILABLE,
        "backend" | "vulkan" | "cuda" | "metal" | "webgpu" | "shader" | "device_lost" | "invalid_launch"
        | "bad_launch" | "wrong_backend" => LAPID_ERR_BACKEND,
        "out_of_memory" | "too_large" | "limit_exceeded" => LAPID_ERR_OUT_OF_MEMORY,
        "index_out_of_range" | "out_of_bounds" => LAPID_ERR_INDEX_OUT_OF_RANGE,
        "abi_mismatch" => LAPID_ERR_ABI_MISMATCH,
        "panic" => LAPID_ERR_PANIC,
        _ => LAPID_ERR_INTERNAL,
    }
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(status: LapidStatus, e: &Error) {
    let json = serde_json::json!({ "status": status, "error": e.code, "message": e.message }).to_string();
    // serde_json escapes control characters, so the JSON has no interior NUL.
    let c = CString::new(json).unwrap_or_default();
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(c));
}

/// Run `f`, turning `Err` and panics into a status code plus the thread-local last error.
pub(crate) fn guard(f: impl FnOnce() -> Result<()>) -> LapidStatus {
    let err = match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => return LAPID_OK,
        Ok(Err(e)) => e,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            Error::new("panic", msg)
        }
    };
    let status = status_of(err.code);
    log(LAPID_LOG_DEBUG, || format!("call failed: {err}"));
    set_last_error(status, &err);
    status
}

pub(crate) fn invalid(msg: impl Into<String>) -> Error {
    Error::new("invalid_argument", msg)
}

/// JSON details of the last failed call on this thread
/// (`{"status": n, "error": "<engine code>", "message": "..."}`), or NULL if no call on this
/// thread has failed since start or `lapid_clear_error`. Successful calls do not reset it.
/// The string is owned by the library and valid until the next failing call or
/// `lapid_clear_error` on the same thread; do not free it.
#[unsafe(no_mangle)]
pub extern "C" fn lapid_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ref().map_or(std::ptr::null(), |c| c.as_ptr()))
}

/// Forget this thread's last error.
#[unsafe(no_mangle)]
pub extern "C" fn lapid_clear_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// Static name of a status code, e.g. "LAPID_ERR_SHAPE_MISMATCH" ("LAPID_ERR_UNKNOWN" for
/// unknown codes). Never free it.
#[unsafe(no_mangle)]
pub extern "C" fn lapid_status_name(status: LapidStatus) -> *const c_char {
    let s: &'static str = match status {
        LAPID_OK => "LAPID_OK\0",
        LAPID_ERR_INVALID_ARGUMENT => "LAPID_ERR_INVALID_ARGUMENT\0",
        LAPID_ERR_SHAPE_MISMATCH => "LAPID_ERR_SHAPE_MISMATCH\0",
        LAPID_ERR_UNSUPPORTED_DTYPE => "LAPID_ERR_UNSUPPORTED_DTYPE\0",
        LAPID_ERR_NOT_FOUND => "LAPID_ERR_NOT_FOUND\0",
        LAPID_ERR_IO => "LAPID_ERR_IO\0",
        LAPID_ERR_FORMAT => "LAPID_ERR_FORMAT\0",
        LAPID_ERR_DEVICE_UNAVAILABLE => "LAPID_ERR_DEVICE_UNAVAILABLE\0",
        LAPID_ERR_BACKEND => "LAPID_ERR_BACKEND\0",
        LAPID_ERR_OUT_OF_MEMORY => "LAPID_ERR_OUT_OF_MEMORY\0",
        LAPID_ERR_INDEX_OUT_OF_RANGE => "LAPID_ERR_INDEX_OUT_OF_RANGE\0",
        LAPID_ERR_ABI_MISMATCH => "LAPID_ERR_ABI_MISMATCH\0",
        LAPID_ERR_PANIC => "LAPID_ERR_PANIC\0",
        LAPID_ERR_INTERNAL => "LAPID_ERR_INTERNAL\0",
        _ => "LAPID_ERR_UNKNOWN\0",
    };
    s.as_ptr().cast()
}

// ---------- logging ----------

struct Sink {
    f: unsafe extern "C" fn(LapidLogLevel, *const c_char, *mut c_void),
    user: usize,
}

static SINK: RwLock<Option<Sink>> = RwLock::new(None);
static LEVEL: AtomicI32 = AtomicI32::new(LAPID_LOG_INFO);

/// Send a message to the log callback (if any and `level` passes the filter).
pub(crate) fn log(level: LapidLogLevel, msg: impl FnOnce() -> String) {
    if level > LEVEL.load(Ordering::Relaxed) {
        return;
    }
    let sink = SINK.read().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = sink.as_ref() {
        let text = msg().replace('\0', " ");
        if let Ok(c) = CString::new(text) {
            // SAFETY: the caller registered `f` and promised it is callable from any thread.
            unsafe { (s.f)(level, c.as_ptr(), s.user as *mut c_void) };
        }
    }
}

/// Install (or remove, with NULL) the process-wide log callback. lapid-c reports device
/// selection, evaluation summaries and failed calls through it; it is silent without
/// one. The engine's own debug output (`lapid_device_set_debug`, env `LAPID_DEBUG=1`)
/// goes to stderr. `user` is passed back verbatim. The callback must not call back into
/// lapid_set_log_callback.
#[unsafe(no_mangle)]
pub extern "C" fn lapid_set_log_callback(callback: LapidLogFn, user: *mut c_void) {
    let mut sink = SINK.write().unwrap_or_else(|e| e.into_inner());
    *sink = callback.map(|f| Sink { f, user: user as usize });
}

/// Messages above `level` are dropped (default `LAPID_LOG_INFO`).
#[unsafe(no_mangle)]
pub extern "C" fn lapid_set_log_level(level: LapidLogLevel) {
    LEVEL.store(level, Ordering::Relaxed);
}

// ---------- library-owned memory ----------

const HEADER: usize = 16;

/// Allocate `len` bytes that the caller releases with `lapid_free` (length-prefixed).
pub(crate) fn alloc_bytes(bytes: &[u8]) -> Result<*mut u8> {
    let size = bytes.len().checked_add(HEADER).ok_or_else(|| Error::new("out_of_memory", "allocation too large"))?;
    let layout = Layout::from_size_align(size, HEADER).map_err(|_| Error::new("out_of_memory", "allocation too large"))?;
    // SAFETY: size >= HEADER > 0; the header stores the total size for `lapid_free`.
    unsafe {
        let base = alloc(layout);
        if base.is_null() {
            return Err(Error::new("out_of_memory", format!("cannot allocate {size} bytes")));
        }
        base.cast::<usize>().write(size);
        let data = base.add(HEADER);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len());
        Ok(data)
    }
}

/// A NUL-terminated copy of `s` owned by the caller (release with `lapid_free`).
pub(crate) fn alloc_string(s: &str) -> Result<*mut c_char> {
    let mut v = Vec::with_capacity(s.len() + 1);
    v.extend_from_slice(s.replace('\0', " ").as_bytes());
    v.push(0);
    Ok(alloc_bytes(&v)?.cast())
}

/// Release memory that lapid returned through a `char **` / `void **` out parameter
/// (JSON strings, dumps). NULL is a no-op. Never pass handles here (they have their own
/// `*_free`) or memory from another allocator.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: `ptr` came from `alloc_bytes`, which put the total size HEADER bytes before it.
    unsafe {
        let base = ptr.cast::<u8>().sub(HEADER);
        let size = base.cast::<usize>().read();
        dealloc(base, Layout::from_size_align_unchecked(size, HEADER));
    }
}
