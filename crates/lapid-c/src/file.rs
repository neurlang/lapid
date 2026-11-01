//! .lapid model files: open from a path or bytes, read tensors, blobs and metadata.

use std::ffi::c_char;
use std::sync::Arc;

use engine::Error;

use crate::dtype_to_c;
use crate::error::{LapidStatus, alloc_string, guard};
use crate::handle::{LapidFile, LapidTensor, get, put, put_tensor, put_value, release, retain, slice, str_arg};

/// Open a .lapid file (read-only memory map). Tensors from it are zero-copy views of the
/// mapping and keep it alive after `lapid_file_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_file_open(path: *const c_char, out: *mut *mut LapidFile) -> LapidStatus {
    guard(|| {
        let path = unsafe { str_arg(path, "path")? };
        put(out, LapidFile(engine::LapidFile::open(path)?))
    })
}

/// Open a complete .lapid image held in memory. The `len` bytes at `data` are copied once,
/// so the caller may free them as soon as this returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_file_open_bytes(data: *const u8, len: usize, out: *mut *mut LapidFile) -> LapidStatus {
    guard(|| {
        let bytes = unsafe { slice(data, len, "data")? }.to_vec();
        put(out, LapidFile(engine::LapidFile::from_bytes(Arc::new(bytes))?))
    })
}

/// Add a reference to a file handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_file_retain(file: *mut LapidFile) {
    unsafe { retain(file) }
}

/// Drop one reference. NULL is a no-op. Tensors taken from the file stay valid; blob
/// pointers from `lapid_file_blob` die with the last file reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_file_free(file: *mut LapidFile) {
    unsafe { release(file) }
}

/// New handle to the tensor `name` (a host leaf, zero copy). `LAPID_ERR_NOT_FOUND` if absent.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_file_tensor(
    file: *const LapidFile,
    name: *const c_char,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    guard(|| {
        let f = unsafe { get(file, "file")? };
        let name = unsafe { str_arg(name, "tensor name")? };
        put_tensor(out, f.0.get(name)?)
    })
}

/// JSON array of the file's tensors, sorted by name:
/// `[{"name": "w", "dtype": "f16", "dtype_code": 1, "shape": [4, 8]}, ...]`.
/// Free with `lapid_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_file_tensors_json(file: *const LapidFile, out_json: *mut *mut c_char) -> LapidStatus {
    guard(|| {
        let f = unsafe { get(file, "file")? };
        let list: Vec<serde_json::Value> = f
            .0
            .tensors
            .iter()
            .map(|(name, t)| {
                serde_json::json!({
                    "name": name,
                    "dtype": t.dtype().name(),
                    "dtype_code": dtype_to_c(t.dtype()),
                    "shape": t.shape(),
                })
            })
            .collect();
        put_value(out_json, alloc_string(&serde_json::Value::Array(list).to_string())?)
    })
}

/// The file's metadata object as JSON. Free with `lapid_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_file_metadata_json(file: *const LapidFile, out_json: *mut *mut c_char) -> LapidStatus {
    guard(|| {
        let f = unsafe { get(file, "file")? };
        put_value(out_json, alloc_string(&f.0.metadata.to_string())?)
    })
}

/// JSON array of the blob index (non-tensor assets such as tokenizer files), e.g.
/// `[{"name": "tokenizer/tokens.json", "media_type": "application/json", "offset": 0, "nbytes": 1234, "hash": "<blake3 hex>"}]`.
/// Free with `lapid_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_file_blobs_json(file: *const LapidFile, out_json: *mut *mut c_char) -> LapidStatus {
    guard(|| {
        let f = unsafe { get(file, "file")? };
        let json = serde_json::to_string(f.0.blobs())?;
        put_value(out_json, alloc_string(&json)?)
    })
}

/// Bytes of the blob `name`, zero copy: `*data` / `*len` point into the file image and stay
/// valid while the file handle lives. Do not free them. `LAPID_ERR_NOT_FOUND` if absent.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_file_blob(
    file: *const LapidFile,
    name: *const c_char,
    data: *mut *const u8,
    len: *mut usize,
) -> LapidStatus {
    guard(|| {
        let f = unsafe { get(file, "file")? };
        let name = unsafe { str_arg(name, "blob name")? };
        let bytes = f.0.blob(name).ok_or_else(|| Error::new("not_found", format!("blob '{name}' not found in file")))?;
        put_value(len, bytes.len())?;
        put_value(data, bytes.as_ptr())
    })
}
