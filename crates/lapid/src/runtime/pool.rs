//! Pool of temp buffers, binned by size class, reused across evals.
//!
//! Every temp slot of a schedule takes one buffer (rounded up to its size class) for the
//! duration of the eval and returns it afterwards. Size classes are 1/8-octave steps, so
//! a buffer is at most 12.5% larger than requested and growing shapes (KV caches) still
//! hit the same class most of the time. Free lists are FIFO, so an eval that takes and
//! returns the same slots each step gets the same buffers in the same slots (stable
//! bindings let GPU backends replay recorded command buffers / graphs).

use lapid_tensor::fxhash::FxHashMap;
use std::collections::VecDeque;
use std::sync::Mutex;

use lapid_compiler::Backend;
use lapid_tensor::{DeviceData, Result};

use super::lock;

/// Smallest buffer handed out.
const MIN_CLASS: usize = 256;
/// Free buffers beyond this many bytes are released instead of pooled.
const MAX_POOLED_BYTES: usize = 2 << 30;

/// Round `bytes` up to its size class.
pub(crate) fn size_class(bytes: usize) -> usize {
    if bytes <= MIN_CLASS {
        return MIN_CLASS;
    }
    let msb = usize::BITS - 1 - bytes.leading_zeros();
    let step = 1usize << msb.saturating_sub(3);
    bytes.div_ceil(step) * step
}

#[derive(Default)]
struct Inner {
    free: FxHashMap<usize, VecDeque<DeviceData>>,
    bytes: usize,
}

pub(crate) struct Pool {
    inner: Mutex<Inner>,
}

impl Pool {
    pub(crate) fn new() -> Pool {
        Pool { inner: Mutex::new(Inner::default()) }
    }

    /// One buffer per slot. On failure, already-taken buffers go back to the pool.
    pub(crate) fn take_slots(&self, backend: &dyn Backend, slot_bytes: &[usize]) -> Result<Vec<DeviceData>> {
        let mut out = Vec::with_capacity(slot_bytes.len());
        for &bytes in slot_bytes {
            match self.take(backend, size_class(bytes)) {
                Ok(b) => out.push(b),
                Err(e) => {
                    self.give_slots(&slot_bytes[..out.len()], out);
                    return Err(e);
                }
            }
        }
        Ok(out)
    }

    /// Return the buffers taken by `take_slots(slot_bytes)`.
    pub(crate) fn give_slots(&self, slot_bytes: &[usize], bufs: Vec<DeviceData>) {
        let mut g = lock(&self.inner);
        for (&bytes, buf) in slot_bytes.iter().zip(bufs) {
            let class = size_class(bytes);
            if g.bytes + class > MAX_POOLED_BYTES {
                continue; // dropped → freed by the backend
            }
            g.bytes += class;
            g.free.entry(class).or_default().push_back(buf);
        }
    }

    fn take(&self, backend: &dyn Backend, class: usize) -> Result<DeviceData> {
        {
            let mut g = lock(&self.inner);
            if let Some(buf) = g.free.get_mut(&class).and_then(VecDeque::pop_front) {
                g.bytes -= class;
                return Ok(buf);
            }
        }
        match backend.alloc(class) {
            Ok(b) => Ok(b),
            Err(_) => {
                // Possibly out of memory: release every pooled buffer and retry once.
                self.clear();
                backend.alloc(class)
            }
        }
    }

    pub(crate) fn clear(&self) {
        let mut g = lock(&self.inner);
        g.free.clear();
        g.bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::size_class;

    #[test]
    fn size_classes() {
        assert_eq!(size_class(0), 256);
        assert_eq!(size_class(256), 256);
        assert_eq!(size_class(257), 288);
        assert_eq!(size_class(1024), 1024);
        assert_eq!(size_class(1025), 1152);
        for n in [1000usize, 4097, 123_457, 18_432_000] {
            let c = size_class(n);
            assert!(c >= n && c - n <= n / 8 + 1, "{n} -> {c}");
            assert_eq!(size_class(c), c);
        }
    }
}
