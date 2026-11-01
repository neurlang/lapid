//! Upload cache: device copies of host leaves, keyed by `HostData::key()`.
//!
//! Each entry keeps a clone of its `HostData`, so the underlying allocation stays alive
//! and its (pointer, len) key cannot be reused by an unrelated allocation while cached.
//!
//! Two classes of entries:
//! * **pinned** — host data backed by an opened `.lapid` file (weights). Kept for the
//!   lifetime of the runtime: weights are uploaded exactly once.
//! * **transient** — everything else (e.g. per-step token ids built with `from_i32`).
//!   Kept in an LRU bounded by bytes and count so long decode loops do not leak.

use std::collections::BTreeMap;

use lapid_tensor::fxhash::FxHashMap;
use std::sync::Mutex;

use lapid_compiler::Backend;
use lapid_tensor::{DeviceData, HostData, Result};

use super::lock;

/// Max bytes of transient (non-file) host data kept on the device.
const TRANSIENT_MAX_BYTES: usize = 256 << 20;
/// Max number of transient entries.
const TRANSIENT_MAX_ENTRIES: usize = 4096;

struct Entry {
    /// Keep-alive: pins the allocation the key points into.
    _host: HostData,
    buf: DeviceData,
    pinned: bool,
    bytes: usize,
    last_used: u64,
}

#[derive(Default)]
struct Inner {
    map: FxHashMap<(usize, usize), Entry>,
    /// Transient entries by `last_used` tick (oldest first) for O(log n) eviction.
    lru: BTreeMap<u64, (usize, usize)>,
    tick: u64,
    transient_bytes: usize,
    transient_count: usize,
}

pub(crate) struct UploadCache {
    inner: Mutex<Inner>,
}

impl UploadCache {
    pub(crate) fn new() -> UploadCache {
        UploadCache { inner: Mutex::new(Inner::default()) }
    }

    /// Device copy of `host`, uploading on a miss. The upload itself runs without the
    /// lock held so concurrent evals on other data are not serialized behind it.
    pub(crate) fn get_or_upload(&self, host: &HostData, backend: &dyn Backend, pinned: bool) -> Result<DeviceData> {
        let key = host.key();
        {
            let mut g = lock(&self.inner);
            g.tick += 1;
            let tick = g.tick;
            if let Some(buf) = g.touch(key, tick) {
                return Ok(buf);
            }
        }

        let bytes = host.bytes();
        // Backends need not support zero-sized buffers.
        let buf = if bytes.is_empty() { backend.upload(&[0u8; 4])? } else { backend.upload(bytes)? };

        let mut g = lock(&self.inner);
        g.tick += 1;
        let tick = g.tick;
        if let Some(buf) = g.touch(key, tick) {
            // Another thread uploaded the same data meanwhile; keep theirs.
            return Ok(buf);
        }
        let len = bytes.len();
        g.map.insert(
            key,
            Entry { _host: host.clone(), buf: buf.clone(), pinned, bytes: len, last_used: tick },
        );
        if !pinned {
            g.lru.insert(tick, key);
            g.transient_bytes += len;
            g.transient_count += 1;
            g.evict_transient(key);
        }
        Ok(buf)
    }
}

impl Inner {
    /// Buffer of `key`, marking it used at `tick`.
    fn touch(&mut self, key: (usize, usize), tick: u64) -> Option<DeviceData> {
        let e = self.map.get_mut(&key)?;
        let old = std::mem::replace(&mut e.last_used, tick);
        let (pinned, buf) = (e.pinned, e.buf.clone());
        if !pinned {
            self.lru.remove(&old);
            self.lru.insert(tick, key);
        }
        Some(buf)
    }

    /// Drop least-recently-used transient entries (never `keep`) until within budget.
    /// Evicted buffers stay alive as long as an in-flight eval still references them.
    fn evict_transient(&mut self, keep: (usize, usize)) {
        while self.transient_bytes > TRANSIENT_MAX_BYTES || self.transient_count > TRANSIENT_MAX_ENTRIES {
            let Some((&tick, &k)) = self.lru.iter().find(|(_, k)| **k != keep) else { break };
            self.lru.remove(&tick);
            if let Some(e) = self.map.remove(&k) {
                self.transient_bytes -= e.bytes;
                self.transient_count -= 1;
            }
        }
    }
}
