//! Visit marks for graph walks without hash maps.
//!
//! Every node carries a scratch word. A [`Marker`] holds a process-wide lock and a fresh
//! stamp; a node marked with the current stamp carries a 24-bit value of this walk (e.g. its
//! position), any other stamp means "not visited". Only one marker exists at a time, so
//! walks never see each other's marks; callers fall back to a map when [`Marker::try_new`]
//! finds the lock taken (a concurrent walk on another thread).

use std::sync::atomic::Ordering;
use std::sync::{Mutex, MutexGuard};

use crate::tensor::Tensor;

/// Largest value a mark holds.
pub const MARK_MAX: u32 = (1 << 24) - 1;

static STAMP: Mutex<u64> = Mutex::new(0);

pub struct Marker {
    /// Stamp in the high 40 bits.
    stamp: u64,
    _guard: MutexGuard<'static, u64>,
}

impl Marker {
    /// Exclusive marker, or `None` while another walk holds one.
    pub fn try_new() -> Option<Marker> {
        let mut g = match STAMP.try_lock() {
            Ok(g) => g,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return None,
        };
        // 40-bit stamps: never wraps in practice (10^12 walks).
        *g = (*g + 1) & ((1 << 40) - 1);
        if *g == 0 {
            *g = 1;
        }
        Some(Marker { stamp: *g << 24, _guard: g })
    }

    /// The value `t` was marked with in this walk.
    #[inline]
    pub fn get(&self, t: &Tensor) -> Option<u32> {
        let m = t.0.mark.load(Ordering::Relaxed);
        (m & !(MARK_MAX as u64) == self.stamp).then_some((m & MARK_MAX as u64) as u32)
    }

    /// Mark `t` with `v` (<= `MARK_MAX`).
    #[inline]
    pub fn set(&self, t: &Tensor, v: u32) {
        debug_assert!(v <= MARK_MAX);
        t.0.mark.store(self.stamp | (v & MARK_MAX) as u64, Ordering::Relaxed);
    }
}
