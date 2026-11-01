//! Monotonic timer for debug / profile timings. `wasm32-unknown-unknown` has no clock
//! (`std::time::Instant::now` panics there), so it reads zero on that target.

use std::time::Duration;

#[derive(Clone, Copy)]
pub(crate) struct Instant {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    t: std::time::Instant,
}

impl Instant {
    pub(crate) fn now() -> Instant {
        Instant {
            #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
            t: std::time::Instant::now(),
        }
    }

    pub(crate) fn elapsed(&self) -> Duration {
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        return self.t.elapsed();
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        return Duration::ZERO;
    }
}
