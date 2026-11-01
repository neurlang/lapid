//! Fork-join thread pool tuned for many tiny launches (decoder steps run ~200 kernels of a
//! few microseconds each), plus the parallel helpers kernels use.
//!
//! * One job at a time per pool. The calling thread publishes the job (a `&dyn Fn()` body
//!   that grabs task indices from a shared atomic counter), runs the body itself, then waits
//!   only for workers that actually joined — never for sleeping ones.
//! * Workers spin for a short while (`LAPID_SPIN_US`, default 1000 µs) after each job before
//!   parking on a condvar, so back-to-back launches of one `execute` cost ~1 µs to dispatch.
//! * The pool used by the helpers is installed per thread by `CpuBackend::execute`
//!   (`Pool::install`). Helpers called outside an installed pool, from a worker, or nested
//!   inside a running job run sequentially on the calling thread.

use std::cell::{Cell, UnsafeCell};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Below this many multiply-accumulates (or comparable element operations) a launch runs on
/// the calling thread: fork/join overhead would dominate.
pub(crate) const PAR_MIN_WORK: usize = 1 << 15;

/// Element count above which elementwise-style kernels go parallel.
pub(crate) const PAR_MIN_ELEMS: usize = 1 << 14;

/// Estimated work (elements x per-element cost, see `kcache::expr_cost`; a plain copy costs
/// 2 per element) above which an expression kernel goes parallel: a fork-join costs ~2 us,
/// about what a serial thread needs for this much work.
pub(crate) const PAR_MIN_COST: usize = 1 << 15;

type Body = dyn Fn() + Sync + 'static;

/// Keeps a value on its own cache line pair (no false sharing between the line every
/// spinning worker polls and the lines that get written per job).
#[repr(align(128))]
struct Pad<T>(T);

/// Per-worker join state, written only by its worker.
struct Slot {
    /// Epoch the worker last tried to enter.
    entered: AtomicU64,
    /// Epoch the worker last left (== `entered` when it is outside any job).
    done: AtomicU64,
}

struct Shared {
    /// Incremented for every published job (polled by spinning workers; written once per job).
    epoch: Pad<AtomicU64>,
    /// Body of the current job; written only while no worker is inside a job.
    job: Pad<UnsafeCell<Option<*const Body>>>,
    /// Set once the caller finished its share: late workers must not enter.
    closed: Pad<AtomicBool>,
    slots: Vec<Pad<Slot>>,
    sleepers: Pad<AtomicUsize>,
    lock: Mutex<()>,
    cv: Condvar,
    shutdown: AtomicBool,
    spin: Duration,
}

// SAFETY: `job` is only written by the single caller between jobs (no worker inside one,
// see `Pool::run`) and read by workers after they registered for the current epoch.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub(crate) struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// Serializes callers; a second concurrent caller runs its job inline.
    running: Mutex<()>,
}

thread_local! {
    /// Pool installed for the current thread (set by `Pool::install`).
    static CURRENT: Cell<*const Pool> = const { Cell::new(std::ptr::null()) };
}

impl Pool {
    /// Pool with `threads` participants (the caller + `threads - 1` workers).
    pub(crate) fn new(threads: usize) -> Pool {
        let spin_us = std::env::var("LAPID_SPIN_US").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(1000);
        let nw = threads.max(1) - 1;
        let shared = Arc::new(Shared {
            epoch: Pad(AtomicU64::new(0)),
            job: Pad(UnsafeCell::new(None)),
            closed: Pad(AtomicBool::new(true)),
            slots: (0..nw).map(|_| Pad(Slot { entered: AtomicU64::new(0), done: AtomicU64::new(0) })).collect(),
            sleepers: Pad(AtomicUsize::new(0)),
            lock: Mutex::new(()),
            cv: Condvar::new(),
            shutdown: AtomicBool::new(false),
            spin: Duration::from_micros(spin_us),
        });
        let mut workers = Vec::new();
        for i in 0..nw {
            let sh = shared.clone();
            match std::thread::Builder::new().name(format!("lapid-cpu-{}", i + 1)).spawn(move || worker(&sh, i)) {
                Ok(h) => workers.push(h),
                Err(_) => break,
            }
        }
        Pool { shared, workers, running: Mutex::new(()) }
    }

    /// Number of participants (workers + caller).
    pub(crate) fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Run `f` with this pool installed for the parallel helpers on the current thread.
    pub(crate) fn install<R>(&self, f: impl FnOnce() -> R) -> R {
        struct Restore(*const Pool);
        impl Drop for Restore {
            fn drop(&mut self) {
                CURRENT.with(|c| c.set(self.0));
            }
        }
        let _restore = Restore(CURRENT.with(|c| c.replace(self as *const Pool)));
        f()
    }

    /// Run `body` on the caller and every available worker; returns when all participants
    /// left the body. `body` must itself distribute work (e.g. via an atomic counter).
    ///
    /// Join protocol (Dekker-style, all SeqCst): a worker stores `entered = e`, then checks
    /// `closed` and `epoch`; the caller stores `closed = true`, then waits for every worker
    /// whose `entered != done`. Either the caller sees the worker's `entered` (and waits for
    /// it) or the worker sees `closed` (and does not run the body).
    fn run(&self, body: &(dyn Fn() + Sync + '_)) {
        let Ok(_guard) = self.running.try_lock() else {
            body();
            return;
        };
        let sh = &*self.shared;
        // SAFETY: lifetime erasure; the pointer is dereferenced only by workers registered
        // for this epoch, and we wait for all of them before returning (body outlives that).
        let ptr: *const Body = unsafe { std::mem::transmute::<&(dyn Fn() + Sync + '_), &'static Body>(body) };
        // SAFETY: no worker is inside a job (the previous run waited for every registered
        // worker; late ones re-check `closed` / `epoch` before reading `job`).
        unsafe { *sh.job.0.get() = Some(ptr) };
        sh.closed.0.store(false, Ordering::SeqCst);
        sh.epoch.0.fetch_add(1, Ordering::SeqCst);
        if sh.sleepers.0.load(Ordering::SeqCst) > 0 {
            let _g = sh.lock.lock().unwrap_or_else(|e| e.into_inner());
            sh.cv.notify_all();
        }
        // Nested helpers inside the body run sequentially on this thread.
        let prev = CURRENT.with(|c| c.replace(std::ptr::null()));
        body();
        CURRENT.with(|c| c.set(prev));
        sh.closed.0.store(true, Ordering::SeqCst);
        for slot in &sh.slots {
            let slot = &slot.0;
            loop {
                let en = slot.entered.load(Ordering::SeqCst);
                if slot.done.load(Ordering::Acquire) == en {
                    break;
                }
                std::hint::spin_loop();
            }
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        {
            let _g = self.shared.lock.lock().unwrap_or_else(|e| e.into_inner());
            self.shared.cv.notify_all();
        }
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

fn worker(sh: &Shared, w: usize) {
    let slot = &sh.slots[w].0;
    let mut seen = 0u64;
    while let Some(e) = wait_epoch(sh, seen) {
        seen = e;
        slot.entered.store(e, Ordering::SeqCst);
        if !sh.closed.0.load(Ordering::SeqCst) && sh.epoch.0.load(Ordering::SeqCst) == e {
            // SAFETY: registered for a published, still open job (see `Pool::run`).
            if let Some(job) = unsafe { *sh.job.0.get() } {
                // A panicking kernel must not leave the slot registered forever.
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { (*job)() }));
                drop(r);
            }
        }
        slot.done.store(e, Ordering::Release);
    }
}

/// Wait until the epoch differs from `seen` (spin, then park). None on shutdown.
fn wait_epoch(sh: &Shared, seen: u64) -> Option<u64> {
    let start = Instant::now();
    let mut i = 0u32;
    loop {
        let e = sh.epoch.0.load(Ordering::Acquire);
        if e != seen {
            return Some(e);
        }
        if sh.shutdown.load(Ordering::Relaxed) {
            return None;
        }
        std::hint::spin_loop();
        i = i.wrapping_add(1);
        if i.is_multiple_of(256) && start.elapsed() >= sh.spin {
            break;
        }
    }
    let mut g = sh.lock.lock().unwrap_or_else(|e| e.into_inner());
    sh.sleepers.0.fetch_add(1, Ordering::SeqCst);
    let r = loop {
        let e = sh.epoch.0.load(Ordering::SeqCst);
        if e != seen {
            break Some(e);
        }
        if sh.shutdown.load(Ordering::SeqCst) {
            break None;
        }
        g = sh.cv.wait(g).unwrap_or_else(|e| e.into_inner());
    };
    sh.sleepers.0.fetch_sub(1, Ordering::SeqCst);
    r
}

/// The pool helpers may use on this thread right now, if any.
fn current() -> Option<&'static Pool> {
    let p = CURRENT.with(|c| c.get());
    // SAFETY: set by `install`, which outlives every helper call on this thread.
    if p.is_null() { None } else { Some(unsafe { &*p }) }
}

/// Raw pointer that may be shared across the pool's threads (disjoint writes only).
pub(crate) struct SyncPtr<T>(pub *mut T);
impl<T> Clone for SyncPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}

/// Run `f(scratch, i)` for i in 0..n, in parallel when `parallel` is set.
/// `init` creates per-participant scratch state (once per thread per call).
pub(crate) fn for_each<S, I, F>(parallel: bool, n: usize, init: I, f: F)
where
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) + Sync + Send,
{
    if n == 0 {
        return;
    }
    let pool = if parallel && n > 1 { current().filter(|p| p.threads() > 1) } else { None };
    let Some(pool) = pool else {
        let mut s = init();
        for i in 0..n {
            f(&mut s, i);
        }
        return;
    };
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    // Grab several tiny tasks at once so the shared counter does not become the bottleneck
    // (e.g. 18000 row copies); ~8 grabs per participant keep the load balanced.
    let grab = (n / (pool.threads() * 8)).max(1);
    let body = || {
        let mut i = next.fetch_add(grab, Ordering::Relaxed);
        if i >= n {
            return;
        }
        let mut s = init();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            while i < n {
                for t in i..(i + grab).min(n) {
                    f(&mut s, t);
                }
                i = next.fetch_add(grab, Ordering::Relaxed);
            }
        }));
        if r.is_err() {
            failed.store(true, Ordering::Relaxed);
            next.store(n, Ordering::Relaxed);
        }
    };
    pool.run(&body);
    // Kernels index-check their writes; re-raise worker panics on the caller.
    assert!(!failed.load(Ordering::Relaxed), "cpu: a parallel kernel task panicked");
}

/// Ordered map over 0..n, in parallel when `parallel` is set.
pub(crate) fn map<T, S, I, F>(parallel: bool, n: usize, init: I, f: F) -> Vec<T>
where
    T: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) -> T + Sync + Send,
{
    let mut out: Vec<std::mem::MaybeUninit<T>> = Vec::with_capacity(n);
    // SAFETY: MaybeUninit needs no initialization.
    unsafe { out.set_len(n) };
    let ptr = SyncPtr(out.as_mut_ptr());
    for_each(parallel, n, init, |s, i| {
        let v = f(s, i);
        let p = ptr;
        // SAFETY: every index is written exactly once by exactly one task.
        unsafe { (*p.0.add(i)).write(v) };
    });
    // SAFETY: for_each ran every index (a panic would have propagated above).
    let mut out = std::mem::ManuallyDrop::new(out);
    unsafe { Vec::from_raw_parts(out.as_mut_ptr() as *mut T, out.len(), out.capacity()) }
}

/// Run `f(scratch, chunk_index, chunk)` over `data.chunks_mut(chunk)`, in parallel when
/// `parallel` is set. `chunk` must be non-zero.
pub(crate) fn chunks_mut<T, S, I, F>(parallel: bool, data: &mut [T], chunk: usize, init: I, f: F)
where
    T: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize, &mut [T]) + Sync + Send,
{
    if data.is_empty() {
        return;
    }
    let len = data.len();
    let n = len.div_ceil(chunk);
    let ptr = SyncPtr(data.as_mut_ptr());
    for_each(parallel, n, init, |s, i| {
        let p = ptr;
        let start = i * chunk;
        let w = chunk.min(len - start);
        // SAFETY: chunks are disjoint sub-slices of `data`, each handed to one task.
        let c = unsafe { std::slice::from_raw_parts_mut(p.0.add(start), w) };
        f(s, i, c);
    });
}

/// Grow-only scratch: returns the first `n` elements, resizing (never shrinking) as needed.
pub(crate) fn grow<T: Clone + Default>(v: &mut Vec<T>, n: usize) -> &mut [T] {
    if v.len() < n {
        v.resize(n, T::default());
    }
    &mut v[..n]
}

/// Split `total` items into blocks of `align`-multiples so that there are roughly
/// `target_tasks` blocks, clamped to [min_block, max_block].
pub(crate) fn block_size(total: usize, target_tasks: usize, align: usize, min_block: usize, max_block: usize) -> usize {
    let per = total.div_ceil(target_tasks.max(1)).max(1);
    let b = per.div_ceil(align) * align;
    b.clamp(min_block.max(align), max_block.max(align))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_runs_every_task_once() {
        let pool = Pool::new(4);
        pool.install(|| {
            for n in [1usize, 2, 7, 100, 1000] {
                let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                for_each(true, n, || (), |_, i| {
                    hits[i].fetch_add(1, Ordering::Relaxed);
                });
                assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
                let v = map(true, n, || 0usize, |c, i| {
                    *c += 1;
                    i * 2
                });
                assert_eq!(v, (0..n).map(|i| i * 2).collect::<Vec<_>>());
                let mut d = vec![0usize; n * 3 + 1];
                chunks_mut(true, &mut d, 3, || (), |_, ci, c| c.iter_mut().for_each(|x| *x = ci));
                assert!(d.iter().enumerate().all(|(i, &x)| x == i / 3));
            }
        });
    }

    #[test]
    fn nested_calls_run_inline() {
        let pool = Pool::new(3);
        pool.install(|| {
            let total = AtomicUsize::new(0);
            for_each(true, 8, || (), |_, _| {
                for_each(true, 4, || (), |_, _| {
                    total.fetch_add(1, Ordering::Relaxed);
                });
            });
            assert_eq!(total.load(Ordering::Relaxed), 32);
        });
    }

    /// Memory read bandwidth through the pool (manual: `cargo test --release bandwidth -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn bandwidth() {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let pool = Pool::new(threads);
        let data = vec![1.0f32; 128 << 20]; // 512 MB
        for tasks in [threads, threads * 4, threads * 16] {
            let chunk = data.len() / tasks;
            pool.install(|| {
                for _ in 0..2 {
                    let t = Instant::now();
                    let sums = map(true, tasks, || (), |_, i| {
                        let mut acc = [0f32; 16];
                        for c in data[i * chunk..(i + 1) * chunk].as_chunks::<16>().0 {
                            for l in 0..16 {
                                acc[l] += c[l];
                            }
                        }
                        acc.iter().sum::<f32>()
                    });
                    let s: f32 = sums.iter().sum();
                    let dt = t.elapsed().as_secs_f64();
                    eprintln!("tasks {tasks}: {:.1} GB/s ({s})", 512e6 / 1.0 / dt / 1e9 * 1.048576);
                }
            });
        }
    }
}
