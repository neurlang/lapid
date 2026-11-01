//! Graph node storage: a compact node with inline operands / shape, behind a pooled,
//! atomically refcounted pointer.
//!
//! A decode step builds and tears down a few thousand nodes per token. With `Arc<Node>` and
//! `Vec` operands / shapes each node cost four heap blocks scattered over the heap, and the
//! teardown after a step (cold cache) was dominated by misses on them. Here a node is one
//! fixed-size block (operands and shapes up to rank 4 inline), blocks are recycled through a
//! per-thread free list (shared batches between threads), and teardown is iterative (no
//! recursion depth limit on long chains).

use std::cell::RefCell;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering, fence};
use std::sync::Mutex;

use smallvec::SmallVec;

use crate::dtype::DType;
use crate::op::Op;

use super::Tensor;

/// Operands of a node (inline up to 3).
pub type Srcs = SmallVec<[Tensor; 3]>;
/// Shape of a node (inline up to rank 4).
pub type Shape = SmallVec<[usize; 4]>;

#[derive(Debug)]
pub struct Node {
    /// Unique, monotonically increasing id (creation order => valid topological order).
    pub id: u64,
    pub op: Op,
    pub src: Srcs,
    pub shape: Shape,
    pub dtype: DType,
    /// Optional debug name (e.g. "encoder.blocks.3.attn.out"). Used by lapid-verify to
    /// match reference activations. Set with `Tensor::named`.
    pub name: Option<Box<str>>,
    /// Scratch word of graph walks (see `crate::walk::Marker`).
    pub(crate) mark: AtomicU64,
}

struct Inner {
    rc: AtomicUsize,
    node: Node,
}

/// Shared pointer to a pooled node (like `Arc<Node>`).
pub(crate) struct NodeRef {
    ptr: NonNull<Inner>,
}

// SAFETY: the pointee is shared immutably (the refcount and walk mark are atomics); `Node`
// is Send + Sync (its leaves hold Send + Sync data), exactly like `Arc<Node>`.
unsafe impl Send for NodeRef {}
unsafe impl Sync for NodeRef {}

impl NodeRef {
    pub(crate) fn new(node: Node) -> NodeRef {
        let ptr = pool::alloc();
        // SAFETY: `ptr` is a valid, unused block with the layout of `Inner`.
        unsafe { ptr.as_ptr().write(Inner { rc: AtomicUsize::new(1), node }) };
        NodeRef { ptr }
    }

    #[inline]
    fn inner(&self) -> &Inner {
        // SAFETY: alive while any NodeRef points to it.
        unsafe { self.ptr.as_ref() }
    }
}

impl std::ops::Deref for NodeRef {
    type Target = Node;
    #[inline]
    fn deref(&self) -> &Node {
        &self.inner().node
    }
}

impl Clone for NodeRef {
    #[inline]
    fn clone(&self) -> NodeRef {
        let old = self.inner().rc.fetch_add(1, Ordering::Relaxed);
        if old > isize::MAX as usize {
            std::process::abort();
        }
        NodeRef { ptr: self.ptr }
    }
}

/// Drop one reference; true when it was the last (the caller then owns the block).
#[inline]
fn release(p: NonNull<Inner>) -> bool {
    // SAFETY: the caller holds a reference.
    let rc = &unsafe { p.as_ref() }.rc;
    // Sole owner: nobody else can clone it concurrently, skip the read-modify-write.
    if rc.load(Ordering::Acquire) == 1 {
        return true;
    }
    if rc.fetch_sub(1, Ordering::Release) != 1 {
        return false;
    }
    fence(Ordering::Acquire);
    true
}

impl Drop for NodeRef {
    fn drop(&mut self) {
        if release(self.ptr) {
            // SAFETY: we held the last reference.
            unsafe { teardown(self.ptr) };
        }
    }
}

/// Lookahead (queue entries) of the teardown prefetch.
const PREFETCH_AHEAD: usize = 12;

#[inline(always)]
fn prefetch(p: *const u8) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: a prefetch hint never faults.
    unsafe {
        std::arch::asm!("prfm pstl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags));
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch hint never faults.
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = p;
}

/// Every cache line of a node block.
#[inline(always)]
fn prefetch_block(p: NonNull<Inner>) {
    let b = p.as_ptr() as *const u8;
    let mut off = 0;
    while off < std::mem::size_of::<Inner>() {
        prefetch(b.wrapping_add(off));
        off += 64;
    }
    prefetch(b.wrapping_add(std::mem::size_of::<Inner>() - 1));
}

/// Destroy a node whose last reference is gone, and every operand that dies with it.
///
/// Iterative (no recursion limit on long chains) and breadth first over a queue of pending
/// operand references: a graph is torn down after its evaluation, when its nodes are long out
/// of cache, so each queued block is prefetched a few entries before it is released and the
/// misses overlap instead of serializing.
///
/// # Safety
/// The caller owns the last reference to `root`.
unsafe fn teardown(root: NonNull<Inner>) {
    // SAFETY (whole body): every pointer in `queue` holds one reference that we own; a block
    // is destroyed exactly once, by whoever drops its last reference.
    type Queue = SmallVec<[NonNull<Inner>; 32]>;
    unsafe fn destroy(p: NonNull<Inner>, queue: &mut Queue, dead: &mut Queue) {
        unsafe {
            let node = &mut (*p.as_ptr()).node;
            for s in std::mem::take(&mut node.src) {
                queue.push(ManuallyDrop::new(s).0.ptr);
            }
            std::ptr::drop_in_place(p.as_ptr());
        }
        dead.push(p);
    }
    let (mut queue, mut dead) = (Queue::new(), Queue::new());
    unsafe { destroy(root, &mut queue, &mut dead) };
    let mut i = 0;
    while i < queue.len() {
        if let Some(&ahead) = queue.get(i + PREFETCH_AHEAD) {
            prefetch_block(ahead);
        }
        let p = queue[i];
        i += 1;
        if i >= 4096 && 2 * i >= queue.len() {
            // Keep the queue bounded by the live frontier on long chains.
            queue.drain(..i);
            i = 0;
        }
        if release(p) {
            unsafe { destroy(p, &mut queue, &mut dead) };
        }
    }
    // SAFETY: every block in `dead` was destroyed above.
    unsafe { pool::free_all(&dead) };
}

impl std::fmt::Debug for NodeRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

/// Free blocks of `Inner`: a per-thread LIFO list, overflowing to / refilling from shared
/// batches; beyond both caps blocks go back to the allocator.
mod pool {
    use super::*;
    use std::alloc::{Layout, dealloc, handle_alloc_error};

    /// Blocks kept per thread.
    const LOCAL_MAX: usize = 16 << 10;
    /// Blocks moved per exchange with the shared list.
    const BATCH: usize = 1 << 10;
    /// Shared batches kept.
    const SHARED_MAX: usize = 64;
    /// Lookahead (free-list entries) of the allocation prefetch.
    const ALLOC_AHEAD: usize = 8;

    struct Block(NonNull<Inner>);
    // SAFETY: an unused, exclusively owned allocation.
    unsafe impl Send for Block {}

    static SHARED: Mutex<Vec<Vec<Block>>> = Mutex::new(Vec::new());

    struct Local(Vec<Block>);
    impl Drop for Local {
        fn drop(&mut self) {
            give_back(std::mem::take(&mut self.0));
        }
    }

    thread_local! {
        static LOCAL: RefCell<Local> = const { RefCell::new(Local(Vec::new())) };
    }

    fn layout() -> Layout {
        Layout::new::<Inner>()
    }

    fn shared() -> std::sync::MutexGuard<'static, Vec<Vec<Block>>> {
        SHARED.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Hand a batch to the shared list, or free it.
    fn give_back(batch: Vec<Block>) {
        if batch.is_empty() {
            return;
        }
        let mut g = shared();
        if g.len() < SHARED_MAX {
            g.push(batch);
            return;
        }
        drop(g);
        for b in batch {
            // SAFETY: allocated by `alloc` below with this layout.
            unsafe { dealloc(b.0.as_ptr().cast(), layout()) };
        }
    }

    pub(super) fn alloc() -> NonNull<Inner> {
        let got = LOCAL
            .try_with(|l| {
                let mut l = l.borrow_mut();
                if let Some(b) = l.0.pop() {
                    // Blocks freed long ago are out of cache: fetch the ones the next
                    // allocations will get while this one is filled.
                    if let Some(ahead) = l.0.len().checked_sub(ALLOC_AHEAD).and_then(|i| l.0.get(i)) {
                        prefetch_block(ahead.0);
                    }
                    return Some(b.0);
                }
                let batch = shared().pop()?;
                l.0 = batch;
                l.0.pop().map(|b| b.0)
            })
            .ok()
            .flatten();
        got.unwrap_or_else(|| {
            // SAFETY: `Inner` has non-zero size.
            let p = unsafe { std::alloc::alloc(layout()) };
            NonNull::new(p.cast()).unwrap_or_else(|| handle_alloc_error(layout()))
        })
    }

    /// # Safety
    /// Every block is an `Inner` block from `alloc` whose contents were already dropped.
    pub(super) unsafe fn free_all(blocks: &[NonNull<Inner>]) {
        let kept = LOCAL.try_with(|l| {
            let Ok(mut l) = l.try_borrow_mut() else { return false };
            for &p in blocks {
                if l.0.len() >= LOCAL_MAX {
                    let at = l.0.len() - BATCH;
                    let batch = l.0.split_off(at);
                    give_back(batch);
                }
                l.0.push(Block(p));
            }
            true
        });
        if !matches!(kept, Ok(true)) {
            for &p in blocks {
                // SAFETY: allocated by `alloc` with this layout.
                unsafe { dealloc(p.as_ptr().cast(), layout()) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_is_compact() {
        assert!(std::mem::size_of::<Inner>() <= 192, "{}", std::mem::size_of::<Inner>());
    }

    #[test]
    fn deep_chain_drops_without_recursion() {
        let mut t = Tensor::zeros(&[4], DType::F32);
        for _ in 0..200_000 {
            t = t.exp().unwrap();
        }
        drop(t);
    }

    #[test]
    fn shared_operands_survive() {
        let a = Tensor::zeros(&[2], DType::F32);
        let b = a.exp().unwrap();
        let c = b.add(&b).unwrap();
        drop(b);
        assert_eq!(c.src()[0].src()[0].id(), a.id());
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let c = c.clone();
                std::thread::spawn(move || {
                    for _ in 0..10_000 {
                        let d = c.mul(&c).unwrap();
                        assert_eq!(d.src()[0].id(), c.id());
                    }
                })
            })
            .collect();
        threads.into_iter().for_each(|t| t.join().unwrap());
        assert_eq!(c.shape(), &[2]);
    }
}
