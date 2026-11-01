//! MSL source → `newLibraryWithSource` → compute pipeline, cached by kernel key.
//!
//! No lapid-owned on-disk cache: the system Metal cache (per user, `com.apple.metal` under
//! `DARWIN_USER_CACHE_DIR`) already keys compiled libraries by source, and with it warm the
//! 33 pipelines of a Whisper transcription take 7.5 ms in total (M4 Pro). The cold cost is
//! `newLibraryWithSource` itself (~350 ms serial); an `MTLBinaryArchive` only covers the
//! pipeline stage (~4 ms) because it still needs the `MTLFunction` of a source-compiled
//! library, and a source-compiled library cannot be serialized without the offline
//! toolchain. Instead `warm` compiles the missing pipelines of an execute in parallel
//! (cold process 0.60 → 0.37 s).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2::sel;
use objc2_foundation::NSString;
use objc2_metal::{MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLLibrary, MTLMathMode};

use lapid_tensor::{Error, Result};

use super::device::Ctx;
use crate::codegen::template::{ENTRY, numbered};
use crate::codegen::{self, Key};

pub(crate) struct Pipeline {
    pub state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Max threads per threadgroup the compiled kernel supports (register pressure).
    pub max_threads: u32,
}

#[derive(Default)]
pub(crate) struct Pipelines {
    map: Mutex<HashMap<Key, Arc<Pipeline>>>,
}

fn debug() -> bool {
    std::env::var("LAPID_DEBUG").is_ok_and(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
}

impl Pipelines {
    /// Cached pipeline for `key`, compiling on a miss (without holding the lock).
    pub(crate) fn get(&self, ctx: &Ctx, key: &Key) -> Result<Arc<Pipeline>> {
        if let Some(p) = self.lock().get(key) {
            return Ok(p.clone());
        }
        let p = Arc::new(compile(ctx, key.label(), &codegen::source(key))?);
        Ok(self.lock().entry(key.clone()).or_insert(p).clone())
    }

    /// Pipelines of `keys` in order: one map lookup per key, the misses compiled in parallel
    /// (`warm`) first.
    pub(crate) fn get_all(&self, ctx: &Ctx, keys: &[&Key]) -> Result<Vec<Arc<Pipeline>>> {
        let mut out: Vec<Option<Arc<Pipeline>>> = {
            let map = self.lock();
            keys.iter().map(|k| map.get(*k).cloned()).collect()
        };
        let missing: Vec<&Key> = keys.iter().zip(&out).filter(|(_, p)| p.is_none()).map(|(k, _)| *k).collect();
        if !missing.is_empty() {
            self.warm(ctx, &missing);
            for (k, p) in keys.iter().zip(out.iter_mut()) {
                if p.is_none() {
                    *p = Some(self.get(ctx, k)?);
                }
            }
        }
        Ok(out.into_iter().flatten().collect())
    }

    /// Compile the missing pipelines among `keys` in parallel (the Metal compiler service
    /// takes ~10 ms per new kernel when its own cache is cold, e.g. after an update, so the
    /// first encoder pass would otherwise wait for dozens of serial compiles). Failures are
    /// left for `get`, which reports them.
    pub(crate) fn warm(&self, ctx: &Ctx, keys: &[&Key]) {
        let missing: Vec<&Key> = {
            let map = self.lock();
            let mut v: Vec<&Key> = Vec::new();
            for k in keys {
                if !map.contains_key(*k) && !v.contains(k) {
                    v.push(k);
                }
            }
            v
        };
        if missing.len() < 2 {
            return;
        }
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8).min(missing.len());
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(key) = missing.get(i) else { break };
                        if let Ok(p) = compile(ctx, key.label(), &codegen::source(key)) {
                            self.lock().entry((*key).clone()).or_insert_with(|| Arc::new(p));
                        }
                    }
                });
            }
        });
    }

    pub(crate) fn len(&self) -> usize {
        self.lock().len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<Key, Arc<Pipeline>>> {
        self.map.lock().unwrap_or_else(|e| e.into_inner())
    }
}

fn shader_error(label: &str, what: &str, msg: &str, src: &str) -> Error {
    let listing = numbered(src);
    if debug() {
        eprintln!("[lapid-metal] {label}: {what} failed: {msg}\n{listing}");
    }
    Error::new("shader", format!("metal {label}: {what} failed: {msg}\n{listing}"))
}

/// Compile one MSL module whose entry point is `lapid_kernel`.
pub(crate) fn compile(ctx: &Ctx, label: &str, src: &str) -> Result<Pipeline> {
    autoreleasepool(|_| {
        let opts = MTLCompileOptions::new();
        // Relaxed: fast-math reassociation but INF / NaN semantics preserved (masks use -inf).
        if opts.respondsToSelector(sel!(setMathMode:)) {
            opts.setMathMode(MTLMathMode::Relaxed);
        } else {
            #[allow(deprecated)]
            opts.setFastMathEnabled(false);
        }
        let lib = ctx
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(src), Some(&opts))
            .map_err(|e| shader_error(label, "compile", &e.localizedDescription().to_string(), src))?;
        let func = lib
            .newFunctionWithName(&NSString::from_str(ENTRY))
            .ok_or_else(|| shader_error(label, "lookup", "entry point lapid_kernel not found", src))?;
        let state = ctx
            .device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| shader_error(label, "pipeline creation", &e.localizedDescription().to_string(), src))?;
        let max_threads = state.maxTotalThreadsPerThreadgroup().min(u32::MAX as usize) as u32;
        Ok(Pipeline { state, max_threads })
    })
}
