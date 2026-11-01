//! CUDA graphs for `execute`: the launches of one execute are captured from the stream
//! into a graph and replayed as one `cuGraphLaunch`, which on GB10 cuts the GPU-side gap
//! between small kernels (~2-4 us → ~1 us) and costs less host time than direct launches.
//! Instantiated graphs are cached by kernel sequence; a later execute with the same
//! kernels (e.g. the next decode step, where only sizes and buffers change) updates the
//! cached graph in place (`cuGraphExecUpdate`) instead of instantiating a new one.
//! `LAPID_CUDA_NO_GRAPHS=1` launches directly.

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::hash::{Hash, Hasher};

use lapid_tensor::{Error, Result};

use crate::driver::{self, CUgraph, CUgraphExec, CUstream, Driver, GraphApi};

/// Cached executable graphs (least recently used beyond this are destroyed).
const MAX_CACHED: usize = 128;

#[derive(Default)]
pub(crate) struct Graphs {
    execs: HashMap<u64, CUgraphExec>,
    lru: VecDeque<u64>,
    /// (updated in place, instantiated, rejected updates) since creation (diagnostics).
    pub stats: (u64, u64, u64),
}

pub(crate) fn enabled(drv: &Driver) -> bool {
    drv.graph.is_some()
        && !std::env::var("LAPID_CUDA_NO_GRAPHS").is_ok_and(|v| !v.is_empty() && v != "0")
}

fn api(drv: &Driver) -> Result<&GraphApi> {
    drv.graph
        .as_ref()
        .ok_or_else(|| Error::new("cuda", "CUDA graphs unavailable"))
}

/// Cache key of a kernel sequence (function handles and block sizes).
pub(crate) fn key(funcs: impl Iterator<Item = (usize, u32)>) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for f in funcs {
        f.hash(&mut h);
    }
    h.finish()
}

pub(crate) fn begin(drv: &Driver, stream: CUstream) -> Result<()> {
    let g = api(drv)?;
    // SAFETY: our stream, not capturing yet.
    drv.check("cuStreamBeginCapture", unsafe {
        (g.cuStreamBeginCapture_v2)(stream, driver::CAPTURE_MODE_RELAXED)
    })
}

/// End the capture started by `begin` (also after a failed launch, so the stream is
/// usable again). Returns the captured graph.
pub(crate) fn end(drv: &Driver, stream: CUstream) -> Result<CUgraph> {
    let g = api(drv)?;
    let mut graph: CUgraph = std::ptr::null_mut();
    // SAFETY: the stream is capturing (begun by `begin`).
    drv.check("cuStreamEndCapture", unsafe {
        (g.cuStreamEndCapture)(stream, &mut graph)
    })?;
    Ok(graph)
}

pub(crate) fn destroy_graph(drv: &Driver, graph: CUgraph) {
    if let Some(g) = drv.graph.as_ref()
        && !graph.is_null()
    {
        // SAFETY: graph created by `end`, not used afterwards.
        unsafe { (g.cuGraphDestroy)(graph) };
    }
}

impl Graphs {
    /// Launch `graph` on `stream` through the cached executable for `key` (updated in
    /// place when possible, otherwise instantiated). Consumes `graph`.
    pub fn launch(
        &mut self,
        drv: &Driver,
        stream: CUstream,
        key: u64,
        graph: CUgraph,
    ) -> Result<()> {
        let r = self.launch_inner(drv, stream, key, graph);
        destroy_graph(drv, graph);
        r
    }

    fn launch_inner(
        &mut self,
        drv: &Driver,
        stream: CUstream,
        key: u64,
        graph: CUgraph,
    ) -> Result<()> {
        let g = api(drv)?;
        let mut exec = None;
        if let Some(&e) = self.execs.get(&key) {
            let mut info = [0u8; 64];
            // SAFETY: `e` is a live executable graph; `info` is larger than
            // CUgraphExecUpdateResultInfo.
            let r = unsafe { (g.cuGraphExecUpdate_v2)(e, graph, info.as_mut_ptr() as *mut c_void) };
            if r == driver::CUDA_SUCCESS {
                exec = Some(e);
                self.touch(key);
                self.stats.0 += 1;
            } else {
                self.remove(drv, key);
                self.stats.2 += 1;
            }
        }
        let exec = match exec {
            Some(e) => e,
            None => {
                let mut e: CUgraphExec = std::ptr::null_mut();
                // SAFETY: instantiates the captured graph; the executable is owned by the cache.
                drv.check("cuGraphInstantiate", unsafe {
                    (g.cuGraphInstantiateWithFlags)(&mut e, graph, 0)
                })?;
                self.insert(drv, key, e);
                self.stats.1 += 1;
                e
            }
        };
        // SAFETY: executable graph and stream owned by this backend.
        drv.check("cuGraphLaunch", unsafe { (g.cuGraphLaunch)(exec, stream) })
    }

    fn touch(&mut self, key: u64) {
        if let Some(p) = self.lru.iter().position(|&k| k == key) {
            self.lru.remove(p);
        }
        self.lru.push_back(key);
    }

    fn insert(&mut self, drv: &Driver, key: u64, e: CUgraphExec) {
        while self.execs.len() >= MAX_CACHED {
            match self.lru.front().copied() {
                Some(old) => self.remove(drv, old),
                None => break,
            }
        }
        self.execs.insert(key, e);
        self.touch(key);
    }

    fn remove(&mut self, drv: &Driver, key: u64) {
        if let Some(e) = self.execs.remove(&key)
            && let Some(g) = drv.graph.as_ref()
        {
            // SAFETY: the stream is idle between executes; the executable is unused.
            unsafe { (g.cuGraphExecDestroy)(e) };
        }
        if let Some(p) = self.lru.iter().position(|&k| k == key) {
            self.lru.remove(p);
        }
    }

    /// Destroy every cached executable (stream idle).
    pub fn clear(&mut self, drv: &Driver) {
        let keys: Vec<u64> = self.execs.keys().copied().collect();
        for k in keys {
            self.remove(drv, k);
        }
    }
}
