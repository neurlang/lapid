//! Runtime behind a `Device`: one backend, an upload cache for host leaves, a pool of
//! temp buffers, and the `eval` pipeline (compile → resolve buffers → execute → wrap).
//!
//! Runtimes are shared process-wide per physical device (`"cpu"`, `"vulkan:<adapter>"`):
//! constructing `Device::cpu()` twice yields the same runtime, so weights are uploaded
//! once and device buffers from one handle are valid in the other.

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Duration;

use lapid_compiler::{Backend, BufferSource, CompileOptions};
use lapid_tensor::{DType, DeviceData, Error, Op, Result, Tensor, bail};

mod cache;
mod clock;
mod convert;
mod plan;
mod pool;
mod predict;

use cache::UploadCache;
use clock::Instant;
use plan::{GraphKey, LeafRef, Lookup, Plan, PlanCache, PlanSrc};
use pool::Pool;

/// Lock ignoring poisoning: every structure guarded here stays consistent across panics
/// (they only hold caches / free lists).
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Live runtimes, used to share backends between `Device` handles and to find the owner
/// of a foreign device buffer.
static REGISTRY: Mutex<Vec<Weak<Runtime>>> = Mutex::new(Vec::new());

pub(crate) struct Runtime {
    /// Registry key: "cpu" or "vulkan:<adapter index>".
    key: String,
    pub(crate) backend: Arc<dyn Backend>,
    cache: UploadCache,
    pool: Pool,
    plans: Mutex<PlanCache>,
    debug: AtomicBool,
    profile: AtomicBool,
}

/// Get the live runtime for `key` or create it with `make`.
pub(crate) fn shared<E>(
    key: &str,
    make: impl FnOnce() -> std::result::Result<Arc<dyn Backend>, E>,
) -> std::result::Result<Arc<Runtime>, E> {
    let mut reg = lock(&REGISTRY);
    reg.retain(|w| w.strong_count() > 0);
    if let Some(rt) = reg.iter().filter_map(Weak::upgrade).find(|rt| rt.key == key) {
        return Ok(rt);
    }
    let rt = Arc::new(Runtime::new(key.to_string(), make()?));
    reg.push(Arc::downgrade(&rt));
    Ok(rt)
}

/// `shared` for backends whose construction cannot fail.
pub(crate) fn shared_infallible(key: &str, make: impl FnOnce() -> Arc<dyn Backend>) -> Arc<Runtime> {
    match shared::<Infallible>(key, || Ok(make())) {
        Ok(rt) => rt,
        Err(never) => match never {},
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

impl Runtime {
    fn new(key: String, backend: Arc<dyn Backend>) -> Runtime {
        Runtime {
            key,
            backend,
            cache: UploadCache::new(),
            pool: Pool::new(),
            plans: Mutex::new(PlanCache::default()),
            debug: AtomicBool::new(env_flag("LAPID_DEBUG")),
            profile: AtomicBool::new(env_flag("LAPID_PROFILE")),
        }
    }

    pub(crate) fn set_debug(&self, on: bool) {
        self.debug.store(on, Ordering::Relaxed);
    }

    fn debug(&self) -> bool {
        self.debug.load(Ordering::Relaxed)
    }

    fn owns(&self, buf: &DeviceData) -> bool {
        buf.backend() == self.backend.name()
    }

    /// Compile and run `outputs` together; see `Device::eval`.
    pub(crate) fn eval(&self, outputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        if outputs.is_empty() {
            return Ok(Vec::new());
        }
        // Fast path: everything requested is already realized on this device.
        if outputs.iter().all(|t| matches!(t.op(), Op::Device(d) if self.owns(d))) {
            return Ok(outputs.iter().map(|t| (*t).clone()).collect());
        }
        let debug = self.debug();
        let profile = debug || self.profile.load(Ordering::Relaxed);
        let t0 = Instant::now();
        let key = GraphKey::new(outputs);
        let cached = lock(&self.plans).get(&key);
        let hit = cached.as_ref().map(|c| c.1);
        let plan = match cached {
            Some((p, _)) => p,
            None => {
                let owned: Vec<Tensor> = outputs.iter().map(|t| (*t).clone()).collect();
                let schedule = lapid_compiler::compile(&owned, &CompileOptions::default())?;
                if debug {
                    crate::log::emit(&format!(
                        "[lapid] {}: compiled {} launches, {} buffers, {} temp slots in {:.3} ms\n{}",
                        self.backend.name(),
                        schedule.launches.len(),
                        schedule.buffers.len(),
                        schedule.slot_bytes.len(),
                        ms(t0.elapsed()),
                        lapid_compiler::dump(&schedule)
                    ));
                }
                let plan = Arc::new(Plan::new(schedule, &key));
                lock(&self.plans).insert(&key, plan.clone());
                plan
            }
        };
        let t_compile = t0.elapsed();

        let temps = self.pool.take_slots(&*self.backend, &plan.slot_bytes)?;
        let result = self.run(&plan, &key.leaves, &temps, debug);
        // `execute` waits for completion, so the temps are free again.
        self.pool.give_slots(&plan.slot_bytes, temps);
        let (buffers, t_resolve, t_exec) = result?;

        if profile {
            let kinds = self.backend.take_profile();
            let mut line = format!(
                "[lapid-profile] {} eval: {} nodes, {} launches, compile {:.3} ms ({}), resolve {:.3} ms, execute {:.3} ms, total {:.3} ms",
                self.backend.name(),
                key.nodes,
                plan.launches.len(),
                ms(t_compile),
                match hit {
                    Some(Lookup::Exact) => "cached",
                    Some(Lookup::Predicted) => "predicted",
                    None => "miss",
                },
                ms(t_resolve),
                ms(t_exec),
                ms(t0.elapsed())
            );
            if !kinds.is_empty() {
                line.push_str("\n[lapid-profile]   kernels:");
                for (kind, n, t) in kinds {
                    line.push_str(&format!(" {kind} {n}x {t:.3} ms;"));
                }
            }
            crate::log::emit(&line);
        }
        plan.outputs
            .iter()
            .map(|o| {
                let buf = buffers.get(o.buf).ok_or_else(|| {
                    Error::new("internal", format!("schedule output refers to missing buffer {}", o.buf))
                })?;
                Ok(Tensor::from_device(buf.clone(), &o.shape, o.dtype))
            })
            .collect()
    }

    fn resolve_source(&self, src: &BufferSource, bytes: usize) -> Result<DeviceData> {
        Ok(match src {
            BufferSource::Host(h) => self.resolve_host(h)?,
            BufferSource::Device(d) => self.resolve_device(d)?,
            BufferSource::Temp { .. } => bail!("internal", "unbound temp buffer"),
            // Never hand a zero-byte allocation to a backend.
            BufferSource::Output => self.backend.alloc(bytes.max(4))?,
        })
    }

    fn resolve_host(&self, h: &lapid_tensor::HostData) -> Result<DeviceData> {
        let pinned = crate::file::is_file_backed(h.key());
        self.cache.get_or_upload(h, &*self.backend, pinned)
    }

    fn resolve_device(&self, d: &DeviceData) -> Result<DeviceData> {
        if self.owns(d) { Ok(d.clone()) } else { self.backend.upload(&download_foreign(self, d)?) }
    }

    /// Resolve every plan buffer to a device buffer (leaves rebound to this graph's) and
    /// execute the launches. Returns the buffers and the resolve / execute times.
    fn run(
        &self,
        plan: &Plan,
        leaves: &[LeafRef],
        temps: &[DeviceData],
        debug: bool,
    ) -> Result<(Vec<DeviceData>, Duration, Duration)> {
        let t0 = Instant::now();
        // Each distinct leaf is resolved once even if several buffers refer to it; weights
        // (file-backed host data) come from the plan's memo when it saw the same data.
        let mut leaf_bufs: Vec<Option<DeviceData>> = vec![None; leaves.len()];
        let mut memo = plan.memo.try_lock().ok();
        if let Some(m) = memo.as_deref_mut() {
            m.resize(leaves.len(), None);
            for ((slot, leaf), entry) in leaf_bufs.iter_mut().zip(leaves).zip(m.iter()) {
                if let (LeafRef::Host(h), Some((mh, md))) = (leaf, entry)
                    && mh.key() == h.key()
                {
                    *slot = Some(md.clone());
                }
            }
        }
        let mut buffers = Vec::with_capacity(plan.bufs.len());
        for (id, b) in plan.bufs.iter().enumerate() {
            let buf = match &b.src {
                PlanSrc::Leaf(i) => {
                    let Some(slot) = leaf_bufs.get_mut(*i) else {
                        bail!("internal", "buffer {id} refers to missing leaf {i}");
                    };
                    match slot {
                        Some(d) => d.clone(),
                        None => {
                            let d = match &leaves[*i] {
                                LeafRef::Host(h) => {
                                    let pinned = crate::file::is_file_backed(h.key());
                                    let d = self.cache.get_or_upload(h, &*self.backend, pinned)?;
                                    if pinned && let Some(m) = memo.as_deref_mut() && let Some(e) = m.get_mut(*i) {
                                        *e = Some((h.clone(), d.clone()));
                                    }
                                    d
                                }
                                LeafRef::Device(d) => self.resolve_device(d)?,
                            };
                            *slot = Some(d.clone());
                            d
                        }
                    }
                }
                PlanSrc::Temp(slot) => match temps.get(*slot) {
                    Some(b) => b.clone(),
                    None => bail!("internal", "buffer {id} refers to missing temp slot {slot}"),
                },
                PlanSrc::Output => self.backend.alloc(b.bytes.max(4))?,
                PlanSrc::Fixed(src) => self.resolve_source(src, b.bytes)?,
            };
            buffers.push(buf);
        }
        drop(memo);
        let t_resolve = t0.elapsed();

        let t1 = Instant::now();
        if debug {
            // Run launches one by one to time them (same semantics: each `execute`
            // completes before the next starts).
            for (i, launch) in plan.launches.iter().enumerate() {
                let tl = Instant::now();
                self.backend.execute(std::slice::from_ref(launch), &buffers)?;
                crate::log::emit(&format!("[lapid]   #{i:<4} {:<12} {:>9.3} ms", launch.kernel.kind(), ms(tl.elapsed())));
            }
        } else {
            self.backend.execute(&plan.launches, &buffers)?;
        }
        let t_exec = t1.elapsed();
        if debug {
            crate::log::emit(&format!("[lapid] resolve buffers {:.3} ms, execute {:.3} ms", ms(t_resolve), ms(t_exec)));
        }
        Ok((buffers, t_resolve, t_exec))
    }

    /// Upload a host leaf now (cached), or realize anything else on this device.
    pub(crate) fn upload(&self, t: &Tensor) -> Result<Tensor> {
        match t.op() {
            Op::Host(h) => {
                let pinned = crate::file::is_file_backed(h.key());
                let buf = self.cache.get_or_upload(h, &*self.backend, pinned)?;
                Ok(Tensor::from_device(buf, t.shape(), t.dtype()))
            }
            Op::Device(d) if self.owns(d) => Ok(t.clone()),
            Op::Device(d) => {
                let bytes = download_foreign(self, d)?;
                Ok(Tensor::from_device(self.backend.upload(&bytes)?, t.shape(), t.dtype()))
            }
            _ => self.eval(&[t])?.pop().ok_or_else(|| Error::new("internal", "eval returned no output")),
        }
    }

    /// Bytes of `t` (contiguous row-major, exactly `storage_bytes`) on the host.
    fn host_bytes(&self, t: &Tensor) -> Result<Vec<u8>> {
        let want = expected_bytes(t)?;
        let mut bytes = match t.op() {
            Op::Host(h) => h.bytes().to_vec(),
            Op::Device(d) if self.owns(d) => self.backend.download(d)?,
            Op::Device(d) => download_foreign(self, d)?,
            _ => {
                let realized = self.eval(&[t])?;
                let Some(Op::Device(d)) = realized.first().map(Tensor::op) else {
                    bail!("internal", "eval did not return a device tensor");
                };
                self.backend.download(d)?
            }
        };
        if bytes.len() < want {
            bail!("backend", "downloaded {} bytes, expected {want}", bytes.len());
        }
        bytes.truncate(want);
        Ok(bytes)
    }

    pub(crate) fn to_vec_f32(&self, t: &Tensor) -> Result<Vec<f32>> {
        if let Op::Host(h) = t.op() {
            // No device round trip for host leaves.
            let want = expected_bytes(t)?;
            let Some(bytes) = h.bytes().get(..want) else {
                bail!("shape_mismatch", "host leaf is smaller than its shape {:?}", t.shape());
            };
            return convert::to_f32(bytes, t.dtype(), t.shape());
        }
        convert::to_f32(&self.host_bytes(t)?, t.dtype(), t.shape())
    }

    /// Raw storage bytes of `t` in its own dtype (quantized: the lapid-quant layout).
    pub(crate) fn to_vec_bytes(&self, t: &Tensor) -> Result<Vec<u8>> {
        let want = expected_bytes(t)?;
        if let Op::Host(h) = t.op() {
            let Some(bytes) = h.bytes().get(..want) else {
                bail!("shape_mismatch", "host leaf is smaller than its shape {:?}", t.shape());
            };
            return Ok(bytes.to_vec());
        }
        let mut bytes = self.host_bytes(t)?;
        bytes.truncate(want);
        Ok(bytes)
    }

    pub(crate) fn to_vec_i32(&self, t: &Tensor) -> Result<Vec<i32>> {
        if let Op::Host(h) = t.op() {
            let want = expected_bytes(t)?;
            let Some(bytes) = h.bytes().get(..want) else {
                bail!("shape_mismatch", "host leaf is smaller than its shape {:?}", t.shape());
            };
            return convert::to_i32(bytes, t.dtype(), t.shape());
        }
        convert::to_i32(&self.host_bytes(t)?, t.dtype(), t.shape())
    }
}

/// Storage size of `t`, validated so quantized shapes never panic.
fn expected_bytes(t: &Tensor) -> Result<usize> {
    let dt = t.dtype();
    let block = dt.quant_format().map_or(DType::QUANT_GROUP, |f| f.block_elems());
    if dt.is_quant() && (!(2..=3).contains(&t.rank()) || t.shape()[t.rank() - 1] % block != 0) {
        bail!("shape_mismatch", "quantized {} tensor has invalid shape {:?}", dt.name(), t.shape());
    }
    Ok(dt.storage_bytes(t.shape()))
}

/// Download a buffer owned by another backend through a live runtime of that backend.
///
/// Limitation: buffers are matched by backend name only; with several live Vulkan
/// devices the first one that accepts the buffer is used.
fn download_foreign(me: &Runtime, buf: &DeviceData) -> Result<Vec<u8>> {
    let name = buf.backend();
    let owners: Vec<Arc<Runtime>> = lock(&REGISTRY)
        .iter()
        .filter_map(Weak::upgrade)
        .filter(|rt| rt.backend.name() == name && !std::ptr::eq(Arc::as_ptr(rt), me))
        .collect();
    let mut last_err = None;
    for rt in owners {
        match rt.backend.download(buf) {
            Ok(bytes) => return Ok(bytes),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| {
        Error::new(
            "backend",
            format!("cannot transfer a '{name}' buffer to '{}': no live '{name}' device", me.backend.name()),
        )
    }))
}
