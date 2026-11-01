//! Encoding and submission. One command buffer per `execute` with a single compute encoder.
//! Hazards are tracked on MTLBuffer identity: a barrier is needed in front of a dispatch that
//! reads a buffer written since the last barrier, or writes one read or written since then
//! (RAW / WAR / WAW). When most dispatches need one (dependency chains, the common case) the
//! encoder is serial (implicit ordering, cheaper than explicit barriers); otherwise it is
//! concurrent with `memoryBarrierWithScope(buffers)` exactly where needed.
//! Commit + waitUntilCompleted.
//!
//! Profiling mode (`LAPID_PROFILE=1`): every launch gets its own command buffer, committed and
//! waited on, and its GPUStartTime..GPUEndTime is added to the launch's kernel kind.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDevice, MTLDispatchType, MTLEvent, MTLSize,
};

use lapid_compiler::Launch;
use lapid_tensor::{DeviceData, Error, Result};

use super::buffer::{OwnedRaw, RawBuffer, as_metal, new_raw};
use super::device::Ctx;
use super::pipeline::{Pipeline, Pipelines};
use crate::codegen::{self, Bind, Stage};

type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;
type CmdBuf = ProtocolObject<dyn MTLCommandBuffer>;

/// Per-kernel-kind GPU time totals.
#[derive(Default)]
pub(crate) struct Profile {
    totals: BTreeMap<&'static str, (usize, f64)>,
}

impl Profile {
    fn add(&mut self, kind: &'static str, ms: f64) {
        let e = self.totals.entry(kind).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += ms;
    }

    pub(crate) fn take(&mut self) -> Vec<(&'static str, usize, f64)> {
        std::mem::take(&mut self.totals).into_iter().map(|(k, (n, ms))| (k, n, ms)).collect()
    }
}

/// Mutable execution state (guarded by the backend's mutex).
pub(crate) struct Exec {
    scratch: Option<OwnedRaw>,
    scratch_len: usize,
    /// Out-of-range status word (`Bind::Status`), allocated on first use.
    status: Option<OwnedRaw>,
    pub profile: Option<Profile>,
    /// Profile whole command buffers (normal batched execution, GPU time reported as
    /// "command_buffer") instead of one command buffer per launch.
    per_cb: bool,
    /// `LAPID_METAL_ENCODER=serial|concurrent` forces the encoder kind (experiments).
    encoder_override: Option<bool>,
    /// Orders the command buffers of a split execute (see `SPLIT_MIN`); last signaled value.
    event: Option<OwnedEvent>,
    event_value: u64,
}

pub(crate) struct OwnedEvent(Retained<ProtocolObject<dyn MTLEvent>>);
// SAFETY: MTLEvent objects are thread-safe; only used under the backend's exec mutex.
unsafe impl Send for OwnedEvent {}
unsafe impl Sync for OwnedEvent {}

/// Executes with at least this many dispatches are split into two command buffers: the first
/// `FIRST_CHUNK` dispatches are committed as soon as they are encoded, so the GPU starts
/// while the CPU encodes the rest (a decoder step: ~0.2 ms of encoding). The second command
/// buffer waits on an MTLEvent signaled by the first (the buffers are untracked).
const SPLIT_MIN: usize = 192;
const FIRST_CHUNK: usize = 48;

/// A stage ready to encode.
struct Ready<'a> {
    stage: &'a Stage,
    launch: usize,
    pipeline: Arc<Pipeline>,
    bufs: Vec<&'a RawBuffer>,
    reads: Vec<usize>,
    writes: Vec<usize>,
}

#[derive(Default)]
struct Hazards {
    written: Vec<usize>,
    read: Vec<usize>,
}

impl Hazards {
    /// Record a dispatch; returns whether a barrier must precede it.
    fn barrier_before(&mut self, reads: &[usize], writes: &[usize]) -> bool {
        let conflict = reads.iter().any(|r| self.written.contains(r))
            || writes.iter().any(|w| self.written.contains(w) || self.read.contains(w));
        if conflict {
            self.written.clear();
            self.read.clear();
        }
        self.read.extend_from_slice(reads);
        self.written.extend_from_slice(writes);
        conflict
    }
}

/// Dispatches with a median grid below this many threads count as "small".
const SMALL_DISPATCH_THREADS: u64 = 16 * 1024;

/// Serial encoder for chains of small dispatches (e.g. a decoder step): most dispatches need
/// a barrier anyway and the implicit ordering of a serial encoder is cheaper than explicit
/// barriers. Large dispatches (an encoder pass) keep the concurrent encoder, where the few
/// independent launches overlap. Measured on M4 Pro: decode step 1.90 -> 1.75 ms GPU with
/// serial; encoder pass 70 -> 81 ms GPU with serial.
fn prefer_serial(ready: &[Ready<'_>], barriers: &[bool]) -> bool {
    if barriers.iter().filter(|&&b| b).count() * 2 <= ready.len() {
        return false;
    }
    let mut sizes: Vec<u64> = ready
        .iter()
        .map(|r| r.stage.groups.iter().map(|&g| g as u64).product::<u64>() * r.stage.threads as u64)
        .collect();
    sizes.sort_unstable();
    sizes[sizes.len() / 2] < SMALL_DISPATCH_THREADS
}

fn raw_id(b: &RawBuffer) -> usize {
    b as *const RawBuffer as *const () as usize
}

fn check(cb: &CmdBuf) -> Result<()> {
    if cb.status() == MTLCommandBufferStatus::Error {
        let msg = cb.error().map(|e| e.localizedDescription().to_string()).unwrap_or_else(|| "unknown error".into());
        return Err(Error::new("metal", format!("command buffer failed: {msg}")));
    }
    Ok(())
}

fn size(w: u32, h: u32, d: u32) -> MTLSize {
    MTLSize { width: w as usize, height: h as usize, depth: d as usize }
}

fn encode(enc: &Encoder, r: &Ready<'_>) {
    enc.setComputePipelineState(&r.pipeline.state);
    let params: &[u32] = if r.stage.params.is_empty() { &[0] } else { &r.stage.params };
    // SAFETY: `params` outlives the call (setBytes copies); every bound buffer is alive until
    // the command buffer completes (the caller holds `buffers` and waits); indices match the
    // kernel's [[buffer(i)]] declarations by construction (codegen binds).
    unsafe {
        let ptr = NonNull::new(params.as_ptr() as *mut c_void).unwrap_or(NonNull::dangling());
        enc.setBytes_length_atIndex(ptr, params.len() * 4, 0);
        for (i, b) in r.bufs.iter().enumerate() {
            enc.setBuffer_offset_atIndex(Some(*b), 0, i + 1);
        }
    }
    let g = r.stage.groups;
    enc.dispatchThreadgroups_threadsPerThreadgroup(size(g[0], g[1], g[2]), size(r.stage.threads, 1, 1));
}

impl Exec {
    pub(crate) fn new(profiling: bool, per_cb: bool) -> Exec {
        let encoder_override = match std::env::var("LAPID_METAL_ENCODER").as_deref().map(str::trim) {
            Ok("serial") => Some(true),
            Ok("concurrent") => Some(false),
            _ => None,
        };
        Exec {
            scratch: None,
            scratch_len: 0,
            status: None,
            profile: profiling.then(Profile::default),
            per_cb,
            encoder_override,
            event: None,
            event_value: 0,
        }
    }

    fn ensure_scratch(&mut self, ctx: &Ctx, bytes: usize) -> Result<()> {
        if bytes > self.scratch_len || self.scratch.is_none() {
            let len = bytes.max(1 << 16).next_power_of_two();
            self.scratch = Some(OwnedRaw(new_raw(ctx, len)?));
            self.scratch_len = len;
        }
        Ok(())
    }

    pub(crate) fn run(
        &mut self,
        ctx: &Ctx,
        pipelines: &Pipelines,
        launches: &[Launch],
        buffers: &[DeviceData],
        times: Option<&mut Vec<f64>>,
    ) -> Result<()> {
        let t0 = std::time::Instant::now();
        let plans = launches.iter().map(|l| codegen::plan(&l.kernel)).collect::<Result<Vec<_>>>()?;
        let need = plans.iter().map(|p| p.scratch).max().unwrap_or(0);
        if need > 0 {
            self.ensure_scratch(ctx, need)?;
        }
        let scratch_owned: Option<Retained<RawBuffer>> = self.scratch.as_ref().map(|s| s.0.clone());
        let scratch = scratch_owned.as_deref();
        let uses_status = plans.iter().any(|p| p.stages.iter().any(|s| s.binds.contains(&Bind::Status)));
        if uses_status && self.status.is_none() {
            self.status = Some(OwnedRaw(new_raw(ctx, 4)?));
        }
        let status_owned: Option<Retained<RawBuffer>> = if uses_status { self.status.as_ref().map(|s| s.0.clone()) } else { None };
        let status = status_owned.as_deref();
        if let Some(st) = status {
            set_status(st, 0);
        }
        let keys: Vec<&codegen::Key> = plans.iter().flat_map(|p| &p.stages).filter(|s| !s.is_empty()).map(|s| &s.key).collect();
        let mut found = pipelines.get_all(ctx, &keys)?.into_iter();
        let mut ready = Vec::with_capacity(keys.len());
        for (li, (launch, plan)) in launches.iter().zip(&plans).enumerate() {
            for stage in &plan.stages {
                if stage.is_empty() {
                    continue;
                }
                let pipeline = found.next().ok_or_else(|| bad_launch(li, "pipeline missing"))?;
                if stage.threads > pipeline.max_threads {
                    return Err(Error::new(
                        "unsupported",
                        format!(
                            "metal {}: needs {} threads per threadgroup, pipeline supports {}",
                            stage.key.label(),
                            stage.threads,
                            pipeline.max_threads
                        ),
                    ));
                }
                let mut bufs = Vec::with_capacity(stage.binds.len());
                let (mut reads, mut writes) = (Vec::new(), Vec::new());
                for bind in &stage.binds {
                    let raw: &RawBuffer = match *bind {
                        Bind::Input(i) => {
                            let id = *launch.inputs.get(i).ok_or_else(|| bad_launch(li, "missing input"))?;
                            let b = buffers.get(id).ok_or_else(|| bad_launch(li, "input buffer id out of range"))?;
                            let raw = as_metal(b, ctx)?.raw();
                            reads.push(raw_id(raw));
                            raw
                        }
                        Bind::Output => {
                            let b = buffers.get(launch.output).ok_or_else(|| bad_launch(li, "output buffer id out of range"))?;
                            let raw = as_metal(b, ctx)?.raw();
                            writes.push(raw_id(raw));
                            raw
                        }
                        Bind::Scratch => {
                            let raw = scratch.ok_or_else(|| bad_launch(li, "scratch buffer missing"))?;
                            reads.push(raw_id(raw));
                            writes.push(raw_id(raw));
                            raw
                        }
                        // Only ever set to the same non-zero value: no ordering needed.
                        Bind::Status => status.ok_or_else(|| bad_launch(li, "status buffer missing"))?,
                    };
                    bufs.push(raw);
                }
                ready.push(Ready { stage, launch: li, pipeline, bufs, reads, writes });
            }
        }
        if times.is_some() || (self.profile.is_some() && !self.per_cb) {
            run_timed(ctx, launches, &ready, self.profile.as_mut(), times)?;
            return check_status(status);
        }
        if ready.is_empty() {
            return Ok(());
        }
        let t_prep = t0.elapsed().as_secs_f64() * 1e3;
        let mut hz = Hazards::default();
        let barriers: Vec<bool> = ready.iter().map(|r| hz.barrier_before(&r.reads, &r.writes)).collect();
        let serial = match self.encoder_override {
            Some(serial) => serial,
            None => prefer_serial(&ready, &barriers),
        };
        if self.event.is_none() && ready.len() >= SPLIT_MIN {
            self.event = ctx.device.newEvent().map(OwnedEvent);
        }
        let event = if ready.len() >= SPLIT_MIN { self.event.as_ref().map(|e| e.0.clone()) } else { None };
        let first = if event.is_some() { FIRST_CHUNK } else { ready.len() };
        autoreleasepool(|_| -> Result<()> {
            let kind = if serial { MTLDispatchType::Serial } else { MTLDispatchType::Concurrent };
            let mut cbs: Vec<Retained<CmdBuf>> = Vec::with_capacity(2);
            let mut t_enc = 0.0;
            let mut start = 0;
            while start < ready.len() {
                let end = if start == 0 { first } else { ready.len() };
                let encoded = (|| -> Result<Retained<CmdBuf>> {
                    let cb = ctx.queue.commandBufferWithUnretainedReferences().ok_or_else(|| Error::new("metal", "commandBuffer returned nil"))?;
                    if let (Some(ev), false) = (&event, cbs.is_empty()) {
                        cb.encodeWaitForEvent_value(ev, self.event_value);
                    }
                    let enc = cb
                        .computeCommandEncoderWithDispatchType(kind)
                        .ok_or_else(|| Error::new("metal", "computeCommandEncoder returned nil"))?;
                    for (r, &barrier) in ready[start..end].iter().zip(&barriers[start..end]) {
                        if barrier && !serial {
                            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                        }
                        encode(&enc, r);
                    }
                    enc.endEncoding();
                    if let (Some(ev), true) = (&event, end < ready.len()) {
                        self.event_value += 1;
                        cb.encodeSignalEvent_value(ev, self.event_value);
                    }
                    cb.commit();
                    Ok(cb)
                })();
                match encoded {
                    Ok(cb) => cbs.push(cb),
                    Err(e) => {
                        // Earlier command buffers reference the caller's buffers: drain them.
                        for cb in &cbs {
                            cb.waitUntilCompleted();
                        }
                        return Err(e);
                    }
                }
                if start == 0 {
                    t_enc = t0.elapsed().as_secs_f64() * 1e3 - t_prep;
                }
                start = end;
            }
            let t_enc_all = t0.elapsed().as_secs_f64() * 1e3 - t_prep;
            for cb in &cbs {
                cb.waitUntilCompleted();
            }
            for cb in &cbs {
                check(cb)?;
            }
            if let (Some(p), Some(f), Some(l)) = (self.profile.as_mut(), cbs.first(), cbs.last()) {
                p.add("command_buffer", (l.GPUEndTime() - f.GPUStartTime()) * 1e3);
                p.add("host_prepare", t_prep);
                p.add("host_encode_first", t_enc);
                p.add("host_encode", t_enc_all);
            }
            Ok(())
        })?;
        check_status(status)
    }
}

/// One command buffer per stage, waited on individually; GPU time per launch.
fn run_timed(
    ctx: &Ctx,
    launches: &[Launch],
    ready: &[Ready<'_>],
    profile: Option<&mut Profile>,
    times: Option<&mut Vec<f64>>,
) -> Result<()> {
    let mut per_launch = vec![0.0f64; launches.len()];
    for r in ready {
        let ms = autoreleasepool(|_| -> Result<f64> {
            let cb = ctx.queue.commandBuffer().ok_or_else(|| Error::new("metal", "commandBuffer returned nil"))?;
            let enc = cb.computeCommandEncoder().ok_or_else(|| Error::new("metal", "computeCommandEncoder returned nil"))?;
            encode(&enc, r);
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            check(&cb)?;
            Ok((cb.GPUEndTime() - cb.GPUStartTime()) * 1e3)
        })?;
        per_launch[r.launch] += ms;
    }
    if let Some(p) = profile {
        for (l, ms) in launches.iter().zip(&per_launch) {
            p.add(l.kernel.kind(), *ms);
        }
    }
    if let Some(t) = times {
        t.extend_from_slice(&per_launch);
    }
    Ok(())
}

fn set_status(b: &RawBuffer, v: u32) {
    // SAFETY: a 4-byte shared buffer owned by the backend; no GPU work is in flight (execute
    // runs under the backend lock and waits for completion).
    unsafe { b.contents().cast::<u32>().as_ptr().write_volatile(v) };
}

/// After the command buffer completed: an out-of-range index flagged by a gather kernel.
fn check_status(status: Option<&RawBuffer>) -> Result<()> {
    let Some(b) = status else { return Ok(()) };
    // SAFETY: as in `set_status`; the GPU work has completed.
    let v = unsafe { b.contents().cast::<u32>().as_ptr().read_volatile() };
    if v != 0 {
        set_status(b, 0);
        return Err(Error::new("index_out_of_range", "metal: gather index out of range (output unspecified)"));
    }
    Ok(())
}

fn bad_launch(launch: usize, msg: &str) -> Error {
    Error::new("bad_launch", format!("metal launch {launch}: {msg}"))
}
