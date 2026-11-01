//! Recording and submitting launches: one command encoder per `execute`, one compute pass
//! (WebGPU makes every dispatch its own synchronization scope, so a launch sees all writes
//! of the previous one), params of all launches packed into one storage buffer.

use std::num::NonZeroU64;
use std::sync::Arc;

use lapid_compiler::Launch;
use lapid_tensor::{DeviceData, Error, Result};
use lapid_wgsl::Extra;

use crate::buffer::{GpuBuffer, as_gpu, read_back};
use crate::context::Ctx;
use crate::pipeline::{Pipe, Pipelines};

/// Persistent per-backend execution state.
#[derive(Default)]
pub(crate) struct Exec {
    params: Option<wgpu::Buffer>,
    scratch: Option<wgpu::Buffer>,
    status: Option<wgpu::Buffer>,
    /// wasm: the error flag of submitted gathers is checked with the next download.
    status_pending: bool,
    /// Storage binding limit below the adapter's (tests: forces split launches).
    pub binding_limit: Option<u64>,
}

struct Item {
    pipe: Arc<Pipe>,
    li: usize,
    groups: [u32; 3],
    params_off: u64,
    params_len: u64,
    extra: Option<Extra>,
    scratch: Option<(u64, u64)>,
    /// per input: bound byte range (`None` = whole buffer; split launches)
    ranges: Vec<Option<(u64, u64)>>,
}

fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

fn storage_buffer(ctx: &Ctx, size: u64, label: &str) -> Result<wgpu::Buffer> {
    if size > ctx.max_buffer {
        return Err(Error::new("too_large", format!("webgpu: {label} of {size} bytes exceeds maxBufferSize")));
    }
    let b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: align_up(size.max(16), 16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.take_error()?;
    Ok(b)
}

/// Reuse `slot` when it holds at least `size` bytes, else replace it (grown 1.5x).
fn ensure<'a>(ctx: &Ctx, slot: &'a mut Option<wgpu::Buffer>, size: u64, label: &str) -> Result<&'a wgpu::Buffer> {
    if slot.as_ref().is_none_or(|b| b.size() < size) {
        let grown = slot.as_ref().map_or(size, |b| size.max(b.size() + b.size() / 2));
        *slot = Some(storage_buffer(ctx, grown.min(ctx.max_binding.max(size)), label)?);
    }
    slot.as_ref().ok_or_else(|| Error::new("webgpu", format!("{label} missing")))
}

impl Exec {
    pub fn run(&mut self, ctx: &Ctx, pipelines: &Pipelines, launches: &[Launch], buffers: &[DeviceData]) -> Result<()> {
        let gbufs: Vec<&GpuBuffer> = buffers.iter().map(|d| as_gpu(d, ctx)).collect::<Result<_>>()?;
        let align = ctx.storage_align;

        // plan every launch; every stage becomes one dispatch
        let mut items: Vec<Item> = Vec::with_capacity(launches.len());
        let mut params: Vec<u8> = Vec::new();
        let mut scratch_bytes = 0u64;
        let limit = self.binding_limit.unwrap_or(ctx.max_binding).min(ctx.max_binding);
        for (li, l) in launches.iter().enumerate() {
            let ectx = |e: Error| Error::new(e.code, format!("launch {li} ({}): {}", l.kernel.kind(), e.message));
            let mut sizes = Vec::with_capacity(l.inputs.len());
            for &id in l.inputs.iter().chain(std::iter::once(&l.output)) {
                let Some(b) = gbufs.get(id) else {
                    return Err(Error::new("bad_launch", format!("launch {li}: buffer id {id} out of range")));
                };
                sizes.push(b.size);
            }
            let out_size = sizes.pop().unwrap_or(0);
            if out_size > limit {
                return Err(ectx(Error::new("too_large", format!("output of {out_size} bytes exceeds maxStorageBufferBindingSize ({limit})"))));
            }
            // launches binding a buffer above the limit run as several sub-range launches
            let parts: Vec<(lapid_wgsl::Plan, lapid_wgsl::Ranges)> = match lapid_wgsl::plan_split(&l.kernel, &sizes, limit, align, &ctx.caps).map_err(&ectx)? {
                Some(subs) => subs.into_iter().map(|s| (s.plan, s.ranges)).collect(),
                None => vec![(lapid_wgsl::plan(&l.kernel, &ctx.caps).map_err(&ectx)?, vec![None; l.inputs.len()])],
            };
            for (plan, ranges) in parts {
                if plan.is_empty() {
                    continue;
                }
                if l.inputs.len() != plan.key.num_inputs() {
                    return Err(Error::new(
                        "bad_launch",
                        format!("launch {li} ({}): {} inputs, kernel expects {}", l.kernel.kind(), l.inputs.len(), plan.key.num_inputs()),
                    ));
                }
                let scratch = (plan.scratch > 0).then(|| {
                    let off = align_up(scratch_bytes, align);
                    let n = align_up(plan.scratch as u64, 4);
                    scratch_bytes = off + n;
                    (off, n)
                });
                let mut stage = Some(plan);
                while let Some(p) = stage {
                    let pipe = pipelines.get(ctx, &p.key)?;
                    let off = align_up(params.len() as u64, align);
                    params.resize(off as usize, 0);
                    if p.params.is_empty() {
                        params.extend_from_slice(&[0u8; 4]);
                    } else {
                        params.extend(p.params.iter().flat_map(|w| w.to_le_bytes()));
                    }
                    let params_len = params.len() as u64 - off;
                    let ranges = ranges.clone();
                    items.push(Item { pipe, li, groups: p.groups, params_off: off, params_len, extra: p.key.extra(), scratch, ranges });
                    stage = p.next.map(|n| *n);
                }
            }
        }
        if items.is_empty() {
            return Ok(());
        }

        params.resize(align_up(params.len() as u64, 16) as usize, 0);
        let params_buf = ensure(ctx, &mut self.params, params.len() as u64, "lapid params")?.clone();
        ctx.queue.write_buffer(&params_buf, 0, &params);
        let scratch_buf = if scratch_bytes > 0 { Some(ensure(ctx, &mut self.scratch, scratch_bytes, "lapid scratch")?.clone()) } else { None };
        let uses_status = items.iter().any(|it| matches!(it.extra, Some(Extra::Status | Extra::ScratchStatus)));
        let status_buf = if uses_status {
            if self.status.is_none() {
                self.status = Some(storage_buffer(ctx, 16, "lapid status")?);
            }
            self.status.clone()
        } else {
            None
        };

        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("lapid") });
        let mut pass: Option<wgpu::ComputePass<'static>> = None;
        let mut temps: Vec<wgpu::Buffer> = Vec::new();
        for it in &items {
            let l = &launches[it.li];
            let out = gbufs[l.output];
            // WebGPU forbids a buffer being read-only and writable in one dispatch: copy
            // inputs aliasing the output first (outside the compute pass)
            let mut input_bufs: Vec<(&wgpu::Buffer, Option<usize>)> = Vec::with_capacity(l.inputs.len());
            for &id in &l.inputs {
                let b = gbufs[id];
                if b.id == out.id {
                    pass = None;
                    let t = storage_buffer(ctx, b.size, "lapid alias copy")?;
                    enc.copy_buffer_to_buffer(&b.buffer, 0, &t, 0, b.size);
                    temps.push(t);
                    input_bufs.push((&b.buffer, Some(temps.len() - 1)));
                } else {
                    input_bufs.push((&b.buffer, None));
                }
            }
            let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(l.inputs.len() + 3);
            entries.push(wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &params_buf,
                    offset: it.params_off,
                    size: NonZeroU64::new(it.params_len),
                }),
            });
            for (i, &(b, t)) in input_bufs.iter().enumerate() {
                let buf = match t {
                    Some(k) => &temps[k],
                    None => b,
                };
                let resource = match it.ranges.get(i).copied().flatten() {
                    Some((offset, size)) => wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: buf, offset, size: NonZeroU64::new(size) }),
                    None => buf.as_entire_binding(),
                };
                entries.push(wgpu::BindGroupEntry { binding: 1 + i as u32, resource });
            }
            let nb = entries.len() as u32;
            entries.push(wgpu::BindGroupEntry { binding: nb, resource: out.buffer.as_entire_binding() });
            match it.extra {
                Some(Extra::Scratch) => {
                    let (off, n) = it.scratch.ok_or_else(|| Error::new("webgpu", "scratch range missing"))?;
                    let sb = scratch_buf.as_ref().ok_or_else(|| Error::new("webgpu", "scratch buffer missing"))?;
                    entries.push(wgpu::BindGroupEntry {
                        binding: nb + 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: sb, offset: off, size: NonZeroU64::new(n) }),
                    });
                }
                Some(Extra::Status) => {
                    let sb = status_buf.as_ref().ok_or_else(|| Error::new("webgpu", "status buffer missing"))?;
                    entries.push(wgpu::BindGroupEntry { binding: nb + 1, resource: sb.as_entire_binding() });
                }
                Some(Extra::ScratchStatus) => {
                    let (off, n) = it.scratch.ok_or_else(|| Error::new("webgpu", "scratch range missing"))?;
                    let sb = scratch_buf.as_ref().ok_or_else(|| Error::new("webgpu", "scratch buffer missing"))?;
                    entries.push(wgpu::BindGroupEntry {
                        binding: nb + 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: sb, offset: off, size: NonZeroU64::new(n) }),
                    });
                    let st = status_buf.as_ref().ok_or_else(|| Error::new("webgpu", "status buffer missing"))?;
                    entries.push(wgpu::BindGroupEntry { binding: nb + 2, resource: st.as_entire_binding() });
                }
                None => {}
            }
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &it.pipe.layout, entries: &entries });
            let p = pass.get_or_insert_with(|| {
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }).forget_lifetime()
            });
            p.set_pipeline(&it.pipe.pipeline);
            p.set_bind_group(0, &bg, &[]);
            let [x, y, z] = it.groups;
            p.dispatch_workgroups(x, y, z);
        }
        drop(pass);
        ctx.queue.submit([enc.finish()]);
        drop(temps);

        #[cfg(not(target_arch = "wasm32"))]
        {
            ctx.wait_idle()?;
            ctx.take_error()?;
            if uses_status {
                self.check_status(ctx)?;
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            ctx.take_error()?;
            self.status_pending |= uses_status;
        }
        Ok(())
    }

    /// Read and clear the device error flag (waits for submitted work).
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn check_status(&mut self, ctx: &Ctx) -> Result<()> {
        let Some(sb) = self.status.as_ref() else { return Ok(()) };
        let words = read_back(ctx, &[(sb, 16)])?;
        self.status_result(ctx, &words[0])
    }

    fn status_result(&mut self, ctx: &Ctx, status: &[u8]) -> Result<()> {
        self.status_pending = false;
        if status.iter().take(4).any(|&b| b != 0) {
            if let Some(sb) = self.status.as_ref() {
                ctx.queue.write_buffer(sb, 0, &[0u8; 16]);
            }
            return Err(Error::new("index_out_of_range", "gather index outside the table / axis"));
        }
        Ok(())
    }

    /// Copy `b` to the host (waits for all submitted work).
    pub fn download(&mut self, ctx: &Ctx, b: &GpuBuffer) -> Result<Vec<u8>> {
        let n = b.len_bytes();
        match (self.status_pending, self.status.as_ref()) {
            (true, Some(sb)) => {
                let mut r = read_back(ctx, &[(&b.buffer, n as u64), (sb, 16)])?;
                let status = r.pop().unwrap_or_default();
                self.status_result(ctx, &status)?;
                let mut out = r.pop().unwrap_or_default();
                out.truncate(n);
                Ok(out)
            }
            _ => {
                let mut out = read_back(ctx, &[(&b.buffer, n as u64)])?.pop().unwrap_or_default();
                out.truncate(n);
                Ok(out)
            }
        }
    }
}
