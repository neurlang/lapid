//! Command recording and submission: one command buffer per `execute`, params ring,
//! descriptor push/allocation, hazard-tracked barriers, fence wait. Also staging copies.

use std::collections::HashSet;
use std::sync::Arc;

use ash::vk;
use lapid_compiler::Launch;
use lapid_tensor::{DeviceData, Error, Result};

use crate::buffer::{VkBuffer, as_vk};
use crate::codegen::{self, Plan};
use crate::context::{Shared, vk_err};
use crate::memory::MemUsage;
use crate::pipeline::{Pipeline, Pipelines};
use crate::profile::Profiler;
use crate::replay::Replay;

/// Per-backend execution state (guarded by a mutex in the backend).
pub(crate) struct Exec {
    pool: vk::CommandPool,
    /// Command buffers of one execute (recorded and submitted in chunks).
    cmds: Vec<vk::CommandBuffer>,
    /// Index of the command buffer being recorded.
    cur: usize,
    fence: vk::Fence,
    params: Option<VkBuffer>,
    /// Device scratch for multi-stage kernels (split attention / split-k partials).
    scratch: Option<VkBuffer>,
    /// Device error flag (one word, host visible): kernels set it on an out-of-range index.
    status: Option<VkBuffer>,
    desc_pool: vk::DescriptorPool,
    desc_sets: u32,
    /// GPU timestamps (`LAPID_PROFILE=1` or `time_launches`).
    pub prof: Option<Profiler>,
    /// Accumulate per-kernel totals for every execute (`LAPID_PROFILE=1`).
    pub profiling: bool,
    /// Recorded command buffers of repeated launch sequences.
    pub replay: Replay,
    /// Storage binding limit below the device's (tests: forces split launches).
    pub binding_limit: Option<u64>,
}

/// What to do with per-launch GPU timings of one `run`.
pub(crate) enum Timing<'a> {
    None,
    /// Store the GPU ms of every (non-empty) launch, in order.
    Collect(&'a mut Vec<f64>),
}

/// Command buffers per execute; launches are submitted in chunks of `CHUNK` (the first
/// after `FIRST_CHUNK`) to overlap recording with execution.
const MAX_CHUNKS: usize = 8;
const FIRST_CHUNK: usize = 16;
const CHUNK: usize = 48;

/// How long `submit_wait` polls the fence before falling back to a blocking wait.
const SPIN_WAIT: std::time::Duration = std::time::Duration::from_millis(20);

/// Descriptor budget per allocated set (kernels bind params + inputs + output).
const MAX_BINDINGS_PER_SET: u32 = 64;

impl Exec {
    pub fn new(shared: &Shared) -> Result<Exec> {
        let dev = &shared.device;
        // SAFETY: valid device; objects destroyed in `destroy`.
        unsafe {
            let pinfo = vk::CommandPoolCreateInfo::default()
                .queue_family_index(shared.queue_family)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            let pool = dev.create_command_pool(&pinfo, None).map_err(|e| vk_err("vkCreateCommandPool", e))?;
            let ainfo = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(MAX_CHUNKS as u32);
            let cmds = match dev.allocate_command_buffers(&ainfo) {
                Ok(v) => v,
                Err(e) => {
                    dev.destroy_command_pool(pool, None);
                    return Err(vk_err("vkAllocateCommandBuffers", e));
                }
            };
            let fence = match dev.create_fence(&vk::FenceCreateInfo::default(), None) {
                Ok(f) => f,
                Err(e) => {
                    dev.destroy_command_pool(pool, None);
                    return Err(vk_err("vkCreateFence", e));
                }
            };
            let profiling = std::env::var("LAPID_PROFILE").map(|v| !v.is_empty() && v != "0").unwrap_or(false);
            Ok(Exec {
                pool,
                cmds,
                cur: 0,
                fence,
                params: None,
                scratch: None,
                status: None,
                desc_pool: vk::DescriptorPool::null(),
                desc_sets: 0,
                prof: Profiler::new(shared),
                profiling,
                replay: Replay::new(shared),
                binding_limit: None,
            })
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        // SAFETY: called at teardown with the device idle.
        unsafe {
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.pool, None);
            if self.desc_pool != vk::DescriptorPool::null() {
                device.destroy_descriptor_pool(self.desc_pool, None);
            }
        }
        if let Some(p) = self.prof.as_mut() {
            p.destroy(device);
        }
        self.replay.destroy(device);
        self.params = None;
        self.scratch = None;
        self.status = None;
    }

    fn begin(&mut self, shared: &Shared) -> Result<()> {
        let dev = &shared.device;
        // SAFETY: the previous submission (if any) completed before we returned from it.
        unsafe {
            dev.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()).map_err(|e| vk_err("vkResetCommandPool", e))?;
            let info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.cur = 0;
            dev.begin_command_buffer(self.cmds[0], &info).map_err(|e| vk_err("vkBeginCommandBuffer", e))
        }
    }

    /// Submit the commands recorded so far (no fence) so the GPU starts while the rest of the
    /// launches are recorded, and continue in the next command buffer. Barriers recorded
    /// later still order against this work (their first scope covers all earlier
    /// submissions on the queue).
    fn flush(&mut self, shared: &Shared) -> Result<bool> {
        if self.cur + 1 >= self.cmds.len() {
            return Ok(false);
        }
        let dev = &shared.device;
        // SAFETY: `cmds[cur]` is recording; the next buffer is unused since the pool reset.
        unsafe {
            dev.end_command_buffer(self.cmds[self.cur]).map_err(|e| vk_err("vkEndCommandBuffer", e))?;
            let cmds = [self.cmds[self.cur]];
            let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
            {
                let _q = shared.lock_queue();
                dev.queue_submit(shared.queue, &submit, vk::Fence::null()).map_err(|e| vk_err("vkQueueSubmit", e))?;
            }
            self.cur += 1;
            let info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            dev.begin_command_buffer(self.cmds[self.cur], &info).map_err(|e| vk_err("vkBeginCommandBuffer", e))?;
        }
        Ok(true)
    }

    fn submit_wait(&mut self, shared: &Shared) -> Result<()> {
        // SAFETY: the command buffer is in the recording state.
        unsafe { shared.device.end_command_buffer(self.cmds[self.cur]) }.map_err(|e| vk_err("vkEndCommandBuffer", e))?;
        let cmd = [self.cmds[self.cur]];
        self.submit_list_wait(shared, &cmd)
    }

    /// Submit executable command buffers (in order, one batch) and wait for them.
    fn submit_list_wait(&mut self, shared: &Shared, cmds: &[vk::CommandBuffer]) -> Result<()> {
        let dev = &shared.device;
        // SAFETY: the command buffers are executable; queue access is serialized by
        // `queue_lock`, the fence by the backend's exec mutex.
        unsafe {
            let submit = [vk::SubmitInfo::default().command_buffers(cmds)];
            dev.reset_fences(&[self.fence]).map_err(|e| vk_err("vkResetFences", e))?;
            {
                let _q = shared.lock_queue();
                dev.queue_submit(shared.queue, &submit, self.fence).map_err(|e| vk_err("vkQueueSubmit", e))?;
            }
            // Poll briefly before blocking: a blocking fence wait adds the driver's wake-up
            // latency (~0.1-0.3 ms) to every execute, which dominates short decode steps.
            let t0 = std::time::Instant::now();
            while t0.elapsed() < SPIN_WAIT {
                match dev.get_fence_status(self.fence) {
                    Ok(true) => return Ok(()),
                    Ok(false) => std::hint::spin_loop(),
                    Err(e) => return Err(vk_err("vkGetFenceStatus", e)),
                }
            }
            dev.wait_for_fences(&[self.fence], true, u64::MAX).map_err(|e| vk_err("vkWaitForFences", e))
        }
    }

    fn ensure_params(&mut self, shared: &Arc<Shared>, bytes: usize) -> Result<()> {
        let have = self.params.as_ref().map(|b| b.size as usize).unwrap_or(0);
        if have >= bytes {
            return Ok(());
        }
        let size = bytes.max(64 << 10).next_power_of_two();
        self.params = None;
        self.params = Some(VkBuffer::create(shared, size, MemUsage::Upload)?);
        Ok(())
    }

    fn ensure_scratch(&mut self, shared: &Arc<Shared>, bytes: usize) -> Result<()> {
        let have = self.scratch.as_ref().map(|b| b.size as usize).unwrap_or(0);
        if have >= bytes {
            return Ok(());
        }
        let size = bytes.max(1 << 20).next_power_of_two();
        self.scratch = None;
        self.scratch = Some(VkBuffer::create(shared, size, MemUsage::Device)?);
        Ok(())
    }

    fn ensure_status(&mut self, shared: &Arc<Shared>) -> Result<vk::Buffer> {
        if self.status.is_none() {
            let b = VkBuffer::create(shared, 16, MemUsage::Readback)?;
            b.write_mapped(0, &[0u8; 16])?;
            self.status = Some(b);
        }
        self.status.as_ref().map(|b| b.buffer).ok_or_else(|| Error::new("vulkan", "status buffer missing"))
    }

    /// Read and clear the error flag after the GPU finished.
    fn check_status(&mut self) -> Result<()> {
        let Some(b) = self.status.as_ref() else { return Ok(()) };
        let mut w = [0u8; 4];
        b.read_mapped(0, &mut w)?;
        if w != [0u8; 4] {
            b.write_mapped(0, &[0u8; 4])?;
            return Err(Error::new("index_out_of_range", "gather / matmul_id index outside the table / axis / expert range"));
        }
        Ok(())
    }

    fn ensure_desc_pool(&mut self, shared: &Shared, sets: u32) -> Result<()> {
        let dev = &shared.device;
        if self.desc_sets >= sets && self.desc_pool != vk::DescriptorPool::null() {
            // SAFETY: no descriptor set from the pool is in use (previous work completed).
            unsafe { dev.reset_descriptor_pool(self.desc_pool, vk::DescriptorPoolResetFlags::empty()) }
                .map_err(|e| vk_err("vkResetDescriptorPool", e))?;
            return Ok(());
        }
        let cap = sets.max(256).next_power_of_two();
        // SAFETY: old pool is idle; new pool sized for `cap` sets.
        unsafe {
            if self.desc_pool != vk::DescriptorPool::null() {
                dev.destroy_descriptor_pool(self.desc_pool, None);
                self.desc_pool = vk::DescriptorPool::null();
                self.desc_sets = 0;
            }
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(cap * MAX_BINDINGS_PER_SET)];
            let info = vk::DescriptorPoolCreateInfo::default().max_sets(cap).pool_sizes(&sizes);
            self.desc_pool = dev.create_descriptor_pool(&info, None).map_err(|e| vk_err("vkCreateDescriptorPool", e))?;
        }
        self.desc_sets = cap;
        Ok(())
    }

    /// Run `launches` and wait for completion.
    pub fn run(&mut self, shared: &Arc<Shared>, pipelines: &Pipelines, launches: &[Launch], buffers: &[DeviceData], mut timing: Timing) -> Result<()> {
        let host_t0 = std::time::Instant::now();
        let caps = &shared.caps;
        let vbufs: Vec<&VkBuffer> = buffers.iter().map(|d| as_vk(d, shared)).collect::<Result<_>>()?;
        let limit = self.binding_limit.unwrap_or(caps.max_storage_range).min(caps.max_storage_range);

        // plan + pipelines; every stage of a launch becomes one dispatch item
        let mut items: Vec<Item> = Vec::with_capacity(launches.len());
        let align = caps.storage_align as usize;
        let mut params_bytes = 0usize;
        let mut scratch_bytes = 0usize;
        for (li, l) in launches.iter().enumerate() {
            for &id in l.inputs.iter().chain(std::iter::once(&l.output)) {
                if id >= vbufs.len() {
                    return Err(Error::new("bad_launch", format!("launch {li}: buffer id {id} out of range")));
                }
            }
            let ctx = |e: Error| Error::new(e.code, format!("launch {li} ({}): {}", l.kernel.kind(), e.message));
            if vbufs[l.output].size > limit {
                return Err(ctx(Error::new("too_large", format!("output of {} bytes exceeds maxStorageBufferRange", vbufs[l.output].size))));
            }
            let sizes: Vec<u64> = l.inputs.iter().map(|&i| vbufs[i].size).collect();
            // launches binding a buffer above the limit run as several sub-range launches
            let parts: Vec<(Plan, lapid_wgsl::Ranges)> = match codegen::plan_split(&l.kernel, &sizes, limit, caps.storage_align, &caps.wgsl()).map_err(&ctx)? {
                Some(subs) => subs.into_iter().map(|s| (s.plan, s.ranges)).collect(),
                None => vec![(codegen::plan(&l.kernel, caps).map_err(&ctx)?, vec![None; l.inputs.len()])],
            };
            for (mut plan, ranges) in parts {
                if plan.is_empty() {
                    continue;
                }
                if l.inputs.len() != plan.key.num_inputs() {
                    return Err(Error::new(
                        "bad_launch",
                        format!("launch {li} ({}): {} inputs, kernel expects {}", l.kernel.kind(), l.inputs.len(), plan.key.num_inputs()),
                    ));
                }
                let scratch = if plan.scratch > 0 {
                    let off = scratch_bytes.div_ceil(align) * align;
                    scratch_bytes = off + plan.scratch;
                    Some((off, plan.scratch))
                } else {
                    None
                };
                let mut follow = false;
                loop {
                    let next = plan.next.take();
                    let pipe = pipelines.get(shared, &plan.key)?;
                    let off = params_bytes.div_ceil(align) * align;
                    params_bytes = off + plan.params.len().max(1) * 4;
                    let mut bufs: Vec<vk::Buffer> = l.inputs.iter().map(|&i| vbufs[i].buffer).collect();
                    bufs.push(vbufs[l.output].buffer);
                    let mut ids: Vec<u64> = l.inputs.iter().map(|&i| vbufs[i].id).collect();
                    ids.push(vbufs[l.output].id);
                    let ranges = ranges.clone();
                    items.push(Item { plan, pipe, li, params_off: off, scratch, follow, barrier: false, bufs, ids, ranges });
                    match next {
                        Some(n) => {
                            plan = *n;
                            follow = true;
                        }
                        None => break,
                    }
                }
            }
        }
        if items.is_empty() {
            return Ok(());
        }
        // barriers: a compute->compute barrier in front of a launch that reads or writes a
        // buffer written since the last barrier, or writes one read since then
        {
            let mut written: HashSet<vk::Buffer> = HashSet::new();
            let mut read: HashSet<vk::Buffer> = HashSet::new();
            for it in items.iter_mut() {
                let (ins, out) = it.bufs.split_at(it.bufs.len() - 1);
                let out = out[0];
                if it.follow || ins.iter().any(|b| written.contains(b)) || written.contains(&out) || read.contains(&out) {
                    it.barrier = true;
                    written.clear();
                    read.clear();
                }
                read.extend(ins.iter().copied());
                written.insert(out);
            }
        }

        // params
        self.ensure_params(shared, params_bytes)?;
        let params = self.params.as_ref().ok_or_else(|| Error::new("vulkan", "params buffer missing"))?;
        for it in &items {
            if it.plan.params.is_empty() {
                params.write_mapped(it.params_off, &[0u8; 4])?;
            } else {
                params.write_mapped(it.params_off, bytemuck::cast_slice::<u32, u8>(it.plan.params.as_slice()))?;
            }
        }
        let params_handle = params.buffer;
        let params_id = params.id;
        let scratch_handle = if scratch_bytes > 0 {
            self.ensure_scratch(shared, scratch_bytes)?;
            self.scratch.as_ref().map(|b| b.buffer).unwrap_or_default()
        } else {
            vk::Buffer::null()
        };
        let uses_status = items.iter().any(|it| matches!(it.plan.key.extra(), Some(codegen::Extra::Status | codegen::Extra::ScratchStatus)));
        let status_handle = if uses_status { self.ensure_status(shared)? } else { vk::Buffer::null() };
        let base_ids = [
            params_id,
            if scratch_bytes > 0 { self.scratch.as_ref().map(|b| b.id).unwrap_or(0) } else { 0 },
            if uses_status { self.status.as_ref().map(|b| b.id).unwrap_or(0) } else { 0 },
        ];
        let rec = Rec { base_ids, dev: &shared.device, push: shared.push_desc.as_ref(), params: params_handle, scratch: scratch_handle, status: status_handle };

        let timed = self.prof.is_some() && (self.profiling || matches!(timing, Timing::Collect(_)));
        if !timed && self.replay.enabled && items.iter().all(|it| it.pipe.push)
            && let Some(cmds) = self.replay.lookup(shared, &rec, &items)? {
                self.submit_list_wait(shared, &cmds)?;
                if uses_status {
                    self.check_status()?;
                }
                return Ok(());
            }

        // descriptor sets (non-push path)
        let need_sets = items.iter().filter(|it| !it.pipe.push).count() as u32;
        let mut sets: Vec<vk::DescriptorSet> = Vec::new();
        if need_sets > 0 {
            self.ensure_desc_pool(shared, need_sets)?;
            let layouts: Vec<vk::DescriptorSetLayout> = items.iter().filter(|it| !it.pipe.push).map(|it| it.pipe.set_layout).collect();
            let ainfo = vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.desc_pool).set_layouts(&layouts);
            // SAFETY: pool has room for `need_sets` sets.
            sets = unsafe { shared.device.allocate_descriptor_sets(&ainfo) }.map_err(|e| vk_err("vkAllocateDescriptorSets", e))?;
        }

        self.begin(shared)?;
        let dev = &shared.device;
        let mut cmd = self.cmds[self.cur];
        // first chunk small so the GPU starts early, then larger ones
        let mut next_flush = FIRST_CHUNK;
        let mut set_iter = sets.into_iter();
        let mut bound: Option<vk::Pipeline> = None;
        let nq = items.len() as u32 + 1;
        if timed {
            let p = self.prof.as_mut().ok_or_else(|| Error::new("vulkan", "profiler missing"))?;
            p.begin(shared, cmd, nq)?;
            p.stamp(dev, cmd, 0);
        }
        for (ii, it) in items.iter().enumerate() {
            let set = if it.pipe.push { None } else { Some(set_iter.next().ok_or_else(|| Error::new("vulkan", "descriptor set underflow"))?) };
            rec.record(cmd, it, set, &mut bound)?;
            if timed
                && let Some(p) = self.prof.as_ref() {
                    p.stamp(dev, cmd, ii as u32 + 1);
                }
            if ii + 1 == next_flush && ii + 1 < items.len() {
                if self.flush(shared)? {
                    // bindings are per command buffer
                    cmd = self.cmds[self.cur];
                    bound = None;
                }
                next_flush += CHUNK;
            }
        }
        final_barrier(dev, cmd);
        let host_ms = host_t0.elapsed().as_secs_f64() * 1e3;
        self.submit_wait(shared)?;
        if uses_status {
            self.check_status()?;
        }
        if timed {
            let profiling = self.profiling;
            let p = self.prof.as_mut().ok_or_else(|| Error::new("vulkan", "profiler missing"))?;
            let ms = p.read(shared, nq)?;
            if profiling {
                for (it, &t) in items.iter().zip(&ms) {
                    p.add(&it.plan.key.name(), t);
                }
                p.host_ms += host_ms;
                p.executes += 1;
            }
            if let Timing::Collect(out) = &mut timing {
                // one entry per launch: sum of its stages
                let mut last = usize::MAX;
                for (it, &t) in items.iter().zip(&ms) {
                    if it.li == last {
                        if let Some(v) = out.last_mut() {
                            *v += t;
                        }
                    } else {
                        out.push(t);
                        last = it.li;
                    }
                }
            }
        }
        Ok(())
    }

    /// Copy `src` → `dst` (whole `bytes`) and wait. Used for staging.
    pub fn copy(&mut self, shared: &Shared, src: &VkBuffer, dst: &VkBuffer, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.begin(shared)?;
        let dev = &shared.device;
        // SAFETY: recording state; both buffers are alive until we return.
        unsafe {
            let pre = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)];
            dev.cmd_pipeline_barrier(
                self.cmds[0],
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &pre,
                &[],
                &[],
            );
            let region = [vk::BufferCopy { src_offset: 0, dst_offset: 0, size: bytes }];
            dev.cmd_copy_buffer(self.cmds[0], src.buffer, dst.buffer, &region);
            let post = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            dev.cmd_pipeline_barrier(
                self.cmds[0],
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &post,
                &[],
                &[],
            );
        }
        self.submit_wait(shared)
    }
}

/// One dispatch: a planned launch stage with its pipeline and buffers.
pub(crate) struct Item {
    pub plan: Plan,
    pub pipe: Arc<Pipeline>,
    /// index into `launches`
    pub li: usize,
    pub params_off: usize,
    /// (offset, bytes) in the scratch buffer, shared by the stages of a launch
    pub scratch: Option<(usize, usize)>,
    /// later stage of a launch (always preceded by a barrier)
    pub follow: bool,
    /// record a compute->compute barrier in front of this dispatch
    pub barrier: bool,
    /// input buffers then the output buffer
    pub bufs: Vec<vk::Buffer>,
    /// their `VkBuffer::id` serials
    pub ids: Vec<u64>,
    /// per input: bound byte range (`None` = whole buffer; split launches)
    pub ranges: Vec<Option<(u64, u64)>>,
}

/// What recording a dispatch needs besides the item.
pub(crate) struct Rec<'a> {
    /// serials of the params / scratch / status buffers (0 = none)
    pub base_ids: [u64; 3],
    pub dev: &'a ash::Device,
    pub push: Option<&'a ash::khr::push_descriptor::Device>,
    pub params: vk::Buffer,
    pub scratch: vk::Buffer,
    pub status: vk::Buffer,
}

impl Rec<'_> {
    /// Record barrier (if any), descriptors and the dispatch of `it` into `cmd`. `set` is
    /// the descriptor set to use when the pipeline does not push descriptors.
    pub fn record(&self, cmd: vk::CommandBuffer, it: &Item, set: Option<vk::DescriptorSet>, bound: &mut Option<vk::Pipeline>) -> Result<()> {
        let dev = self.dev;
        if it.barrier {
            compute_barrier(dev, cmd);
        }
        let mut infos: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(it.bufs.len() + 2);
        infos.push(
            vk::DescriptorBufferInfo::default()
                .buffer(self.params)
                .offset(it.params_off as u64)
                .range((it.plan.params.len().max(1) * 4) as u64),
        );
        for (i, &b) in it.bufs.iter().enumerate() {
            let (off, len) = it.ranges.get(i).copied().flatten().unwrap_or((0, vk::WHOLE_SIZE));
            infos.push(vk::DescriptorBufferInfo::default().buffer(b).offset(off).range(len));
        }
        match it.plan.key.extra() {
            Some(codegen::Extra::Scratch) => {
                let (off, bytes) = it.scratch.ok_or_else(|| Error::new("vulkan", "scratch range missing"))?;
                infos.push(vk::DescriptorBufferInfo::default().buffer(self.scratch).offset(off as u64).range(bytes as u64));
            }
            Some(codegen::Extra::Status) => {
                infos.push(vk::DescriptorBufferInfo::default().buffer(self.status).offset(0).range(vk::WHOLE_SIZE));
            }
            Some(codegen::Extra::ScratchStatus) => {
                let (off, bytes) = it.scratch.ok_or_else(|| Error::new("vulkan", "scratch range missing"))?;
                infos.push(vk::DescriptorBufferInfo::default().buffer(self.scratch).offset(off as u64).range(bytes as u64));
                infos.push(vk::DescriptorBufferInfo::default().buffer(self.status).offset(0).range(vk::WHOLE_SIZE));
            }
            None => {}
        }
        let writes: Vec<vk::WriteDescriptorSet> = (0..infos.len())
            .map(|i| {
                let w = vk::WriteDescriptorSet::default()
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&infos[i]));
                match set {
                    Some(s) => w.dst_set(s),
                    None => w,
                }
            })
            .collect();
        // SAFETY: recording into a command buffer we own; all handles stay alive until the
        // submission completes (buffers are borrowed from the caller for the execute).
        unsafe {
            if *bound != Some(it.pipe.pipeline) {
                dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, it.pipe.pipeline);
                *bound = Some(it.pipe.pipeline);
            }
            match set {
                None => {
                    let pd = self.push.ok_or_else(|| Error::new("vulkan", "push descriptors unavailable"))?;
                    pd.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::COMPUTE, it.pipe.layout, 0, &writes);
                }
                Some(s) => {
                    dev.update_descriptor_sets(&writes, &[]);
                    dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, it.pipe.layout, 0, &[s], &[]);
                }
            }
            let [x, y, z] = it.plan.groups;
            dev.cmd_dispatch(cmd, x, y, z);
        }
        Ok(())
    }
}

/// Make the results visible to the host (mapped downloads) and to later transfers.
pub(crate) fn final_barrier(dev: &ash::Device, cmd: vk::CommandBuffer) {
    let mb = [vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::SHADER_READ)];
    // SAFETY: recording state.
    unsafe {
        dev.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &mb,
            &[],
            &[],
        );
    }
}

fn compute_barrier(dev: &ash::Device, cmd: vk::CommandBuffer) {
    let mb = [vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
    // SAFETY: recording state.
    unsafe {
        dev.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &mb,
            &[],
            &[],
        );
    }
}
