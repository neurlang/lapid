//! Command-buffer replay for repeated launch sequences (decoder steps).
//!
//! A launch sequence's *signature* is everything recorded into the command buffer except
//! buffer identities: pipelines, dispatch sizes, params-ring offsets / lengths, scratch
//! ranges and barrier positions. Sizes that change every step (KV length, offsets) travel
//! in the params ring, which is rewritten before every submit, so a decode step usually has
//! the same signature as the previous one.
//!
//! The second time a signature is seen it is recorded into persistent command buffers, in
//! chunks of `CHUNK` dispatches; afterwards an execute with that signature re-records only
//! the chunks whose buffers (matched by `VkBuffer::id`, never by handle) changed — e.g.
//! the per-step output or token buffer — and submits all chunks at once. A few variants
//! per signature are kept because pooled temp buffers may rotate between steps.
//! `LAPID_VK_NO_REPLAY=1` disables replay.

use ash::vk;
use lapid_tensor::Result;

use crate::context::{Shared, vk_err};
use crate::execute::{Item, Rec, final_barrier};

/// Dispatches per recorded chunk (the unit of re-recording).
const CHUNK: usize = 16;
/// Recorded sequences kept (LRU).
const MAX_ENTRIES: usize = 8;
/// Variants (different buffers) per signature.
const MAX_VARIANTS: usize = 3;
/// Signatures remembered as "seen once".
const MAX_SEEN: usize = 64;

struct Entry {
    hash: u64,
    sig: Vec<u64>,
    /// buffer serials per item (inputs then output)
    ids: Vec<Vec<u64>>,
    base_ids: [u64; 3],
    cmds: Vec<vk::CommandBuffer>,
    last_used: u64,
}

pub(crate) struct Replay {
    pool: vk::CommandPool,
    pub enabled: bool,
    entries: Vec<Entry>,
    seen: Vec<u64>,
    tick: u64,
    /// (full hits, partial hits, chunks re-recorded, sequences recorded)
    pub stats: [u64; 4],
}

fn fnv(words: &[u64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for w in words {
        h ^= w;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn signature(items: &[Item]) -> Vec<u64> {
    let mut s = Vec::with_capacity(items.len() * 10);
    for it in items {
        use ash::vk::Handle;
        s.push(it.pipe.pipeline.as_raw());
        s.extend(it.plan.groups.iter().map(|&g| g as u64));
        s.push(it.params_off as u64);
        s.push(it.plan.params.len() as u64);
        let (so, sl) = it.scratch.unwrap_or((usize::MAX, 0));
        s.push(so as u64);
        s.push(sl as u64);
        s.push(it.barrier as u64 | (it.follow as u64) << 1 | (it.bufs.len() as u64) << 2);
    }
    s
}

impl Replay {
    pub fn new(shared: &Shared) -> Replay {
        let disabled = std::env::var("LAPID_VK_NO_REPLAY").is_ok_and(|v| !v.is_empty() && v != "0");
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(shared.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: valid device; the pool is destroyed in `destroy`.
        let pool = unsafe { shared.device.create_command_pool(&info, None) }.unwrap_or(vk::CommandPool::null());
        Replay {
            enabled: !disabled && pool != vk::CommandPool::null() && shared.push_desc.is_some(),
            pool,
            entries: Vec::new(),
            seen: Vec::new(),
            tick: 0,
            stats: [0; 4],
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        if self.pool != vk::CommandPool::null() {
            // SAFETY: device idle at teardown; freeing the pool frees its command buffers.
            unsafe { device.destroy_command_pool(self.pool, None) };
            self.pool = vk::CommandPool::null();
        }
        self.entries.clear();
    }

    /// Executable command buffers for `items` (all with push descriptors; params already
    /// written), or `None` when the sequence should be recorded the normal way. The GPU
    /// must be idle with respect to this backend's previous submissions (every execute waits).
    pub fn lookup(&mut self, shared: &Shared, rec: &Rec, items: &[Item]) -> Result<Option<Vec<vk::CommandBuffer>>> {
        let sig = signature(items);
        let hash = fnv(&sig);
        self.tick += 1;
        let nchunks = items.len().div_ceil(CHUNK);
        // best recorded variant: fewest chunks to re-record
        let mut best: Option<(usize, Vec<usize>)> = None;
        let mut variants = 0;
        for (ei, e) in self.entries.iter().enumerate() {
            if e.hash != hash || e.sig != sig {
                continue;
            }
            variants += 1;
            let dirty: Vec<usize> = if e.base_ids != rec.base_ids {
                (0..nchunks).collect()
            } else {
                (0..nchunks)
                    .filter(|&c| {
                        let r = c * CHUNK..((c + 1) * CHUNK).min(items.len());
                        items[r.clone()].iter().zip(&e.ids[r]).any(|(it, ids)| it.ids != *ids)
                    })
                    .collect()
            };
            if best.as_ref().is_none_or(|(_, d)| dirty.len() < d.len()) {
                best = Some((ei, dirty));
            }
        }
        match best {
            Some((ei, dirty)) if dirty.is_empty() || dirty.len() * 2 <= nchunks || variants >= MAX_VARIANTS => {
                self.stats[if dirty.is_empty() { 0 } else { 1 }] += 1;
                self.stats[2] += dirty.len() as u64;
                for &c in &dirty {
                    self.record_chunk(shared, rec, items, ei, c)?;
                }
                let e = &mut self.entries[ei];
                for &c in &dirty {
                    let r = c * CHUNK..((c + 1) * CHUNK).min(items.len());
                    for i in r {
                        e.ids[i].clone_from(&items[i].ids);
                    }
                }
                e.base_ids = rec.base_ids;
                e.last_used = self.tick;
                Ok(Some(e.cmds.clone()))
            }
            _ => {
                if variants == 0 && !self.seen.contains(&hash) {
                    if self.seen.len() >= MAX_SEEN {
                        self.seen.remove(0);
                    }
                    self.seen.push(hash);
                    return Ok(None);
                }
                let ei = self.new_entry(shared, hash, sig, items, rec.base_ids, nchunks)?;
                for c in 0..nchunks {
                    self.record_chunk(shared, rec, items, ei, c)?;
                }
                self.stats[3] += 1;
                Ok(Some(self.entries[ei].cmds.clone()))
            }
        }
    }

    fn new_entry(&mut self, shared: &Shared, hash: u64, sig: Vec<u64>, items: &[Item], base_ids: [u64; 3], nchunks: usize) -> Result<usize> {
        if self.entries.len() >= MAX_ENTRIES
            && let Some((i, _)) = self.entries.iter().enumerate().min_by_key(|(_, e)| e.last_used) {
                let e = self.entries.swap_remove(i);
                // SAFETY: the command buffers are not pending (previous executes completed).
                unsafe { shared.device.free_command_buffers(self.pool, &e.cmds) };
            }
        let ainfo = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(nchunks as u32);
        // SAFETY: valid pool.
        let cmds = unsafe { shared.device.allocate_command_buffers(&ainfo) }.map_err(|e| vk_err("vkAllocateCommandBuffers", e))?;
        self.entries.push(Entry {
            hash,
            sig,
            ids: items.iter().map(|it| it.ids.clone()).collect(),
            base_ids,
            cmds,
            last_used: self.tick,
        });
        Ok(self.entries.len() - 1)
    }

    fn record_chunk(&self, shared: &Shared, rec: &Rec, items: &[Item], ei: usize, c: usize) -> Result<()> {
        let dev = &shared.device;
        let cmd = self.entries[ei].cmds[c];
        let r = c * CHUNK..((c + 1) * CHUNK).min(items.len());
        let last = r.end == items.len();
        // SAFETY: the command buffer is not pending; its pool allows individual resets.
        unsafe {
            dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).map_err(|e| vk_err("vkResetCommandBuffer", e))?;
            dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).map_err(|e| vk_err("vkBeginCommandBuffer", e))?;
        }
        let mut bound = None;
        for it in &items[r] {
            rec.record(cmd, it, None, &mut bound)?;
        }
        if last {
            final_barrier(dev, cmd);
        }
        // SAFETY: recording state.
        unsafe { dev.end_command_buffer(cmd) }.map_err(|e| vk_err("vkEndCommandBuffer", e))
    }
}
