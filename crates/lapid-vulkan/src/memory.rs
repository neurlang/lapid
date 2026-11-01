//! Device memory sub-allocator: large blocks per memory type, first-fit free lists with
//! coalescing, persistent mapping for host-visible coherent memory.

use std::ptr::NonNull;

use ash::vk;
use lapid_tensor::{Error, Result};

use crate::context::vk_err;

const MAX_BLOCK: u64 = 256 << 20;
const MIN_BLOCK: u64 = 16 << 20;

/// Intended use of an allocation; drives memory type preference.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum MemUsage {
    /// Tensor storage: device-local, ideally also host-visible (unified memory / ReBAR).
    Device,
    /// Host-visible coherent memory written by the CPU, read by the GPU (params, staging up).
    Upload,
    /// Host-visible coherent memory read back by the CPU (staging down). Prefers cached.
    Readback,
}

/// Raw mapped pointer that may cross threads (access is externally synchronized).
#[derive(Copy, Clone, Debug)]
pub(crate) struct MappedPtr(pub NonNull<u8>);
// SAFETY: the pointer refers to persistently mapped device memory owned by the allocator;
// all accesses go through `VkBuffer` methods that respect the execution lock.
unsafe impl Send for MappedPtr {}
// SAFETY: see above.
unsafe impl Sync for MappedPtr {}

#[derive(Debug)]
pub(crate) struct Allocation {
    pub memory: vk::DeviceMemory,
    pub offset: u64,
    pub size: u64,
    /// Pointer to the first byte of this allocation if host-visible + coherent.
    pub mapped: Option<MappedPtr>,
    block_id: u64,
    type_index: u32,
}

struct Block {
    id: u64,
    memory: vk::DeviceMemory,
    #[allow(dead_code)] // diagnostics (Debug)
    size: u64,
    mapped: Option<MappedPtr>,
    /// Free ranges `(offset, size)`, sorted by offset, non-adjacent.
    free: Vec<(u64, u64)>,
    used: u64,
    dedicated: bool,
}

impl Block {
    fn try_alloc(&mut self, size: u64, align: u64) -> Option<u64> {
        for i in 0..self.free.len() {
            let (off, len) = self.free[i];
            let start = off.div_ceil(align) * align;
            let pad = start - off;
            if len < pad + size {
                continue;
            }
            let end = start + size;
            let tail = off + len - end;
            match (pad > 0, tail > 0) {
                (false, false) => {
                    self.free.remove(i);
                }
                (true, false) => self.free[i] = (off, pad),
                (false, true) => self.free[i] = (end, tail),
                (true, true) => {
                    self.free[i] = (off, pad);
                    self.free.insert(i + 1, (end, tail));
                }
            }
            self.used += size;
            return Some(start);
        }
        None
    }

    fn release(&mut self, offset: u64, size: u64) {
        self.used = self.used.saturating_sub(size);
        let pos = self.free.partition_point(|&(o, _)| o < offset);
        self.free.insert(pos, (offset, size));
        // merge with next
        if pos + 1 < self.free.len() {
            let (o, l) = self.free[pos];
            let (no, nl) = self.free[pos + 1];
            if o + l == no {
                self.free[pos] = (o, l + nl);
                self.free.remove(pos + 1);
            }
        }
        // merge with previous
        if pos > 0 {
            let (po, pl) = self.free[pos - 1];
            let (o, l) = self.free[pos];
            if po + pl == o {
                self.free[pos - 1] = (po, pl + l);
                self.free.remove(pos);
            }
        }
    }
}

pub(crate) struct Allocator {
    props: vk::PhysicalDeviceMemoryProperties,
    /// Blocks per memory type index.
    blocks: Vec<Vec<Block>>,
    next_id: u64,
    /// Types that failed with out-of-memory once; skipped while alternatives exist.
    exhausted: Vec<bool>,
}

impl Allocator {
    pub fn new(props: vk::PhysicalDeviceMemoryProperties) -> Allocator {
        let n = props.memory_type_count as usize;
        Allocator { props, blocks: (0..n).map(|_| Vec::new()).collect(), next_id: 1, exhausted: vec![false; n] }
    }

    fn flags(&self, t: u32) -> vk::MemoryPropertyFlags {
        self.props.memory_types_as_slice()[t as usize].property_flags
    }

    fn heap_size(&self, t: u32) -> u64 {
        let heap = self.props.memory_types_as_slice()[t as usize].heap_index as usize;
        self.props.memory_heaps_as_slice().get(heap).map(|h| h.size).unwrap_or(MAX_BLOCK)
    }

    /// Candidate memory types for `type_bits`, best first.
    fn candidates(&self, type_bits: u32, usage: MemUsage) -> Vec<u32> {
        use vk::MemoryPropertyFlags as F;
        let mut c: Vec<(i32, u32)> = Vec::new();
        for (i, t) in self.props.memory_types_as_slice().iter().enumerate() {
            if type_bits & (1 << i) == 0 {
                continue;
            }
            let f = t.property_flags;
            if f.contains(F::PROTECTED) || f.contains(F::LAZILY_ALLOCATED) {
                continue;
            }
            let host = f.contains(F::HOST_VISIBLE | F::HOST_COHERENT);
            let local = f.contains(F::DEVICE_LOCAL);
            let cached = f.contains(F::HOST_CACHED);
            let score = match usage {
                MemUsage::Device => (local as i32) * 8 + (host as i32) * 4 + (cached as i32),
                MemUsage::Upload => {
                    if !host {
                        continue;
                    }
                    (local as i32) * 2 + 1
                }
                MemUsage::Readback => {
                    if !host {
                        continue;
                    }
                    (cached as i32) * 4 + (local as i32)
                }
            };
            let penalty = if self.exhausted[i] { -100 } else { 0 };
            c.push((score + penalty, i as u32));
        }
        c.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        c.into_iter().map(|(_, i)| i).collect()
    }

    pub fn alloc(&mut self, device: &ash::Device, req: vk::MemoryRequirements, usage: MemUsage) -> Result<Allocation> {
        let cands = self.candidates(req.memory_type_bits, usage);
        if cands.is_empty() {
            return Err(Error::new("vulkan", format!("no memory type for {usage:?} (bits {:#x})", req.memory_type_bits)));
        }
        let mut last_err = None;
        for t in cands {
            match self.alloc_in_type(device, t, req.size.max(1), req.alignment.max(1)) {
                Ok(a) => return Ok(a),
                Err(e) => {
                    if e.code == "out_of_memory" {
                        self.exhausted[t as usize] = true;
                    }
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| Error::new("out_of_memory", "vulkan allocation failed")))
    }

    fn alloc_in_type(&mut self, device: &ash::Device, t: u32, size: u64, align: u64) -> Result<Allocation> {
        let make = |b: &Block, off: u64| Allocation {
            memory: b.memory,
            offset: off,
            size,
            // SAFETY: off + size lies inside the mapped block.
            mapped: b.mapped.map(|p| MappedPtr(unsafe { NonNull::new_unchecked(p.0.as_ptr().add(off as usize)) })),
            block_id: b.id,
            type_index: t,
        };
        for b in self.blocks[t as usize].iter_mut() {
            if b.dedicated {
                continue;
            }
            if let Some(off) = b.try_alloc(size, align) {
                return Ok(make(b, off));
            }
        }
        // new block
        let block_size = (self.heap_size(t) / 4).clamp(MIN_BLOCK, MAX_BLOCK);
        let dedicated = size > block_size / 2;
        let alloc_size = if dedicated { size } else { block_size };
        let mut block = self.new_block(device, t, alloc_size, dedicated)?;
        let off = block
            .try_alloc(size, align)
            .ok_or_else(|| Error::new("vulkan", "fresh memory block cannot hold allocation"))?;
        let a = make(&block, off);
        self.blocks[t as usize].push(block);
        Ok(a)
    }

    fn new_block(&mut self, device: &ash::Device, t: u32, size: u64, dedicated: bool) -> Result<Block> {
        let flags = self.flags(t);
        let info = vk::MemoryAllocateInfo::default().allocation_size(size).memory_type_index(t);
        // SAFETY: valid device and allocate info.
        let memory = unsafe { device.allocate_memory(&info, None) }.map_err(|e| vk_err("vkAllocateMemory", e))?;
        let mapped = if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT) {
            // SAFETY: memory is host visible and not yet mapped.
            match unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) } {
                Ok(p) => NonNull::new(p as *mut u8).map(MappedPtr),
                Err(e) => {
                    // SAFETY: memory was just allocated and is unused.
                    unsafe { device.free_memory(memory, None) };
                    return Err(vk_err("vkMapMemory", e));
                }
            }
        } else {
            None
        };
        let id = self.next_id;
        self.next_id += 1;
        Ok(Block { id, memory, size, mapped, free: vec![(0, size)], used: 0, dedicated })
    }

    pub fn free(&mut self, device: &ash::Device, a: &Allocation) {
        let list = &mut self.blocks[a.type_index as usize];
        let Some(pos) = list.iter().position(|b| b.id == a.block_id) else {
            return;
        };
        list[pos].release(a.offset, a.size);
        let empty = list[pos].used == 0;
        // Keep one empty shared block per type around for reuse; release dedicated ones and
        // surplus empty blocks so memory goes back to the system.
        let other_empty = list.iter().enumerate().any(|(i, b)| i != pos && b.used == 0 && !b.dedicated);
        if empty && (list[pos].dedicated || other_empty) {
            let b = list.swap_remove(pos);
            // SAFETY: the block has no live allocations.
            unsafe {
                if b.mapped.is_some() {
                    device.unmap_memory(b.memory);
                }
                device.free_memory(b.memory, None);
            }
            self.exhausted[a.type_index as usize] = false;
        }
    }

    /// Free every block (device teardown).
    pub fn destroy(&mut self, device: &ash::Device) {
        for list in self.blocks.iter_mut() {
            for b in list.drain(..) {
                // SAFETY: called once at device teardown after the device is idle.
                unsafe {
                    if b.mapped.is_some() {
                        device.unmap_memory(b.memory);
                    }
                    device.free_memory(b.memory, None);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(size: u64) -> Block {
        Block { id: 1, memory: vk::DeviceMemory::null(), size, mapped: None, free: vec![(0, size)], used: 0, dedicated: false }
    }

    #[test]
    fn first_fit_alignment_and_coalescing() {
        let mut b = block(1024);
        let a = b.try_alloc(100, 64).unwrap();
        assert_eq!(a, 0);
        let c = b.try_alloc(100, 256).unwrap();
        assert_eq!(c, 256);
        // the gap [100, 256) stays free and is used by a small allocation
        let d = b.try_alloc(50, 4).unwrap();
        assert_eq!(d, 100);
        b.release(a, 100);
        b.release(d, 50);
        b.release(c, 100);
        assert_eq!(b.free, vec![(0, 1024)]);
        assert_eq!(b.used, 0);
        assert!(b.try_alloc(2000, 4).is_none());
    }
}
