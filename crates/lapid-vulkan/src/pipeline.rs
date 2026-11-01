//! WGSL → naga → SPIR-V → compute pipeline, cached by `codegen::Key`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ash::vk;
use lapid_tensor::{Error, Result};

use crate::codegen::{self, Key};
use crate::context::{Caps, Shared, vk_err};
use crate::disk_cache::{DiskCache, hash_hex};

pub(crate) struct Pipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub set_layout: vk::DescriptorSetLayout,
    #[allow(dead_code)] // diagnostics
    pub bindings: u32,
    /// Descriptors are pushed (`VK_KHR_push_descriptor`) instead of allocated.
    pub push: bool,
    /// Final WGSL (for `dump_kernels`).
    pub source: Arc<str>,
    /// File name used when dumping: `<label>-<key hash>.wgsl`.
    pub file_name: String,
}

#[derive(Default)]
struct Inner {
    pipelines: HashMap<Key, Arc<Pipeline>>,
    /// (binding count, push) → layouts
    layouts: HashMap<(u32, bool), (vk::DescriptorSetLayout, vk::PipelineLayout)>,
}

pub(crate) struct Pipelines {
    inner: Mutex<Inner>,
    /// Driver pipeline cache (seeded from / saved to the disk cache).
    vk_cache: vk::PipelineCache,
    disk: Option<DiskCache>,
    /// Pipelines were created since the last `save`.
    dirty: std::sync::atomic::AtomicBool,
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).map(|v| !v.is_empty() && v != "0").unwrap_or(false)
}

fn numbered(src: &str) -> String {
    src.lines().enumerate().map(|(i, l)| format!("{:4} | {l}\n", i + 1)).collect()
}

fn shader_err(label: &str, msg: String, src: &str) -> Error {
    if env_flag("LAPID_DEBUG") {
        eprintln!("lapid-vulkan: shader error in {label}:\n{msg}\n{}", numbered(src));
    }
    Error::new("shader", format!("{label}: {msg}\n--- generated WGSL ---\n{}", numbered(src)))
}

/// Compile WGSL to SPIR-V words for this device.
pub(crate) fn compile_wgsl(label: &str, src: &str, caps: &Caps) -> Result<Vec<u32>> {
    use naga::back::spv;
    use naga::valid::{Capabilities as VC, ValidationFlags, Validator};

    let module = naga::front::wgsl::parse_str(src).map_err(|e| shader_err(label, e.emit_to_string(src), src))?;
    let mut vcaps = VC::SHADER_FLOAT16_IN_FLOAT32;
    if caps.f16 {
        vcaps |= VC::SHADER_FLOAT16;
    }
    if caps.subgroup {
        vcaps |= VC::SUBGROUP;
    }
    if caps.int16 {
        vcaps |= VC::SHADER_INT16;
    }
    if caps.coopmat {
        vcaps |= VC::COOPERATIVE_MATRIX;
    }
    let flags = ValidationFlags::all().difference(ValidationFlags::CONTROL_FLOW_UNIFORMITY);
    let info = Validator::new(flags, vcaps)
        .validate(&module)
        .map_err(|e| shader_err(label, e.emit_to_string(src), src))?;

    let mut set = naga::FastHashSet::default();
    set.insert(spv::Capability::Shader);
    if caps.f16 {
        set.insert(spv::Capability::Float16);
        set.insert(spv::Capability::StorageBuffer16BitAccess);
        set.insert(spv::Capability::UniformAndStorageBuffer16BitAccess);
    }
    if caps.int8 {
        set.insert(spv::Capability::Int8);
    }
    if caps.int16 {
        set.insert(spv::Capability::Int16);
    }
    if caps.coopmat {
        set.insert(spv::Capability::CooperativeMatrixKHR);
        set.insert(spv::Capability::VulkanMemoryModel);
    }
    if caps.int_dot {
        set.insert(spv::Capability::DotProduct);
        set.insert(spv::Capability::DotProductInput4x8BitPacked);
    }
    if caps.subgroup {
        set.insert(spv::Capability::GroupNonUniform);
        set.insert(spv::Capability::GroupNonUniformArithmetic);
        if caps.subgroup_ballot {
            set.insert(spv::Capability::GroupNonUniformBallot);
        }
        if caps.subgroup_shuffle {
            set.insert(spv::Capability::GroupNonUniformShuffle);
        }
        if caps.subgroup_vote {
            set.insert(spv::Capability::GroupNonUniformVote);
        }
    }
    let opts = spv::Options {
        lang_version: (1, 3),
        flags: spv::WriterFlags::empty(),
        capabilities: Some(set),
        zero_initialize_workgroup_memory: spv::ZeroInitializeWorkgroupMemoryMode::None,
        force_loop_bounding: false,
        use_storage_input_output_16: false,
        emit_int_div_checks: false,
        ..spv::Options::default()
    };
    let popts = spv::PipelineOptions { shader_stage: naga::ShaderStage::Compute, entry_point: "main".into() };
    spv::write_vec(&module, &info, &opts, Some(&popts)).map_err(|e| shader_err(label, e.to_string(), src))
}

impl Pipelines {
    pub fn new(shared: &Shared) -> Pipelines {
        let disk = DiskCache::open(&shared.dev_id);
        let initial = disk.as_ref().and_then(|d| d.load_pipeline_cache()).unwrap_or_default();
        let create = |data: &[u8]| {
            let info = vk::PipelineCacheCreateInfo::default().initial_data(data);
            // SAFETY: valid device; the data header was validated against this device (the
            // driver validates the rest and may ignore it).
            unsafe { shared.device.create_pipeline_cache(&info, None) }
        };
        // no driver cache object at all when the disk cache is disabled
        let vk_cache = if disk.is_some() { create(&initial).or_else(|_| create(&[])).unwrap_or(vk::PipelineCache::null()) } else { vk::PipelineCache::null() };
        Pipelines { inner: Mutex::new(Inner::default()), vk_cache, disk, dirty: false.into() }
    }

    /// Write the driver pipeline cache to disk if pipelines were created since the last save.
    pub fn save(&self, shared: &Shared) {
        let Some(disk) = &self.disk else { return };
        if self.vk_cache == vk::PipelineCache::null() || !self.dirty.swap(false, std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let _g = self.inner.lock();
        // SAFETY: valid cache; creation calls are serialized by the `inner` lock.
        if let Ok(data) = unsafe { shared.device.get_pipeline_cache_data(self.vk_cache) } {
            disk.store_pipeline_cache(&data);
        }
    }

    /// SPIR-V for `src`: from the disk cache, else compiled with naga (and stored).
    fn spirv(&self, key: &Key, src: &str, caps: &Caps) -> Result<Vec<u32>> {
        let hash = self.disk.as_ref().map(|_| hash_hex(&[src.as_bytes(), format!("{caps:?}").as_bytes()]));
        if let (Some(d), Some(h)) = (&self.disk, &hash)
            && let Some(w) = d.load_spirv(h) {
                return Ok(w);
            }
        let words = compile_wgsl(key.label(), src, caps)?;
        if let (Some(d), Some(h)) = (&self.disk, &hash) {
            d.store_spirv(h, &words);
        }
        Ok(words)
    }

    /// Get (compiling on first use) the pipeline for `key`.
    pub fn get(&self, shared: &Shared, key: &Key) -> Result<Arc<Pipeline>> {
        let mut inner = self.inner.lock().map_err(|_| Error::new("vulkan", "pipeline cache poisoned"))?;
        if let Some(p) = inner.pipelines.get(key) {
            return Ok(p.clone());
        }
        let caps = &shared.caps;
        let src = codegen::source(key, caps);
        let file_name = kernel_file_name(key);
        if env_flag("LAPID_DEBUG") {
            debug_dump(&file_name, &src);
        }
        let words = self.spirv(key, &src, caps)?;
        let bindings = key.num_bindings() as u32;
        let push = caps.push_descriptor && shared.push_desc.is_some() && bindings <= caps.max_push_descriptors;
        let (set_layout, layout) = match inner.layouts.get(&(bindings, push)) {
            Some(&l) => l,
            None => {
                let l = create_layouts(&shared.device, bindings, push)?;
                inner.layouts.insert((bindings, push), l);
                l
            }
        };
        let full = key.uses_subgroups(&caps.wgsl()) && caps.full_subgroups;
        let pipeline = create_pipeline(&shared.device, self.vk_cache, &words, layout, full)?;
        self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        let p = Arc::new(Pipeline { pipeline, layout, set_layout, bindings, push, source: src.into(), file_name });
        inner.pipelines.insert(key.clone(), p.clone());
        Ok(p)
    }

    /// Write the WGSL of every compiled pipeline into `dir`; returns the file count.
    pub fn dump(&self, dir: &std::path::Path) -> Result<usize> {
        let inner = self.inner.lock().map_err(|_| Error::new("vulkan", "pipeline cache poisoned"))?;
        std::fs::create_dir_all(dir).map_err(|e| Error::new("io", format!("{}: {e}", dir.display())))?;
        for p in inner.pipelines.values() {
            let path = dir.join(&p.file_name);
            std::fs::write(&path, p.source.as_bytes()).map_err(|e| Error::new("io", format!("{}: {e}", path.display())))?;
        }
        Ok(inner.pipelines.len())
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map(|i| i.pipelines.len()).unwrap_or(0)
    }

    /// Destroy every pipeline and layout (backend teardown, device idle).
    pub fn destroy(&self, device: &ash::Device) {
        if let Ok(mut inner) = self.inner.lock() {
            // SAFETY: the device is idle; nothing references these objects anymore.
            unsafe {
                for (_, p) in inner.pipelines.drain() {
                    device.destroy_pipeline(p.pipeline, None);
                }
                for (_, (set, layout)) in inner.layouts.drain() {
                    device.destroy_pipeline_layout(layout, None);
                    device.destroy_descriptor_set_layout(set, None);
                }
                if self.vk_cache != vk::PipelineCache::null() {
                    device.destroy_pipeline_cache(self.vk_cache, None);
                }
            }
        }
    }
}

/// `<label>-<16 hex digits of the key hash>.wgsl` (stable within a build).
pub(crate) fn kernel_file_name(key: &Key) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut h);
    format!("{}-{:016x}.wgsl", key.label(), h.finish())
}

/// `LAPID_DEBUG=1`: write each shader as it is compiled to `$LAPID_KERNEL_DIR`
/// (default `<tmp>/lapid-kernels`).
fn debug_dump(name: &str, src: &str) {
    let dir = std::env::var_os("LAPID_KERNEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("lapid-kernels"));
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| eprintln!("lapid-vulkan: writing compiled kernels to {}", dir.display()));
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join(name), src);
}

fn create_layouts(device: &ash::Device, n: u32, push: bool) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout)> {
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let flags = if push { vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR } else { vk::DescriptorSetLayoutCreateFlags::empty() };
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings).flags(flags);
    // SAFETY: valid device and create infos.
    unsafe {
        let set = device.create_descriptor_set_layout(&info, None).map_err(|e| vk_err("vkCreateDescriptorSetLayout", e))?;
        let sets = [set];
        let pinfo = vk::PipelineLayoutCreateInfo::default().set_layouts(&sets);
        match device.create_pipeline_layout(&pinfo, None) {
            Ok(layout) => Ok((set, layout)),
            Err(e) => {
                device.destroy_descriptor_set_layout(set, None);
                Err(vk_err("vkCreatePipelineLayout", e))
            }
        }
    }
}

fn create_pipeline(device: &ash::Device, cache: vk::PipelineCache, words: &[u32], layout: vk::PipelineLayout, full_subgroups: bool) -> Result<vk::Pipeline> {
    // SAFETY: SPIR-V produced and validated by naga; the module is destroyed after use.
    unsafe {
        let minfo = vk::ShaderModuleCreateInfo::default().code(words);
        let module = device.create_shader_module(&minfo, None).map_err(|e| vk_err("vkCreateShaderModule", e))?;
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        if full_subgroups {
            stage = stage.flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS);
        }
        let info = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)];
        let res = device.create_compute_pipelines(cache, &info, None);
        device.destroy_shader_module(module, None);
        match res {
            Ok(p) => p.into_iter().next().ok_or_else(|| Error::new("vulkan", "no pipeline returned")),
            Err((_, e)) => Err(vk_err("vkCreateComputePipelines", e)),
        }
    }
}
