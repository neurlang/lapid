//! Vulkan instance / physical device / logical device setup and capability detection.

use std::ffi::{CStr, c_char};
use std::sync::{Mutex, MutexGuard, OnceLock};

use ash::vk;
use lapid_tensor::{Error, Result};

use crate::AdapterInfo;
use crate::memory::Allocator;

/// Map a raw Vulkan error to a lapid error.
pub(crate) fn vk_err(what: &str, r: vk::Result) -> Error {
    match r {
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
            Error::new("out_of_memory", format!("vulkan: {what}: {r}"))
        }
        vk::Result::ERROR_DEVICE_LOST => Error::new("device_lost", format!("vulkan: {what}: {r}")),
        _ => Error::new("vulkan", format!("{what}: {r}")),
    }
}

/// What the device can do, as far as code generation and execution are concerned.
#[derive(Clone, Debug)]
pub(crate) struct Caps {
    /// `enable f16;` + `array<f16>` storage (shaderFloat16 + 16-bit storage access).
    pub f16: bool,
    /// shaderInt8 (lets naga emit native 8-bit packing instead of polyfills).
    pub int8: bool,
    /// shaderInt16.
    pub int16: bool,
    /// Packed 4x8 integer dot product (`dot4I8Packed` → OpSDot).
    pub int_dot: bool,
    /// Subgroup arithmetic (+ basic) usable from compute shaders.
    pub subgroup: bool,
    pub subgroup_ballot: bool,
    pub subgroup_shuffle: bool,
    pub subgroup_vote: bool,
    pub subgroup_size: u32,
    /// `REQUIRE_FULL_SUBGROUPS` can be requested for compute pipelines.
    pub full_subgroups: bool,
    pub max_groups: [u32; 3],
    pub max_shared: u32,
    #[allow(dead_code)] // reported in `Debug` (part of the kernel cache key)
    pub max_invocations: u32,
    pub storage_align: u64,
    pub max_storage_range: u64,
    pub push_descriptor: bool,
    pub max_push_descriptors: u32,
    /// Cooperative matrices (VK_KHR_cooperative_matrix) with a subgroup-scope
    /// 16x16x16 f16 x f16 -> f32 shape, plus the Vulkan memory model naga requires for them.
    pub coopmat: bool,
    /// Nanoseconds per timestamp tick; 0 = timestamps unsupported on the compute queue.
    pub timestamp_period: f32,
}

/// Device-wide state shared by the backend and every buffer it hands out.
pub(crate) struct Shared {
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub caps: Caps,
    pub name: String,
    pub allocator: Mutex<Allocator>,
    pub push_desc: Option<ash::khr::push_descriptor::Device>,
    /// Queue submissions and device-wide waits need external synchronization: the device
    /// is shared by every backend opened on the same adapter in this process.
    pub queue_lock: Mutex<()>,
    /// Device + driver identity (on-disk kernel cache key).
    pub dev_id: crate::disk_cache::DeviceId,
    instance: ash::Instance,
}

impl Shared {
    pub fn lock_queue(&self) -> MutexGuard<'_, ()> {
        self.queue_lock.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// `vkDeviceWaitIdle` under the queue lock.
    pub fn wait_idle(&self) {
        let _q = self.lock_queue();
        // SAFETY: queue access is serialized by `queue_lock`.
        unsafe {
            let _ = self.device.device_wait_idle();
        }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: every object created from `device` is owned by something holding an
        // `Arc<Shared>` and has been destroyed by now; the allocator frees its blocks here.
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Ok(mut a) = self.allocator.lock() {
                a.destroy(&self.device);
            }
            let _g = loader_lock();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).map(|v| !v.is_empty() && v != "0").unwrap_or(false)
}

/// The Vulkan loader, loaded once per process and never unloaded. Dropping an
/// `ash::Entry` dlcloses libvulkan, which crashes any other thread that is inside a loader
/// call at that moment (this was the parallel-test crash/hang).
fn load_entry() -> Result<&'static ash::Entry> {
    static ENTRY: OnceLock<std::result::Result<ash::Entry, String>> = OnceLock::new();
    // SAFETY: loading the system Vulkan loader; failures are reported, never panics.
    ENTRY
        .get_or_init(|| unsafe { ash::Entry::load() }.map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| Error::new("vulkan_unavailable", format!("cannot load the Vulkan loader: {e}")))
}

/// Serializes instance / device creation and destruction. Drivers are not required to be
/// (and in practice are not always) safe against concurrent ICD initialization.
fn loader_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

fn create_instance(entry: &ash::Entry) -> Result<ash::Instance> {
    // SAFETY: plain Vulkan calls on a valid entry.
    unsafe {
        let loader_version = entry.try_enumerate_instance_version().ok().flatten().unwrap_or(vk::API_VERSION_1_0);
        if loader_version < vk::API_VERSION_1_1 {
            return Err(Error::new("vulkan_unavailable", "Vulkan loader older than 1.1"));
        }
        let api = loader_version.min(vk::API_VERSION_1_3);
        let app = vk::ApplicationInfo::default()
            .application_name(c"lapid")
            .application_version(1)
            .engine_name(c"lapid")
            .engine_version(1)
            .api_version(api);

        let mut layers: Vec<*const c_char> = Vec::new();
        if env_flag("LAPID_VK_VALIDATION") {
            let wanted = c"VK_LAYER_KHRONOS_validation";
            let available = entry.enumerate_instance_layer_properties().unwrap_or_default();
            if available.iter().any(|l| l.layer_name_as_c_str().map(|n| n == wanted).unwrap_or(false)) {
                layers.push(wanted.as_ptr());
            } else {
                eprintln!("lapid-vulkan: LAPID_VK_VALIDATION set but VK_LAYER_KHRONOS_validation is not installed");
            }
        }
        let mut exts: Vec<*const c_char> = Vec::new();
        let mut flags = vk::InstanceCreateFlags::empty();
        let inst_exts = entry.enumerate_instance_extension_properties(None).unwrap_or_default();
        let has_ext = |n: &CStr| inst_exts.iter().any(|e| e.extension_name_as_c_str().map(|x| x == n).unwrap_or(false));
        if has_ext(vk::KHR_PORTABILITY_ENUMERATION_NAME) {
            exts.push(vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }
        let info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&exts)
            .flags(flags);
        entry
            .create_instance(&info, None)
            .map_err(|e| Error::new("vulkan_unavailable", format!("vkCreateInstance failed: {e}")))
    }
}

fn kind_name(t: vk::PhysicalDeviceType) -> &'static str {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
        vk::PhysicalDeviceType::CPU => "cpu",
        _ => "other",
    }
}

/// Physical devices usable for compute (API >= 1.1 and a compute queue).
fn usable_devices(instance: &ash::Instance) -> Result<Vec<(vk::PhysicalDevice, vk::PhysicalDeviceProperties)>> {
    // SAFETY: valid instance.
    unsafe {
        let devs = instance.enumerate_physical_devices().map_err(|e| vk_err("vkEnumeratePhysicalDevices", e))?;
        let mut out = Vec::new();
        for pd in devs {
            let props = instance.get_physical_device_properties(pd);
            if props.api_version < vk::API_VERSION_1_1 {
                continue;
            }
            let qf = instance.get_physical_device_queue_family_properties(pd);
            if !qf.iter().any(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE)) {
                continue;
            }
            out.push((pd, props));
        }
        Ok(out)
    }
}

fn device_name(props: &vk::PhysicalDeviceProperties) -> String {
    props
        .device_name_as_c_str()
        .map(|c| c.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "unknown vulkan device".to_string())
}

/// Adapter list, enumerated once per process (instance churn is avoided for the same
/// reason as device churn, see `shared_device`).
pub(crate) fn enumerate_adapters() -> Result<Vec<AdapterInfo>> {
    static ADAPTERS: OnceLock<Result<Vec<AdapterInfo>>> = OnceLock::new();
    ADAPTERS.get_or_init(enumerate_adapters_uncached).clone()
}

fn enumerate_adapters_uncached() -> Result<Vec<AdapterInfo>> {
    let entry = load_entry()?;
    let _g = loader_lock();
    let instance = create_instance(entry)?;
    let result = usable_devices(&instance).map(|devs| {
        devs.iter()
            .enumerate()
            .map(|(index, (_, props))| AdapterInfo {
                index,
                name: device_name(props),
                kind: kind_name(props.device_type).to_string(),
                is_software: props.device_type == vk::PhysicalDeviceType::CPU,
            })
            .collect()
    });
    // SAFETY: nothing else was created from this instance.
    unsafe { instance.destroy_instance(None) };
    result
}

const fn b(v: bool) -> vk::Bool32 {
    if v { vk::TRUE } else { vk::FALSE }
}

/// The device for adapter `index`: created on first use and kept for the life of the
/// process, shared by every backend on that adapter.
///
/// Creating and destroying VkDevices repeatedly while other threads use the GPU hung the
/// NVIDIA driver inside `vkDestroyDevice` (the parallel test suite hang); one long-lived
/// device per adapter is also what lets backends share memory and avoids re-initializing
/// the driver per backend. Memory is still returned to the device as buffers drop.
pub(crate) fn shared_device(index: usize) -> Result<std::sync::Arc<Shared>> {
    static DEVICES: Mutex<Vec<(usize, std::sync::Arc<Shared>)>> = Mutex::new(Vec::new());
    let mut list = DEVICES.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((_, s)) = list.iter().find(|(i, _)| *i == index) {
        return Ok(s.clone());
    }
    let s = std::sync::Arc::new(create_shared(index)?);
    list.push((index, s.clone()));
    Ok(s)
}

/// Create the device for adapter `index` (index into `enumerate_adapters()`).
fn create_shared(index: usize) -> Result<Shared> {
    let entry = load_entry()?;
    let _g = loader_lock();
    let instance = create_instance(entry)?;
    match create_device(entry, &instance, index) {
        Ok(parts) => Ok(Shared {
            device: parts.device,
            queue: parts.queue,
            queue_family: parts.queue_family,
            caps: parts.caps,
            name: parts.name,
            allocator: Mutex::new(Allocator::new(parts.mem_props)),
            push_desc: parts.push_desc,
            queue_lock: Mutex::new(()),
            dev_id: parts.dev_id,
            instance,
        }),
        Err(e) => {
            // SAFETY: the device was not created (or was destroyed) on error.
            unsafe { instance.destroy_instance(None) };
            Err(e)
        }
    }
}

struct DeviceParts {
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    caps: Caps,
    name: String,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    push_desc: Option<ash::khr::push_descriptor::Device>,
    dev_id: crate::disk_cache::DeviceId,
}

fn create_device(entry: &ash::Entry, instance: &ash::Instance, index: usize) -> Result<DeviceParts> {
    let devs = usable_devices(instance)?;
    let Some(&(pd, props)) = devs.get(index) else {
        return Err(Error::new(
            "invalid_device",
            format!("vulkan adapter index {index} out of range ({} adapters)", devs.len()),
        ));
    };
    let api12 = props.api_version >= vk::API_VERSION_1_2;
    let api13 = props.api_version >= vk::API_VERSION_1_3;

    // SAFETY: valid instance + physical device; all chained structs outlive the calls.
    unsafe {
        // ---- queue family: prefer a compute-only queue, else any compute queue.
        let qf = instance.get_physical_device_queue_family_properties(pd);
        let compute_only = qf.iter().position(|q| {
            q.queue_flags.contains(vk::QueueFlags::COMPUTE) && !q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        });
        let any_compute = qf.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE));
        let queue_family = compute_only.or(any_compute).ok_or_else(|| Error::new("vulkan", "no compute queue"))? as u32;

        // ---- extensions
        let dev_exts = instance.enumerate_device_extension_properties(pd).map_err(|e| vk_err("enumerate extensions", e))?;
        let has_ext = |n: &CStr| dev_exts.iter().any(|e| e.extension_name_as_c_str().map(|x| x == n).unwrap_or(false));
        let mut ext_names: Vec<*const c_char> = Vec::new();
        let push_descriptor = has_ext(vk::KHR_PUSH_DESCRIPTOR_NAME);
        if push_descriptor {
            ext_names.push(vk::KHR_PUSH_DESCRIPTOR_NAME.as_ptr());
        }
        let coop_ext = has_ext(vk::KHR_COOPERATIVE_MATRIX_NAME);
        if has_ext(vk::KHR_PORTABILITY_SUBSET_NAME) {
            ext_names.push(vk::KHR_PORTABILITY_SUBSET_NAME.as_ptr());
        }

        // ---- supported features
        let mut s11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut s12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut s13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut scm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        {
            let mut f2 = vk::PhysicalDeviceFeatures2::default();
            if api12 {
                f2 = f2.push_next(&mut s11).push_next(&mut s12);
            }
            if api13 {
                f2 = f2.push_next(&mut s13);
            }
            if coop_ext {
                f2 = f2.push_next(&mut scm);
            }
            instance.get_physical_device_features2(pd, &mut f2);
        }

        // ---- properties
        let mut sg_props = vk::PhysicalDeviceSubgroupProperties::default();
        let mut push_props = vk::PhysicalDevicePushDescriptorPropertiesKHR::default();
        {
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sg_props);
            if push_descriptor {
                p2 = p2.push_next(&mut push_props);
            }
            instance.get_physical_device_properties2(pd, &mut p2);
        }
        let mem_props = instance.get_physical_device_memory_properties(pd);

        let disable_f16 = env_flag("LAPID_VK_NO_F16");
        let disable_sg = env_flag("LAPID_VK_NO_SUBGROUPS");
        let disable_dot = env_flag("LAPID_VK_NO_INT_DOT");

        let f16 = api12
            && !disable_f16
            && s12.shader_float16 == vk::TRUE
            && s11.storage_buffer16_bit_access == vk::TRUE
            && s11.uniform_and_storage_buffer16_bit_access == vk::TRUE;
        let int8 = api12 && s12.shader_int8 == vk::TRUE;
        let base = instance.get_physical_device_features(pd);
        let int16 = base.shader_int16 == vk::TRUE;
        let int_dot = api13 && !disable_dot && s13.shader_integer_dot_product == vk::TRUE;
        let sg_compute = sg_props.supported_stages.contains(vk::ShaderStageFlags::COMPUTE);
        let ops = sg_props.supported_operations;
        let subgroup = !disable_sg
            && sg_compute
            && ops.contains(vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC)
            && sg_props.subgroup_size >= 4;
        let full_subgroups = api13 && s13.subgroup_size_control == vk::TRUE && s13.compute_full_subgroups == vk::TRUE;

        // cooperative matrix: need the 16x16x16 f16*f16+f32 subgroup shape and the Vulkan
        // memory model (naga emits MemoryModel Vulkan for modules using coop types).
        let coopmat = coop_ext
            && api13
            && !env_flag("LAPID_VK_NO_COOPMAT")
            && f16
            && subgroup
            && full_subgroups
            && sg_props.subgroup_size == 32
            && scm.cooperative_matrix == vk::TRUE
            && s12.vulkan_memory_model == vk::TRUE
            && {
                let ci = ash::khr::cooperative_matrix::Instance::new(entry, instance);
                let props = ci.get_physical_device_cooperative_matrix_properties(pd).unwrap_or_default();
                props.iter().any(|p| {
                    p.m_size == 16
                        && p.n_size == 16
                        && p.k_size == 16
                        && p.a_type == vk::ComponentTypeKHR::FLOAT16
                        && p.b_type == vk::ComponentTypeKHR::FLOAT16
                        && p.c_type == vk::ComponentTypeKHR::FLOAT32
                        && p.result_type == vk::ComponentTypeKHR::FLOAT32
                        && p.scope == vk::ScopeKHR::SUBGROUP
                })
            };
        if coopmat {
            ext_names.push(vk::KHR_COOPERATIVE_MATRIX_NAME.as_ptr());
        }

        // ---- enabled features (only what we use and is supported)
        let base_enable = vk::PhysicalDeviceFeatures { shader_int16: b(int16), ..Default::default() };
        let mut e11 = vk::PhysicalDeviceVulkan11Features {
            storage_buffer16_bit_access: b(f16),
            uniform_and_storage_buffer16_bit_access: b(f16),
            ..Default::default()
        };
        let mut e12 = vk::PhysicalDeviceVulkan12Features {
            shader_float16: b(f16),
            shader_int8: b(int8),
            storage_buffer8_bit_access: s12.storage_buffer8_bit_access,
            uniform_and_storage_buffer8_bit_access: s12.uniform_and_storage_buffer8_bit_access,
            vulkan_memory_model: b(coopmat),
            ..Default::default()
        };
        let mut ecm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(coopmat);
        let mut e13 = vk::PhysicalDeviceVulkan13Features {
            subgroup_size_control: b(full_subgroups),
            compute_full_subgroups: b(full_subgroups),
            shader_integer_dot_product: b(int_dot),
            synchronization2: s13.synchronization2,
            maintenance4: s13.maintenance4,
            ..Default::default()
        };

        let prio = [1.0f32];
        let qinfo = [vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family).queue_priorities(&prio)];
        let mut dinfo = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qinfo)
            .enabled_extension_names(&ext_names)
            .enabled_features(&base_enable);
        if api12 {
            dinfo = dinfo.push_next(&mut e11).push_next(&mut e12);
        }
        if api13 {
            dinfo = dinfo.push_next(&mut e13);
        }
        if coopmat {
            dinfo = dinfo.push_next(&mut ecm);
        }
        let device = instance.create_device(pd, &dinfo, None).map_err(|e| vk_err("vkCreateDevice", e))?;
        let queue = device.get_device_queue(queue_family, 0);
        let push_desc = push_descriptor.then(|| ash::khr::push_descriptor::Device::new(instance, &device));

        let lim = &props.limits;
        let caps = Caps {
            f16,
            int8,
            int16,
            int_dot,
            subgroup,
            subgroup_ballot: subgroup && ops.contains(vk::SubgroupFeatureFlags::BALLOT),
            subgroup_shuffle: subgroup && ops.contains(vk::SubgroupFeatureFlags::SHUFFLE),
            subgroup_vote: subgroup && ops.contains(vk::SubgroupFeatureFlags::VOTE),
            subgroup_size: sg_props.subgroup_size.max(1),
            full_subgroups,
            max_groups: lim.max_compute_work_group_count,
            max_shared: lim.max_compute_shared_memory_size,
            max_invocations: lim.max_compute_work_group_invocations,
            storage_align: lim.min_storage_buffer_offset_alignment.max(4),
            max_storage_range: lim.max_storage_buffer_range as u64,
            push_descriptor,
            max_push_descriptors: if push_descriptor { push_props.max_push_descriptors } else { 0 },
            coopmat,
            timestamp_period: if qf[queue_family as usize].timestamp_valid_bits > 0 { lim.timestamp_period } else { 0.0 },
        };
        let dev_id = crate::disk_cache::DeviceId {
            name: device_name(&props),
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            driver_version: props.driver_version,
            pipeline_cache_uuid: props.pipeline_cache_uuid,
        };
        Ok(DeviceParts { device, queue, queue_family, caps, name: device_name(&props), mem_props, push_desc, dev_id })
    }
}
