//! Instance / adapter / device setup, capability detection and error capture.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use lapid_tensor::{Error, Result};

#[cfg(not(target_arch = "wasm32"))]
use crate::AdapterInfo;

/// Device-wide state shared by the backend and every buffer it hands out.
pub(crate) struct Ctx {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub caps: lapid_wgsl::Caps,
    pub name: String,
    pub backend: &'static str,
    /// maxStorageBufferBindingSize
    pub max_binding: u64,
    /// maxBufferSize
    pub max_buffer: u64,
    /// minStorageBufferOffsetAlignment
    pub storage_align: u64,
    /// Process-unique id (buffers are checked against it).
    pub id: u64,
    /// First error reported by wgpu since the last `take_error` (uncaptured errors).
    errors: Arc<Mutex<Option<Error>>>,
}

fn backend_name(b: wgpu::Backend) -> &'static str {
    match b {
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Metal => "metal",
        wgpu::Backend::Dx12 => "dx12",
        wgpu::Backend::Gl => "gl",
        wgpu::Backend::BrowserWebGpu => "browser",
        _ => "other",
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn kind_name(t: wgpu::DeviceType) -> &'static str {
    match t {
        wgpu::DeviceType::DiscreteGpu => "discrete",
        wgpu::DeviceType::IntegratedGpu => "integrated",
        wgpu::DeviceType::VirtualGpu => "virtual",
        wgpu::DeviceType::Cpu => "cpu",
        wgpu::DeviceType::Other => "other",
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

#[cfg(not(target_arch = "wasm32"))]
fn instance() -> &'static wgpu::Instance {
    static INSTANCE: std::sync::OnceLock<wgpu::Instance> = std::sync::OnceLock::new();
    INSTANCE.get_or_init(|| {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        desc.flags = if env_flag("LAPID_WGPU_VALIDATION") { wgpu::InstanceFlags::debugging() } else { wgpu::InstanceFlags::empty() };
        wgpu::Instance::new(desc)
    })
}

#[cfg(not(target_arch = "wasm32"))]
async fn native_adapters() -> Vec<wgpu::Adapter> {
    instance().enumerate_adapters(wgpu::Backends::PRIMARY).await
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn enumerate() -> Result<Vec<AdapterInfo>> {
    let list = pollster::block_on(native_adapters());
    Ok(list
        .iter()
        .enumerate()
        .map(|(index, a)| {
            let info = a.get_info();
            AdapterInfo { index, name: info.name, backend: backend_name(info.backend).to_string(), kind: kind_name(info.device_type).to_string() }
        })
        .collect())
}

#[cfg(not(target_arch = "wasm32"))]
async fn pick_adapter(index: usize) -> Result<wgpu::Adapter> {
    let mut list = native_adapters().await;
    if index >= list.len() {
        return Err(Error::new("device_unavailable", format!("webgpu:{index}: only {} wgpu adapter(s) found", list.len())));
    }
    Ok(list.swap_remove(index))
}

#[cfg(target_arch = "wasm32")]
async fn pick_adapter(index: usize) -> Result<wgpu::Adapter> {
    if index != 0 {
        return Err(Error::new("device_unavailable", format!("webgpu:{index}: the browser exposes one adapter (webgpu:0)")));
    }
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(desc);
    let opts = wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() };
    instance
        .request_adapter(&opts)
        .await
        .map_err(|e| Error::new("webgpu_unavailable", format!("no WebGPU adapter (navigator.gpu missing or blocked): {e}")))
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

impl Ctx {
    pub async fn open(index: usize) -> Result<Ctx> {
        let adapter = pick_adapter(index).await?;
        let info = adapter.get_info();
        let feats = adapter.features();
        let limits = adapter.limits();
        let f16 = feats.contains(wgpu::Features::SHADER_F16) && !env_flag("LAPID_WEBGPU_NO_F16");
        // subgroup kernels assume one fixed subgroup size (lanes packed in local-index order)
        let sg_size = info.subgroup_min_size;
        let subgroup = feats.contains(wgpu::Features::SUBGROUP)
            && sg_size == info.subgroup_max_size
            && (4..=128).contains(&sg_size)
            && !env_flag("LAPID_WEBGPU_NO_SUBGROUPS");
        let mut required = wgpu::Features::empty();
        if f16 {
            required |= wgpu::Features::SHADER_F16;
        }
        if subgroup {
            required |= wgpu::Features::SUBGROUP;
        }
        let desc = wgpu::DeviceDescriptor {
            label: Some("lapid"),
            required_features: required,
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        };
        let (device, queue) = adapter
            .request_device(&desc)
            .await
            .map_err(|e| Error::new("webgpu_unavailable", format!("{}: request_device failed: {e}", info.name)))?;
        let errors: Arc<Mutex<Option<Error>>> = Arc::new(Mutex::new(None));
        let sink = errors.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            let err = match &e {
                wgpu::Error::OutOfMemory { .. } => Error::new("out_of_memory", format!("webgpu: {e}")),
                _ => Error::new("webgpu", format!("{e}")),
            };
            if let Ok(mut slot) = sink.lock()
                && slot.is_none()
            {
                *slot = Some(err);
            }
        }));
        let caps = lapid_wgsl::Caps {
            f16,
            subgroup,
            subgroup_size: if subgroup { sg_size } else { 32 },
            full_subgroups: false,
            max_groups: [limits.max_compute_workgroups_per_dimension; 3],
            max_shared: limits.max_compute_workgroup_storage_size,
            coopmat: false,
            webgpu: true,
        };
        if limits.max_compute_invocations_per_workgroup < 256 || limits.max_storage_buffers_per_shader_stage < 8 {
            return Err(Error::new(
                "webgpu_unavailable",
                format!(
                    "{}: needs 256 invocations and 8 storage buffers per stage (has {} / {})",
                    info.name, limits.max_compute_invocations_per_workgroup, limits.max_storage_buffers_per_shader_stage
                ),
            ));
        }
        Ok(Ctx {
            device,
            queue,
            caps,
            // browsers often hide the adapter name: fall back to vendor / description
            name: [&info.name, &info.driver_info, &info.driver]
                .into_iter()
                .find(|s| !s.trim().is_empty())
                .cloned()
                .unwrap_or_else(|| "GPU".to_string()),
            backend: backend_name(info.backend),
            max_binding: limits.max_storage_buffer_binding_size,
            max_buffer: limits.max_buffer_size,
            storage_align: (limits.min_storage_buffer_offset_alignment as u64).max(4),
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            errors,
        })
    }

    /// Return (and clear) the first error wgpu reported since the last call.
    pub fn take_error(&self) -> Result<()> {
        match self.errors.lock() {
            Ok(mut slot) => match slot.take() {
                Some(e) => Err(e),
                None => Ok(()),
            },
            Err(_) => Err(Error::new("webgpu", "error slot poisoned")),
        }
    }

    /// Block until all submitted work completed (native); no-op in the browser.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn wait_idle(&self) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.device
                .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
                .map_err(|e| Error::new("device_lost", format!("webgpu: poll: {e}")))?;
        }
        Ok(())
    }
}
