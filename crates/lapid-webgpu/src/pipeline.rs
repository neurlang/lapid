//! WGSL (lapid-wgsl) → shader module → compute pipeline, cached by `lapid_wgsl::Key`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use lapid_tensor::{Error, Result};
use lapid_wgsl::Key;

use crate::context::Ctx;

/// A compiled kernel.
pub(crate) struct Pipe {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
    source: Arc<str>,
    file_name: String,
}

#[derive(Default)]
struct Inner {
    pipelines: HashMap<Key, Arc<Pipe>>,
    /// (bind group layout, pipeline layout) per (bindings, read-only inputs)
    layouts: HashMap<(usize, usize), (wgpu::BindGroupLayout, wgpu::PipelineLayout)>,
}

#[derive(Default)]
pub(crate) struct Pipelines {
    inner: Mutex<Inner>,
}

fn numbered(src: &str) -> String {
    src.lines().enumerate().map(|(i, l)| format!("{:4} | {l}\n", i + 1)).collect()
}

fn kernel_file_name(key: &Key, src: &str) -> String {
    // FNV-1a of the source: stable, dependency free
    let mut h: u64 = 0xcbf29ce484222325;
    for b in src.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    format!("{}-{h:016x}.wgsl", key.label())
}

/// Layout: binding 0 = params and 1..=inputs are read-only storage; the output and the
/// optional extra buffer are read-write.
fn create_layouts(device: &wgpu::Device, bindings: usize, inputs: usize) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..bindings)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: i <= inputs },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    (bgl, pl)
}

impl Pipelines {
    /// Get (compiling on first use) the pipeline for `key`.
    pub fn get(&self, ctx: &Ctx, key: &Key) -> Result<Arc<Pipe>> {
        let mut inner = self.inner.lock().map_err(|_| Error::new("webgpu", "pipeline cache poisoned"))?;
        if let Some(p) = inner.pipelines.get(key) {
            return Ok(p.clone());
        }
        let src = lapid_wgsl::source(key, &ctx.caps);
        let file_name = kernel_file_name(key, &src);
        // naga (native wgpu) does not parse `enable subgroups;` yet; browsers require it
        #[cfg(not(target_arch = "wasm32"))]
        let compiled = src.replace("enable subgroups;", "");
        #[cfg(target_arch = "wasm32")]
        let compiled = src.clone();
        let bindings = key.num_bindings();
        let inputs = key.num_inputs();
        let (bgl, pl) = inner
            .layouts
            .entry((bindings, inputs))
            .or_insert_with(|| create_layouts(&ctx.device, bindings, inputs))
            .clone();
        let pipeline = compile(ctx, key.label(), &compiled, &pl).map_err(|e| {
            Error::new(e.code, format!("{}\n--- generated WGSL ---\n{}", e.message, numbered(&compiled)))
        })?;
        let p = Arc::new(Pipe { pipeline, layout: bgl, source: src.into(), file_name });
        inner.pipelines.insert(key.clone(), p.clone());
        Ok(p)
    }

    pub fn sources(&self) -> Vec<(String, String)> {
        self.inner
            .lock()
            .map(|i| i.pipelines.values().map(|p| (p.file_name.clone(), p.source.to_string())).collect())
            .unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map(|i| i.pipelines.len()).unwrap_or(0)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn compile(ctx: &Ctx, label: &str, src: &str, pl: &wgpu::PipelineLayout) -> Result<wgpu::ComputePipeline> {
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let desc = wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) };
    // SAFETY: the kernels never index out of bounds (sizes come from validated launches;
    // gathers clamp and report through the status buffer), so runtime bounds checks only
    // cost time. Browsers always add their own checks.
    let module = unsafe { ctx.device.create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked()) };
    let pipeline = create_pipeline(ctx, label, &module, pl);
    if let Some(e) = pollster::block_on(scope.pop()) {
        return Err(Error::new("shader", format!("{label}: {e}")));
    }
    Ok(pipeline)
}

#[cfg(target_arch = "wasm32")]
fn compile(ctx: &Ctx, label: &str, src: &str, pl: &wgpu::PipelineLayout) -> Result<wgpu::ComputePipeline> {
    // errors arrive asynchronously through the uncaptured-error handler
    let desc = wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) };
    let module = ctx.device.create_shader_module(desc);
    Ok(create_pipeline(ctx, label, &module, pl))
}

fn create_pipeline(ctx: &Ctx, label: &str, module: &wgpu::ShaderModule, pl: &wgpu::PipelineLayout) -> wgpu::ComputePipeline {
    ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(pl),
        module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions { constants: &[], zero_initialize_workgroup_memory: false },
        cache: None,
    })
}
