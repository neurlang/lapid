//! `lapid kernels dump` and `lapid bench kernels`: developer tooling over the raw backends.
//!
//! The suite (`suite.rs`) is compiled with `lapid_compiler`, its buffers are materialized on
//! a backend opened here (not through `lapid::Device`, which hides the backend), and the
//! backend's own tooling is used: `dump_kernels` (Vulkan WGSL, CUDA PTX) and
//! `time_launches` (GPU timestamps / events; Metal: per-launch command buffers). The CPU
//! backend has no generated source and is timed by wall clock.

use std::path::Path;
use std::time::Instant;

use lapid::compiler::{Backend, BufferSource, CompileOptions, Launch, Schedule};
use lapid::{DeviceData, Error, Result, bail};
use serde_json::{Value, json};

use crate::suite::{self, Case};

/// A backend opened for tooling.
enum Tool {
    Cpu(lapid_cpu::CpuBackend),
    #[cfg(feature = "vulkan")]
    Vulkan(lapid_vulkan::VulkanBackend),
    #[cfg(feature = "cuda")]
    Cuda(lapid_cuda::CudaBackend),
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    Metal(lapid_metal::MetalBackend),
}

fn unavailable(what: &str) -> Error {
    Error::new("device_unavailable", format!("{what}: lapid was built without this backend"))
}

/// `cpu`, `vulkan[:N]`, `cuda[:N]`, `metal[:N]`; `None` = the backend of `Device::best()`.
fn open(device: Option<&str>) -> Result<Tool> {
    let name = match device {
        Some(d) => d.trim().to_ascii_lowercase(),
        None => lapid::Device::best().name().to_string(),
    };
    let (kind, index) = match name.split_once(':') {
        Some((k, i)) => {
            let i = i.trim().parse::<usize>().map_err(|_| {
                Error::new("invalid_argument", format!("bad device index in {name:?}"))
            })?;
            (k.to_string(), i)
        }
        None => (name.clone(), 0),
    };
    match kind.as_str() {
        "cpu" => Ok(Tool::Cpu(lapid_cpu::CpuBackend::new())),
        "vulkan" => open_vulkan(index),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Tool::Cuda(lapid_cuda::CudaBackend::new(index)?)),
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        "metal" => Ok(Tool::Metal(lapid_metal::MetalBackend::new(index)?)),
        #[cfg(not(feature = "cuda"))]
        "cuda" => Err(unavailable(&name)),
        #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
        "metal" => Err(unavailable(&name)),
        _ => bail!("invalid_argument", "unknown device {name:?} (expected cpu, vulkan[:N], cuda[:N] or metal[:N])"),
    }
}

/// `vulkan:N` indexes non-software adapters, like `lapid::Device::vulkan`.
#[cfg(feature = "vulkan")]
fn open_vulkan(index: usize) -> Result<Tool> {
    let allow_sw = std::env::var("LAPID_VULKAN_ALLOW_SOFTWARE").is_ok_and(|v| v.trim() == "1");
    let usable: Vec<_> = lapid_vulkan::adapters()?.into_iter().filter(|a| allow_sw || !a.is_software).collect();
    let a = usable.get(index).ok_or_else(|| {
        Error::new("device_unavailable", format!("vulkan:{index}: only {} usable adapter(s)", usable.len()))
    })?;
    Ok(Tool::Vulkan(lapid_vulkan::VulkanBackend::new(a.index)?))
}

#[cfg(not(feature = "vulkan"))]
fn open_vulkan(index: usize) -> Result<Tool> {
    Err(unavailable(&format!("vulkan:{index}")))
}

impl Tool {
    fn backend(&self) -> &dyn Backend {
        match self {
            Tool::Cpu(b) => b,
            #[cfg(feature = "vulkan")]
            Tool::Vulkan(b) => b,
            #[cfg(feature = "cuda")]
            Tool::Cuda(b) => b,
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            Tool::Metal(b) => b,
        }
    }

    /// Generated kernel source written to `dir`; `None` when the backend has none to dump.
    fn dump(&self, dir: &Path) -> Result<Option<usize>> {
        match self {
            #[cfg(feature = "vulkan")]
            Tool::Vulkan(b) => b.dump_kernels(dir).map(Some),
            #[cfg(feature = "cuda")]
            Tool::Cuda(b) => b.dump_kernels(dir).map(Some),
            _ => {
                let _ = dir;
                Ok(None)
            }
        }
    }

    /// Per-launch device time in ms (GPU timers), or the wall time of the whole run on CPU.
    fn time(&self, launches: &[Launch], bufs: &[DeviceData]) -> Result<f64> {
        let per_launch: Vec<f64> = match self {
            #[cfg(feature = "vulkan")]
            Tool::Vulkan(b) => b.time_launches(launches, bufs)?,
            #[cfg(feature = "cuda")]
            Tool::Cuda(b) => b.time_launches(launches, bufs)?,
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            Tool::Metal(b) => b.time_launches(launches, bufs)?,
            _ => {
                let t = Instant::now();
                self.backend().execute(launches, bufs)?;
                return Ok(t.elapsed().as_secs_f64() * 1e3);
            }
        };
        Ok(per_launch.iter().sum())
    }
}

/// Resolve every schedule buffer on `be`: host leaves uploaded, temps allocated per slot.
fn materialize(be: &dyn Backend, s: &Schedule) -> Result<Vec<DeviceData>> {
    let mut slots: Vec<Option<DeviceData>> = vec![None; s.slot_bytes.len()];
    let mut out = Vec::with_capacity(s.buffers.len());
    for b in &s.buffers {
        let buf = match &b.source {
            BufferSource::Host(h) => be.upload(h.bytes())?,
            BufferSource::Device(d) => d.clone(),
            BufferSource::Output => be.alloc(b.bytes.max(4))?,
            BufferSource::Temp { slot } => {
                let bytes = *s.slot_bytes.get(*slot).ok_or_else(|| Error::new("internal", "temp slot out of range"))?;
                match slots.get_mut(*slot) {
                    Some(Some(d)) => d.clone(),
                    Some(entry) => entry.insert(be.alloc(bytes.max(4))?).clone(),
                    None => bail!("internal", "temp slot {slot} out of range"),
                }
            }
        };
        out.push(buf);
    }
    Ok(out)
}

fn describe(be: &dyn Backend) -> String {
    let name = be.device_name();
    if name.starts_with(be.name()) { name } else { format!("{}: {name}", be.name()) }
}

struct Prepared {
    case: Case,
    schedule: Schedule,
    bufs: Vec<DeviceData>,
}

fn prepare(tool: &Tool, case: Case) -> Result<Prepared> {
    let schedule = lapid::compiler::compile(&case.outputs, &CompileOptions::default())
        .map_err(|e| Error::new(e.code, format!("{}: {}", case.name, e.message)))?;
    let be = tool.backend();
    let bufs = materialize(be, &schedule)?;
    be.prepare(&schedule.launches)?;
    be.execute(&schedule.launches, &bufs)?;
    Ok(Prepared { case, schedule, bufs })
}

/// Compile and run the suite on `device`, then write every generated kernel (WGSL / PTX)
/// and each case's schedule (`<case>.schedule.txt`) to `dir`.
pub fn dump(dir: &Path, device: Option<&str>, filter: Option<&str>) -> Result<Value> {
    let tool = open(device)?;
    std::fs::create_dir_all(dir).map_err(|e| Error::new("io", format!("create {}: {e}", dir.display())))?;
    let mut schedules = Vec::new();
    for case in suite::cases(filter)? {
        let p = prepare(&tool, case)?;
        let file = format!("{}.schedule.txt", p.case.name.replace([' ', '/'], "_"));
        std::fs::write(dir.join(&file), lapid::compiler::dump(&p.schedule))
            .map_err(|e| Error::new("io", format!("write {file}: {e}")))?;
        schedules.push(file);
    }
    let kernels = tool.dump(dir)?;
    Ok(json!({
        "device": describe(tool.backend()),
        "dir": dir.display().to_string(),
        "kernels": kernels,
        "kernel_source": if kernels.is_some() { "dumped" } else { "none (backend has no generated source)" },
        "schedules": schedules,
    }))
}

/// Time every suite case: one warmup, then `runs` measured runs (median / min).
pub fn bench(device: Option<&str>, filter: Option<&str>, runs: usize) -> Result<Value> {
    let tool = open(device)?;
    let runs = runs.max(1);
    let mut rows = Vec::new();
    for case in suite::cases(filter)? {
        let p = prepare(&tool, case)?;
        let launches = &p.schedule.launches;
        tool.time(launches, &p.bufs)?;
        let mut t: Vec<f64> = (0..runs).map(|_| tool.time(launches, &p.bufs)).collect::<Result<_>>()?;
        t.sort_by(f64::total_cmp);
        let med = t[t.len() / 2];
        let per_s = |x: f64| if med > 0.0 { x / (med * 1e-3) } else { 0.0 };
        rows.push(json!({
            "case": p.case.name,
            "launches": launches.len(),
            "kernels": launches.iter().map(|l| l.kernel.kind()).collect::<Vec<_>>().join(" + "),
            "median_ms": med,
            "min_ms": t[0],
            "tflops": per_s(p.case.flops) / 1e12,
            "gbps": per_s(p.case.bytes) / 1e9,
        }));
    }
    Ok(json!({
        "device": describe(tool.backend()),
        "timing": if matches!(tool, Tool::Cpu(_)) { "wall clock" } else { "device timers" },
        "runs": runs,
        "cases": rows,
    }))
}
