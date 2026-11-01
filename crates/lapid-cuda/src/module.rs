//! PTX → device code through the driver's JIT (`cuModuleLoadDataEx`), cached per kernel key.
//!
//! Sources are generated without a header; `header` adds `.version` / `.target` for the
//! device (`.target sm_XY` at the highest ISA the driver knows). If the JIT rejects that
//! (old driver tables, unknown target), the kernel is retried at a conservative target
//! (`sm_80` / ISA 7.0, or `sm_70` / ISA 6.3) and that fallback sticks for the device.
//! `LAPID_DEBUG=1` writes every kernel to `$LAPID_KERNEL_DIR` (default `<tmp>/lapid-kernels`).

use std::collections::HashMap;
use std::ffi::{CString, c_int, c_void};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use lapid_tensor::{Error, Result};

use crate::cache::DiskCache;
use crate::codegen::{self, Key};
use crate::context::{Caps, Ctx, min_isa_for_sm};
use crate::driver::{self, CUfunction, CUmodule};

/// A compiled kernel (one module with entry `lapid_main`).
pub(crate) struct CuKernel {
    module: CUmodule,
    pub func: CUfunction,
    /// Profile label.
    pub name: String,
    pub source: String,
}

// SAFETY: module / function handles are plain driver handles usable from any thread with
// the context current.
unsafe impl Send for CuKernel {}
// SAFETY: immutable after creation.
unsafe impl Sync for CuKernel {}

#[derive(Default)]
pub(crate) struct Modules {
    map: Mutex<HashMap<Key, Arc<CuKernel>>>,
    fallback: AtomicBool,
    disk: OnceLock<DiskCache>,
}

pub(crate) fn debug_enabled() -> bool {
    std::env::var("LAPID_DEBUG").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// `.version` / `.target` / `.address_size` lines.
pub(crate) fn header(caps: &Caps, fallback: bool) -> String {
    let (isa, sm) = if !fallback && min_isa_for_sm(caps.sm) <= caps.ptx_isa {
        (caps.ptx_isa, caps.sm)
    } else if caps.sm >= 80 {
        (70, 80)
    } else {
        (63, 70)
    };
    format!(
        ".version {}.{}\n.target sm_{sm}\n.address_size 64\n\n",
        isa / 10,
        isa % 10
    )
}

/// Source with line numbers (for error messages).
pub(crate) fn numbered(src: &str) -> String {
    let mut s = String::with_capacity(src.len() + src.len() / 8);
    for (i, l) in src.lines().enumerate() {
        let _ = writeln!(s, "{:5} | {l}", i + 1);
    }
    s
}

impl Modules {
    pub fn len(&self) -> usize {
        self.map.lock().map(|m| m.len()).unwrap_or(0)
    }

    pub fn get(&self, ctx: &Ctx, key: &Key) -> Result<Arc<CuKernel>> {
        let mut g = self
            .map
            .lock()
            .map_err(|_| Error::new("cuda", "module cache poisoned"))?;
        if let Some(k) = g.get(key) {
            return Ok(k.clone());
        }
        let body = codegen::source(key, &ctx.caps);
        let k = Arc::new(self.compile(ctx, &key.name(), &body)?);
        g.insert(key.clone(), k.clone());
        Ok(k)
    }

    fn compile(&self, ctx: &Ctx, name: &str, body: &str) -> Result<CuKernel> {
        let fb = self.fallback.load(Ordering::Relaxed);
        let src = format!("{}{body}", header(&ctx.caps, fb));
        match self.build(ctx, &src) {
            Ok((module, func)) => Ok(CuKernel {
                module,
                func,
                name: name.to_string(),
                source: dump_debug(name, src),
            }),
            Err(first) if !fb => {
                let src2 = format!("{}{body}", header(&ctx.caps, true));
                match self.build(ctx, &src2) {
                    Ok((module, func)) => {
                        self.fallback.store(true, Ordering::Relaxed);
                        Ok(CuKernel {
                            module,
                            func,
                            name: name.to_string(),
                            source: dump_debug(name, src2),
                        })
                    }
                    Err(_) => Err(first),
                }
            }
            Err(e) => Err(e),
        }
    }

    /// Compile `src`: cached cubin when present, else JIT through `cuLink*` and cache the
    /// cubin, else (no link API / link failure) the plain JIT path, whose errors carry
    /// the log and numbered source.
    fn build(&self, ctx: &Ctx, src: &str) -> Result<(CUmodule, CUfunction)> {
        let disk = self.disk.get_or_init(|| {
            let ver = ctx.drv.version().unwrap_or((0, 0));
            DiskCache::new(&ctx.name, ctx.caps.sm, ver)
        });
        if disk.enabled() {
            if let Some(bin) = disk.get(src)
                && let Ok(m) = load_image(ctx, &bin)
            {
                return Ok(m);
            }
            if let Some(bin) = link_ptx(ctx, src)
                && let Ok(m) = load_image(ctx, &bin)
            {
                disk.put(src, &bin);
                return Ok(m);
            }
        }
        jit(ctx, src)
    }

    /// Write every compiled kernel to `dir`; returns the number of files.
    pub fn dump(&self, dir: &std::path::Path) -> Result<usize> {
        std::fs::create_dir_all(dir)?;
        let g = self
            .map
            .lock()
            .map_err(|_| Error::new("cuda", "module cache poisoned"))?;
        for (i, k) in g.values().enumerate() {
            std::fs::write(
                dir.join(format!("{:03}-{}.ptx", i, file_name(&k.name))),
                &k.source,
            )?;
        }
        Ok(g.len())
    }

    /// Unload all modules (context current, device idle).
    pub fn unload_all(&self, ctx: &Ctx) {
        if let Ok(mut g) = self.map.lock() {
            for (_, k) in g.drain() {
                // SAFETY: no launch of this module is pending.
                unsafe { (ctx.drv.cuModuleUnload)(k.module) };
            }
        }
    }
}

fn file_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

fn dump_debug(name: &str, src: String) -> String {
    if debug_enabled() {
        let dir = std::env::var_os("LAPID_KERNEL_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("lapid-kernels"));
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&src, &mut h);
        let path = dir.join(format!(
            "cuda-{}-{:016x}.ptx",
            file_name(name),
            std::hash::Hasher::finish(&h)
        ));
        if std::fs::create_dir_all(&dir)
            .and_then(|_| std::fs::write(&path, &src))
            .is_err()
        {
            eprintln!("[lapid-cuda] could not write {}", path.display());
        }
    }
    src
}

const LOG_BYTES: usize = 32 << 10;

/// JIT `src` to a cubin with the driver linker; `None` on any failure (the caller falls
/// back to `jit`, which reports errors with the log).
fn link_ptx(ctx: &Ctx, src: &str) -> Option<Vec<u8>> {
    let drv = ctx.drv;
    let l = drv.link.as_ref()?;
    ctx.bind().ok()?;
    let image = CString::new(src).ok()?;
    let mut opts: Vec<c_int> = vec![driver::JIT_OPTIMIZATION_LEVEL];
    let mut vals: Vec<*mut c_void> = vec![4usize as *mut c_void];
    let mut state: driver::CUlinkState = std::ptr::null_mut();
    // SAFETY: options / values arrays have equal length and outlive the link state.
    let r = unsafe { (l.cuLinkCreate_v2)(1, opts.as_mut_ptr(), vals.as_mut_ptr(), &mut state) };
    if r != driver::CUDA_SUCCESS {
        return None;
    }
    let bytes = image.as_bytes_with_nul();
    // SAFETY: `state` is live; the image (NUL-terminated PTX) outlives the call.
    let mut out = None;
    unsafe {
        let r = (l.cuLinkAddData_v2)(
            state,
            driver::JIT_INPUT_PTX,
            bytes.as_ptr() as *mut c_void,
            bytes.len(),
            c"lapid".as_ptr(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        if r == driver::CUDA_SUCCESS {
            let mut cubin: *mut c_void = std::ptr::null_mut();
            let mut size = 0usize;
            if (l.cuLinkComplete)(state, &mut cubin, &mut size) == driver::CUDA_SUCCESS
                && !cubin.is_null()
                && size > 0
            {
                // the image is owned by the link state: copy before destroying it
                out = Some(std::slice::from_raw_parts(cubin as *const u8, size).to_vec());
            }
        }
        (l.cuLinkDestroy)(state);
    }
    out
}

/// Load a cubin image and resolve `lapid_main`.
fn load_image(ctx: &Ctx, image: &[u8]) -> Result<(CUmodule, CUfunction)> {
    let drv = ctx.drv;
    ctx.bind()?;
    let mut module: CUmodule = std::ptr::null_mut();
    // SAFETY: `image` is a complete cubin (validated by the driver); no options.
    let r = unsafe {
        (drv.cuModuleLoadDataEx)(
            &mut module,
            image.as_ptr() as *const c_void,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    drv.check("cuModuleLoadData(cubin)", r)?;
    entry(ctx, module)
}

/// `lapid_main` of a freshly loaded module (unloads it on failure).
fn entry(ctx: &Ctx, module: CUmodule) -> Result<(CUmodule, CUfunction)> {
    let drv = ctx.drv;
    let mut func: CUfunction = std::ptr::null_mut();
    // SAFETY: module just loaded; the entry name is NUL-terminated.
    let r = unsafe { (drv.cuModuleGetFunction)(&mut func, module, c"lapid_main".as_ptr()) };
    if r != driver::CUDA_SUCCESS {
        // SAFETY: unloading the module we own.
        unsafe { (drv.cuModuleUnload)(module) };
        return Err(Error::new(
            "shader",
            format!("cuModuleGetFunction(lapid_main): {}", drv.error_text(r)),
        ));
    }
    Ok((module, func))
}

fn jit(ctx: &Ctx, src: &str) -> Result<(CUmodule, CUfunction)> {
    let drv = ctx.drv;
    ctx.bind()?;
    let image =
        CString::new(src).map_err(|_| Error::new("shader", "PTX source contains a NUL byte"))?;
    let mut err_log = vec![0u8; LOG_BYTES];
    let mut info_log = vec![0u8; LOG_BYTES];
    let verbose = debug_enabled();
    let mut opts: Vec<c_int> = vec![
        driver::JIT_ERROR_LOG_BUFFER,
        driver::JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        driver::JIT_INFO_LOG_BUFFER,
        driver::JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        driver::JIT_OPTIMIZATION_LEVEL,
    ];
    let mut vals: Vec<*mut c_void> = vec![
        err_log.as_mut_ptr() as *mut c_void,
        LOG_BYTES as *mut c_void,
        info_log.as_mut_ptr() as *mut c_void,
        LOG_BYTES as *mut c_void,
        4usize as *mut c_void,
    ];
    if verbose {
        opts.push(driver::JIT_LOG_VERBOSE);
        vals.push(std::ptr::dangling_mut::<c_void>());
    }
    let mut module: CUmodule = std::ptr::null_mut();
    // SAFETY: options / values arrays have equal length; log buffers outlive the call.
    let r = unsafe {
        (drv.cuModuleLoadDataEx)(
            &mut module,
            image.as_ptr() as *const c_void,
            opts.len() as u32,
            opts.as_mut_ptr(),
            vals.as_mut_ptr(),
        )
    };
    let text = |b: &[u8]| {
        String::from_utf8_lossy(&b[..b.iter().position(|&c| c == 0).unwrap_or(b.len())])
            .trim()
            .to_string()
    };
    if r != driver::CUDA_SUCCESS {
        let log = text(&err_log);
        let msg = format!(
            "PTX JIT failed: {}\n{log}\n--- source ---\n{}",
            drv.error_text(r),
            numbered(src)
        );
        return Err(Error::new("shader", msg));
    }
    if verbose {
        let info = text(&info_log);
        if !info.is_empty() {
            eprintln!("[lapid-cuda] jit: {info}");
        }
    }
    entry(ctx, module)
}
