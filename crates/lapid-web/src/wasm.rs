//! wasm-bindgen exports and the JSPI bridge.
//!
//! - `lapid_jspi.lapid_yield` (raw import, wrapped in `WebAssembly.Suspending` by
//!   `js/lapid_jspi.js`): awaits one event-loop turn. Registered as the lapid-webgpu wait
//!   hook, so a WebGPU readback suspends the wasm stack until `mapAsync` resolves.
//! - `lapid_web_run(job)` (raw export): runs a queued job. JS calls it through
//!   `WebAssembly.promising`, which gives the job its own suspendable stack.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use lapid::Device;
use js_sys::Promise;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

#[link(wasm_import_module = "lapid_jspi")]
unsafe extern "C" {
    /// Suspends until the next event-loop task (JSPI). Only valid inside `lapid_web_run`.
    fn lapid_yield();
}

#[wasm_bindgen(module = "/js/lapid_jspi.js")]
extern "C" {
    /// `WebAssembly.promising(exports.lapid_web_run)(job)`.
    #[wasm_bindgen(js_name = runJob)]
    fn run_job(exports: &JsValue, job: u32) -> Promise;
    /// JSPI (`WebAssembly.Suspending` + `WebAssembly.promising`) available.
    #[wasm_bindgen(js_name = hasJspi)]
    fn has_jspi() -> bool;
    /// `navigator.gpu` exists.
    #[wasm_bindgen(js_name = hasWebGpu)]
    fn has_webgpu() -> bool;
}

type Job = Box<dyn FnOnce()>;

thread_local! {
    static JOBS: RefCell<HashMap<u32, Job>> = RefCell::new(HashMap::new());
    static NEXT_JOB: Cell<u32> = const { Cell::new(1) };
    /// lapid is not re-entrant while a job is suspended: one job at a time.
    static BUSY: Cell<bool> = const { Cell::new(false) };
}

fn wait_hook() {
    // SAFETY: plain import call; only reached from inside `lapid_web_run` (a promising stack).
    unsafe { lapid_yield() }
}

/// Run a queued job (called by `runJob` through `WebAssembly.promising`).
#[unsafe(no_mangle)]
pub extern "C" fn lapid_web_run(job: u32) {
    let f = JOBS.with(|j| j.borrow_mut().remove(&job));
    if let Some(f) = f {
        f();
    }
}

/// Run `f` on a suspendable stack when `jspi` (WebGPU), else directly (CPU).
async fn run<T: 'static>(jspi: bool, f: impl FnOnce() -> T + 'static) -> Result<T, JsError> {
    if BUSY.with(|b| b.replace(true)) {
        return Err(JsError::new("lapid is busy with another call; await it first"));
    }
    let out = run_inner(jspi, f).await;
    BUSY.with(|b| b.set(false));
    out
}

async fn run_inner<T: 'static>(jspi: bool, f: impl FnOnce() -> T + 'static) -> Result<T, JsError> {
    if !jspi {
        return Ok(f());
    }
    let slot: Rc<RefCell<Option<T>>> = Rc::new(RefCell::new(None));
    let s2 = slot.clone();
    let id = NEXT_JOB.with(|n| {
        let id = n.get();
        n.set(id.wrapping_add(1).max(1));
        id
    });
    JOBS.with(|j| j.borrow_mut().insert(id, Box::new(move || *s2.borrow_mut() = Some(f()))));
    let r = JsFuture::from(run_job(&wasm_bindgen::exports(), id)).await;
    JOBS.with(|j| j.borrow_mut().remove(&id));
    if let Err(e) = r {
        return Err(JsError::new(&format!("lapid job failed: {}", js_err_string(&e))));
    }
    slot.borrow_mut().take().ok_or_else(|| JsError::new("lapid job did not complete"))
}

fn js_err_string(e: &JsValue) -> String {
    e.as_string()
        .or_else(|| js_sys::Reflect::get(e, &"message".into()).ok().and_then(|m| m.as_string()))
        .unwrap_or_else(|| format!("{e:?}"))
}

fn err(e: lapid::Error) -> JsError {
    JsError::new(&format!("{}: {}", e.code, e.message))
}

fn install_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            web_sys::console::error_1(&format!("lapid panicked: {info}").into());
        }));
    });
}

/// Whether this browser can run lapid on the GPU (WebGPU + JSPI).
#[wasm_bindgen(js_name = gpuSupported)]
pub fn gpu_supported() -> bool {
    has_webgpu() && has_jspi()
}

/// Open the device for `name` ("auto" | "webgpu" | "cpu").
async fn open_device(name: &str) -> Result<Device, JsError> {
    match name {
        "cpu" => Ok(Device::cpu()),
        "webgpu" | "gpu" => {
            if !has_webgpu() {
                return Err(JsError::new("WebGPU is not available in this browser (navigator.gpu missing)"));
            }
            if !has_jspi() {
                return Err(JsError::new(
                    "WebGPU needs JavaScript Promise Integration (WebAssembly.Suspending / promising), \
                     e.g. Chrome 137+; use device \"cpu\"",
                ));
            }
            Device::webgpu_async(0).await.map_err(err)
        }
        "auto" | "" => {
            if gpu_supported() {
                match Device::webgpu_async(0).await {
                    Ok(d) => return Ok(d),
                    Err(e) => web_sys::console::warn_1(&format!("lapid: WebGPU unavailable ({e}); using the CPU").into()),
                }
            }
            Ok(Device::cpu())
        }
        other => Err(JsError::new(&format!("unknown device '{other}' (expected auto, webgpu or cpu)"))),
    }
}

/// Whisper speech recognition.
#[wasm_bindgen]
pub struct Whisper {
    model: Rc<lapid_hub::Whisper>,
    device: String,
}

#[wasm_bindgen]
impl Whisper {
    /// Load a `.lapid` Whisper model (`Uint8Array`) on `device` ("auto" by default:
    /// WebGPU when available, else CPU).
    pub async fn load(bytes: Vec<u8>, device: Option<String>) -> Result<Whisper, JsError> {
        install_panic_hook();
        lapid_webgpu::set_wait_hook(wait_hook);
        let name = device.unwrap_or_default().trim().to_ascii_lowercase();
        let dev = open_device(&name).await?;
        let jspi = dev.name() == "webgpu";
        let description = dev.description();
        let model = run(jspi, move || crate::load_whisper(bytes, &dev)).await?.map_err(err)?;
        Ok(Whisper { model: Rc::new(model), device: description })
    }

    /// The device the model runs on, e.g. "webgpu: NVIDIA ..." or "cpu: ...".
    #[wasm_bindgen(getter)]
    pub fn device(&self) -> String {
        self.device.clone()
    }

    /// Transcribe 16 kHz mono samples (`Float32Array`). `language`: code or "auto";
    /// `timestamps` (default true). Resolves to `{text, language, segments: [{start, end,
    /// text, tokens}]}`.
    pub fn transcribe(&self, samples: Vec<f32>, language: Option<String>, timestamps: Option<bool>) -> Promise {
        let model = self.model.clone();
        let jspi = model.device().name() == "webgpu";
        wasm_bindgen_futures::future_to_promise(async move {
            let opts = crate::options(language, timestamps);
            let json = run(jspi, move || crate::transcribe_json(&model, &samples, &opts)).await?.map_err(err)?;
            js_sys::JSON::parse(&json).map_err(|e| JsError::new(&js_err_string(&e)).into())
        })
    }
}
