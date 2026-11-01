// lapid-web JSPI helpers (imported by the wasm-bindgen glue as a snippet).

let promisingRun = null;

/** JavaScript Promise Integration available (Chrome 137+, ...). */
export function hasJspi() {
  return typeof WebAssembly.Suspending === "function" && typeof WebAssembly.promising === "function";
}

/** WebGPU available. */
export function hasWebGpu() {
  return typeof navigator !== "undefined" && !!navigator.gpu;
}

/** Run lapid job `job` on a suspendable stack: `WebAssembly.promising(lapid_web_run)(job)`. */
export function runJob(exports, job) {
  if (promisingRun === null) {
    promisingRun = WebAssembly.promising(exports.lapid_web_run);
  }
  return promisingRun(job);
}
