// Raw wasm import module "lapid_jspi" (the build script points the glue's import here).
// `lapid_yield` is a WebAssembly.Suspending import: the calling wasm stack (inside
// `WebAssembly.promising`) suspends until the returned promise resolves, i.e. until the
// next event-loop task, which lets WebGPU `mapAsync` callbacks run.

const channel = typeof MessageChannel !== "undefined" ? new MessageChannel() : null;
const waiters = [];
if (channel) {
  channel.port1.onmessage = () => {
    const w = waiters.shift();
    if (w) w();
  };
}

let spins = 0;
let last = 0;

function nextTask() {
  const now = typeof performance !== "undefined" ? performance.now() : Date.now();
  spins = now - last < 1 ? spins + 1 : 0;
  last = now;
  return new Promise((resolve) => {
    // a fast MessageChannel hop first; back off to timers when the GPU is busy for longer
    if (channel && spins < 32) {
      waiters.push(resolve);
      channel.port2.postMessage(0);
    } else {
      setTimeout(resolve, 0);
    }
  });
}

export const lapid_yield =
  typeof WebAssembly.Suspending === "function"
    ? new WebAssembly.Suspending(nextTask)
    : () => {
        throw new Error("lapid: WebAssembly JSPI is unavailable (WebAssembly.Suspending)");
      };
