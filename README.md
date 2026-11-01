# lapid

**Fast ML inference on every device. Pure Rust.**

lapid is a tensor engine: build a graph from simple ops, and lapid compiles it into fused kernels for whatever hardware is available — from one `cargo add`, with nothing else to install.

## Features

- **Every backend in one build** — CUDA, Vulkan, Metal, WebGPU and SIMD CPU (NEON, AVX2) in a single binary; the best device is picked at runtime.
- **Nothing to install** — no CUDA toolkit, no Vulkan SDK, no shader compilers, no C/C++ toolchain. GPU drivers are loaded at runtime; kernels are generated on the fly and cached.
- **A compiler, not a kernel zoo** — ~25 primitives, automatic fusion (`matmul + bias + activation` is one kernel), zero-copy views, memory planning.
- **Quantized** — `fp16`, `int8`, `int4` and GGUF K-quants (`q4_k`, `q5_k`, `q6_k`), with tensor-core / simdgroup matmuls where the hardware has them.
- **One file per model** — `.lapid`: weights, tokenizer and metadata in a single mmap-able file, zero-copy, also loadable from memory. Imports Hugging Face safetensors and GGUF.
- **Verified** — models are checked layer by layer against their PyTorch originals.
- **Small** — a few MB, no runtime dependencies beyond the OS and the GPU driver.
- **Runs in the browser** — the same code builds for WebAssembly + WebGPU.

## Backends

| Backend | Platforms | Needs to build | Needs to run |
|---|---|---|---|
| CUDA | Linux, Windows (NVIDIA) | nothing | NVIDIA driver |
| Vulkan | Linux, Windows, Android | nothing | Vulkan driver |
| Metal | macOS, iOS | nothing | built into the OS |
| WebGPU | browsers, native | nothing | a WebGPU-capable browser |
| CPU | everywhere (NEON, AVX2) | nothing | nothing |

## Quick start

```sh
cargo add lapid
```

```rust
use lapid::{Device, Tensor};

let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
let y = x.matmul(&x)?.gelu()?.softmax()?;          // builds a graph
println!("{:?}", Device::best().to_vec_f32(&y)?);  // compiles, fuses, runs
```

Pick a device with `Device::from_name("cuda:0")` (or `vulkan`, `metal`, `webgpu`, `cpu`), or set `LAPID_DEVICE`.

Examples: [crates/lapid-hub/examples/](crates/lapid-hub/examples/) · Docs: [docs/](docs/)

## License

MIT or Apache-2.0.
