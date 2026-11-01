<div align="center">

# 🔥 Lapid

**Fast ML inference on every device. Pure Rust.**

[![CI](https://github.com/thewh1teagle/lapid/actions/workflows/ci.yml/badge.svg)](https://github.com/thewh1teagle/lapid/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

</div>

## Highlights

- 📦 **No SDKs**: no CUDA toolkit, Vulkan SDK or C++ compiler. Drivers load at runtime.
- 🪶 **Small models**: runs quantized weights (`fp16`, `int8`, `int4`, GGUF) to use less memory.
- 🖥️ **Every GPU**: CUDA, Vulkan, Metal and WebGPU, with a fast CPU fallback. The best one is picked at runtime.
- 🌐 **Runs in the browser**: the same code builds for WebAssembly and WebGPU.

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

<div align="center">

[Examples](crates/lapid-hub/examples/) · [Docs](docs/)

</div>
