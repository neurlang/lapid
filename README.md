<div align="center">

# 🔥 Lapid

**Fast ML inference on every device. Pure Rust.**

[![CI](https://github.com/thewh1teagle/lapid/actions/workflows/ci.yml/badge.svg)](https://github.com/thewh1teagle/lapid/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

</div>

## Highlights

- 📦 **Just `cargo add`**: builds anywhere with plain Rust; GPU drivers are picked up at runtime.
- 🛠️ **Hackable**: models and GPU kernels are plain, readable code you can change.
- 🖥️ **Every GPU**: CUDA, Vulkan, Metal and WebGPU, with a fast CPU fallback.
- 🤗 **Hugging Face and GGUF**: load models straight from the Hub or a `.gguf` file.

## Why

- **ONNX Runtime** flies on CUDA, crawls elsewhere, and ships as a heavy C++ black box.
- **llama.cpp** is fast, but needs one build per GPU and a hand-written kernel for every op.
- **Rust frameworks** each pick a corner: candle (few GPUs), burn (training), tract (CPU only).

**lapid** is one small Rust build for every GPU, with kernels written by a compiler and models you can actually read.

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
