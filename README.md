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

- **ONNX Runtime** is fast on NVIDIA, slow on other GPUs, and big and hard to change.
- **llama.cpp** is fast, but you need a different build for each GPU.
- **Other Rust libraries** each miss something: candle supports few GPUs, burn is made for training, tract runs only on CPU.

**lapid** is one small Rust library that runs fast on every GPU, and you can read and change all of it.

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
