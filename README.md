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

ONNX Runtime is slow outside CUDA and is a heavy C++ dependency that's hard to change. llama.cpp is fast, but every GPU needs its own build and SDK, and every op is a hand-written kernel per backend. Rust options trade one thing for another: candle has limited GPU support, burn is built for training, tract is CPU only.

lapid gives you one small Rust build that runs on every GPU, a compiler that generates and fuses the kernels, and models you can read and change.

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
