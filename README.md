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

## Why Lapid

- **vs ONNX Runtime**: fast on every GPU, not only NVIDIA, and a small Rust library you can change.
- **vs llama.cpp**: one build for all GPUs, no SDKs to install.
- **vs Candle, Burn, Tract**: more GPUs, made for inference, and runs on both CPU and GPU.

## Quick start

```sh
cargo add lapid lapid-hub
```

Transcribe audio with Parakeet TDT:

```rust
use lapid::Device;
use lapid_hub::{Parakeet, parakeet::TranscribeOptions};

let model = Parakeet::from_hub("nvidia/parakeet-tdt-0.6b-v3", "int8", &Device::best())?;
let out = model.transcribe_file("audio.wav".as_ref(), &TranscribeOptions::default())?;
println!("{}", out.text);
```

Or use the tensor engine directly:

```rust
use lapid::{Device, Tensor};

let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
let y = x.matmul(&x)?.gelu()?.softmax()?;          // builds a graph
println!("{:?}", Device::best().to_vec_f32(&y)?);  // compiles, fuses, runs
```

<div align="center">

[Examples](crates/lapid-hub/examples/) · [Docs](docs/)

</div>
