# lapid — project rules

- **Keep this file general.** Principles only — no file paths, crate names, commands or numbers, so it never goes stale. Specifics belong in plans and specs.
- **Pure Rust.** No C/C++ dependencies; native drivers and system frameworks are loaded at runtime.
- **Generic code stays generic.** Model-specific logic (conversion, tokenizers, metadata, verification wiring) lives with the model in the hub layer, never in core crates.
- **Layers.** Core = tensors, ops, compiler, backends. Hub = ready models. CLI = thin routing; logic lives in the crate that owns it.
- **Contracts first.** Cross-crate interfaces are designed up front and changed deliberately, never as a side effect.
- **One compiler.** Models are written with primitives and composites; backends implement the kernel IR only. No per-model kernels.
- **Readable kernels.** Kernels are real source files with small substitution slots, not code assembled in strings; generated output can always be dumped and diffed.
- **Standard names.** Dtypes and formats follow ONNX-style naming (`fp16`, `int8`, `int4`); details like group size are attributes, not part of the name.
- **Proof, not vibes.** Correctness is checked layer by layer against a reference implementation; speed claims come from reproducible benchmarks against established engines.
- **Fast loops only.** Never wait on something slow when a cheaper check gives the same answer: narrow builds and targeted tests, tiny fixtures, isolated layers and kernels, cached expensive artifacts. Full end-to-end runs are a final check, run in the background.
- **Errors** carry stable machine-readable codes; no panics on fallible library paths.
- **Specs** are versioned with the code and updated in the same change.
