//! lapid-nn — the shared model-building library for lapid (think `torch.nn` plus the common
//! model utilities every ready model needs).
//!
//! Everything here is built only from lapid's public API (`Tensor`, `Device`, `LapidFile`), so
//! it stays generic: model-specific logic (weight names, decoding rules, conversion) lives
//! with the model in `lapid-hub`.
//!
//! * [`layers`] — `Linear`, `Embedding`, `LayerNorm`, `RmsNorm`, `Conv1d` (strided, dilated,
//!   grouped / depthwise, causal with streaming state).
//! * [`ops`] — activations built from primitives (`silu`, `sigmoid`, `gelu_tanh`, `relu`, `elu`, `glu`).
//! * [`attention`] — multi-head attention with GQA / MQA, causal / sliding-window / custom
//!   masks, cross attention and a [`attention::KvCache`].
//! * [`rope`] — rotary position embeddings (half-split or interleaved, partial rotary,
//!   configurable theta and linear scaling).
//! * [`mlp`] — `Mlp` (`down(act(up x))`) and `GatedMlp` (SwiGLU / GeGLU).
//! * [`weights`] — `Loader` (weights from a `.lapid` file, fake-quant reference precisions).
//! * [`llm`]: decoder-LLM blocks (MoE experts and routing, gated attention, Gated DeltaNet,
//!   hyper-connections, RoPE variants incl. llama3 / YaRN / M-RoPE, LM head).
//! * [`audio`] — WAV loading, resampling, STFT and mel filterbanks.
//! * [`vision`]: images (decode, torchvision-exact resize), patching, 2-D RoPE, patch pooling.
//! * [`tokenizer`] — Hugging Face `tokenizers` wrapper (`tokenizer.json`, byte-level vocabularies).
//! * [`sampling`] / [`generate`] — logits processing, seeded sampling, a generation iterator.
//!
//! Rule of two: a piece lives here only when at least two models need it.

pub mod attention;
pub mod audio;
pub mod generate;
pub mod layers;
pub mod llm;
pub mod mlp;
pub mod ops;
pub mod rope;
pub mod sampling;
pub mod speech;
pub mod tokenizer;
pub mod vision;
pub mod weights;

#[cfg(test)]
pub(crate) mod testutil;

pub use attention::{Attention, KvCache, Mask};
pub use layers::{Conv1d, Embedding, LayerNorm, Linear, RmsNorm};
pub use mlp::{Activation, GatedMlp, Mlp};
pub use rope::{Rope, RopeLayout, RopeScaling, ZeroTailRope};
pub use sampling::{Sampler, SamplingParams, argmax};
pub use tokenizer::Tokenizer;
pub use weights::{Loader, Precision};
