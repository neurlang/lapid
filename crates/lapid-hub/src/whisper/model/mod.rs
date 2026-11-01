//! Whisper network (HF `WhisperForConditionalGeneration` layout) built as lazy lapid graphs
//! from `lapid_nn` blocks. Only Whisper's weight names and wiring live here.

pub(crate) mod config;
pub(crate) mod decoder;
pub(crate) mod encoder;

use lapid::{Error, Result, Tensor};
use lapid_nn::{Activation, Attention, LayerNorm, Loader, Mlp};

/// LayerNorm epsilon of every Whisper LayerNorm.
pub(crate) const LN_EPS: f32 = 1e-5;

/// First tensor of an eval result.
pub(crate) fn first(v: Vec<Tensor>) -> Result<Tensor> {
    v.into_iter()
        .next()
        .ok_or_else(|| Error::new("backend", "device.eval returned no outputs"))
}

pub(crate) fn layer_norm(l: &Loader<'_>, prefix: &str) -> Result<LayerNorm> {
    l.layer_norm(prefix, LN_EPS)
}

/// Multi-head attention: `q_proj`, `k_proj` (no bias), `v_proj`, `out_proj`.
pub(crate) fn attention(l: &Loader<'_>, prefix: &str, n_state: usize, n_head: usize) -> Result<Attention> {
    let q = l.linear(&format!("{prefix}.q_proj"), true)?;
    l.expect_shape(&q.weight, &format!("{prefix}.q_proj.weight"), &[n_state, n_state])?;
    Attention::new(
        q,
        l.linear(&format!("{prefix}.k_proj"), false)?,
        l.linear(&format!("{prefix}.v_proj"), true)?,
        l.linear(&format!("{prefix}.out_proj"), true)?,
        n_head,
        n_head,
    )
}

/// fc2(gelu(fc1(x))).
pub(crate) fn mlp(l: &Loader<'_>, prefix: &str) -> Result<Mlp> {
    Ok(Mlp::new(l.linear(&format!("{prefix}.fc1"), true)?, l.linear(&format!("{prefix}.fc2"), true)?, Activation::Gelu))
}
