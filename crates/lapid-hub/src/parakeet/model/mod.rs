//! Parakeet TDT network (HF `ParakeetForTDT` layout) built as lazy lapid graphs: conv
//! subsampling, FastConformer encoder (conformer blocks with relative-position attention,
//! `lapid_nn::speech`), LSTM prediction network and the TDT joint. Only Parakeet's wiring
//! and weight names live here.

pub(crate) mod decoder;
pub(crate) mod encoder;
pub(crate) mod subsampling;

use lapid::Result;
use lapid_nn::{Linear, Loader};

/// LayerNorm epsilon of every Parakeet LayerNorm (PyTorch default).
pub(crate) const LN_EPS: f32 = 1e-5;

/// `{prefix}.weight` (with `{prefix}.bias` when the file has one), shape-checked.
pub(crate) fn linear_checked(l: &Loader<'_>, prefix: &str, out: usize, inp: usize) -> Result<Linear> {
    l.linear_checked(prefix, out, inp, l.contains(&format!("{prefix}.bias")))
}
