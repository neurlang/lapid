//! Parakeet TDT network (HF `ParakeetForTDT` layout) built as lazy lapid graphs:
//! conv subsampling, FastConformer encoder with relative-position attention, LSTM
//! prediction network and the TDT joint. Only Parakeet's wiring lives here; the
//! model-local building blocks (2-D convs, rel-pos attention, LSTM) are candidates for
//! `lapid-nn` once a second model needs them.

pub(crate) mod attention;
pub(crate) mod decoder;
pub(crate) mod encoder;
pub(crate) mod subsampling;

use lapid::{Error, Result, Tensor};
use lapid_nn::{Linear, Loader};

/// LayerNorm epsilon of every Parakeet LayerNorm (PyTorch default).
pub(crate) const LN_EPS: f32 = 1e-5;

/// `{prefix}.weight` with `{prefix}.bias` when the file has one.
pub(crate) fn linear(l: &Loader<'_>, prefix: &str) -> Result<Linear> {
    let bias = l.contains(&format!("{prefix}.bias"));
    l.linear(prefix, bias)
}

/// Linear with a shape check of the weight.
pub(crate) fn linear_checked(l: &Loader<'_>, prefix: &str, out: usize, inp: usize) -> Result<Linear> {
    let lin = linear(l, prefix)?;
    l.expect_shape(&lin.weight, &format!("{prefix}.weight"), &[out, inp])?;
    Ok(lin)
}

/// `x[.., start + i*stride, ..]` for `i in 0..n` along `axis` (needs `start + n*stride <=
/// len`): a window slice viewed as [n, stride] and its first column.
pub(crate) fn strided(x: &Tensor, axis: usize, start: usize, n: usize, stride: usize) -> Result<Tensor> {
    let w = x.slice(axis as isize, start, start + n * stride)?;
    if stride == 1 {
        return Ok(w);
    }
    let mut shape = w.shape().to_vec();
    shape[axis] = n;
    shape.insert(axis + 1, stride);
    w.reshape(&shape)?.slice(axis as isize + 1, 0, 1)?.squeeze(axis as isize + 1)
}

/// Per-channel weights of a depthwise kernel: `w` [C, K] (realized f32) -> K tensors [C].
pub(crate) fn kernel_taps(w: &Tensor) -> Result<Vec<Tensor>> {
    let k = w.shape().get(1).copied().unwrap_or(0);
    (0..k).map(|t| w.slice(1, t, t + 1)?.squeeze(1)).collect()
}

/// Σ_t taps[t] * weights[t] (+ bias), broadcasting [C] weights over the last axis. Kept as
/// one elementwise chain so the compiler fuses it into a single kernel.
pub(crate) fn weighted_sum(taps: &[Tensor], weights: &[Tensor], bias: Option<&Tensor>) -> Result<Tensor> {
    let mut acc: Option<Tensor> = None;
    for (x, w) in taps.iter().zip(weights) {
        let term = x.mul(w)?;
        acc = Some(match acc {
            Some(a) => a.add(&term)?,
            None => term,
        });
    }
    let acc = acc.ok_or_else(|| Error::new("internal", "weighted sum of zero taps"))?;
    match bias {
        Some(b) => acc.add(b),
        None => Ok(acc),
    }
}
