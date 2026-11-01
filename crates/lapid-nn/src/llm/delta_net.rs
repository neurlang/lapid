//! Gated DeltaNet token mixer (HF `Qwen3_5GatedDeltaNet`, llama.cpp `build_layer_attn_linear`;
//! Qwen4-Exp uses it with a sigmoid output gate):
//!
//! ```text
//! qkv = silu(conv1d_causal(in_proj_qkv(h)))   (depthwise, streamed window)
//! q, k = l2norm(split(qkv))  (repeated to the v heads, tiled order: v head h reads k head h % Hk)
//! beta = sigmoid(in_proj_b(h)),  g = A * softplus(in_proj_a(h) + dt_bias)
//! o, S = gated_delta_rule(q, k, v, g, beta, S, scale = 1/sqrt(Dk))
//! out = out_proj(rmsnorm(o) * act(in_proj_z(h)))   (act = silu, or sigmoid)
//! ```
//!
//! File conventions (lapid's, = llama.cpp's GGUF values): `A` holds `-exp(A_log)`, v heads are
//! tiled (HF groups them; the converters permute).

use lapid::{DType, Result, Tensor, bail};

use super::{l2norm, softplus};
use crate::layers::{Conv1d, Linear, RmsNorm};
use crate::ops;
use crate::weights::Loader;

/// Shapes of a DeltaNet layer.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DeltaNetDims {
    pub hidden: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub conv_kernel: usize,
    pub eps: f32,
}

impl DeltaNetDims {
    /// Channels of the causal conv (q, k and v concatenated).
    pub fn conv_dim(&self) -> usize {
        2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim
    }
}

pub struct DeltaNet {
    in_qkv: Linear,
    in_z: Linear,
    in_b: Linear,
    in_a: Linear,
    conv: Conv1d,
    /// -exp(A_log) [Hv] f32.
    a: Tensor,
    dt_bias: Tensor,
    norm: RmsNorm,
    out: Linear,
    dims: DeltaNetDims,
    sigmoid_gate: bool,
}

impl DeltaNet {
    /// `{p}.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,conv1d,norm}.weight`, `{p}.A`,
    /// `{p}.dt_bias` (`p` e.g. `model.layers.0.linear_attn`).
    pub fn load(l: &Loader<'_>, p: &str, dims: DeltaNetDims, sigmoid_gate: bool) -> Result<DeltaNet> {
        let DeltaNetDims { hidden: d, v_heads: hv, v_dim: dv, .. } = dims;
        let c = dims.conv_dim();
        let lin = |n: &str, o: usize, i: usize| l.linear_checked(&format!("{p}.{n}"), o, i, false);
        let f32_param = |n: &str| l.tensor_checked(&format!("{p}.{n}"), &[hv])?.cast(DType::F32);
        Ok(DeltaNet {
            in_qkv: lin("in_proj_qkv", c, d)?,
            in_z: lin("in_proj_z", hv * dv, d)?,
            in_b: lin("in_proj_b", hv, d)?,
            in_a: lin("in_proj_a", hv, d)?,
            conv: Conv1d::depthwise(l.tensor_checked(&format!("{p}.conv1d.weight"), &[c, 1, dims.conv_kernel])?, None),
            a: f32_param("A")?,
            dt_bias: f32_param("dt_bias")?,
            norm: l.rms_norm_checked(&format!("{p}.norm"), dv, dims.eps)?,
            out: lin("out_proj", d, hv * dv)?,
            dims,
            sigmoid_gate,
        })
    }

    /// h [B, n, D] -> [B, n, D]; `conv` / `ssm` are the committed states (`None` = zeros) and
    /// are replaced by the new (lazy) ones.
    pub fn forward(&self, h: &Tensor, conv: &mut Option<Tensor>, ssm: &mut Option<Tensor>) -> Result<Tensor> {
        let (b, n) = (h.shape()[0], h.shape()[1]);
        let DeltaNetDims { k_heads: hk, v_heads: hv, k_dim: dk, v_dim: dv, .. } = self.dims;
        let kd = hk * dk;
        let qkv = self.in_qkv.forward(h)?.cast(DType::F32)?; // [B, n, C]
        let c = qkv.shape()[2];
        if c != 2 * kd + hv * dv {
            bail!("shape_mismatch", "deltanet: in_proj_qkv gives {c} channels, expected {}", 2 * kd + hv * dv);
        }
        let mixed = self.conv.forward_stream(&qkv.transpose(1, 2)?, conv)?; // [B, C, n]
        let mixed = ops::silu(&mixed.cast(DType::F32)?)?.transpose(1, 2)?; // [B, n, C]
        let q = l2norm(&mixed.slice(2, 0, kd)?.reshape(&[b, n, hk, dk])?, 1e-6)?;
        let k = l2norm(&mixed.slice(2, kd, 2 * kd)?.reshape(&[b, n, hk, dk])?, 1e-6)?;
        let v = mixed.slice(2, 2 * kd, c)?.reshape(&[b, n, hv, dv])?;
        let tile = |x: Tensor| -> Result<Tensor> {
            if hv == hk {
                return Ok(x);
            }
            x.unsqueeze(2)?.expand(&[b, n, hv / hk, hk, dk])?.reshape(&[b, n, hv, dk])
        };
        let beta = ops::sigmoid(&self.in_b.forward(h)?.cast(DType::F32)?)?; // [B, n, Hv]
        let g = softplus(&self.in_a.forward(h)?.cast(DType::F32)?.add(&self.dt_bias)?)?.mul(&self.a)?; // <= 0
        let state = ssm.take().unwrap_or_else(|| Tensor::zeros(&[b, hv, dk, dv], DType::F32));
        let out = Tensor::gated_delta_net(&tile(q)?, &tile(k)?, &v, &g, &beta, &state, 1.0 / (dk as f32).sqrt())?; // [B, Hv, n + Dk, Dv]
        *ssm = Some(out.slice(2, n, n + dk)?);
        let o = out.slice(2, 0, n)?.permute(&[0, 2, 1, 3])?; // [B, n, Hv, Dv]
        let z = self.in_z.forward(h)?.cast(DType::F32)?.reshape(&[b, n, hv, dv])?;
        let z = if self.sigmoid_gate { ops::sigmoid(&z)? } else { ops::silu(&z)? };
        self.out.forward(&self.norm.forward(&o)?.mul(&z)?.reshape(&[b, n, hv * dv])?)
    }
}
