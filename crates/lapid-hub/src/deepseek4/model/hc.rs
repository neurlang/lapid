//! Manifold-constrained hyper-connections (mHC): the residual stream is `hc` parallel copies
//! `x [n, hc, D]`. Around each block (llama.cpp `DSV4_HC_PRE` / `DSV4_HC_COMB` / `DSV4_HC_POST`,
//! here compositions of lapid primitives):
//!
//! ```text
//! mix  = rmsnorm(flatten(x)) @ fn^T                      [n, (2 + hc) hc]
//! pre  = sigmoid(mix[:hc] s0 + b[:hc]) + eps             collapse weights
//! post = 2 sigmoid(mix[hc:2hc] s1 + b[hc:2hc])           expand weights
//! comb = sinkhorn(softmax_row(mix[2hc:] s2 + b[2hc:]) + eps)   [n, src, dst], doubly stochastic
//! block input  = sum_i pre[i] x[i]
//! block output = post[j] y + sum_i comb[i, j] x[i]       for every stream j
//! ```
//! The final head collapses the streams with `pre = sigmoid(mix s + b) + eps` only.

use lapid::{DType, Result, Tensor};
use lapid_nn::Loader;

pub(crate) struct HyperConn {
    /// `[mix, hc * D]` f32.
    func: Tensor,
    /// `[mix]` f32.
    base: Tensor,
    scale: [f32; 3],
    hc: usize,
    iters: usize,
    eps: f32,
    norm_eps: f32,
    ones: Tensor,
}

pub(crate) struct HcPre {
    pub collapsed: Tensor,
    pub post: Tensor,
    pub comb: Tensor,
}

/// Host copy of a small f32 parameter.
pub(crate) fn host_f32(l: &Loader<'_>, name: &str) -> Result<Vec<f32>> {
    let t = l.tensor_f32(name)?;
    l.device.to_vec_f32(&t)
}

fn param(l: &Loader<'_>, name: &str, shape: &[usize]) -> Result<Tensor> {
    let t = l.tensor_f32(name)?;
    l.expect_shape(&t, name, shape)?;
    Ok(t)
}

fn mixes(x: &Tensor, func: &Tensor, ones: &Tensor, norm_eps: f32) -> Result<Tensor> {
    let (n, hc, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    lapid_nn::RmsNorm::new(ones.clone(), norm_eps).forward(&x.reshape(&[n, hc * d])?)?.matmul_t(func)
}

impl HyperConn {
    /// `{p}_fn`, `{p}_base`, `{p}_scale` (e.g. `layers.3.hc_attn`).
    pub fn load(l: &Loader<'_>, p: &str, hc: usize, d: usize, iters: usize, eps: f32, norm_eps: f32) -> Result<HyperConn> {
        let mix = (2 + hc) * hc;
        let s = host_f32(l, &format!("{p}_scale"))?;
        if s.len() != 3 {
            return Err(lapid::Error::new("shape_mismatch", format!("{p}_scale: expected 3 values, got {}", s.len())));
        }
        Ok(HyperConn {
            func: param(l, &format!("{p}_fn"), &[mix, hc * d])?,
            base: param(l, &format!("{p}_base"), &[mix])?,
            scale: [s[0], s[1], s[2]],
            hc,
            iters,
            eps,
            norm_eps,
            ones: l.device.upload(&Tensor::full(1.0, &[hc * d], DType::F32))?,
        })
    }

    pub fn pre(&self, x: &Tensor) -> Result<HcPre> {
        let (n, hc) = (x.shape()[0], self.hc);
        let m = mixes(x, &self.func, &self.ones, self.norm_eps)?; // [n, mix]
        let eps = f64::from(self.eps);
        let part = |i: usize, a: usize, b: usize| -> Result<Tensor> {
            m.slice(1, a, b)?.mul_scalar(f64::from(self.scale[i]))?.add(&self.base.slice(0, a, b)?)
        };
        let pre = lapid_nn::ops::sigmoid(&part(0, 0, hc)?)?.add_scalar(eps)?; // [n, hc]
        let post = lapid_nn::ops::sigmoid(&part(1, hc, 2 * hc)?)?.mul_scalar(2.0)?;
        // comb[n, src, dst]: softmax over dst, then Sinkhorn (column = over src, row = over dst).
        let mut comb = part(2, 2 * hc, 2 * hc + hc * hc)?.reshape(&[n, hc, hc])?.softmax()?.add_scalar(eps)?;
        let norm_src = |c: &Tensor| -> Result<Tensor> { c.div(&c.sum(1, true)?.add_scalar(eps)?) };
        let norm_dst = |c: &Tensor| -> Result<Tensor> { c.div(&c.sum(2, true)?.add_scalar(eps)?) };
        comb = norm_src(&comb)?;
        for _ in 1..self.iters {
            comb = norm_src(&norm_dst(&comb)?)?;
        }
        let collapsed = pre.unsqueeze(2)?.mul(x)?.sum(1, false)?; // [n, D]
        Ok(HcPre { collapsed, post, comb })
    }

    /// `y [n, D]`, `residual [n, hc, D]` → `[n, hc, D]`.
    pub fn post(&self, y: &Tensor, residual: &Tensor, post: &Tensor, comb: &Tensor) -> Result<Tensor> {
        let expand = post.unsqueeze(2)?.mul(&y.unsqueeze(1)?)?; // [n, hc, D]
        let mixed = comb.unsqueeze(3)?.mul(&residual.unsqueeze(2)?)?.sum(1, false)?; // [n, dst, D]
        expand.add(&mixed)
    }
}

pub(crate) struct HyperHead {
    func: Tensor,
    base: Tensor,
    scale: f32,
    eps: f32,
    norm_eps: f32,
    ones: Tensor,
}

impl HyperHead {
    pub fn load(l: &Loader<'_>, hc: usize, d: usize, eps: f32, norm_eps: f32) -> Result<HyperHead> {
        let s = host_f32(l, "hc_head_scale")?;
        Ok(HyperHead {
            func: param(l, "hc_head_fn", &[hc, hc * d])?,
            base: param(l, "hc_head_base", &[hc])?,
            scale: s.first().copied().unwrap_or(1.0),
            eps,
            norm_eps,
            ones: l.device.upload(&Tensor::full(1.0, &[hc * d], DType::F32))?,
        })
    }

    /// `[n, hc, D]` → `[n, D]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let m = mixes(x, &self.func, &self.ones, self.norm_eps)?;
        let pre = lapid_nn::ops::sigmoid(&m.mul_scalar(f64::from(self.scale))?.add(&self.base)?)?.add_scalar(f64::from(self.eps))?;
        pre.unsqueeze(2)?.mul(x)?.sum(1, false)
    }
}
