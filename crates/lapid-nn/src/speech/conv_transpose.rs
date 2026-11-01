//! Transposed 1-D convolution (PyTorch `ConvTranspose1d`, dense or depthwise), with the
//! causal streaming state of streaming codecs (Mimi, EnCodec-style SEANet decoders).
//!
//! y[co, t*S + k] += x[ci, t] * w[ci, co, k]. The kernel is zero-padded to `m * S` taps and
//! viewed as `m` chunks of `S`: every input frame t contributes one [Cout, m, S] block (one
//! matmul for all frames), and output block j sums chunk c of frame j - c (overlap-add of
//! `m` shifted slices). No zero insertion, no wasted multiply-adds beyond the padding.
//!
//! Streaming (causal, PyTorch `StreamingConvTranspose1d`): each call returns `T * S`
//! samples; the trailing `(m - 1) * S` partial sums (without bias) are carried to the next
//! call. Concatenated outputs equal the full transposed conv trimmed by `K - S` on the right.

use lapid::{DType, Device, Result, Tensor, bail};

pub struct ConvTranspose1d {
    /// Dense: [Cin, Cout * m * S]; depthwise: [1, C, 1, m * S] (both f32, realized).
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    chunks: usize,
    cout: usize,
    depthwise: bool,
}

impl ConvTranspose1d {
    /// `weight` [Cin, Cout / groups, K] (PyTorch layout); `groups` 1 or Cin (depthwise,
    /// Cout = Cin). Prepares the padded weight on `device`.
    pub fn new(
        weight: &Tensor,
        bias: Option<Tensor>,
        stride: usize,
        groups: usize,
        device: &Device,
    ) -> Result<ConvTranspose1d> {
        if weight.rank() != 3 || stride == 0 {
            bail!(
                "shape_mismatch",
                "ConvTranspose1d: weight must be [Cin, Cout/groups, K] with stride >= 1, got {:?}",
                weight.shape()
            );
        }
        let (cin, cpg, k) = (weight.shape()[0], weight.shape()[1], weight.shape()[2]);
        let chunks = k.div_ceil(stride).max(1);
        let pad = chunks * stride - k;
        let w = weight.cast(DType::F32)?.pad(2, 0, pad, 0.0)?;
        let (w, cout, depthwise) = match groups {
            1 => (w.reshape(&[cin, cpg * chunks * stride])?, cpg, false),
            g if g == cin && cpg == 1 => (w.reshape(&[1, cin, 1, chunks * stride])?, cin, true),
            g => bail!("unsupported", "ConvTranspose1d: {g} groups (only 1 or depthwise)"),
        };
        let w = device
            .eval(&[&w.contiguous()?])?
            .pop()
            .ok_or_else(|| lapid::Error::new("internal", "eval returned nothing"))?;
        let bias = match bias {
            Some(b) => Some(b.cast(DType::F32)?.reshape(&[cout, 1])?),
            None => None,
        };
        Ok(ConvTranspose1d { weight: w, bias, stride, chunks, cout, depthwise })
    }

    /// Overlap-added contributions of x [B, Cin, T]: [B, Cout, (T + m - 1) * S], no bias.
    fn overlap_add(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t) = (x.shape()[0], x.shape()[2]);
        let (m, s, c) = (self.chunks, self.stride, self.cout);
        let z = if self.depthwise {
            // [B, C, T, 1] * [1, C, 1, m*S]
            x.unsqueeze(3)?.mul(&self.weight)?.reshape(&[b, c, t, m, s])?
        } else {
            let y = x.permute(&[0, 2, 1])?.matmul(&self.weight)?; // [B, T, Cout*m*S]
            y.reshape(&[b, t, c, m, s])?.permute(&[0, 2, 1, 3, 4])?
        };
        let mut acc: Option<Tensor> = None;
        for ci in 0..m {
            let zc = z.slice(3, ci, ci + 1)?.reshape(&[b, c, t, s])?.pad(2, ci, m - 1 - ci, 0.0)?;
            acc = Some(match acc {
                Some(a) => a.add(&zc)?,
                None => zc,
            });
        }
        let acc = acc.ok_or_else(|| lapid::Error::new("internal", "no chunks"))?;
        acc.reshape(&[b, c, (t + m - 1) * s])
    }

    fn add_bias(&self, y: Tensor) -> Result<Tensor> {
        match &self.bias {
            Some(bias) => y.add(bias),
            None => Ok(y),
        }
    }

    /// Causal full-sequence transposed conv: [B, Cout, T * S] (right `K - S` trimmed).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.shape()[2];
        self.add_bias(self.overlap_add(x)?.slice(2, 0, t * self.stride)?)
    }

    /// Streaming step: x [B, Cin, T] -> [B, Cout, T * S]. `state` holds the partial sums
    /// [B, Cout, (m - 1) * S] (`None` = zeros) and is replaced by a lazy view; evaluate it
    /// with the output and store the realized tensor.
    pub fn forward_stream(&self, x: &Tensor, state: &mut Option<Tensor>) -> Result<Tensor> {
        if x.rank() != 3 {
            bail!("shape_mismatch", "ConvTranspose1d expects x [B, Cin, T], got {:?}", x.shape());
        }
        let (b, t) = (x.shape()[0], x.shape()[2]);
        let n = t * self.stride;
        let p = (self.chunks - 1) * self.stride;
        let mut y = self.overlap_add(x)?;
        if p == 0 {
            *state = None;
            return self.add_bias(y);
        }
        if let Some(prev) = state.take() {
            if prev.shape() != [b, self.cout, p] {
                bail!(
                    "shape_mismatch",
                    "ConvTranspose1d stream state must be [{b}, {}, {p}], got {:?}",
                    self.cout,
                    prev.shape()
                );
            }
            let total = y.shape()[2];
            let head = y.slice(2, 0, p)?.add(&prev)?;
            y = Tensor::cat(&[&head, &y.slice(2, p, total)?], 2)?;
        }
        *state = Some(y.slice(2, n, n + p)?);
        self.add_bias(y.slice(2, 0, n)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct definition, [Cin, Cout, K] weight, groups 1 or depthwise, full length.
    fn naive(x: &[f32], cin: usize, t: usize, w: &[f32], cpg: usize, k: usize, s: usize, depthwise: bool) -> Vec<f32> {
        let cout = if depthwise { cin } else { cpg };
        let len = (t - 1) * s + k;
        let mut y = vec![0.0f32; cout * len];
        for ci in 0..cin {
            for ti in 0..t {
                for kk in 0..k {
                    if depthwise {
                        y[ci * len + ti * s + kk] += x[ci * t + ti] * w[ci * k + kk];
                    } else {
                        for co in 0..cout {
                            y[co * len + ti * s + kk] += x[ci * t + ti] * w[(ci * cpg + co) * k + kk];
                        }
                    }
                }
            }
        }
        y
    }

    fn check(cin: usize, cpg: usize, k: usize, s: usize, depthwise: bool) {
        let dev = Device::cpu();
        let t = 5;
        let x: Vec<f32> = (0..cin * t).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.1).collect();
        let w: Vec<f32> = (0..cin * cpg * k).map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.05).collect();
        let wt = Tensor::from_f32(&w, &[cin, cpg, k]).expect("w");
        let groups = if depthwise { cin } else { 1 };
        let conv = ConvTranspose1d::new(&wt, None, s, groups, &dev).expect("conv");
        let cout = if depthwise { cin } else { cpg };
        let full = naive(&x, cin, t, &w, cpg, k, s, depthwise);
        let len = (t - 1) * s + k;
        // streaming in chunks of 2 + 3 frames equals the first T*S samples
        let mut st = None;
        let mut got = vec![Vec::new(); cout];
        for (a, bnd) in [(0, 2), (2, 5)] {
            let xs: Vec<f32> = (0..cin).flat_map(|c| x[c * t + a..c * t + bnd].to_vec()).collect();
            let xt = Tensor::from_f32(&xs, &[1, cin, bnd - a]).expect("x");
            let y = conv.forward_stream(&xt, &mut st).expect("fwd");
            let mut outs = vec![&y];
            if let Some(s) = st.as_ref() {
                outs.push(s);
            }
            let mut r = dev.eval(&outs).expect("eval");
            if r.len() > 1 {
                st = r.pop();
            }
            let v = dev.to_vec_f32(&r[0]).expect("vec");
            let n = (bnd - a) * s;
            for co in 0..cout {
                got[co].extend_from_slice(&v[co * n..(co + 1) * n]);
            }
        }
        // and so does the one-call forward
        let y = conv.forward(&Tensor::from_f32(&x, &[1, cin, t]).expect("x")).expect("forward");
        let whole = dev.to_vec_f32(&dev.eval(&[&y]).expect("eval")[0]).expect("vec");
        for co in 0..cout {
            for i in 0..t * s {
                let e = full[co * len + i];
                assert!((got[co][i] - e).abs() < 1e-4, "co {co} i {i}: {} vs {e}", got[co][i]);
                assert!((whole[co * t * s + i] - e).abs() < 1e-4, "forward co {co} i {i}");
            }
        }
    }

    #[test]
    fn dense_matches_naive() {
        check(3, 4, 12, 6, false);
        check(2, 3, 7, 3, false); // K not a multiple of S
    }

    #[test]
    fn depthwise_matches_naive() {
        check(4, 1, 32, 16, true);
    }
}
