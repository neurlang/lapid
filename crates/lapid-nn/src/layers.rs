//! Basic layers: `Linear`, `Embedding`, `LayerNorm`, `RmsNorm`, `Conv1d`.
//!
//! Layers hold device tensors and build lazy graph nodes; nothing runs until `Device::eval`.

use lapid::{DType, Error, Result, Tensor, bail};

use crate::weights::fake_quant_act;

/// y = x @ W^T + b. W: [out, in] (any float or quant dtype), b: [out] optional.
#[derive(Clone)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    /// Fake-quantize the input to int8 blocks first (reference models of quantized kernels).
    pub act_quant: bool,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Linear {
        Linear { weight, bias, act_quant: false }
    }

    pub fn with_act_quant(mut self, on: bool) -> Linear {
        self.act_quant = on;
        self
    }

    pub fn in_features(&self) -> usize {
        self.weight.shape().get(1).copied().unwrap_or(0)
    }

    pub fn out_features(&self) -> usize {
        self.weight.shape().first().copied().unwrap_or(0)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.weight.rank() != 2 {
            bail!("shape_mismatch", "Linear weight must be [out, in], got {:?}", self.weight.shape());
        }
        let x = if self.act_quant { fake_quant_act(x)? } else { x.clone() };
        // The compiler fuses the bias add into the matmul epilogue.
        let y = x.matmul_t(&self.weight)?;
        add_bias(y, self.bias.as_ref())
    }
}

/// Token embedding lookup: table [V, D], ids I32 [...] -> [..., D] f32.
#[derive(Clone)]
pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {
    pub fn new(weight: Tensor) -> Embedding {
        Embedding { weight }
    }

    pub fn vocab_size(&self) -> usize {
        self.weight.shape().first().copied().unwrap_or(0)
    }

    pub fn dim(&self) -> usize {
        self.weight.shape().get(1).copied().unwrap_or(0)
    }

    /// `ids` is an I32 tensor of any shape.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        self.weight.gather(ids)
    }

    /// Host ids -> [1, n, D], checking every id against the vocabulary.
    pub fn lookup(&self, ids: &[u32]) -> Result<Tensor> {
        if ids.is_empty() {
            bail!("invalid_argument", "embedding lookup of zero ids");
        }
        let v = self.vocab_size();
        let mut ids_i32 = Vec::with_capacity(ids.len());
        for &id in ids {
            if id as usize >= v {
                bail!("invalid_argument", "token id {id} outside vocabulary of {v}");
            }
            ids_i32.push(id as i32);
        }
        self.forward(&Tensor::from_i32(&ids_i32, &[1, ids.len()])?)
    }

    /// Tied output head: x [..., D] @ table^T -> [..., V].
    pub fn logits(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul_t(&self.weight)
    }
}

/// (x - mean) / sqrt(var + eps) * w + b over the last dim (biased variance).
#[derive(Clone)]
pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f32) -> LayerNorm {
        LayerNorm { weight, bias, eps }
    }

    /// LayerNorm without a bias (a zero bias of the weight's shape and dtype).
    pub fn no_bias(weight: Tensor, eps: f32) -> LayerNorm {
        let bias = Tensor::zeros(weight.shape(), weight.dtype());
        LayerNorm { weight, bias, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.layer_norm(&self.weight, &self.bias, self.eps)
    }
}

/// x / sqrt(mean(x^2) + eps) * (w + weight_offset) over the last dim, computed in f32 and
/// returned in x's dtype. `weight_offset` is 1.0 for Gemma-style norms (`x * (1 + w)`).
///
/// One `Tensor::rms_norm` composite (a native kernel on every backend).
#[derive(Clone)]
pub struct RmsNorm {
    pub weight: Tensor,
    pub eps: f32,
    pub weight_offset: f32,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f32) -> RmsNorm {
        RmsNorm { weight, eps, weight_offset: 0.0 }
    }

    pub fn with_weight_offset(mut self, offset: f32) -> RmsNorm {
        self.weight_offset = offset;
        self
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.shape().last().copied().unwrap_or(0);
        if self.weight.shape() != [d] {
            bail!("shape_mismatch", "RmsNorm over last dim {d}: weight {:?} must be [{d}]", self.weight.shape());
        }
        let mut w = self.weight.clone();
        if self.weight_offset != 0.0 {
            w = w.cast(DType::F32)?.add_scalar(f64::from(self.weight_offset))?;
        }
        x.rms_norm(&w, self.eps)
    }
}

/// 1-D convolution. x: [B, Cin, L], weight: [Cout, Cin / groups, K] (float), bias [Cout].
/// Out: [B, Cout, Lout], Lout = (L + left + right - dilation * (K - 1) - 1) / stride + 1.
///
/// One group: im2col (pad + per-tap slices, strided taps via reshape) + matmul. Depthwise
/// (groups == Cin == Cout): per-tap multiply-add. Other group counts run one im2col per group.
#[derive(Clone)]
pub struct Conv1d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    /// Zero padding (left, right).
    pub padding: (usize, usize),
    pub dilation: usize,
    pub groups: usize,
}

impl Conv1d {
    /// Stride 1, no padding, dilation 1, one group.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Conv1d {
        Conv1d { weight, bias, stride: 1, padding: (0, 0), dilation: 1, groups: 1 }
    }

    /// Depthwise conv: weight [C, 1, K], groups = C.
    pub fn depthwise(weight: Tensor, bias: Option<Tensor>) -> Conv1d {
        let groups = weight.shape().first().copied().unwrap_or(1).max(1);
        Conv1d { groups, ..Conv1d::new(weight, bias) }
    }

    pub fn with_stride(mut self, stride: usize) -> Conv1d {
        self.stride = stride;
        self
    }

    /// Symmetric zero padding.
    pub fn with_padding(mut self, padding: usize) -> Conv1d {
        self.padding = (padding, padding);
        self
    }

    pub fn with_padding_lr(mut self, left: usize, right: usize) -> Conv1d {
        self.padding = (left, right);
        self
    }

    pub fn with_dilation(mut self, dilation: usize) -> Conv1d {
        self.dilation = dilation;
        self
    }

    pub fn with_groups(mut self, groups: usize) -> Conv1d {
        self.groups = groups;
        self
    }

    /// Causal padding: `dilation * (K - 1)` zeros on the left, none on the right (set the
    /// dilation first).
    pub fn causal(mut self) -> Conv1d {
        let k = self.kernel_size();
        self.padding = (self.dilation * k.saturating_sub(1), 0);
        self
    }

    pub fn kernel_size(&self) -> usize {
        self.weight.shape().get(2).copied().unwrap_or(0)
    }

    /// Input frames a causal conv needs from the past: `dilation * (K - 1)`.
    pub fn context(&self) -> usize {
        self.dilation * self.kernel_size().saturating_sub(1)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_padded(x, self.padding)
    }

    /// Streaming causal conv (stride 1): `state` holds the last `context()` input frames
    /// [B, Cin, ctx] of the previous call (`None` = zeros, i.e. the start of the stream) and
    /// is replaced by the frames this call leaves behind. Concatenating the outputs of
    /// consecutive calls equals one `forward` of a `causal()` conv over the whole input.
    ///
    /// The new state is a lazy view; evaluate it together with the output (and store the
    /// realized tensor) so graphs do not grow across calls.
    pub fn forward_stream(&self, x: &Tensor, state: &mut Option<Tensor>) -> Result<Tensor> {
        if self.stride != 1 {
            bail!("invalid_argument", "streaming Conv1d needs stride 1, got {}", self.stride);
        }
        if x.rank() != 3 {
            bail!("shape_mismatch", "Conv1d expects x [B, Cin, L], got {:?}", x.shape());
        }
        let ctx = self.context();
        if ctx == 0 {
            return self.forward_padded(x, (0, 0));
        }
        let (b, cin) = (x.shape()[0], x.shape()[1]);
        let prev = match state.take() {
            Some(s) => {
                if s.shape() != [b, cin, ctx] {
                    bail!("shape_mismatch", "Conv1d stream state must be [{b}, {cin}, {ctx}], got {:?}", s.shape());
                }
                s.cast(x.dtype())?
            }
            None => Tensor::zeros(&[b, cin, ctx], x.dtype()),
        };
        if x.shape()[2] == 1 && self.dilation == 1 && self.is_depthwise(cin) {
            return self.depthwise_step(x, prev, state);
        }
        let xin = Tensor::cat(&[&prev, x], 2)?;
        let len = xin.shape()[2];
        *state = Some(xin.slice(2, len - ctx, len)?);
        self.forward_padded(&xin, (0, 0))
    }

    fn is_depthwise(&self, cin: usize) -> bool {
        let w = self.weight.shape();
        w.len() == 3 && self.groups == cin && cin > 1 && w[0] == cin && w[1] == 1
    }

    /// One frame of a streaming depthwise conv (decode): `y = sum_t<ctx w_t prev_t + w_ctx x`
    /// as a reduce over the state plus a fused term, so the window is never materialized;
    /// the new state is `prev[1..] ++ x`, written straight into its own buffer.
    fn depthwise_step(&self, x: &Tensor, prev: Tensor, state: &mut Option<Tensor>) -> Result<Tensor> {
        let (c, k) = (self.weight.shape()[0], self.kernel_size());
        let ctx = k - 1;
        let w = self.weight.cast(x.dtype())?.reshape(&[c, k])?;
        let head = prev.mul(&w.slice(1, 0, ctx)?)?.sum(-1, true)?; // [B, C, 1]
        let mut y = head.add(&x.mul(&w.slice(1, ctx, k)?)?)?;
        if let Some(b) = &self.bias {
            y = y.add(&b.cast(y.dtype())?.reshape(&[c, 1])?)?;
        }
        *state = Some(if ctx > 1 { Tensor::cat(&[&prev.slice(2, 1, ctx)?, x], 2)? } else { x.clone() });
        Ok(y)
    }

    fn forward_padded(&self, x: &Tensor, (left, right): (usize, usize)) -> Result<Tensor> {
        let w = &self.weight;
        if x.rank() != 3 || w.rank() != 3 {
            bail!(
                "shape_mismatch",
                "Conv1d expects x [B, Cin, L] and weight [Cout, Cin/groups, K], got {:?} and {:?}",
                x.shape(),
                w.shape()
            );
        }
        let (b, cin, l) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (cout, wcin, k) = (w.shape()[0], w.shape()[1], w.shape()[2]);
        let g = self.groups;
        let (s, d) = (self.stride, self.dilation);
        if g == 0 || cin % g != 0 || cout % g != 0 {
            bail!("invalid_argument", "Conv1d: {g} groups must divide Cin {cin} and Cout {cout}");
        }
        if wcin * g != cin {
            bail!("shape_mismatch", "Conv1d: input has {cin} channels, weight expects {wcin} x {g} groups");
        }
        if s == 0 || k == 0 || d == 0 {
            bail!("invalid_argument", "Conv1d needs stride, kernel and dilation >= 1 (stride {s}, kernel {k}, dilation {d})");
        }
        let lp = l + left + right;
        let span = d * (k - 1) + 1;
        if lp < span {
            bail!("shape_mismatch", "Conv1d: padded length {lp} is shorter than the kernel span {span}");
        }
        let lout = (lp - span) / s + 1;
        // Tap t reads padded positions t*d, t*d + s, ..., t*d + (lout-1)*s. Taking the window
        // [t*d, t*d + lout*s) and viewing it as [lout, s] makes the strided pick a plain
        // slice of the last dim; pad the tail so the last tap's window fits.
        let need = d * (k - 1) + lout * s;
        let xp = x.pad(2, left, right + need.saturating_sub(lp), 0.0)?;
        let tap = |xs: &Tensor, c: usize, t: usize| -> Result<Tensor> {
            let mut tap = xs.slice(2, t * d, t * d + lout * s)?; // [B, c, lout*s]
            if s > 1 {
                tap = tap.reshape(&[b, c, lout, s])?.slice(3, 0, 1)?.squeeze(3)?;
            }
            Ok(tap)
        };

        if g > 1 && wcin == 1 && cout == cin {
            // Depthwise: y[c] = sum_t tap_t[c] * w[c, 0, t].
            let taps: Vec<Tensor> =
                (0..k).map(|t| tap(&xp, cin, t)?.unsqueeze(3)).collect::<Result<_>>()?;
            let refs: Vec<&Tensor> = taps.iter().collect();
            let cols = Tensor::cat(&refs, 3)?; // [B, C, Lout, K]
            let wk = w.cast(x.dtype())?.reshape(&[cout, 1, k])?;
            let y = cols.mul(&wk)?.sum(-1, false)?; // [B, C, Lout]
            return match &self.bias {
                Some(bias) => y.add(&bias.cast(y.dtype())?.reshape(&[cout, 1])?),
                None => Ok(y),
            };
        }

        let (cin_g, cout_g) = (cin / g, cout / g);
        let mut outs = Vec::with_capacity(g);
        for gi in 0..g {
            let xg = if g == 1 { xp.clone() } else { xp.slice(1, gi * cin_g, (gi + 1) * cin_g)? };
            let wg = if g == 1 { w.clone() } else { w.slice(0, gi * cout_g, (gi + 1) * cout_g)? };
            let mut taps = Vec::with_capacity(k);
            for t in 0..k {
                // [B, Cin, Lout] -> [B, Lout, Cin, 1]
                taps.push(tap(&xg, cin_g, t)?.unsqueeze(3)?.permute(&[0, 2, 1, 3])?);
            }
            // Columns [B, Lout, Cin, K] -> [B, Lout, Cin*K]: column cin*K + t, which matches
            // the row-major weight [Cout, Cin, K] viewed as [Cout, Cin*K].
            let refs: Vec<&Tensor> = taps.iter().collect();
            let cols = Tensor::cat(&refs, 3)?.reshape(&[b, lout, cin_g * k])?;
            let wm = wg.reshape(&[cout_g, cin_g * k])?;
            outs.push(cols.matmul_t(&wm)?); // [B, Lout, Cout_g]
        }
        let y = if outs.len() == 1 {
            outs.pop().ok_or_else(|| Error::new("internal", "conv produced no output"))?
        } else {
            let refs: Vec<&Tensor> = outs.iter().collect();
            Tensor::cat(&refs, 2)?
        };
        let y = add_bias(y, self.bias.as_ref())?; // [B, Lout, Cout]
        y.permute(&[0, 2, 1])
    }
}

/// `y + bias` broadcast over the last dim; the bias is cast to y's dtype if needed.
fn add_bias(y: Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    match bias {
        Some(bias) => y.add(&bias.cast(y.dtype())?),
        None => Ok(y),
    }
}

#[cfg(test)]
mod tests;
