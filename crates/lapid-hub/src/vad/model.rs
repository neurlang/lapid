//! The Silero VAD v5 network.
//!
//! Per 512-sample window (with the previous 64 samples prepended as context):
//! reflect pad right by 64 → STFT as a matmul with the stored basis (4 frames of 256, hop 128,
//! 129 bins of magnitude) → 4 × [Conv1d(k3, pad 1) + ReLU] (strides 1, 2, 2, 1: 4 frames → 1)
//! → LSTMCell(128) → ReLU → Conv1d(128 → 1, k1) → sigmoid.
//!
//! Split between device and host:
//! * **Device (batched):** everything up to and including the LSTM input projection
//!   (`W_ih x + b_ih + b_hh`) runs as one lazy graph over many windows at once
//!   ([`Net::front`]). Nothing in it depends on the recurrent state.
//! * **Host (sequential):** the recurrence `W_hh h` + gates and the 128 → 1 head
//!   ([`Net::step`]), in f32. That is ~66k multiply-adds per window,
//!   ~13 µs per window measured on the GB10 (60 s of audio: 25-60 ms), far cheaper than a
//!   device evaluation per window (0.3-0.6 ms on CPU), so long audio
//!   costs one device evaluation per batch of windows plus a host loop. Streaming uses the same
//!   two functions, and device batches are never smaller than `MIN_BATCH_WINDOWS` (padded
//!   with silence), so stream and batch are bit-identical on every backend.

use lapid::{Device, Error, Result, Tensor};
use lapid_nn::{Conv1d, Linear, Loader};

use super::config::VadConfig;

/// Realized recurrent state (h, c), each `hidden_size` floats.
#[derive(Clone, Debug, PartialEq)]
pub struct LstmState {
    pub h: Vec<f32>,
    pub c: Vec<f32>,
}

impl LstmState {
    pub fn zeros(hidden: usize) -> LstmState {
        LstmState { h: vec![0.0; hidden], c: vec![0.0; hidden] }
    }
}

/// Host results of the batched front-end over `n` windows.
pub(crate) struct Front {
    /// LSTM input gates `W_ih x + b_ih + b_hh`, [n, 4H].
    pub gates: Vec<f32>,
    /// STFT magnitude [n, bins, frames] (traced runs only).
    pub stft: Option<Vec<f32>>,
    /// Encoder output [n, H] (traced runs only).
    pub encoder: Option<Vec<f32>>,
}

pub(crate) struct Net {
    cfg: VadConfig,
    /// STFT basis [2 · bins, filter_length] (real rows, then imaginary rows).
    basis: Tensor,
    encoder: Vec<Conv1d>,
    input_proj: Linear,
    /// Host copies: recurrent weight [4H, H] and the head.
    w_hh: Vec<f32>,
    head_w: Vec<f32>,
    head_b: f32,
}

fn host(l: &Loader<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let t = l.tensor_f32(name)?;
    l.expect_shape(&t, name, shape)?;
    l.device.to_vec_f32(&t)
}

impl Net {
    pub fn load(l: &Loader<'_>, cfg: &VadConfig) -> Result<Net> {
        let (h, bins, fl) = (cfg.hidden_size, cfg.n_bins(), cfg.filter_length);
        let basis = l.tensor_f32("stft.forward_basis_buffer")?;
        l.expect_shape(&basis, "stft.forward_basis_buffer", &[2 * bins, fl])?;
        let mut encoder = Vec::with_capacity(cfg.encoder.len());
        for (i, b) in cfg.encoder.iter().enumerate() {
            let p = format!("encoder.{i}.reparam_conv");
            let w = l.tensor_f32(&format!("{p}.weight"))?;
            l.expect_shape(&w, &format!("{p}.weight"), &[b.out_channels, b.in_channels, 3])?;
            let bias = l.tensor_f32(&format!("{p}.bias"))?;
            encoder.push(Conv1d::new(w, Some(bias)).with_stride(b.stride).with_padding(1));
        }
        let w_ih = l.tensor_f32("decoder.rnn.weight_ih")?;
        l.expect_shape(&w_ih, "decoder.rnn.weight_ih", &[4 * h, h])?;
        let b_ih = host(l, "decoder.rnn.bias_ih", &[4 * h])?;
        let b_hh = host(l, "decoder.rnn.bias_hh", &[4 * h])?;
        let bias: Vec<f32> = b_ih.iter().zip(&b_hh).map(|(a, b)| a + b).collect();
        let bias = l.device.upload(&Tensor::from_f32(&bias, &[4 * h])?)?;
        let head_b = host(l, "decoder.decoder.2.bias", &[1])?;
        Ok(Net {
            cfg: cfg.clone(),
            basis,
            encoder,
            input_proj: Linear::new(w_ih, Some(bias)),
            w_hh: host(l, "decoder.rnn.weight_hh", &[4 * h, h])?,
            head_w: host(l, "decoder.decoder.2.weight", &[1, h])?,
            head_b: head_b.first().copied().unwrap_or(0.0),
        })
    }

    pub fn config(&self) -> &VadConfig {
        &self.cfg
    }

    /// Length of one window's model input after reflect padding (640).
    pub fn input_len(&self) -> usize {
        self.cfg.input_size() + self.cfg.stft_pad()
    }

    /// Append one window's model input to `out`: `context ++ chunk` (zero-padded to the
    /// window size) reflect-padded on the right like `F.pad(x, (0, pad), "reflect")`.
    pub fn push_input(&self, context: &[f32], chunk: &[f32], out: &mut Vec<f32>) {
        let (c, w, pad) = (self.cfg.context_size, self.cfg.window_size, self.cfg.stft_pad());
        let start = out.len();
        out.extend(context.iter().copied().chain(std::iter::repeat(0.0)).take(c));
        out.extend(chunk.iter().copied().chain(std::iter::repeat(0.0)).take(w));
        let len = c + w;
        for j in 0..pad {
            let v = out[start + len - 2 - j];
            out.push(v);
        }
    }

    /// Batched front-end over `n` windows of [`push_input`](Self::push_input) data.
    pub fn front(&self, device: &Device, inputs: &[f32], n: usize, traced: bool) -> Result<Front> {
        let cfg = &self.cfg;
        let (len, fl, hop, bins, h) = (self.input_len(), cfg.filter_length, cfg.hop_length, cfg.n_bins(), cfg.hidden_size);
        if n == 0 || inputs.len() != n * len {
            return Err(Error::new("invalid_argument", format!("vad: {} input samples for {n} windows of {len}", inputs.len())));
        }
        let frames = cfg.stft_frames();
        let x = Tensor::from_f32(inputs, &[n, len])?;
        let taps: Vec<Tensor> = (0..frames).map(|t| x.slice(1, t * hop, t * hop + fl)?.unsqueeze(1)).collect::<Result<_>>()?;
        let refs: Vec<&Tensor> = taps.iter().collect();
        let spec = Tensor::cat(&refs, 1)?.matmul_t(&self.basis)?; // [n, frames, 2 bins]
        let re = spec.slice(2, 0, bins)?;
        let im = spec.slice(2, bins, 2 * bins)?;
        let mag = re.mul(&re)?.add(&im.mul(&im)?)?.sqrt()?.permute(&[0, 2, 1])?; // [n, bins, frames]
        let mut y = mag.clone();
        for conv in &self.encoder {
            y = lapid_nn::ops::relu(&conv.forward(&y)?)?;
        }
        let enc = y.reshape(&[n, h])?;
        let gates = self.input_proj.forward(&enc)?;
        let mut outs = vec![&gates];
        let mag_c = mag.contiguous()?;
        if traced {
            outs.push(&mag_c);
            outs.push(&enc);
        }
        let realized = device.eval(&outs)?;
        let get = |i: usize| -> Result<Vec<f32>> {
            let t = realized.get(i).ok_or_else(|| Error::new("backend", "device.eval returned too few outputs"))?;
            device.to_vec_f32(t)
        };
        Ok(Front {
            gates: get(0)?,
            stft: if traced { Some(get(1)?) } else { None },
            encoder: if traced { Some(get(2)?) } else { None },
        })
    }

    /// One recurrent step on input gates `gx` [4H]: updates `state`, returns the speech
    /// probability. PyTorch `LSTMCell` gate order (i, f, g, o).
    pub fn step(&self, state: &mut LstmState, gx: &[f32]) -> f32 {
        let h = self.cfg.hidden_size;
        let mut gates = vec![0f32; 4 * h];
        for (r, g) in gates.iter_mut().enumerate() {
            let row = &self.w_hh[r * h..(r + 1) * h];
            *g = gx[r] + dot(row, &state.h);
        }
        for j in 0..h {
            let i = sigmoid(gates[j]);
            let f = sigmoid(gates[h + j]);
            let g = gates[2 * h + j].tanh();
            let o = sigmoid(gates[3 * h + j]);
            let c = f * state.c[j] + i * g;
            state.c[j] = c;
            state.h[j] = o * c.tanh();
        }
        let logit = self.head_b + self.head_w.iter().zip(&state.h).map(|(w, x)| w * x.max(0.0)).sum::<f32>();
        sigmoid(logit)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Dot product with 8 independent accumulators (vectorizes; order fixed, so results are
/// deterministic across runs and callers).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ac, ar) = a.as_chunks::<8>();
    let (bc, br) = b.as_chunks::<8>();
    for (x, y) in ac.iter().zip(bc) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (x, y) in ar.iter().zip(br) {
        s += x * y;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_naive() {
        let a: Vec<f32> = (0..131).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..131).map(|i| (i as f32 * 0.11).cos()).collect();
        let naive: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
        assert!((dot(&a, &b) as f64 - naive).abs() < 1e-4);
    }
}
