//! LSTM (PyTorch `nn.LSTM` / `nn.LSTMCell`, gate order i, f, g, o), one step at a time:
//!
//! ```text
//! gates = W_ih x + b_ih + W_hh h + b_hh;   c' = σ(f) c + σ(i) tanh(g);   h' = σ(o) tanh(c')
//! ```
//!
//! * [`Lstm`]: stacked layers as a lazy graph (e.g. a transducer prediction network, one
//!   step per emitted token, the realized state carried by the host loop).
//! * [`HostLstmCell`]: the recurrence on the host in f32 for small cells whose input
//!   projection `W_ih x + b` was batched on the device (e.g. a VAD over many windows): a
//!   [H, H] mat-vec per step is far cheaper than a device evaluation per step.

use lapid::{DType, Result, Tensor};

use crate::Linear;
use crate::ops::sigmoid;

/// One layer: input and recurrent projections, each [4H, in] with its bias.
pub struct LstmLayer {
    pub ih: Linear,
    pub hh: Linear,
}

/// Lazy per-layer state: h and c, each [1, H].
#[derive(Clone)]
pub struct LstmState {
    pub h: Vec<Tensor>,
    pub c: Vec<Tensor>,
}

impl LstmState {
    /// h of every layer, then c of every layer.
    pub fn tensors(&self) -> Vec<&Tensor> {
        self.h.iter().chain(&self.c).collect()
    }

    /// Rebuild from [`tensors`](Self::tensors) order.
    pub fn from_tensors(mut v: Vec<Tensor>, layers: usize) -> LstmState {
        let c = v.split_off(layers.min(v.len()));
        LstmState { h: v, c }
    }
}

/// Stacked LSTM, batch 1.
pub struct Lstm {
    pub layers: Vec<LstmLayer>,
    pub hidden: usize,
}

impl Lstm {
    pub fn zero_state(&self) -> LstmState {
        let z = || Tensor::zeros(&[1, self.hidden], DType::F32);
        LstmState { h: self.layers.iter().map(|_| z()).collect(), c: self.layers.iter().map(|_| z()).collect() }
    }

    /// One step on x [1, in]: (top layer h [1, H], new state), both lazy.
    pub fn step(&self, x: &Tensor, state: &LstmState) -> Result<(Tensor, LstmState)> {
        let h = self.hidden;
        let mut x = x.clone();
        let mut next = LstmState { h: Vec::new(), c: Vec::new() };
        for (i, layer) in self.layers.iter().enumerate() {
            let gates = layer.ih.forward(&x)?.add(&layer.hh.forward(&state.h[i])?)?; // [1, 4H]
            let gate = |k: usize| gates.slice(1, k * h, (k + 1) * h);
            let ig = sigmoid(&gate(0)?)?;
            let fg = sigmoid(&gate(1)?)?;
            let gg = gate(2)?.tanh()?;
            let og = sigmoid(&gate(3)?)?;
            let c = fg.mul(&state.c[i])?.add(&ig.mul(&gg)?)?;
            let hn = og.mul(&c.tanh()?)?;
            next.h.push(hn.clone());
            next.c.push(c);
            x = hn;
        }
        Ok((x, next))
    }
}

/// Realized recurrent state of a [`HostLstmCell`]: h and c, each `hidden` floats.
#[derive(Clone, Debug, PartialEq)]
pub struct HostLstmState {
    pub h: Vec<f32>,
    pub c: Vec<f32>,
}

impl HostLstmState {
    pub fn zeros(hidden: usize) -> HostLstmState {
        HostLstmState { h: vec![0.0; hidden], c: vec![0.0; hidden] }
    }
}

/// The recurrent half of an LSTM cell on the host: `W_hh` [4H, H] row-major.
pub struct HostLstmCell {
    pub w_hh: Vec<f32>,
    pub hidden: usize,
}

impl HostLstmCell {
    /// One step on precomputed input gates `gx` = `W_ih x + b_ih + b_hh` [4H]: updates
    /// `state` (deterministic: fixed summation order).
    pub fn step(&self, state: &mut HostLstmState, gx: &[f32]) {
        let h = self.hidden;
        let mut gates = vec![0f32; 4 * h];
        for (r, g) in gates.iter_mut().enumerate() {
            let row = &self.w_hh[r * h..(r + 1) * h];
            *g = gx[r] + dot(row, &state.h);
        }
        for j in 0..h {
            let i = host_sigmoid(gates[j]);
            let f = host_sigmoid(gates[h + j]);
            let g = gates[2 * h + j].tanh();
            let o = host_sigmoid(gates[3 * h + j]);
            let c = f * state.c[j] + i * g;
            state.c[j] = c;
            state.h[j] = o * c.tanh();
        }
    }
}

pub fn host_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Dot product with 8 independent accumulators (vectorizes; order fixed, so results are
/// deterministic across runs and callers).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
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

    #[test]
    fn device_step_matches_host_cell() {
        let (h, n) = (3usize, 4 * 3);
        let w_ih: Vec<f32> = (0..n * h).map(|i| (i as f32 * 0.13).sin() * 0.5).collect();
        let w_hh: Vec<f32> = (0..n * h).map(|i| (i as f32 * 0.29).cos() * 0.5).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let t = |d: &[f32], s: &[usize]| Tensor::from_f32(d, s).expect("t");
        let lstm = Lstm {
            layers: vec![LstmLayer { ih: Linear::new(t(&w_ih, &[n, h]), Some(t(&b, &[n]))), hh: Linear::new(t(&w_hh, &[n, h]), None) }],
            hidden: h,
        };
        let x = [0.3f32, -0.7, 1.1];
        let (mut state, mut host) = (lstm.zero_state(), HostLstmState::zeros(h));
        let cell = HostLstmCell { w_hh: w_hh.clone(), hidden: h };
        let dev = lapid::Device::cpu();
        for _ in 0..3 {
            let (out, next) = lstm.step(&t(&x, &[1, h]), &state).expect("step");
            let r = dev.eval(&[&out, &next.c[0]]).expect("eval");
            state = LstmState { h: vec![r[0].clone()], c: vec![r[1].clone()] };
            let gx: Vec<f32> = (0..n).map(|r| b[r] + (0..h).map(|j| w_ih[r * h + j] * x[j]).sum::<f32>()).collect();
            cell.step(&mut host, &gx);
            let got = dev.to_vec_f32(&r[0]).expect("h");
            assert!(got.iter().zip(&host.h).all(|(a, b)| (a - b).abs() < 1e-5), "{got:?} vs {:?}", host.h);
        }
    }
}
