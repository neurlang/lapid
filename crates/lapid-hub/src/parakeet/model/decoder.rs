//! TDT prediction network and joint.
//!
//! * Prediction: token embedding → stacked LSTM (PyTorch gate order i, f, g, o) →
//!   `decoder_projector`. One step per emitted token; the host loop keeps the realized
//!   (h, c) per layer between steps.
//! * Joint: `head(relu(enc_t + dec))` → [vocab + blank | durations] logits.

use lapid::{DType, Result, Tensor};
use lapid_nn::{Embedding, Linear, Loader};

use super::linear_checked;
use crate::parakeet::config::ParakeetConfig;

struct LstmLayer {
    ih: Linear,
    hh: Linear,
}

/// Realized LSTM state: per layer h and c, each [1, H].
#[derive(Clone)]
pub(crate) struct LstmState {
    pub h: Vec<Tensor>,
    pub c: Vec<Tensor>,
}

impl LstmState {
    pub fn tensors(&self) -> Vec<&Tensor> {
        self.h.iter().chain(&self.c).collect()
    }

    /// Rebuild from `tensors()` order.
    pub fn from_tensors(mut v: Vec<Tensor>, layers: usize) -> LstmState {
        let c = v.split_off(layers.min(v.len()));
        LstmState { h: v, c }
    }
}

pub(crate) struct Prediction {
    embed: Embedding,
    layers: Vec<LstmLayer>,
    proj: Linear,
    hidden: usize,
}

impl Prediction {
    pub fn load(l: &Loader<'_>, cfg: &ParakeetConfig) -> Result<Prediction> {
        let h = cfg.decoder_hidden;
        let embed = l.embedding("decoder.embedding")?;
        l.expect_shape(&embed.weight, "decoder.embedding.weight", &[cfg.vocab_size, h])?;
        let mut layers = Vec::with_capacity(cfg.decoder_layers);
        for i in 0..cfg.decoder_layers {
            let lin = |kind: &str| -> Result<Linear> {
                let w = format!("decoder.lstm.weight_{kind}_l{i}");
                let weight = l.tensor(&w)?;
                l.expect_shape(&weight, &w, &[4 * h, h])?;
                let bias = l.tensor(&format!("decoder.lstm.bias_{kind}_l{i}"))?;
                Ok(Linear::new(weight, Some(bias)).with_act_quant(l.act_quant(&w)?))
            };
            layers.push(LstmLayer { ih: lin("ih")?, hh: lin("hh")? });
        }
        Ok(Prediction {
            embed,
            layers,
            proj: linear_checked(l, "decoder.decoder_projector", h, h)?,
            hidden: h,
        })
    }

    pub fn zero_state(&self) -> LstmState {
        let z = || Tensor::zeros(&[1, self.hidden], DType::F32);
        LstmState { h: (0..self.layers.len()).map(|_| z()).collect(), c: (0..self.layers.len()).map(|_| z()).collect() }
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// One step on `token`: (projected output [1, H], new state), both lazy.
    pub fn step(&self, token: u32, state: &LstmState) -> Result<(Tensor, LstmState)> {
        let h = self.hidden;
        let mut x = self.embed.lookup(&[token])?.reshape(&[1, h])?;
        let mut next = LstmState { h: Vec::new(), c: Vec::new() };
        for (i, layer) in self.layers.iter().enumerate() {
            let gates = layer.ih.forward(&x)?.add(&layer.hh.forward(&state.h[i])?)?; // [1, 4H]
            let gate = |k: usize| gates.slice(1, k * h, (k + 1) * h);
            let ig = lapid_nn::ops::sigmoid(&gate(0)?)?;
            let fg = lapid_nn::ops::sigmoid(&gate(1)?)?;
            let gg = gate(2)?.tanh()?;
            let og = lapid_nn::ops::sigmoid(&gate(3)?)?;
            let c = fg.mul(&state.c[i])?.add(&ig.mul(&gg)?)?;
            let hn = og.mul(&c.tanh()?)?;
            next.h.push(hn.clone());
            next.c.push(c);
            x = hn;
        }
        Ok((self.proj.forward(&x)?, next))
    }
}

pub(crate) struct Joint {
    head: Linear,
}

impl Joint {
    pub fn load(l: &Loader<'_>, cfg: &ParakeetConfig) -> Result<Joint> {
        Ok(Joint { head: linear_checked(l, "joint.head", cfg.joint_dim(), cfg.decoder_hidden)? })
    }

    /// enc [1, H] (projected encoder frame), dec [1, H] -> logits [1, vocab + durations].
    pub fn logits(&self, enc: &Tensor, dec: &Tensor) -> Result<Tensor> {
        self.head.forward(&lapid_nn::ops::relu(&enc.add(dec)?)?)
    }
}
