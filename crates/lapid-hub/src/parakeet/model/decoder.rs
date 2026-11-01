//! TDT prediction network and joint.
//!
//! * Prediction: token embedding → stacked LSTM (`lapid_nn::speech::Lstm`) →
//!   `decoder_projector`. One step per emitted token; the host loop keeps the realized
//!   (h, c) per layer between steps.
//! * Joint: `head(relu(enc_t + dec))` → [vocab + blank | durations] logits.

use lapid::{Result, Tensor};
use lapid_nn::speech::{Lstm, LstmLayer, LstmState};
use lapid_nn::{Embedding, Linear, Loader};

use super::linear_checked;
use crate::parakeet::config::ParakeetConfig;

pub(crate) struct Prediction {
    embed: Embedding,
    pub lstm: Lstm,
    proj: Linear,
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
        Ok(Prediction { embed, lstm: Lstm { layers, hidden: h }, proj: linear_checked(l, "decoder.decoder_projector", h, h)? })
    }

    /// One step on `token`: (projected output [1, H], new state), both lazy.
    pub fn step(&self, token: u32, state: &LstmState) -> Result<(Tensor, LstmState)> {
        let x = self.embed.lookup(&[token])?.reshape(&[1, self.lstm.hidden])?;
        let (h, next) = self.lstm.step(&x, state)?;
        Ok((self.proj.forward(&h)?, next))
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
