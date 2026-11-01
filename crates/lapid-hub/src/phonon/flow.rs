//! The LSD flow head (`flow_lm.flow_net`, "SimpleMLPAdaLN"): an MLP with adaptive layer
//! norm conditioned on the backbone state and on the start / end times of a flow step. One
//! Euler step per `num_flow_steps` maps Gaussian noise to the next latent.

use lapid::{DType, Result, Tensor};
use lapid_nn::ops::silu;
use lapid_nn::{LayerNorm, Linear, Loader, RmsNorm};

use super::config::FlowConfig;

/// Sinusoidal embedding of a flow time → Linear → SiLU → Linear → RMSNorm (mean square).
struct TimeEmbed {
    /// Checkpoint `freqs` (host, f32).
    freqs: Vec<f32>,
    fc1: Linear,
    fc2: Linear,
    norm: RmsNorm,
}

impl TimeEmbed {
    fn forward(&self, t: f32) -> Result<Tensor> {
        let mut emb: Vec<f32> = self.freqs.iter().map(|&f| (t * f).cos()).collect();
        emb.extend(self.freqs.iter().map(|&f| (t * f).sin()));
        let x = Tensor::from_f32(&emb, &[1, emb.len()])?;
        self.norm.forward(&self.fc2.forward(&silu(&self.fc1.forward(&x)?)?)?)
    }
}

struct ResBlock {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    ada: Linear,
}

pub(crate) struct FlowHead {
    input_proj: Linear,
    cond_embed: Linear,
    time: Vec<TimeEmbed>,
    blocks: Vec<ResBlock>,
    final_ada: Linear,
    final_norm: LayerNorm,
    final_linear: Linear,
    dim: usize,
}

impl FlowHead {
    pub fn load(loader: &Loader, cfg: &FlowConfig) -> Result<FlowHead> {
        let p = "flow_lm.flow_net";
        let lin = |n: &str| loader.linear(&format!("{p}.{n}"), true);
        let mut time = Vec::new();
        for i in 0..2 {
            let q = format!("{p}.time_embed.{i}");
            let freqs = loader.device.to_vec_f32(&loader.tensor_f32(&format!("{q}.freqs"))?)?;
            if freqs.len() * 2 != cfg.frequency_embedding_size {
                lapid::bail!(
                    "shape_mismatch",
                    "{q}.freqs has {} entries, expected {}",
                    freqs.len(),
                    cfg.frequency_embedding_size / 2
                );
            }
            time.push(TimeEmbed {
                freqs,
                fc1: loader.linear(&format!("{q}.mlp.0"), true)?,
                fc2: loader.linear(&format!("{q}.mlp.2"), true)?,
                norm: RmsNorm::new(loader.tensor(&format!("{q}.mlp.3.alpha"))?, cfg.time_norm_eps),
            });
        }
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let q = format!("res_blocks.{i}");
            blocks.push(ResBlock {
                norm: loader.layer_norm(&format!("{p}.{q}.in_ln"), cfg.norm_eps)?,
                fc1: lin(&format!("{q}.mlp.0"))?,
                fc2: lin(&format!("{q}.mlp.2"))?,
                ada: lin(&format!("{q}.adaLN_modulation.1"))?,
            });
        }
        let d = cfg.dim;
        Ok(FlowHead {
            input_proj: lin("input_proj")?,
            cond_embed: lin("cond_embed")?,
            time,
            blocks,
            final_ada: lin("final_layer.adaLN_modulation.1")?,
            final_norm: LayerNorm::new(
                Tensor::full(1.0, &[d], DType::F32),
                Tensor::full(0.0, &[d], DType::F32),
                cfg.norm_eps,
            ),
            final_linear: lin("final_layer.linear")?,
            dim: d,
        })
    }

    /// Time condition of a step from `s` to `t`: the mean of the two time embeddings [1, dim].
    pub fn time_condition(&self, s: f32, t: f32) -> Result<Tensor> {
        self.time[0].forward(s)?.add(&self.time[1].forward(t)?)?.mul_scalar(0.5)
    }

    fn chunk(&self, x: &Tensor, i: usize) -> Result<Tensor> {
        x.slice(-1, i * self.dim, (i + 1) * self.dim)
    }

    /// Velocity for latents x [N, ldim], backbone state c [N, D], time condition [1, dim].
    pub fn velocity(&self, x: &Tensor, c: &Tensor, time_cond: &Tensor) -> Result<Tensor> {
        let mut h = self.input_proj.forward(x)?;
        let y = self.cond_embed.forward(c)?.add(time_cond)?;
        let ys = silu(&y)?;
        for b in &self.blocks {
            let m = b.ada.forward(&ys)?;
            let (shift, scale, gate) = (self.chunk(&m, 0)?, self.chunk(&m, 1)?, self.chunk(&m, 2)?);
            let u = b.norm.forward(&h)?.mul(&scale.add_scalar(1.0)?)?.add(&shift)?;
            let u = b.fc2.forward(&silu(&b.fc1.forward(&u)?)?)?;
            h = h.add(&gate.mul(&u)?)?;
        }
        let m = self.final_ada.forward(&ys)?;
        let (shift, scale) = (self.chunk(&m, 0)?, self.chunk(&m, 1)?);
        let u = self.final_norm.forward(&h)?.mul(&scale.add_scalar(1.0)?)?.add(&shift)?;
        self.final_linear.forward(&u)
    }

    /// Latent from `noise` [N, ldim] in `time_conds.len()` Euler steps (LSD sampler).
    pub fn sample(&self, c: &Tensor, noise: &Tensor, time_conds: &[Tensor]) -> Result<Tensor> {
        let n = time_conds.len().max(1) as f64;
        let mut x = noise.clone();
        for tc in time_conds {
            x = x.add(&self.velocity(&x, c, tc)?.mul_scalar(1.0 / n)?)?;
        }
        Ok(x)
    }
}
