//! Feed-forward blocks: `Mlp` (`down(act(up x))`, Whisper / BERT / conformer style) and
//! `GatedMlp` (`down(act(gate x) * up x)`: SwiGLU with SiLU, GeGLU with tanh-GELU).

use lapid::{Result, Tensor};

use crate::layers::Linear;
use crate::ops;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Activation {
    /// Exact (erf) GELU.
    Gelu,
    /// Tanh-approximated GELU (`gelu_pytorch_tanh`).
    GeluTanh,
    /// SiLU / swish.
    Silu,
    Relu,
}

impl Activation {
    pub fn apply(self, x: &Tensor) -> Result<Tensor> {
        match self {
            Activation::Gelu => ops::gelu(x),
            Activation::GeluTanh => ops::gelu_tanh(x),
            Activation::Silu => ops::silu(x),
            Activation::Relu => ops::relu(x),
        }
    }

    /// HF `hidden_act` / `hidden_activation` names.
    pub fn from_name(name: &str) -> Option<Activation> {
        Some(match name {
            "gelu" => Activation::Gelu,
            "gelu_pytorch_tanh" | "gelu_new" | "gelu_tanh" => Activation::GeluTanh,
            "silu" | "swish" => Activation::Silu,
            "relu" => Activation::Relu,
            _ => return None,
        })
    }
}

/// down(act(up(x))).
#[derive(Clone)]
pub struct Mlp {
    pub up: Linear,
    pub down: Linear,
    pub act: Activation,
}

impl Mlp {
    pub fn new(up: Linear, down: Linear, act: Activation) -> Mlp {
        Mlp { up, down, act }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down.forward(&self.act.apply(&self.up.forward(x)?)?)
    }
}

/// down(act(gate(x)) * up(x)).
#[derive(Clone)]
pub struct GatedMlp {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    pub act: Activation,
}

impl GatedMlp {
    pub fn new(gate: Linear, up: Linear, down: Linear, act: Activation) -> GatedMlp {
        GatedMlp { gate, up, down, act }
    }

    /// SwiGLU (LLaMA / LFM): SiLU gate.
    pub fn swiglu(gate: Linear, up: Linear, down: Linear) -> GatedMlp {
        GatedMlp::new(gate, up, down, Activation::Silu)
    }

    /// GeGLU (Gemma): tanh-GELU gate.
    pub fn geglu(gate: Linear, up: Linear, down: Linear) -> GatedMlp {
        GatedMlp::new(gate, up, down, Activation::GeluTanh)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.act.apply(&self.gate.forward(x)?)?;
        self.down.forward(&g.mul(&self.up.forward(x)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{assert_close, eval, matmul_t, rand_vec};

    fn lin(out: usize, inp: usize, seed: u64) -> (Linear, Vec<f32>) {
        let w = rand_vec(out * inp, seed, 0.5);
        (Linear::new(Tensor::from_f32(&w, &[out, inp]).unwrap(), None), w)
    }

    fn act_ref(a: Activation, x: f32) -> f32 {
        let x64 = x as f64;
        (match a {
            Activation::Gelu => 0.5 * x64 * (1.0 + erf(x64 / 2f64.sqrt())),
            Activation::GeluTanh => {
                let c = (2.0 / std::f64::consts::PI).sqrt();
                0.5 * x64 * (1.0 + (c * (x64 + 0.044715 * x64.powi(3))).tanh())
            }
            Activation::Silu => x64 / (1.0 + (-x64).exp()),
            Activation::Relu => x64.max(0.0),
        }) as f32
    }

    /// Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7).
    fn erf(x: f64) -> f64 {
        let s = x.signum();
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let y = 1.0
            - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
                + 0.254_829_592)
                * t
                * (-x * x).exp();
        s * y
    }

    #[test]
    fn mlp_matches_reference() {
        let (d, f, n) = (8, 16, 3);
        let xv = rand_vec(n * d, 1, 1.0);
        let x = Tensor::from_f32(&xv, &[1, n, d]).unwrap();
        for act in [Activation::Gelu, Activation::GeluTanh, Activation::Silu, Activation::Relu] {
            let (up, uw) = lin(f, d, 2);
            let (down, dw) = lin(d, f, 3);
            let got = eval(&Mlp::new(up, down, act).forward(&x).unwrap());
            let h: Vec<f32> = matmul_t(&xv, &uw, n, d, f).into_iter().map(|v| act_ref(act, v)).collect();
            assert_close(&got, &matmul_t(&h, &dw, n, f, d), 1e-4);
        }
    }

    #[test]
    fn gated_mlp_matches_reference() {
        let (d, f, n) = (8, 12, 2);
        let xv = rand_vec(n * d, 4, 1.0);
        let x = Tensor::from_f32(&xv, &[n, d]).unwrap();
        for (mk, act) in [
            (GatedMlp::swiglu as fn(Linear, Linear, Linear) -> GatedMlp, Activation::Silu),
            (GatedMlp::geglu, Activation::GeluTanh),
        ] {
            let (gate, gw) = lin(f, d, 5);
            let (up, uw) = lin(f, d, 6);
            let (down, dw) = lin(d, f, 7);
            let got = eval(&mk(gate, up, down).forward(&x).unwrap());
            let g = matmul_t(&xv, &gw, n, d, f);
            let u = matmul_t(&xv, &uw, n, d, f);
            let h: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| act_ref(act, a) * b).collect();
            assert_close(&got, &matmul_t(&h, &dw, n, f, d), 1e-4);
        }
    }

    #[test]
    fn activation_names() {
        assert_eq!(Activation::from_name("gelu_pytorch_tanh"), Some(Activation::GeluTanh));
        assert_eq!(Activation::from_name("silu"), Some(Activation::Silu));
        assert_eq!(Activation::from_name("mish"), None);
    }
}
