//! Activations built from lapid primitives (the compiler fuses them into neighbouring
//! kernels, e.g. a matmul epilogue).

use lapid::{BinaryOp, Result, Tensor, UnaryOp, bail};

/// 1 / (1 + exp(-x)).
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    x.neg()?.exp()?.add_scalar(1.0)?.unary(UnaryOp::Recip)
}

/// SiLU / swish: x * sigmoid(x).
pub fn silu(x: &Tensor) -> Result<Tensor> {
    x.mul(&sigmoid(x)?)
}

/// Exact (erf) GELU.
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    x.gelu()
}

/// Tanh-approximated GELU (`gelu_pytorch_tanh`, `gelu_new`):
/// 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    const C: f64 = 0.797_884_560_802_865_4; // sqrt(2 / pi)
    let x3 = x.mul(x)?.mul(x)?;
    let inner = x.add(&x3.mul_scalar(0.044_715)?)?.mul_scalar(C)?;
    x.mul(&inner.tanh()?.add_scalar(1.0)?.mul_scalar(0.5)?)
}

/// max(x, 0).
pub fn relu(x: &Tensor) -> Result<Tensor> {
    x.binary_scalar(BinaryOp::Max, 0.0)
}

/// ELU (alpha 1): max(x, 0) + exp(min(x, 0)) - 1.
pub fn elu(x: &Tensor) -> Result<Tensor> {
    let neg = x.binary_scalar(BinaryOp::Min, 0.0)?.exp()?.add_scalar(-1.0)?;
    relu(x)?.add(&neg)
}

/// Gated linear unit along `axis`: the first half times sigmoid of the second half.
pub fn glu(x: &Tensor, axis: isize) -> Result<Tensor> {
    let ax = x.axis(axis)?;
    let n = x.shape()[ax];
    if !n.is_multiple_of(2) {
        bail!("shape_mismatch", "glu: axis {ax} of {:?} has odd size {n}", x.shape());
    }
    let a = x.slice(ax as isize, 0, n / 2)?;
    let b = x.slice(ax as isize, n / 2, n)?;
    a.mul(&sigmoid(&b)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{assert_close, eval};

    fn input() -> (Tensor, Vec<f32>) {
        let v: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.7).collect();
        (Tensor::from_f32(&v, &[2, 8]).unwrap(), v)
    }

    #[test]
    fn sigmoid_silu_relu() {
        let (x, v) = input();
        let sig: Vec<f32> = v.iter().map(|&a| 1.0 / (1.0 + (-a).exp())).collect();
        assert_close(&eval(&sigmoid(&x).unwrap()), &sig, 1e-6);
        let si: Vec<f32> = v.iter().zip(&sig).map(|(a, s)| a * s).collect();
        assert_close(&eval(&silu(&x).unwrap()), &si, 1e-6);
        let re: Vec<f32> = v.iter().map(|&a| a.max(0.0)).collect();
        assert_close(&eval(&relu(&x).unwrap()), &re, 0.0);
    }

    #[test]
    fn elu_matches_formula() {
        let (x, v) = input();
        let want: Vec<f32> = v.iter().map(|&a| if a > 0.0 { a } else { a.exp() - 1.0 }).collect();
        assert_close(&eval(&elu(&x).unwrap()), &want, 1e-6);
    }

    #[test]
    fn gelu_tanh_matches_formula() {
        let (x, v) = input();
        let want: Vec<f32> = v
            .iter()
            .map(|&a| {
                let a = a as f64;
                let c = (2.0 / std::f64::consts::PI).sqrt();
                (0.5 * a * (1.0 + (c * (a + 0.044715 * a * a * a)).tanh())) as f32
            })
            .collect();
        assert_close(&eval(&gelu_tanh(&x).unwrap()), &want, 1e-5);
        // close to exact gelu
        let exact = eval(&gelu(&x).unwrap());
        assert_close(&eval(&gelu_tanh(&x).unwrap()), &exact, 1e-3);
    }

    #[test]
    fn glu_halves() {
        let (x, v) = input();
        let y = glu(&x, -1).unwrap();
        assert_eq!(y.shape(), &[2, 4]);
        let mut want = Vec::new();
        for r in 0..2 {
            for c in 0..4 {
                let a = v[r * 8 + c];
                let b = v[r * 8 + 4 + c];
                want.push(a / (1.0 + (-b).exp()));
            }
        }
        assert_close(&eval(&y), &want, 1e-6);
        assert!(glu(&Tensor::zeros(&[2, 3], lapid::DType::F32), -1).is_err());
    }
}
