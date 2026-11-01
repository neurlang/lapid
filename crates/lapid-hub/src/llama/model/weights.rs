//! Weight loading helpers: shape-checked layers, and the fake-quant reference for every
//! quantized format this module stores (K-quants and `[E, N, K]` expert stacks included,
//! which `Loader::tensor` (2-D int8 / int4 only) does not dequantize yet).

use lapid::{DType, Error, Result, Tensor};
use lapid_nn::{Linear, Loader, Precision, RmsNorm};

/// Weight `name` on the device (dequantized to f32 in the fake-quant precisions).
pub(crate) fn weight(l: &Loader<'_>, name: &str) -> Result<Tensor> {
    if l.precision == Precision::Stored {
        return l.tensor(name);
    }
    let t = l.file.get(name)?;
    let Some(fmt) = t.dtype().quant_format() else { return l.tensor(name) };
    let shape = t.shape().to_vec();
    let (mats, rows, cols) = match shape.as_slice() {
        [r, c] => (1, *r, *c),
        [e, r, c] => (*e, *r, *c),
        _ => return Err(Error::new("shape_mismatch", format!("{name}: quantized weight must be 2-D or 3-D, got {shape:?}"))),
    };
    if mats == 1 && !fmt.is_aos() {
        return l.tensor(name);
    }
    let bytes = l.file.bytes(name)?;
    let per = lapid::quant::layout(fmt, rows, cols).total;
    if bytes.len() != per * mats {
        return Err(Error::new("format", format!("{name}: {} bytes, layout needs {}", bytes.len(), per * mats)));
    }
    let mut data = Vec::with_capacity(mats * rows * cols);
    for m in 0..mats {
        data.extend(lapid::quant::dequantize(fmt, &bytes[m * per..(m + 1) * per], rows, cols));
    }
    l.device.upload(&Tensor::from_f32(&data, &shape)?)
}

pub(crate) fn tensor(l: &Loader<'_>, name: &str, shape: &[usize]) -> Result<Tensor> {
    let t = weight(l, name)?;
    l.expect_shape(&t, name, shape)?;
    Ok(t)
}

pub(crate) fn f32_param(l: &Loader<'_>, name: &str, shape: &[usize]) -> Result<Tensor> {
    tensor(l, name, shape)?.cast(DType::F32)
}

/// `{name}.weight` [out, inp] plus `{name}.bias` [out] (f32) when `bias`.
pub(crate) fn linear(l: &Loader<'_>, name: &str, out: usize, inp: usize, bias: bool) -> Result<Linear> {
    let w = format!("{name}.weight");
    let b = if bias { Some(f32_param(l, &format!("{name}.bias"), &[out])?) } else { None };
    let lin = Linear::new(weight(l, &w)?, b).with_act_quant(l.act_quant(&w)?);
    l.expect_shape(&lin.weight, &w, &[out, inp])?;
    Ok(lin)
}

pub(crate) fn norm(l: &Loader<'_>, name: &str, d: usize, eps: f32) -> Result<RmsNorm> {
    let n = l.rms_norm(name, eps)?;
    l.expect_shape(&n.weight, &format!("{name}.weight"), &[d])?;
    Ok(n)
}
