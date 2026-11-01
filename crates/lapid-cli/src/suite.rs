//! The kernel suite of `lapid kernels dump` / `lapid bench kernels`: small graphs built from
//! lapid primitives at typical transformer shapes (decode GEMV, prefill GEMM in fp16 / int8 /
//! int4, attention, softmax, layer norm, fused elementwise). Generic: no model code.

use std::sync::Arc;

use lapid::{DType, HostData, Result, Tensor};

/// One case: the graph outputs plus the work it does (for TFLOP/s and GB/s columns).
pub struct Case {
    pub name: String,
    pub outputs: Vec<Tensor>,
    pub flops: f64,
    pub bytes: f64,
}

/// Deterministic pseudo-random values in [-1, 1).
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn f32_tensor(shape: &[usize], seed: u64) -> Result<Tensor> {
    Tensor::from_f32(&values(shape.iter().product(), seed), shape)
}

/// A `[rows, cols]` weight in `dtype` (fp32, fp16, int8 or int4).
fn weight(rows: usize, cols: usize, dtype: DType, seed: u64) -> Result<Tensor> {
    let v = values(rows * cols, seed);
    let bytes: Vec<u8> = match dtype {
        DType::F16 => v.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect(),
        DType::Int8 => lapid::quant::quantize(lapid::quant::Format::Int8, &v, rows, cols),
        DType::Int4 => lapid::quant::quantize(lapid::quant::Format::Int4, &v, rows, cols),
        _ => return f32_tensor(&[rows, cols], seed),
    };
    Tensor::from_host(HostData::new(Arc::new(bytes)), &[rows, cols], dtype)
}

fn weight_bytes(rows: usize, cols: usize, dtype: DType) -> f64 {
    dtype.storage_bytes(&[rows, cols]) as f64
}

fn linear(name: &str, m: usize, k: usize, n: usize, dtype: DType) -> Result<Case> {
    let x = f32_tensor(&[m, k], 1)?;
    let w = weight(n, k, dtype, 2)?;
    let y = x.matmul_t(&w)?;
    Ok(Case {
        name: format!("{name} {m}x{k}x{n} {}", dtype.name()),
        outputs: vec![y],
        flops: 2.0 * (m * k * n) as f64,
        bytes: weight_bytes(n, k, dtype) + 4.0 * (m * k + m * n) as f64,
    })
}

fn attention(name: &str, heads: usize, sq: usize, sk: usize, d: usize, causal: bool) -> Result<Case> {
    let q = f32_tensor(&[1, heads, sq, d], 3)?;
    let k = f32_tensor(&[1, heads, sk, d], 4)?;
    let v = f32_tensor(&[1, heads, sk, d], 5)?;
    let o = Tensor::attention(&q, &k, &v, None, 1.0 / (d as f32).sqrt(), causal)?;
    Ok(Case {
        name: format!("{name} h{heads} q{sq} k{sk} d{d}"),
        outputs: vec![o],
        flops: 4.0 * (heads * sq * sk * d) as f64,
        bytes: 4.0 * (heads * (2 * sq + 2 * sk) * d) as f64,
    })
}

fn rows_case(name: &str, rows: usize, cols: usize, f: impl Fn(&Tensor) -> Result<Tensor>) -> Result<Case> {
    let x = f32_tensor(&[rows, cols], 6)?;
    Ok(Case {
        name: format!("{name} {rows}x{cols}"),
        outputs: vec![f(&x)?],
        flops: 0.0,
        bytes: 8.0 * (rows * cols) as f64,
    })
}

/// Every case, in a stable order. `filter` keeps cases whose name contains it.
pub fn cases(filter: Option<&str>) -> Result<Vec<Case>> {
    let mut out = Vec::new();
    for dtype in [DType::F16, DType::Int8, DType::Int4] {
        out.push(linear("gemv", 1, 1024, 4096, dtype)?);
        out.push(linear("gemv", 1, 4096, 1024, dtype)?);
        out.push(linear("gemm", 512, 1024, 4096, dtype)?);
    }
    out.push(attention("attn-decode", 16, 1, 1024, 64, false)?);
    out.push(attention("attn-prefill", 16, 512, 512, 64, true)?);
    out.push(rows_case("softmax", 512, 1024, |x| x.softmax())?);
    out.push(rows_case("layernorm", 512, 1024, |x| {
        let w = f32_tensor(&[1024], 7)?;
        let b = f32_tensor(&[1024], 8)?;
        x.layer_norm(&w, &b, 1e-5)
    })?);
    out.push(rows_case("gelu-mul", 512, 4096, |x| x.gelu()?.mul(x))?);
    out.retain(|c| filter.is_none_or(|f| c.name.contains(f)));
    Ok(out)
}
