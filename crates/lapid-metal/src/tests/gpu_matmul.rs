//! GPU tests: GEMM / GEMV for every B dtype against a naive f64 reference.

use lapid_compiler::*;
use lapid_tensor::{BinaryOp, DType};

use super::keys::{gelu_expr, load, mat_spec};
use super::util::*;
use crate::MetalBackend;

/// B values as the kernel sees them and the bytes to upload. Float B is stored [n, k]
/// (transposed) or [k, n].
fn make_b(dt: DType, n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let raw = rand(n * k, seed);
    match dt {
        DType::Int8 | DType::Int4 => {
            let f = if dt == DType::Int8 { lapid_quant::Format::Int8 } else { lapid_quant::Format::Int4 };
            let q = lapid_quant::quantize(f, &raw, n, k);
            let d = lapid_quant::dequantize(f, &q, n, k);
            (q, d)
        }
        _ => (encode(dt, &raw), round_to(dt, &raw)),
    }
}

struct Case {
    a: DType,
    b: DType,
    bt: bool,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    /// bias [n] + gelu epilogue
    epilogue: bool,
    out: DType,
}

fn check(be: &MetalBackend, c: &Case, tol: f32) {
    let (batch, m, n, k) = (c.batch, c.m, c.n, c.k);
    // A: [batch, m, k] stored with a padded row stride and offset.
    let a_rs = k + 3;
    let a_off = 5;
    let a_bs = m * a_rs;
    let a_vals = round_to(c.a, &rand(a_off + batch * a_bs, 11));
    let quant = c.b.is_quant();
    let b_batch = if quant { 1 } else { batch };
    // B: per batch [n, k] (transposed) or [k, n]; float B gets an offset and padded rows.
    let (b_off, b_rs) = if quant { (0, k) } else if c.bt { (7, k + 1) } else { (7, n + 2) };
    let b_bs = if quant { 0 } else if c.bt { n * b_rs } else { k * b_rs };
    let (b_bytes, b_vals) = if quant {
        make_b(c.b, n, k, 12)
    } else {
        let (_, v) = make_b(c.b, b_off + b_batch * b_bs, 1, 12);
        (encode(c.b, &v), v)
    };
    let bias = rand(n, 13);
    let extra = View { offset: 0, strides: vec![0, 0, 1] };
    let spec = mat_spec(
        c.a,
        c.b,
        c.bt,
        if c.epilogue { vec![DType::F32] } else { vec![] },
        c.epilogue.then(|| Expr::Binary(BinaryOp::Add, Box::new(gelu_expr(0)), load(1))),
        c.out,
    );
    let dims = MatMulDims {
        batch,
        m,
        n,
        k,
        a: MatView { offset: a_off, batch_stride: a_bs, row_stride: a_rs },
        b: MatView { offset: b_off, batch_stride: b_bs, row_stride: b_rs },
        extra_views: if c.epilogue { vec![extra] } else { vec![] },
        out: MatView::contiguous(m, n),
    };
    let mut ins = vec![encode(c.a, &a_vals), b_bytes];
    if c.epilogue {
        ins.push(encode(DType::F32, &bias));
    }
    let total = batch * m * n;
    let got = decode(c.out, &run(be, Kernel::MatMul(spec, dims), &ins, total * c.out.size()), total);
    let mut want = vec![0f32; total];
    for bt in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f64;
                for kk in 0..k {
                    let av = a_vals[a_off + bt * a_bs + i * a_rs + kk] as f64;
                    let bv = if quant {
                        b_vals[j * k + kk]
                    } else if c.bt {
                        b_vals[b_off + bt * b_bs + j * b_rs + kk]
                    } else {
                        b_vals[b_off + bt * b_bs + kk * b_rs + j]
                    } as f64;
                    s += av * bv;
                }
                if c.epilogue {
                    s = gelu_ref(s) + bias[j] as f64;
                }
                want[(bt * m + i) * n + j] = s as f32;
            }
        }
    }
    let what = format!("matmul {:?}x{:?} bt={} b{} m{} n{} k{} epi={} -> {:?}", c.a, c.b, c.bt, batch, m, n, k, c.epilogue, c.out);
    assert_close(&what, &got, &round_to(c.out, &want), tol);
}

/// Tolerance (scaled by max |want|) for B tiles staged as f16 (quantized / f16 weights).
fn tol_for(b: DType, out: DType) -> f32 {
    let t: f32 = match b {
        DType::F16 | DType::Int8 | DType::Int4 => 2e-3,
        _ => 2e-5,
    };
    if out == DType::F32 { t } else { t.max(8e-3) }
}

#[test]
fn gemm_float_b() {
    let Some(be) = backend() else { return };
    for b in [DType::F32, DType::F16, DType::BF16] {
        for bt in [true, false] {
            for &(batch, m, n, k) in &[(2usize, 70usize, 100usize, 96usize), (1, 130, 65, 33), (3, 5, 9, 7)] {
                for a in [DType::F32, DType::F16] {
                    let c = Case { a, b, bt, batch, m, n, k, epilogue: bt && m == 70, out: DType::F32 };
                    check(&be, &c, tol_for(b, DType::F32));
                }
            }
        }
    }
}

#[test]
fn gemm_quantized_b() {
    let Some(be) = backend() else { return };
    for b in [DType::Int8, DType::Int4] {
        for &(batch, m, n, k, epi) in &[(1usize, 70usize, 72usize, 128usize, true), (2, 1500 / 10, 130, 96, false), (1, 64, 64, 32, false)] {
            let c = Case { a: DType::F32, b, bt: true, batch, m, n, k, epilogue: epi, out: DType::F32 };
            check(&be, &c, tol_for(b, DType::F32));
        }
        let c = Case { a: DType::F16, b, bt: true, batch: 1, m: 33, n: 40, k: 64, epilogue: true, out: DType::F16 };
        check(&be, &c, tol_for(b, DType::F16));
    }
}

#[test]
fn gemv_every_b_dtype() {
    let Some(be) = backend() else { return };
    for b in [DType::F32, DType::F16, DType::BF16, DType::Int8, DType::Int4] {
        for m in 1..=4 {
            let ks: &[usize] = if b.is_quant() { &[96, 768] } else { &[96, 40, 13] };
            for &k in ks {
                for a in [DType::F32, DType::F16] {
                    let c = Case { a, b, bt: true, batch: if b.is_quant() { 1 } else { 2 }, m, n: 37, k, epilogue: m == 2, out: DType::F32 };
                    let tol = if b == DType::BF16 { 1e-5 } else { 2e-5 };
                    check(&be, &c, tol);
                }
            }
        }
    }
}
