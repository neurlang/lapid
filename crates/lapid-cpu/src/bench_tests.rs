//! Kernel micro-benchmarks on Whisper-small decoder / cross-K/V shapes (ignored tests).
//! `cargo test --release -p lapid-cpu bench_ -- --ignored --nocapture --test-threads=1`

use std::sync::Arc;
use std::time::Instant;

use lapid_compiler::*;
use lapid_quant::Format;
use lapid_tensor::{DType, DeviceData};
use half::f16;

use super::*;

fn backend() -> CpuBackend {
    CpuBackend::new()
}

fn rnd(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn up_f32(be: &CpuBackend, v: &[f32]) -> DeviceData {
    be.upload(bytemuck::cast_slice(v)).unwrap()
}

fn up_f16(be: &CpuBackend, v: &[f32]) -> DeviceData {
    let h: Vec<f16> = v.iter().map(|&x| f16::from_f32(x)).collect();
    be.upload(bytemuck::cast_slice(&h)).unwrap()
}

/// Median ms of `reps` executions of `launches` (after warmup).
fn time(be: &CpuBackend, name: &str, launches: &[Launch], bufs: &[DeviceData], bytes: f64) {
    be.prepare(launches).unwrap();
    for _ in 0..3 {
        be.execute(launches, bufs).unwrap();
    }
    let mut v = Vec::new();
    for _ in 0..30 {
        let t = Instant::now();
        be.execute(launches, bufs).unwrap();
        v.push(t.elapsed().as_secs_f64() * 1e3 / launches.len() as f64);
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let ms = v[v.len() / 2];
    eprintln!("{name:<40} {ms:>8.4} ms (min {:.4})  {:>6.1} GB/s", v[0], bytes / ms / 1e6);
}

fn mm(b_dtype: DType, m: usize, n: usize, k: usize) -> Kernel {
    Kernel::MatMul(
        Arc::new(MatMulSpec {
            a_dtype: DType::F32,
            b_dtype,
            b_transposed: true,
            extra_dtypes: vec![],
            out_dtype: DType::F32,
            epilogue: None,
        }),
        MatMulDims {
            batch: 1,
            m,
            n,
            k,
            a: MatView { offset: 0, batch_stride: 0, row_stride: k },
            b: MatView { offset: 0, batch_stride: 0, row_stride: k },
            extra_views: vec![],
            out: MatView::contiguous(m, n),
        },
    )
}

#[test]
#[ignore]
fn bench_gemv() {
    let be = backend();
    for (n, k) in [(768usize, 768usize), (3072, 768), (768, 3072), (51865, 768)] {
        let w = rnd(n * k, 7);
        let a = up_f32(&be, &rnd(k, 3));
        let out = be.alloc(n * 4).unwrap();
        // 8 distinct weight matrices (cycled) so small ones do not live in cache.
        let copies = if n * k < 4_000_000 { 8 } else { 1 };
        for (dt, name) in [(DType::F16, "f16"), (DType::Int8, "int8")] {
            let mut bufs = vec![a.clone(), out.clone()];
            let mut launches = Vec::new();
            for _ in 0..copies {
                let b = match dt {
                    DType::F16 => up_f16(&be, &w),
                    _ => be.upload(&lapid_quant::quantize(Format::Int8, &w, n, k)).unwrap(),
                };
                bufs.push(b);
                launches.push(Launch { kernel: mm(dt, 1, n, k), inputs: vec![0, bufs.len() - 1], output: 1 });
            }
            let wbytes = (n * k) as f64 * if dt == DType::F16 { 2.0 } else { 1.0625 };
            time(&be, &format!("gemv {name} n={n} k={k}"), &launches, &bufs, wbytes);
        }
    }
}

#[test]
#[ignore]
fn bench_cross() {
    let be = backend();
    let (h, s, d) = (12usize, 1500usize, 64usize);
    // Permute copy [1500, 12, 64] -> [12, 1500, 64] (cross K/V realization).
    let x = up_f32(&be, &rnd(h * s * d, 1));
    let out = be.alloc(h * s * d * 4).unwrap();
    let k = Kernel::Elementwise(
        Arc::new(EwSpec { rank: 3, in_dtypes: vec![DType::F32], out_dtype: DType::F32, expr: Expr::Load(0) }),
        EwDims {
            shape: vec![h, s, d],
            in_views: vec![View { offset: 0, strides: vec![d as isize, (h * d) as isize, 1] }],
            out_view: View::contiguous(&[h, s, d]),
        },
    );
    let l = [Launch { kernel: k, inputs: vec![0], output: 1 }];
    time(&be, "permute copy 12x1500x64", &l, &[x, out], (h * s * d * 8) as f64);

    // Single-query cross attention over 1500 keys.
    let q = up_f32(&be, &rnd(h * d, 2));
    let kk = up_f32(&be, &rnd(h * s * d, 3));
    let vv = up_f32(&be, &rnd(h * s * d, 4));
    let o = be.alloc(h * d * 4).unwrap();
    let kv = View::contiguous(&[1, h, s, d]);
    let att = Kernel::Attention(
        Arc::new(AttentionSpec { dtype: DType::F32, mask_dtype: None, causal: false, out_dtype: DType::F32 }),
        AttentionDims {
            batch: 1,
            heads: h,
            sq: 1,
            sk: s,
            d,
            scale: 0.125f32.to_bits(),
            q: View::contiguous(&[1, h, 1, d]),
            k: kv.clone(),
            v: kv,
            mask: None,
        },
    );
    let l = [Launch { kernel: att, inputs: vec![0, 1, 2], output: 3 }];
    time(&be, "attention sq=1 sk=1500", &l, &[q, kk, vv, o], (h * s * d * 8) as f64);

    // Cross K projection m=1500.
    let a = up_f32(&be, &rnd(s * 768, 5));
    let w = rnd(768 * 768, 6);
    let out = be.alloc(s * 768 * 4).unwrap();
    for (dt, name) in [(DType::F16, "f16"), (DType::Int8, "int8")] {
        let b = match dt {
            DType::F16 => up_f16(&be, &w),
            _ => be.upload(&lapid_quant::quantize(Format::Int8, &w, 768, 768)).unwrap(),
        };
        let l = [Launch { kernel: mm(dt, s, 768, 768), inputs: vec![0, 1], output: 2 }];
        time(&be, &format!("gemm {name} m=1500 n=768 k=768"), &l, &[a.clone(), b, out.clone()], 0.0);
    }
}

#[test]
#[ignore]
fn bench_encoder_attention() {
    let be = backend();
    let (h, s, d) = (12usize, 1500usize, 64usize);
    // q/k/v as permuted views of [1500, 12, 64] projection outputs.
    let pv = View { offset: 0, strides: vec![0, d as isize, (h * d) as isize, 1] };
    let q = up_f32(&be, &rnd(h * s * d, 2));
    let k = up_f32(&be, &rnd(h * s * d, 3));
    let v = up_f32(&be, &rnd(h * s * d, 4));
    let o = be.alloc(h * s * d * 4).unwrap();
    let att = Kernel::Attention(
        Arc::new(AttentionSpec { dtype: DType::F32, mask_dtype: None, causal: false, out_dtype: DType::F32 }),
        AttentionDims {
            batch: 1,
            heads: h,
            sq: s,
            sk: s,
            d,
            scale: 0.125f32.to_bits(),
            q: pv.clone(),
            k: pv.clone(),
            v: pv,
            mask: None,
        },
    );
    let l = [Launch { kernel: att, inputs: vec![0, 1, 2], output: 3 }];
    time(&be, "encoder attention 12x1500x1500x64", &l, &[q, k, v, o], 0.0);
}

#[test]
#[ignore]
fn bench_exp() {
    let mut v: Vec<f32> = (0..1 << 16).map(|i| -(i as f32) * 0.001).collect();
    for (name, f) in [("exp", crate::math::exp as fn(f32) -> f32), ("exp_neg", crate::math::exp_neg)] {
        let t = Instant::now();
        let mut acc = 0f32;
        for _ in 0..200 {
            for x in v.iter_mut() {
                *x = f(*x) - 1.0;
            }
            acc += v[7];
        }
        eprintln!("{name}: {:.3} ns/elem ({acc})", t.elapsed().as_secs_f64() * 1e9 / (200.0 * 65536.0));
    }
}

/// Encoder-shaped matmuls (m=1500). The rate column is GFLOP/s here. Quantized B uses f32
/// activations x dequantized weights unless `LAPID_CPU_INT8_ACT=1`.
#[test]
#[ignore]
fn bench_encoder_matmul() {
    let be = backend();
    let m = 1500;
    for (n, k) in [(768usize, 768usize), (3072, 768), (768, 3072)] {
        let a = up_f32(&be, &rnd(m * k, 5));
        let w = rnd(n * k, 6);
        let out = be.alloc(m * n * 4).unwrap();
        for (dt, name) in [(DType::F16, "f16"), (DType::Int8, "int8"), (DType::Int4, "int4")] {
            let b = match dt {
                DType::F16 => up_f16(&be, &w),
                DType::Int8 => be.upload(&lapid_quant::quantize(Format::Int8, &w, n, k)).unwrap(),
                _ => be.upload(&lapid_quant::quantize(Format::Int4, &w, n, k)).unwrap(),
            };
            let l = [Launch { kernel: mm(dt, m, n, k), inputs: vec![0, 1], output: 2 }];
            time(&be, &format!("gemm {name} m={m} n={n} k={k}"), &l, &[a.clone(), b, out.clone()], (2 * m * n * k) as f64 / 1e3);
        }
    }
}
