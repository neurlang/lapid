//! Kernel micro-benchmarks on Whisper-small shapes (GPU time per launch, median of runs).
//! `cargo test -p lapid-metal --release bench_ -- --ignored --nocapture`

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::{DType, DeviceData};

use super::keys::mat_spec;
use super::util::*;
use crate::MetalBackend;

fn time(be: &MetalBackend, kernel: Kernel, inputs: &[Vec<u8>], out_bytes: usize, runs: usize) -> f64 {
    let mut bufs: Vec<DeviceData> = inputs.iter().map(|b| be.upload(b).unwrap()).collect();
    bufs.push(be.alloc(out_bytes).unwrap());
    let launch = Launch { kernel, inputs: (0..inputs.len()).collect(), output: inputs.len() };
    let mut t: Vec<f64> = (0..runs).map(|_| be.time_launches(std::slice::from_ref(&launch), &bufs).unwrap()[0]).collect();
    t.sort_by(|a, b| a.total_cmp(b));
    t[runs / 2]
}

/// Wall time per launch of `reps` identical back-to-back launches in one execute (min of 5):
/// small kernels as they run in a decoder step.
fn time_batched(be: &MetalBackend, kernel: Kernel, inputs: &[Vec<u8>], out_bytes: usize, reps: usize) -> f64 {
    let mut bufs: Vec<DeviceData> = inputs.iter().map(|b| be.upload(b).unwrap()).collect();
    bufs.push(be.alloc(out_bytes).unwrap());
    let launch = Launch { kernel, inputs: (0..inputs.len()).collect(), output: inputs.len() };
    let launches = vec![launch; reps];
    (0..6)
        .map(|_| {
            let t = std::time::Instant::now();
            be.execute(&launches, &bufs).unwrap();
            t.elapsed().as_secs_f64() * 1e3 / reps as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn weights(b: DType, n: usize, k: usize) -> Vec<u8> {
    let raw = rand(n * k, 1);
    match b {
        DType::Int8 => lapid_quant::quantize(lapid_quant::Format::Int8, &raw, n, k),
        DType::Int4 => lapid_quant::quantize(lapid_quant::Format::Int4, &raw, n, k),
        d if d.is_quant() => super::gpu_kquant::kq_matrix(d, n, k, 1).0,
        _ => encode(b, &raw),
    }
}

fn bench_gemv(be: &MetalBackend, b: DType, m: usize, n: usize, k: usize) {
    let bbytes = weights(b, n, k);
    let spec = mat_spec(DType::F32, b, true, vec![], None, DType::F32);
    let dims = MatMulDims {
        batch: 1,
        m,
        n,
        k,
        a: MatView { offset: 0, batch_stride: 0, row_stride: k },
        b: MatView { offset: 0, batch_stride: 0, row_stride: k },
        extra_views: vec![],
        out: MatView::contiguous(m, n),
    };
    let ins = [encode(DType::F32, &rand(m * k, 2)), bbytes.clone()];
    let ms = time_batched(be, Kernel::MatMul(spec.clone(), dims.clone()), &ins, m * n * 4, 200);
    let gpu = time(be, Kernel::MatMul(spec, dims), &ins, m * n * 4, 21);
    let gbs = bbytes.len() as f64 / ms / 1e6;
    let gpu_gbs = bbytes.len() as f64 / gpu / 1e6;
    eprintln!("gemv {b:?} m{m} n{n} k{k}: batched {:.1} us {gbs:.0} GB/s, single GPU {:.1} us {gpu_gbs:.0} GB/s", ms * 1e3, gpu * 1e3);
}

fn bench_matmul(be: &MetalBackend, a: DType, b: DType, m: usize, n: usize, k: usize) {
    let bbytes = weights(b, n, k);
    let spec = mat_spec(a, b, true, vec![], None, DType::F32);
    let dims = MatMulDims {
        batch: 1,
        m,
        n,
        k,
        a: MatView { offset: 0, batch_stride: 0, row_stride: k },
        b: MatView { offset: 0, batch_stride: 0, row_stride: k },
        extra_views: vec![],
        out: MatView::contiguous(m, n),
    };
    let ms = time(be, Kernel::MatMul(spec, dims), &[encode(a, &rand(m * k, 2)), bbytes], m * n * 4, 9);
    let gflops = 2.0 * (m * n * k) as f64 / ms / 1e6;
    let gbs = ((n * k) as f64 * if b.is_quant() { 1.0 } else { b.size() as f64 }) / ms / 1e6;
    eprintln!("matmul {a:?}x{b:?} m{m} n{n} k{k}: {ms:.3} ms  {gflops:.0} GFLOP/s  {gbs:.0} GB/s(B)");
}

fn bench_attention(be: &MetalBackend, h: usize, sq: usize, sk: usize, d: usize, bshd: bool) {
    let spec = Arc::new(AttentionSpec { dtype: DType::F32, mask_dtype: None, causal: false, out_dtype: DType::F32 });
    let view = |s: usize| {
        if bshd {
            View { offset: 0, strides: vec![(s * h * d) as isize, d as isize, (h * d) as isize, 1] }
        } else {
            View::contiguous(&[1, h, s, d])
        }
    };
    let dims = AttentionDims { batch: 1, heads: h, sq, sk, d, scale: 0.125f32.to_bits(), q: view(sq), k: view(sk), v: view(sk), mask: None };
    let ins = [encode(DType::F32, &rand(h * sq * d, 3)), encode(DType::F32, &rand(h * sk * d, 4)), encode(DType::F32, &rand(h * sk * d, 5))];
    let ms = time(be, Kernel::Attention(spec, dims), &ins, h * sq * d * 4, 9);
    let gflops = 4.0 * (h * sq * sk * d) as f64 / ms / 1e6;
    eprintln!("attention h{h} sq{sq} sk{sk} d{d} bshd={bshd}: {ms:.3} ms  {gflops:.0} GFLOP/s");
}

fn bench_matmul_quiet(be: &MetalBackend) {
    let (m, n, k) = (1500, 3072, 768);
    let spec = mat_spec(DType::F32, DType::F16, true, vec![], None, DType::F32);
    let dims = MatMulDims {
        batch: 1,
        m,
        n,
        k,
        a: MatView { offset: 0, batch_stride: 0, row_stride: k },
        b: MatView { offset: 0, batch_stride: 0, row_stride: k },
        extra_views: vec![],
        out: MatView::contiguous(m, n),
    };
    time(be, Kernel::MatMul(spec, dims), &[encode(DType::F32, &rand(m * k, 2)), encode(DType::F16, &rand(n * k, 1))], m * n * 4, 3);
}

#[test]
#[ignore]
fn bench_whisper_shapes() {
    let Some(be) = backend() else { return };
    // Warm up (GPU clocks ramp up over ~100 ms of work).
    for _ in 0..20 {
        bench_matmul_quiet(&be);
    }
    for b in [DType::F16, DType::Int8, DType::Int4, DType::F32] {
        bench_matmul(&be, DType::F32, b, 1500, 768, 768);
        bench_matmul(&be, DType::F32, b, 1500, 3072, 768);
        bench_matmul(&be, DType::F32, b, 1500, 768, 3072);
    }
    for b in [DType::F16, DType::Int8, DType::Int4] {
        bench_gemv(&be, b, 1, 768, 768);
        bench_gemv(&be, b, 1, 3072, 768);
        bench_gemv(&be, b, 1, 768, 3072);
        bench_gemv(&be, b, 1, 51865, 768);
        bench_gemv(&be, b, 3, 3072, 768);
    }
    bench_attention(&be, 12, 1500, 1500, 64, true);
    bench_attention(&be, 12, 1500, 1500, 64, false);
    bench_attention(&be, 12, 1, 1500, 64, true);
    bench_attention(&be, 12, 1, 100, 64, true);
}

/// LFM3-1.7B decode / prefill shapes (hidden 2048, ffn 7680, conv in_proj 6144, kv 512).
/// `cargo test -p lapid-metal --release bench_lfm3 -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_lfm3_shapes() {
    let Some(be) = backend() else { return };
    for _ in 0..20 {
        bench_matmul_quiet(&be);
    }
    for b in [DType::F16, DType::Int8, DType::Int4, DType::Q4K, DType::Q6K] {
        for &(n, k) in &[(2048usize, 2048usize), (512, 2048), (6144, 2048), (7680, 2048), (2048, 7680), (65536, 2048)] {
            bench_gemv(&be, b, 1, n, k);
        }
    }
    for b in [DType::F16, DType::Int8, DType::Q4K] {
        bench_matmul(&be, DType::F32, b, 512, 7680, 2048);
        bench_matmul(&be, DType::F32, b, 16, 7680, 2048);
    }
}

/// GEMV over weights that do not fit the system-level cache: `copies` distinct matrices
/// (>= 256 MB in total) multiplied back to back in one execute, as in a decode step.
/// Returns DRAM GB/s (wall time of the execute, best of 5).
fn bench_gemv_stream(be: &MetalBackend, b: DType, n: usize, k: usize) {
    let w = weights(b, n, k);
    let copies = (256usize << 20).div_ceil(w.len()).max(4);
    let mut bufs: Vec<DeviceData> = vec![be.upload(&encode(DType::F32, &rand(k, 2))).unwrap()];
    for _ in 0..copies {
        bufs.push(be.upload(&w).unwrap());
    }
    // LAPID_BENCH_INDEPENDENT=1: one output per launch (no barriers between the GEMVs).
    let independent = std::env::var("LAPID_BENCH_INDEPENDENT").is_ok();
    let out = bufs.len();
    for _ in 0..if independent { copies } else { 1 } {
        bufs.push(be.alloc(n * 4).unwrap());
    }
    let spec = mat_spec(DType::F32, b, true, vec![], None, DType::F32);
    let dims = MatMulDims {
        batch: 1,
        m: 1,
        n,
        k,
        a: MatView { offset: 0, batch_stride: 0, row_stride: k },
        b: MatView { offset: 0, batch_stride: 0, row_stride: k },
        extra_views: vec![],
        out: MatView::contiguous(1, n),
    };
    let launches: Vec<Launch> =
        (0..copies).map(|c| Launch { kernel: Kernel::MatMul(spec.clone(), dims.clone()), inputs: vec![0, 1 + c], output: if independent { out + c } else { out } }).collect();
    let ms = (0..6)
        .map(|_| {
            let t = std::time::Instant::now();
            be.execute(&launches, &bufs).unwrap();
            t.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min);
    let gbs = (w.len() * copies) as f64 / ms / 1e6;
    eprintln!("gemv stream {b:?} n{n} k{k} x{copies}: {:.1} us each, {gbs:.0} GB/s", ms * 1e3 / copies as f64);
}

#[test]
#[ignore]
fn bench_gemv_dram() {
    let Some(be) = backend() else { return };
    for _ in 0..20 {
        bench_matmul_quiet(&be);
    }
    for b in [DType::F16, DType::Int8, DType::Int4, DType::Q4K, DType::Q5K, DType::Q6K] {
        for &(n, k) in &[(2048usize, 2048usize), (512, 2048), (7680, 2048), (2048, 7680)] {
            bench_gemv_stream(&be, b, n, k);
        }
    }
}
