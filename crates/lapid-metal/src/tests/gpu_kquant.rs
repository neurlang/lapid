//! GPU tests: K-quant (Q4_K / Q5_K / Q6_K) GEMV, GEMM and gather against a naive reference
//! (dequantization transcribed from ggml's `dequantize_row_q*_K`).

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::DType;
use half::f16;

use super::keys::mat_spec;
use super::util::*;
use crate::MetalBackend;

pub(crate) const KQ: [DType; 3] = [DType::Q4K, DType::Q5K, DType::Q6K];

fn block_bytes(d: DType) -> usize {
    match d {
        DType::Q4K => 144,
        DType::Q5K => 176,
        DType::Q6K => 210,
        _ => unreachable!(),
    }
}

fn scale_min(j: usize, q: &[u8]) -> (f32, f32) {
    if j < 4 {
        ((q[j] & 63) as f32, (q[j + 4] & 63) as f32)
    } else {
        (((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)) as f32, ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as f32)
    }
}

fn h(b: &[u8], o: usize) -> f32 {
    f16::from_le_bytes([b[o], b[o + 1]]).to_f32()
}

/// ggml dequantization of one 256-element block.
fn deq_block(d: DType, b: &[u8], y: &mut [f32]) {
    match d {
        DType::Q4K | DType::Q5K => {
            let (dd, dmin) = (h(b, 0), h(b, 2));
            let scales = &b[4..16];
            let (qh, qs) = if d == DType::Q5K { (&b[16..48], &b[48..176]) } else { (&b[0..0], &b[16..144]) };
            let mut is = 0;
            let (mut u1, mut u2) = (1u8, 2u8);
            for j in (0..256).step_by(64) {
                let q = &qs[j / 2..j / 2 + 32];
                let (sc, m) = scale_min(is, scales);
                let (d1, m1) = (dd * sc, dmin * m);
                let (sc, m) = scale_min(is + 1, scales);
                let (d2, m2) = (dd * sc, dmin * m);
                for l in 0..32 {
                    let hi1 = if d == DType::Q5K && qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                    let hi2 = if d == DType::Q5K && qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                    y[j + l] = d1 * ((q[l] & 0xF) as f32 + hi1) - m1;
                    y[j + 32 + l] = d2 * ((q[l] >> 4) as f32 + hi2) - m2;
                }
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        DType::Q6K => {
            let dd = h(b, 208);
            for n in 0..2 {
                let ql = &b[64 * n..];
                let qh = &b[128 + 32 * n..];
                let sc = &b[192 + 8 * n..];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    let s = |i: usize| sc[i] as i8 as f32;
                    y[128 * n + l] = dd * s(is) * q1 as f32;
                    y[128 * n + l + 32] = dd * s(is + 2) * q2 as f32;
                    y[128 * n + l + 64] = dd * s(is + 4) * q3 as f32;
                    y[128 * n + l + 96] = dd * s(is + 6) * q4 as f32;
                }
            }
        }
        _ => unreachable!(),
    }
}

/// Random K-quant `[rows, cols]` matrix (GGUF AoS bytes, padded to the lapid-quant layout)
/// and its dequantized values.
pub(crate) fn kq_matrix(d: DType, rows: usize, cols: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let bb = block_bytes(d);
    let nb = rows * cols / 256;
    let r = rand(nb * bb + 2 * nb, seed);
    let mut bytes: Vec<u8> = r[..nb * bb].iter().map(|&x| ((x + 1.0) * 128.0) as u8).collect();
    for blk in 0..nb {
        let b = &mut bytes[blk * bb..(blk + 1) * bb];
        let u = |i: usize| (r[nb * bb + 2 * blk + i] + 1.0) * 0.5;
        if d == DType::Q6K {
            b[208..210].copy_from_slice(&f16::from_f32(0.0005 + 0.002 * u(0)).to_le_bytes());
        } else {
            b[0..2].copy_from_slice(&f16::from_f32(0.001 + 0.01 * u(0)).to_le_bytes());
            b[2..4].copy_from_slice(&f16::from_f32(0.001 + 0.02 * u(1)).to_le_bytes());
        }
    }
    let mut vals = vec![0f32; rows * cols];
    for blk in 0..nb {
        deq_block(d, &bytes[blk * bb..(blk + 1) * bb], &mut vals[blk * 256..(blk + 1) * 256]);
    }
    let fmt = d.quant_format().unwrap();
    bytes.resize(lapid_quant::layout(fmt, rows, cols).total, 0);
    (bytes, vals)
}

fn matmul_case(be: &MetalBackend, b: DType, a: DType, m: usize, n: usize, k: usize, tol: f32) {
    let (bq, bv) = kq_matrix(b, n, k, 21 + m as u64);
    let av = round_to(a, &rand(m * k, 22));
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
    let got = decode(DType::F32, &run(be, Kernel::MatMul(spec, dims), &[encode(a, &av), bq], m * n * 4), m * n);
    let mut want = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            want[i * n + j] = (0..k).map(|kk| av[i * k + kk] as f64 * bv[j * k + kk] as f64).sum::<f64>() as f32;
        }
    }
    assert_close(&format!("matmul {a:?}x{b:?} m{m} n{n} k{k}"), &got, &want, tol);
}

#[test]
fn kquant_gemv() {
    let Some(be) = backend() else { return };
    for b in KQ {
        for m in 1..=4 {
            for &(n, k) in &[(37usize, 256usize), (72, 768)] {
                for a in [DType::F32, DType::F16] {
                    matmul_case(&be, b, a, m, n, k, 2e-5);
                }
            }
        }
    }
}

#[test]
fn kquant_gemm() {
    let Some(be) = backend() else { return };
    for b in KQ {
        for &(m, n, k) in &[(70usize, 72usize, 512usize), (130, 65, 256), (5, 64, 256)] {
            // B is staged as f16: the dequantized value is rounded once.
            matmul_case(&be, b, DType::F32, m, n, k, 2e-3);
        }
    }
}

#[test]
fn kquant_gather() {
    let Some(be) = backend() else { return };
    for t in KQ {
        let (rows, cols) = (7, 512);
        let (bytes, vals) = kq_matrix(t, rows, cols, 5);
        let idx = [3i32, 0, 6, 3];
        let spec = Arc::new(GatherSpec { table_dtype: t, out_dtype: DType::F32 });
        let dims = GatherDims { rows, cols, count: idx.len() };
        let ib: Vec<u8> = idx.iter().flat_map(|x| x.to_le_bytes()).collect();
        let got = decode(DType::F32, &run(&be, Kernel::Gather(spec, dims), &[bytes, ib], idx.len() * cols * 4), idx.len() * cols);
        let want: Vec<f32> = idx.iter().flat_map(|&r| vals[r as usize * cols..(r as usize + 1) * cols].to_vec()).collect();
        assert_close(&format!("gather {t:?}"), &got, &want, 1e-6);
    }
}
