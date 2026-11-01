//! K-quant (Q4_K / Q5_K / Q6_K) matmul + gather, MatMulId and GatedDeltaNet tests on CUDA
//! device 0 against naive f64 references. K-quant inputs are random super-blocks (any bytes
//! are a valid block once d / dmin are small finite halves), dequantized by a port of
//! ggml's `dequantize_row_q*_K` in this file.

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::{DType, DeviceData};
use half::f16;

use crate::CudaBackend;
use crate::tests::{bytes, check, cuda, floats, mat_spec, quant, rnd, run_on};

// ------------------------------------------------------------------ K-quant reference

fn k_format(d: DType) -> lapid_quant::Format {
    d.quant_format().expect("quantized dtype")
}

fn h(b: &[u8], o: usize) -> f32 {
    f16::from_le_bytes([b[o], b[o + 1]]).to_f32()
}

fn scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    if j < 4 {
        ((q[j] & 63) as f32, (q[j + 4] & 63) as f32)
    } else {
        (
            ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)) as f32,
            ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as f32,
        )
    }
}

/// ggml dequantize_row_q{4,5,6}_K of one block.
fn deq_block(d: DType, b: &[u8], y: &mut [f32]) {
    match d {
        DType::Q4K | DType::Q5K => {
            let (dd, dmin) = (h(b, 0), h(b, 2));
            let sc = &b[4..16];
            let (qh, mut ql) = if d == DType::Q5K { (&b[16..48], &b[48..]) } else { (&b[0..0], &b[16..]) };
            let (mut u1, mut u2) = (1u8, 2u8);
            for (jj, yy) in y.chunks_mut(64).enumerate() {
                let (s1, m1) = scale_min_k4(2 * jj, sc);
                let (s2, m2) = scale_min_k4(2 * jj + 1, sc);
                let (d1, m1) = (dd * s1, dmin * m1);
                let (d2, m2) = (dd * s2, dmin * m2);
                for l in 0..32 {
                    let mut q1 = (ql[l] & 0xF) as f32;
                    let mut q2 = (ql[l] >> 4) as f32;
                    if d == DType::Q5K {
                        q1 += if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                        q2 += if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                    }
                    yy[l] = d1 * q1 - m1;
                    yy[l + 32] = d2 * q2 - m2;
                }
                ql = &ql[32..];
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
                let yy = &mut y[128 * n..];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    let s = |i: usize| sc[i] as i8 as f32;
                    yy[l] = dd * s(is) * q1 as f32;
                    yy[l + 32] = dd * s(is + 2) * q2 as f32;
                    yy[l + 64] = dd * s(is + 4) * q3 as f32;
                    yy[l + 96] = dd * s(is + 6) * q4 as f32;
                }
            }
        }
        _ => unreachable!(),
    }
}

pub(crate) fn k_dequant(d: DType, b: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let bb = k_format(d).block_bytes();
    let mut y = vec![0f32; rows * cols];
    for (i, yy) in y.chunks_mut(256).enumerate() {
        deq_block(d, &b[i * bb..(i + 1) * bb], yy);
    }
    y
}

/// Random K-quant matrix (`layout(..).total` bytes) with small finite scales.
pub(crate) fn k_random(d: DType, rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let f = k_format(d);
    let total = lapid_quant::layout(f, rows, cols).total;
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    let mut b: Vec<u8> = (0..total)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect();
    let bb = f.block_bytes();
    let n = rows * cols / 256;
    let r = rnd(2 * n, seed + 1);
    for i in 0..n {
        let blk = &mut b[i * bb..];
        let put = |blk: &mut [u8], o: usize, v: f32| blk[o..o + 2].copy_from_slice(&f16::from_f32(v).to_le_bytes());
        match d {
            DType::Q6K => put(blk, 208, 0.002 * (1.0 + r[2 * i].abs())),
            _ => {
                put(blk, 0, 0.004 * (1.0 + r[2 * i].abs()));
                put(blk, 2, 0.004 * (1.0 + r[2 * i + 1].abs()));
            }
        }
    }
    b
}

fn naive(a: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let s: f64 = (0..k).map(|t| a[i * k + t] as f64 * w[j * k + t] as f64).sum();
            out[i * n + j] = s as f32;
        }
    }
    out
}

/// A values as the kernel sees them (f16 activations rounded).
fn as_seen(d: DType, v: &[f32]) -> Vec<f32> {
    floats(d, &bytes(d, v))
}

fn kmatmul_case(be: &CudaBackend, a_t: DType, b_t: DType, m: usize, n: usize, k: usize, tol: f32) {
    let spec = mat_spec(a_t, b_t, true, vec![], None);
    let a = as_seen(a_t, &rnd(m * k, 3));
    let wb = k_random(b_t, n, k, 4);
    let w = k_dequant(b_t, &wb, n, k);
    let dims = MatMulDims {
        batch: 1,
        m,
        n,
        k,
        a: MatView { offset: 0, batch_stride: m * k, row_stride: k },
        b: MatView { offset: 0, batch_stride: 0, row_stride: k },
        extra_views: vec![],
        out: MatView::contiguous(m, n),
    };
    let got = floats(DType::F32, &run_on(be, &Kernel::MatMul(spec, dims), &[bytes(a_t, &a), wb], m * n * 4));
    check(&format!("matmul {a_t:?}x{b_t:?} {m}x{n}x{k}"), &got, &naive(&a, &w, m, n, k), tol);
}

const KQ: [DType; 3] = [DType::Q4K, DType::Q5K, DType::Q6K];

#[test]
fn kquant_reference_self_check() {
    // the reference dequant is linear in d: doubling d (and dmin) doubles every element
    for d in KQ {
        let b = k_random(d, 2, 512, 9);
        let y = k_dequant(d, &b, 2, 512);
        assert!(y.iter().all(|v| v.is_finite()) && y.iter().any(|&v| v != 0.0), "{d:?}");
    }
}

#[test]
fn kquant_gemv() {
    let Some(be) = cuda() else { return };
    for d in KQ {
        for m in [1, 2, 3] {
            kmatmul_case(&be, DType::F32, d, m, 67, 512, 1e-5);
        }
        kmatmul_case(&be, DType::F16, d, 1, 40, 768, 1e-5);
    }
}

#[test]
fn kquant_gemm() {
    let Some(be) = cuda() else { return };
    for d in KQ {
        // tensor-core path: f16 operands
        kmatmul_case(&be, DType::F32, d, 150, 200, 512, 2e-3);
        kmatmul_case(&be, DType::F16, d, 70, 130, 256, 2e-3);
    }
}

#[test]
fn kquant_gather() {
    let Some(be) = cuda() else { return };
    for d in KQ {
        let (rows, cols) = (37, 512);
        let tb = k_random(d, rows, cols, 5);
        let t = k_dequant(d, &tb, rows, cols);
        let idx = [3i32, 0, 36, 17, 3];
        let ib: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
        let spec = Arc::new(GatherSpec { table_dtype: d, out_dtype: DType::F32 });
        let k = Kernel::Gather(spec, GatherDims { rows, cols, count: idx.len() });
        let got = floats(DType::F32, &run_on(&be, &k, &[tb, ib], idx.len() * cols * 4));
        let want: Vec<f32> = idx.iter().flat_map(|&i| t[i as usize * cols..(i as usize + 1) * cols].to_vec()).collect();
        check(&format!("gather {d:?}"), &got, &want, 1e-6);
    }
}

// ------------------------------------------------------------------ MatMulId

/// Expert stack bytes + its dequantized f32 values [e, n, k].
fn experts(w_t: DType, e: usize, n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let mut b = Vec::new();
    let mut f = Vec::new();
    for x in 0..e {
        let (eb, ef) = match w_t {
            DType::Q4K | DType::Q5K | DType::Q6K => {
                let eb = k_random(w_t, n, k, 20 + x as u64);
                let ef = k_dequant(w_t, &eb, n, k);
                (eb, ef)
            }
            DType::Int8 | DType::Int4 => {
                let v: Vec<f32> = rnd(n * k, 30 + x as u64).iter().map(|v| v * 0.1).collect();
                let eb = quant(w_t, &v, n, k);
                let ef = lapid_quant::dequantize(k_format(w_t), &eb, n, k);
                (eb, ef)
            }
            _ => {
                let v: Vec<f32> = rnd(n * k, 30 + x as u64).iter().map(|v| v * 0.1).collect();
                (bytes(w_t, &v), as_seen(w_t, &v))
            }
        };
        b.extend(eb);
        f.extend(ef);
    }
    (b, f)
}

fn run_id(
    be: &CudaBackend,
    x_t: DType,
    w_t: DType,
    (t, r, e, n, k): (usize, usize, usize, usize, usize),
    ids: &[i32],
) -> lapid_tensor::Result<(Vec<f32>, Vec<f32>)> {
    let x = as_seen(x_t, &rnd(t * k, 11));
    let (wb, wf) = experts(w_t, e, n, k);
    let spec = Arc::new(MatMulIdSpec { x_dtype: x_t, w_dtype: w_t, out_dtype: DType::F32 });
    let kernel = Kernel::MatMulId(spec, MatMulIdDims { t, r, e, n, k });
    let ib: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let inputs = [bytes(x_t, &x), wb, ib];
    let mut bufs: Vec<DeviceData> = inputs.iter().map(|b| be.upload(b)).collect::<lapid_tensor::Result<_>>()?;
    bufs.push(be.upload(&vec![0u8; t * r * n * 4])?);
    be.execute(&[Launch { kernel, inputs: vec![0, 1, 2], output: 3 }], &bufs)?;
    let got = floats(DType::F32, &be.download(&bufs[3])?);
    let mut want = vec![0f32; t * r * n];
    for ti in 0..t {
        for ri in 0..r {
            let ex = ids[ti * r + ri] as usize;
            let o = naive(&x[ti * k..(ti + 1) * k], &wf[ex * n * k..(ex + 1) * n * k], 1, n, k);
            want[(ti * r + ri) * n..(ti * r + ri + 1) * n].copy_from_slice(&o);
        }
    }
    Ok((got, want))
}

fn ids_for(t: usize, r: usize, e: usize, seed: u64) -> Vec<i32> {
    // distinct experts per token, skewed so some experts get many rows and some none
    let v = rnd(t, seed);
    (0..t)
        .flat_map(|ti| {
            let s = ((v[ti].abs() * 1000.0) as usize) % e;
            (0..r).map(move |ri| ((s + ri * ri + ri) % e) as i32)
        })
        .collect()
}

#[test]
fn matmul_id() {
    let Some(be) = cuda() else { return };
    for w_t in [DType::F16, DType::Int8, DType::Int4, DType::Q4K, DType::Q6K] {
        for (t, r, e, n, k, tol) in [(1, 2, 5, 70, 256, 1e-5), (3, 2, 4, 33, 512, 1e-5), (70, 3, 7, 150, 256, 2e-3), (200, 2, 3, 150, 256, 2e-3), (64, 8, 64, 96, 256, 2e-3), (8, 1, 6, 40, 256, 1e-5)] {
            let ids = ids_for(t, r, e, 5 + t as u64);
            let (got, want) = run_id(&be, DType::F32, w_t, (t, r, e, n, k), &ids).unwrap();
            check(&format!("matmul_id {w_t:?} t{t} r{r} e{e} n{n} k{k}"), &got, &want, tol);
        }
    }
    let ids = ids_for(9, 2, 4, 1);
    let (got, want) = run_id(&be, DType::F16, DType::Q5K, (9, 2, 4, 64, 256), &ids).unwrap();
    check("matmul_id f16 x q5k", &got, &want, 2e-3);
}

#[test]
fn matmul_id_out_of_range() {
    let Some(be) = cuda() else { return };
    for t in [1, 20] {
        let mut ids = ids_for(t, 2, 4, 3);
        ids[1] = 4;
        let e = run_id(&be, DType::F32, DType::F16, (t, 2, 4, 32, 64), &ids).unwrap_err();
        assert_eq!(e.code, "index_out_of_range", "t={t}");
        // the backend keeps working
        let ids = ids_for(t, 2, 4, 3);
        assert!(run_id(&be, DType::F32, DType::F16, (t, 2, 4, 32, 64), &ids).is_ok());
    }
}

// ------------------------------------------------------------------ GatedDeltaNet

#[allow(clippy::too_many_arguments)]
fn gdn_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    st: &[f32],
    (b, t, h, dk, dv): (usize, usize, usize, usize, usize),
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; b * h * (t + dk) * dv];
    for bi in 0..b {
        for hi in 0..h {
            let mut s: Vec<f64> = st[(bi * h + hi) * dk * dv..][..dk * dv].iter().map(|&x| x as f64).collect();
            for ti in 0..t {
                let tok = (bi * t + ti) * h + hi;
                let (qt, kt, vt) = (&q[tok * dk..][..dk], &k[tok * dk..][..dk], &v[tok * dv..][..dv]);
                let gt = (g[tok] as f64).exp();
                s.iter_mut().for_each(|x| *x *= gt);
                for j in 0..dv {
                    let sk: f64 = (0..dk).map(|i| s[i * dv + j] * kt[i] as f64).sum();
                    let d = beta[tok] as f64 * (vt[j] as f64 - sk);
                    for i in 0..dk {
                        s[i * dv + j] += kt[i] as f64 * d;
                    }
                    let o: f64 = (0..dk).map(|i| s[i * dv + j] * qt[i] as f64 * scale as f64).sum();
                    out[((bi * h + hi) * (t + dk) + ti) * dv + j] = o as f32;
                }
            }
            for i in 0..dk * dv {
                out[((bi * h + hi) * (t + dk) + t) * dv + i] = s[i] as f32;
            }
        }
    }
    out
}

fn gdn_case(be: &CudaBackend, (b, t, h, dk, dv): (usize, usize, usize, usize, usize)) {
    let n = b * t * h;
    let norm = |v: Vec<f32>| -> Vec<f32> {
        v.chunks(dk)
            .flat_map(|c| {
                let l = c.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                c.iter().map(move |x| x / l).collect::<Vec<_>>()
            })
            .collect()
    };
    let q = norm(rnd(n * dk, 1));
    let k = norm(rnd(n * dk, 2));
    let v = rnd(n * dv, 3);
    let g: Vec<f32> = rnd(n, 4).iter().map(|x| -0.5 * x.abs() - 0.01).collect();
    let beta: Vec<f32> = rnd(n, 5).iter().map(|x| 0.5 + 0.5 * x.abs()).collect();
    let st: Vec<f32> = rnd(b * h * dk * dv, 6).iter().map(|x| 0.3 * x).collect();
    let scale = 1.0 / (dk as f32).sqrt();
    let spec = Arc::new(GatedDeltaNetSpec { out_dtype: DType::F32 });
    let kernel = Kernel::GatedDeltaNet(spec, GatedDeltaNetDims { b, t, h, dk, dv, scale: scale.to_bits() });
    let f = |x: &[f32]| bytes(DType::F32, x);
    let inputs = [f(&q), f(&k), f(&v), f(&g), f(&beta), f(&st)];
    let got = floats(DType::F32, &run_on(be, &kernel, &inputs, b * h * (t + dk) * dv * 4));
    let want = gdn_ref(&q, &k, &v, &g, &beta, &st, (b, t, h, dk, dv), scale);
    check(&format!("gated_delta_net b{b} t{t} h{h} dk{dk} dv{dv}"), &got, &want, 1e-5);
}

#[test]
fn gated_delta_net() {
    let Some(be) = cuda() else { return };
    gdn_case(&be, (2, 5, 3, 128, 40)); // vector loads, partial column block
    gdn_case(&be, (1, 1, 4, 128, 128)); // decode step
    gdn_case(&be, (1, 9, 2, 20, 16)); // Dk not a multiple of 8 rows per part
    gdn_case(&be, (1, 0, 2, 64, 32)); // no tokens: state passes through
    gdn_case(&be, (1, 300, 1, 64, 64)); // long sequence
}
