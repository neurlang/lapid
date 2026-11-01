//! Round-3 kernels vs naive f64 references: K-quant (Q4_K / Q5_K / Q6_K) matmul + gather,
//! MatMulId (mixture of experts) and GatedDeltaNet. The K-quant reference is a direct port
//! of ggml's `dequantize_row_q{4,5,6}_K` over randomly generated blocks.

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::{DType, DeviceData};
use half::f16;

use crate::codegen::{self, Key};
use crate::pipeline::compile_wgsl;
use crate::tests::{assert_close, backend, caps, f16b, f32b, from_f32b, mat_spec, rnd, run};

const KS: [DType; 3] = [DType::Q4K, DType::Q5K, DType::Q6K];

fn fmt(d: DType) -> lapid_quant::Format {
    d.quant_format().unwrap()
}

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 7) as u8
    }
    fn f16_in(&mut self, lo: f32, hi: f32) -> [u8; 2] {
        let u = (self.next() >> 8) as f32 / (1u32 << 24) as f32;
        f16::from_f32(lo + (hi - lo) * u).to_le_bytes()
    }
}

/// Random K-quant matrix `[rows, cols]` (layout total bytes, padding zeroed).
fn random_k(d: DType, rows: usize, cols: usize, seed: u32) -> Vec<u8> {
    let f = fmt(d);
    let mut out = vec![0u8; lapid_quant::layout(f, rows, cols).total];
    let bb = f.block_bytes();
    let mut r = Rng(seed.wrapping_mul(2654435761) | 1);
    for blk in out[..rows * cols / 256 * bb].chunks_exact_mut(bb) {
        for b in blk.iter_mut() {
            *b = r.byte();
        }
        match d {
            DType::Q6K => blk[208..210].copy_from_slice(&r.f16_in(0.001, 0.01)),
            _ => {
                blk[0..2].copy_from_slice(&r.f16_in(0.001, 0.01));
                blk[2..4].copy_from_slice(&r.f16_in(0.0, 0.02));
            }
        }
    }
    out
}

fn h(b: &[u8], o: usize) -> f64 {
    f16::from_le_bytes([b[o], b[o + 1]]).to_f64()
}

fn scale_min(j: usize, q: &[u8]) -> (f64, f64) {
    if j < 4 {
        ((q[j] & 63) as f64, (q[j + 4] & 63) as f64)
    } else {
        (((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)) as f64, ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as f64)
    }
}

/// ggml dequantize_row_q{4,5,6}_K for one block.
fn deq_block(d: DType, b: &[u8]) -> Vec<f64> {
    let mut y = Vec::with_capacity(256);
    match d {
        DType::Q4K | DType::Q5K => {
            let (dd, dmin) = (h(b, 0), h(b, 2));
            let scales = &b[4..16];
            let (qh, ql) = if d == DType::Q5K { (&b[16..48], &b[48..176]) } else { (&b[0..0], &b[16..144]) };
            let (mut u1, mut u2) = (1u8, 2u8);
            for (c, j) in (0..256).step_by(64).enumerate() {
                let _ = j;
                let q = &ql[c * 32..c * 32 + 32];
                let (sc, m) = scale_min(2 * c, scales);
                let (d1, m1) = (dd * sc, dmin * m);
                let (sc, m) = scale_min(2 * c + 1, scales);
                let (d2, m2) = (dd * sc, dmin * m);
                for l in 0..32 {
                    let hb = if d == DType::Q5K && qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                    y.push(d1 * ((q[l] & 0xF) as f64 + hb) - m1);
                }
                for l in 0..32 {
                    let hb = if d == DType::Q5K && qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                    y.push(d2 * ((q[l] >> 4) as f64 + hb) - m2);
                }
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        _ => {
            let dd = h(b, 208);
            y.resize(256, 0.0);
            for n in 0..2 {
                let ql = &b[n * 64..];
                let qh = &b[128 + n * 32..];
                let sc = &b[192 + n * 8..];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    let s = |i: usize| sc[i] as i8 as f64;
                    y[n * 128 + l] = dd * s(is) * q1 as f64;
                    y[n * 128 + l + 32] = dd * s(is + 2) * q2 as f64;
                    y[n * 128 + l + 64] = dd * s(is + 4) * q3 as f64;
                    y[n * 128 + l + 96] = dd * s(is + 6) * q4 as f64;
                }
            }
        }
    }
    y
}

fn deq_matrix(d: DType, bytes: &[u8], rows: usize, cols: usize) -> Vec<f64> {
    let bb = fmt(d).block_bytes();
    bytes[..rows * cols / 256 * bb].chunks_exact(bb).flat_map(|b| deq_block(d, b)).collect()
}

fn matmul_ref(a: &[f32], w: &[f64], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..k).map(|t| a[i * k + t] as f64 * w[j * k + t]).sum::<f64>() as f32;
        }
    }
    out
}

#[test]
fn kquant_matmul_gemv_and_gemm() {
    let Some(be) = backend() else { return };
    let (n, k) = (37usize, 512usize);
    for d in KS {
        let wb = random_k(d, n, k, 11 + d as u32);
        let w = deq_matrix(d, &wb, n, k);
        for m in [1usize, 3, 70] {
            let a = rnd(m * k, 7 + m as u32);
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
            let kernel = Kernel::MatMul(mat_spec(DType::F32, d, true, vec![], None), dims);
            let got = from_f32b(&run(&be, kernel, &[f32b(&a), wb.clone()], m * n * 4));
            let tol = if m > 4 { 3e-3 } else { 1e-5 };
            assert_close(&got, &matmul_ref(&a, &w, m, n, k), tol, &format!("{} m={m}", d.name()));
        }
    }
}

#[test]
fn kquant_gather_rows() {
    let Some(be) = backend() else { return };
    let (rows, cols) = (9usize, 512usize);
    let idx: Vec<i32> = vec![3, 0, 8, 3, 5];
    for d in KS {
        let tb = random_k(d, rows, cols, 21 + d as u32);
        let t = deq_matrix(d, &tb, rows, cols);
        let spec = Arc::new(GatherSpec { table_dtype: d, out_dtype: DType::F32 });
        let dims = GatherDims { rows, cols, count: idx.len() };
        let ib: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
        let got = from_f32b(&run(&be, Kernel::Gather(spec, dims), &[tb, ib], idx.len() * cols * 4));
        let want: Vec<f32> = idx.iter().flat_map(|&r| t[r as usize * cols..(r as usize + 1) * cols].iter().map(|&x| x as f32)).collect();
        assert_close(&got, &want, 1e-6, &format!("gather {}", d.name()));
    }
}

/// Expert stack bytes + dequantized reference [E][N*K].
fn experts(d: DType, e: usize, n: usize, k: usize, seed: u32) -> (Vec<u8>, Vec<Vec<f64>>) {
    let mut bytes = Vec::new();
    let mut refs = Vec::new();
    for x in 0..e {
        let s = seed + x as u32 * 101;
        match d {
            d if d.is_quant() && fmt(d).is_aos() => {
                let b = random_k(d, n, k, s);
                refs.push(deq_matrix(d, &b, n, k));
                bytes.extend(b);
            }
            d if d.is_quant() => {
                let b = lapid_quant::quantize(fmt(d), &rnd(n * k, s), n, k);
                refs.push(lapid_quant::dequantize(fmt(d), &b, n, k).iter().map(|&v| v as f64).collect());
                bytes.extend(b);
            }
            DType::F16 => {
                let v = rnd(n * k, s);
                refs.push(v.iter().map(|&x| f16::from_f32(x).to_f64()).collect());
                bytes.extend(f16b(&v));
            }
            _ => {
                let v = rnd(n * k, s);
                refs.push(v.iter().map(|&x| x as f64).collect());
                bytes.extend(f32b(&v));
            }
        }
    }
    (bytes, refs)
}

fn ids_bytes(ids: &[i32]) -> Vec<u8> {
    ids.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn matmul_id_all_weight_dtypes() {
    let Some(be) = backend() else { return };
    let (t, r, e, n, k) = (3usize, 2usize, 4usize, 40usize, 256usize);
    let ids: Vec<i32> = vec![2, 0, 3, 3, 1, 2];
    let x = rnd(t * k, 31);
    for d in [DType::F32, DType::F16, DType::Int8, DType::Int4, DType::Q4K, DType::Q5K, DType::Q6K] {
        let (wb, w) = experts(d, e, n, k, 40);
        let spec = Arc::new(MatMulIdSpec { x_dtype: DType::F32, w_dtype: d, out_dtype: DType::F32 });
        let dims = MatMulIdDims { t, r, e, n, k };
        let got = from_f32b(&run(&be, Kernel::MatMulId(spec, dims), &[f32b(&x), wb, ids_bytes(&ids)], t * r * n * 4));
        let mut want = Vec::new();
        for ti in 0..t {
            for ri in 0..r {
                let wx = &w[ids[ti * r + ri] as usize];
                want.extend(matmul_ref(&x[ti * k..(ti + 1) * k], wx, 1, n, k));
            }
        }
        assert_close(&got, &want, 1e-5, &format!("matmul_id {}", d.name()));
    }
}

#[test]
fn matmul_id_out_of_range_is_an_error() {
    let Some(be) = backend() else { return };
    let (t, r, e, n, k) = (1usize, 2usize, 2usize, 8usize, 32usize);
    let (wb, _) = experts(DType::F32, e, n, k, 3);
    let spec = Arc::new(MatMulIdSpec { x_dtype: DType::F32, w_dtype: DType::F32, out_dtype: DType::F32 });
    let kernel = Kernel::MatMulId(spec, MatMulIdDims { t, r, e, n, k });
    let bufs: Vec<DeviceData> = vec![
        be.upload(&f32b(&rnd(t * k, 1))).unwrap(),
        be.upload(&wb).unwrap(),
        be.upload(&ids_bytes(&[1, 2])).unwrap(),
        be.alloc(t * r * n * 4).unwrap(),
    ];
    let launch = Launch { kernel, inputs: vec![0, 1, 2], output: 3 };
    let err = be.execute(std::slice::from_ref(&launch), &bufs).unwrap_err();
    assert_eq!(err.code, "index_out_of_range");
}

#[allow(clippy::too_many_arguments)]
fn gdn_ref(q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32], s0: &[f32], b: usize, t: usize, hh: usize, dk: usize, dv: usize, scale: f64) -> Vec<f32> {
    let mut out = vec![0.0f32; b * hh * (t + dk) * dv];
    for bi in 0..b {
        for h in 0..hh {
            let bh = bi * hh + h;
            let mut s: Vec<f64> = s0[bh * dk * dv..(bh + 1) * dk * dv].iter().map(|&x| x as f64).collect();
            let ob = bh * (t + dk) * dv;
            for ti in 0..t {
                let r = (bi * t + ti) * hh + h;
                let eg = (g[r] as f64).exp();
                s.iter_mut().for_each(|x| *x *= eg);
                for j in 0..dv {
                    let sk: f64 = (0..dk).map(|i| s[i * dv + j] * k[r * dk + i] as f64).sum();
                    let dl = beta[r] as f64 * (v[r * dv + j] as f64 - sk);
                    for i in 0..dk {
                        s[i * dv + j] += k[r * dk + i] as f64 * dl;
                    }
                }
                for j in 0..dv {
                    let o: f64 = (0..dk).map(|i| s[i * dv + j] * q[r * dk + i] as f64).sum();
                    out[ob + ti * dv + j] = (o * scale) as f32;
                }
            }
            for (i, &x) in s.iter().enumerate() {
                out[ob + t * dv + i] = x as f32;
            }
        }
    }
    out
}

#[test]
fn gated_delta_net_matches_reference() {
    let Some(be) = backend() else { return };
    for &(b, t, hh, dk, dv) in &[(2usize, 3usize, 2usize, 40usize, 24usize), (1, 5, 3, 16, 33)] {
        let q = rnd(b * t * hh * dk, 1);
        let k = rnd(b * t * hh * dk, 2);
        let v = rnd(b * t * hh * dv, 3);
        let g: Vec<f32> = rnd(b * t * hh, 4).iter().map(|x| -0.5 * x.abs()).collect();
        let beta: Vec<f32> = rnd(b * t * hh, 5).iter().map(|x| 0.5 + 0.5 * x.abs()).collect();
        let s0 = rnd(b * hh * dk * dv, 6);
        let scale = 1.0 / (dk as f32).sqrt();
        let spec = Arc::new(GatedDeltaNetSpec { out_dtype: DType::F32 });
        let dims = GatedDeltaNetDims { b, t, h: hh, dk, dv, scale: scale.to_bits() };
        let ins = [f32b(&q), f32b(&k), f32b(&v), f32b(&g), f32b(&beta), f32b(&s0)];
        let n_out = b * hh * (t + dk) * dv;
        let got = from_f32b(&run(&be, Kernel::GatedDeltaNet(spec, dims), &ins, n_out * 4));
        let want = gdn_ref(&q, &k, &v, &g, &beta, &s0, b, t, hh, dk, dv, scale as f64);
        assert_close(&got, &want, 1e-5, &format!("gated_delta_net {b}x{t}x{hh}x{dk}x{dv}"));
    }
}

/// Every new key compiles on every capability mix (naga only, no GPU).
#[test]
fn round3_kernels_compile_on_all_caps() {
    let mut keys = Vec::new();
    for d in KS {
        for a in [DType::F32, DType::F16] {
            let s = mat_spec(a, d, true, vec![DType::F32], Some(lapid_compiler::Expr::Binary(lapid_tensor::BinaryOp::Add, crate::tests::load(0), crate::tests::load(1))));
            keys.push(Key::Gemv(s.clone(), false, false));
            keys.push(Key::Gemv(s.clone(), true, false));
            keys.push(Key::Gemm(s.clone()));
            keys.push(Key::GemmCoop(s));
        }
        keys.push(Key::Gather(Arc::new(GatherSpec { table_dtype: d, out_dtype: DType::F16 })));
    }
    for d in [DType::F32, DType::F16, DType::BF16, DType::Int8, DType::Int4, DType::Q4K, DType::Q5K, DType::Q6K] {
        keys.push(Key::MatMulId(Arc::new(MatMulIdSpec { x_dtype: DType::F16, w_dtype: d, out_dtype: DType::F32 })));
    }
    keys.push(Key::GatedDeltaNet(Arc::new(GatedDeltaNetSpec { out_dtype: DType::F32 })));
    keys.push(Key::GatedDeltaNet(Arc::new(GatedDeltaNetSpec { out_dtype: DType::F16 })));
    let mut coop = caps(true, true, true);
    coop.coopmat = true;
    coop.full_subgroups = true;
    for c in [caps(true, true, true), caps(false, false, false), caps(true, false, false), coop] {
        for key in &keys {
            if matches!(key, Key::GemmCoop(_)) && !c.coopmat {
                continue;
            }
            let src = codegen::source(key, &c);
            assert!(!src.contains("{{"), "{key:?}: unfilled slot");
            if let Err(e) = compile_wgsl(key.label(), &src, &c) {
                panic!("{key:?} failed (caps {c:?}):\n{}", e.message);
            }
        }
    }
}
