//! GPU tests: MatMulId (MoE, GEMV and routed GEMM paths) and GatedDeltaNet against naive
//! f64 references.

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::DType;

use super::gpu_kquant::kq_matrix;
use super::util::*;
use crate::MetalBackend;

/// Expert stack bytes ([e] matrices back to back) and per-expert dequantized values.
fn experts(w: DType, e: usize, n: usize, k: usize) -> (Vec<u8>, Vec<Vec<f32>>) {
    let mut bytes = Vec::new();
    let mut vals = Vec::new();
    for x in 0..e {
        let seed = 40 + x as u64;
        let (b, v) = match w {
            DType::Int8 | DType::Int4 => {
                let f = w.quant_format().unwrap();
                let raw = rand(n * k, seed);
                let q = lapid_quant::quantize(f, &raw, n, k);
                let d = lapid_quant::dequantize(f, &q, n, k);
                (q, d)
            }
            d if d.is_quant() => kq_matrix(d, n, k, seed),
            _ => {
                let raw = rand(n * k, seed);
                (encode(w, &raw), round_to(w, &raw))
            }
        };
        bytes.extend_from_slice(&b);
        vals.push(v);
    }
    (bytes, vals)
}

fn moe_case(be: &MetalBackend, x_dt: DType, w: DType, dims: MatMulIdDims, tol: f32) {
    // Ids: a deterministic mix with repeats (also the same expert twice for one token).
    let ids: Vec<i32> = (0..dims.t * dims.r).map(|p| ((p * 7 + p / 3) % dims.e) as i32).collect();
    moe_case_ids(be, x_dt, w, dims, tol, &ids);
}

fn moe_case_ids(be: &MetalBackend, x_dt: DType, w: DType, dims: MatMulIdDims, tol: f32, ids: &[i32]) {
    let MatMulIdDims { t, r, e, n, k } = dims;
    let (wb, wv) = experts(w, e, n, k);
    let x = round_to(x_dt, &rand(t * k, 3));
    let spec = Arc::new(MatMulIdSpec { x_dtype: x_dt, w_dtype: w, out_dtype: DType::F32 });
    let ib: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let got = decode(DType::F32, &run(be, Kernel::MatMulId(spec, dims), &[encode(x_dt, &x), wb, ib], t * r * n * 4), t * r * n);
    let mut want = vec![0f32; t * r * n];
    for p in 0..t * r {
        let ex = &wv[ids[p] as usize];
        let tt = p / r;
        for j in 0..n {
            want[p * n + j] = (0..k).map(|kk| x[tt * k + kk] as f64 * ex[j * k + kk] as f64).sum::<f64>() as f32;
        }
    }
    assert_close(&format!("matmul_id {x_dt:?}x{w:?} t{t} r{r} e{e} n{n} k{k}"), &got, &want, tol);
}

#[test]
fn matmul_id_every_w_dtype() {
    let Some(be) = backend() else { return };
    for w in [DType::F32, DType::F16, DType::Int8, DType::Int4, DType::Q4K, DType::Q6K] {
        let k = if w.is_quant() { 256 } else { 96 };
        for &t in &[1usize, 3, 70] {
            // GEMV path exact-ish; the GEMM path stages B as f16 for f16 / quantized weights.
            let tol = if t <= 4 || w == DType::F32 { 2e-5 } else { 2e-3 };
            moe_case(&be, DType::F32, w, MatMulIdDims { t, r: 4, e: 5, n: 40, k }, tol);
        }
    }
    moe_case(&be, DType::F16, DType::Q5K, MatMulIdDims { t: 2, r: 2, e: 3, n: 70, k: 512 }, 2e-5);
    moe_case(&be, DType::F16, DType::Q5K, MatMulIdDims { t: 129, r: 2, e: 3, n: 70, k: 512 }, 2e-3);
}

#[test]
fn matmul_id_skewed_routing() {
    // Many experts (one row-tile threadgroup per expert) with one hot expert holding several
    // 64-row tiles: the threadgroups loop over the hot expert's tiles.
    let Some(be) = backend() else { return };
    let (t, r, e) = (300usize, 2usize, 64usize);
    let ids: Vec<i32> = (0..t * r).map(|p| if p % r == 0 { 5 } else { ((p * 13 + 1) % e) as i32 }).collect();
    moe_case_ids(&be, DType::F32, DType::Int8, MatMulIdDims { t, r, e, n: 72, k: 64 }, 2e-3, &ids);
    moe_case_ids(&be, DType::F32, DType::Q4K, MatMulIdDims { t, r, e, n: 40, k: 256 }, 2e-3, &ids);
}

#[test]
fn matmul_id_out_of_range() {
    let Some(be) = backend() else { return };
    for t in [1usize, 9] {
        let (n, k, e) = (8, 32, 2);
        let spec = Arc::new(MatMulIdSpec { x_dtype: DType::F32, w_dtype: DType::F32, out_dtype: DType::F32 });
        let ids: Vec<u8> = (0..t).flat_map(|i| (if i == t - 1 { 2i32 } else { 0 }).to_le_bytes()).collect();
        let ins = [encode(DType::F32, &rand(t * k, 1)), encode(DType::F32, &rand(e * n * k, 2)), ids];
        let err = run_on(&be, Kernel::MatMulId(spec, MatMulIdDims { t, r: 1, e, n, k }), &ins, &vec![0u8; t * n * 4]).unwrap_err();
        assert_eq!(err.code, "index_out_of_range", "t = {t}");
        // The backend recovers (status cleared).
        assert!(be.download(&be.upload(&[1, 2, 3, 4]).unwrap()).is_ok());
    }
}

fn delta_case(be: &MetalBackend, b: usize, t: usize, h: usize, dk: usize, dv: usize) {
    let q = rand(b * t * h * dk, 1);
    let k: Vec<f32> = rand(b * t * h * dk, 2).iter().map(|x| x * 0.3).collect();
    let v = rand(b * t * h * dv, 3);
    let g: Vec<f32> = rand(b * t * h, 4).iter().map(|x| -0.5 * (x + 1.0)).collect();
    let beta: Vec<f32> = rand(b * t * h, 5).iter().map(|x| 0.5 * (x + 1.0)).collect();
    let state = rand(b * h * dk * dv, 6);
    let scale = 1.0 / (dk as f32).sqrt();
    let spec = Arc::new(GatedDeltaNetSpec { out_dtype: DType::F32 });
    let dims = GatedDeltaNetDims { b, t, h, dk, dv, scale: scale.to_bits() };
    let ins: Vec<Vec<u8>> = [&q, &k, &v, &g, &beta, &state].iter().map(|x| encode(DType::F32, x)).collect();
    let total = b * h * (t + dk) * dv;
    let got = decode(DType::F32, &run(be, Kernel::GatedDeltaNet(spec, dims), &ins, total * 4), total);
    let mut want = vec![0f32; total];
    for bb in 0..b {
        for hh in 0..h {
            let mut s: Vec<f64> = (0..dk * dv).map(|i| state[(bb * h + hh) * dk * dv + i] as f64).collect();
            let base = (bb * h + hh) * (t + dk) * dv;
            for tt in 0..t {
                let row = (bb * t + tt) * h + hh;
                let decay = (g[row] as f64).exp();
                s.iter_mut().for_each(|x| *x *= decay);
                let kv = &k[row * dk..(row + 1) * dk];
                let qv = &q[row * dk..(row + 1) * dk];
                for j in 0..dv {
                    let sk: f64 = (0..dk).map(|i| s[i * dv + j] * kv[i] as f64).sum();
                    let d = (v[row * dv + j] as f64 - sk) * beta[row] as f64;
                    for i in 0..dk {
                        s[i * dv + j] += kv[i] as f64 * d;
                    }
                    let y: f64 = (0..dk).map(|i| s[i * dv + j] * qv[i] as f64 * scale as f64).sum();
                    want[base + tt * dv + j] = y as f32;
                }
            }
            for i in 0..dk * dv {
                want[base + t * dv + i] = s[i] as f32;
            }
        }
    }
    assert_close(&format!("gated_delta_net b{b} t{t} h{h} dk{dk} dv{dv}"), &got, &want, 2e-5);
}

#[test]
fn gated_delta_net() {
    let Some(be) = backend() else { return };
    delta_case(&be, 2, 5, 3, 40, 24);
    delta_case(&be, 1, 1, 4, 128, 128);
    delta_case(&be, 1, 17, 2, 64, 33);
    delta_case(&be, 1, 0, 2, 32, 8);
}
