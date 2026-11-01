//! Attention kernel vs a naive f64 reference through the `Backend` API: flash path and
//! decode path, causal / masked, KV-cache shapes (sq < sk, strided [B, cap, H, D] caches),
//! f32 and f16 storage, block-boundary key counts.

use std::sync::Arc;

use lapid_compiler::{AttentionDims, AttentionSpec, Backend, Kernel, Launch, View};
use lapid_tensor::{DType, DeviceData};
use half::f16;

use crate::{CpuBackend, Isa};

fn rnd(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Values as stored in `dt` (what the kernel sees) and their bytes.
fn store(v: &[f32], dt: DType) -> (Vec<f32>, Vec<u8>) {
    match dt {
        DType::F16 => {
            let h: Vec<f16> = v.iter().map(|&x| f16::from_f32(x)).collect();
            (h.iter().map(|x| x.to_f32()).collect(), bytemuck::cast_slice(&h).to_vec())
        }
        _ => (v.to_vec(), bytemuck::cast_slice(v).to_vec()),
    }
}

#[derive(Clone, Copy, Debug)]
struct Case {
    b: usize,
    h: usize,
    sq: usize,
    sk: usize,
    d: usize,
    causal: bool,
    mask: bool,
    dt: DType,
    /// KV cache capacity: k/v stored [B, cap, H, D] (+ offset) and read as [B, H, sk, D].
    cap: Option<usize>,
}

/// Element (b, h, s, c) of a rank-4 view.
fn at(x: &[f32], v: &View, b: usize, h: usize, s: usize, c: usize) -> f64 {
    let st = &v.strides;
    let o = v.offset as isize + b as isize * st[0] + h as isize * st[1] + s as isize * st[2] + c as isize * st[3];
    x[o as usize] as f64
}

#[allow(clippy::too_many_arguments)]
fn reference(c: &Case, q: &[f32], k: &[f32], v: &[f32], m: &[f32], dims: &AttentionDims, scale: f64) -> Vec<f32> {
    let mut out = vec![0f32; c.b * c.h * c.sq * c.d];
    for b in 0..c.b {
        for h in 0..c.h {
            for i in 0..c.sq {
                let sc: Vec<f64> = (0..c.sk)
                    .map(|j| {
                        if c.causal && j as isize > i as isize + c.sk as isize - c.sq as isize {
                            return f64::NEG_INFINITY;
                        }
                        let dot: f64 = (0..c.d).map(|e| at(q, &dims.q, b, h, i, e) * at(k, &dims.k, b, h, j, e)).sum();
                        let mv = dims.mask.as_ref().map_or(0.0, |mv| at(m, mv, b, h, i, j));
                        dot * scale + mv
                    })
                    .collect();
                let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if mx == f64::NEG_INFINITY {
                    continue;
                }
                let e: Vec<f64> = sc.iter().map(|&x| (x - mx).exp()).collect();
                let sum: f64 = e.iter().sum();
                for dd in 0..c.d {
                    let acc: f64 = (0..c.sk).map(|j| e[j] * at(v, &dims.v, b, h, j, dd)).sum();
                    out[((b * c.h + h) * c.sq + i) * c.d + dd] = (acc / sum) as f32;
                }
            }
        }
    }
    out
}

fn run_case(bes: &[CpuBackend], c: Case, seed: u64) {
    let (b, h, sq, sk, d) = (c.b, c.h, c.sq, c.sk, c.d);
    let scale = 1.0 / (d as f32).sqrt();
    // q stored [B, Sq, H, D] (projection output), read as [B, H, Sq, D].
    let qv = View { offset: 0, strides: vec![(sq * h * d) as isize, d as isize, (h * d) as isize, 1] };
    let (kv_view, kv_len) = match c.cap {
        Some(cap) => {
            let off = 3 * h * d; // cache slice starting past a few rows
            (View { offset: off, strides: vec![(cap * h * d) as isize, d as isize, (h * d) as isize, 1] }, off + b * cap * h * d)
        }
        None => (View::contiguous(&[b, h, sk, d]), b * h * sk * d),
    };
    // Mask [B, 1, Sq, Sk] broadcast over heads, a few -inf entries.
    let mview = View { offset: 0, strides: vec![(sq * sk) as isize, 0, sk as isize, 1] };
    let mask: Vec<f32> = rnd(b * sq * sk, seed + 3)
        .iter()
        .enumerate()
        .map(|(e, &x)| if e % 7 == 3 { f32::NEG_INFINITY } else { 2.0 * x })
        .collect();
    let (q, qb) = store(&rnd(b * sq * h * d, seed), c.dt);
    let (k, kb) = store(&rnd(kv_len, seed + 1), c.dt);
    // Non-zero-mean values so max|v| scaling is meaningful.
    let (v, vb) = store(&rnd(kv_len, seed + 2).iter().map(|x| x + 0.25).collect::<Vec<_>>(), c.dt);
    let spec = Arc::new(AttentionSpec {
        dtype: c.dt,
        mask_dtype: c.mask.then_some(DType::F32),
        causal: c.causal,
        out_dtype: DType::F32,
    });
    let dims = AttentionDims {
        batch: b,
        heads: h,
        sq,
        sk,
        d,
        scale: scale.to_bits(),
        q: qv,
        k: kv_view.clone(),
        v: kv_view,
        mask: c.mask.then_some(mview),
    };
    let want = reference(&c, &q, &k, &v, &mask, &dims, scale as f64);
    let vmax = v.iter().fold(0f32, |a, &x| a.max(x.abs()));
    let tol = if c.dt == DType::F16 { 2e-3 } else { 1e-5 } * vmax;
    for be in bes {
        let mut bytes = vec![qb.clone(), kb.clone(), vb.clone()];
        if c.mask {
            bytes.push(bytemuck::cast_slice(&mask).to_vec());
        }
        let mut bufs: Vec<DeviceData> = bytes.iter().map(|x| be.upload(x).unwrap()).collect();
        let n = bufs.len();
        bufs.push(be.alloc(want.len() * 4).unwrap());
        let launch = [Launch { kernel: Kernel::Attention(spec.clone(), dims.clone()), inputs: (0..n).collect(), output: n }];
        be.prepare(&launch).unwrap();
        be.execute(&launch, &bufs).unwrap();
        let got: Vec<f32> = bytemuck::pod_collect_to_vec(&be.download(&bufs[n]).unwrap());
        let (e, worst) = got
            .iter()
            .zip(&want)
            .map(|(&g, &w)| (g - w).abs())
            .enumerate()
            .fold((0, 0f32), |a, (e, x)| if x > a.1 || x.is_nan() { (e, x) } else { a });
        assert!(
            worst <= tol && got.iter().all(|x| x.is_finite()),
            "{c:?} threads={}: max abs err {worst} at {e} (got {}, want {}) > tol {tol}",
            be.threads(),
            got[e],
            want[e]
        );
    }
}

fn backends() -> Vec<CpuBackend> {
    vec![CpuBackend::with_threads(4), CpuBackend::build(1, Isa::default())]
}

fn case(h: usize, sq: usize, sk: usize, d: usize, causal: bool, mask: bool) -> Case {
    Case { b: 1, h, sq, sk, d, causal, mask, dt: DType::F32, cap: None }
}

/// Regression: the causal limit cut a key block short of its packed length, and the P·V
/// product then read V panels at the wrong stride whenever d > 16 (found at h=1, 64x64x64).
#[test]
fn attention_causal_square_regression() {
    let bes = backends();
    run_case(&bes, case(1, 64, 64, 64, true, false), 1);
    run_case(&bes, case(2, 100, 100, 128, true, true), 2);
    run_case(&bes, case(1, 48, 300, 80, true, false), 3);
}

#[test]
fn attention_sweep_f32() {
    let bes = backends();
    let mut seed = 10;
    for d in [32, 64, 80, 128] {
        // (h, sq, sk): decode, square, KV-cache prefill/chunk, odd block remainders
        for (h, sq, sk) in [(1, 1, 1), (3, 1, 77), (12, 1, 300), (2, 4, 33), (1, 5, 5), (4, 7, 131), (1, 64, 64), (6, 50, 50), (2, 37, 129), (5, 13, 200)] {
            for (causal, mask) in [(false, false), (true, false), (false, true), (true, true)] {
                seed += 1;
                run_case(&bes, case(h, sq, sk, d, causal, mask), seed);
            }
        }
    }
}

#[test]
fn attention_sweep_f16_and_batches() {
    let bes = backends();
    let mut seed = 500;
    for d in [32, 64, 80, 128] {
        for (b, h, sq, sk) in [(1, 1, 1, 90), (2, 3, 1, 260), (1, 8, 64, 64), (2, 2, 20, 97), (1, 12, 9, 9)] {
            for (causal, mask) in [(false, false), (true, false), (true, true)] {
                seed += 1;
                run_case(&bes, Case { b, h, sq, sk, d, causal, mask, dt: DType::F16, cap: None }, seed);
                run_case(&bes, Case { b, h, sq, sk, d, causal, mask, dt: DType::F32, cap: None }, seed);
            }
        }
    }
}

/// k/v read out of a [B, cap, H, D] KV cache (strides b=cap*H*D, h=D, s=H*D, d=1) with len < cap.
#[test]
fn attention_strided_kv_cache() {
    let bes = backends();
    let mut seed = 900;
    for dt in [DType::F32, DType::F16] {
        for d in [32, 64, 80, 128] {
            for (b, h, sq, sk, cap) in [(1, 4, 1, 37, 64), (2, 6, 1, 600, 700), (1, 2, 8, 50, 64), (2, 3, 30, 130, 200), (1, 12, 64, 64, 100)] {
                for (causal, mask) in [(true, false), (false, true), (true, true)] {
                    seed += 1;
                    run_case(&bes, Case { b, h, sq, sk, d, causal, mask, dt, cap: Some(cap) }, seed);
                }
            }
        }
    }
}
