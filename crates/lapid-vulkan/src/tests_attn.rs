//! Attention regression tests against a naive f64 reference over many shapes: causal / not,
//! mask / none, sq = 1, sq = sk, sq < sk, odd sk, head dims 32..128, 1..12 heads, f32 and
//! f16 storage, and a KV-cache layout (K/V read from a [B, cap, H, D] buffer).

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::{DType, DeviceData};

use crate::tests::{backend, f16b, f32b, from_f32b, rnd};

#[derive(Clone, Copy)]
struct Case {
    b: usize,
    h: usize,
    sq: usize,
    sk: usize,
    d: usize,
    causal: bool,
    mask: bool,
    f16: bool,
    /// K/V live in a [B, cap, H, D] cache with cap > sk
    cache: bool,
}

fn idx(v: &View, c: [usize; 4]) -> usize {
    (v.offset as isize + (0..4).map(|i| c[i] as isize * v.strides[i]).sum::<isize>()) as usize
}

#[allow(clippy::too_many_arguments)]
fn reference(c: &Case, q: &[f32], k: &[f32], v: &[f32], m: Option<&[f32]>, qv: &View, kv: &View, vv: &View, mv: Option<&View>, scale: f32) -> Vec<f32> {
    let mut out = vec![0f32; c.b * c.h * c.sq * c.d];
    for b in 0..c.b {
        for h in 0..c.h {
            for i in 0..c.sq {
                let mut s = vec![f64::NEG_INFINITY; c.sk];
                for (j, sj) in s.iter_mut().enumerate() {
                    if c.causal && j as isize > i as isize + c.sk as isize - c.sq as isize {
                        continue;
                    }
                    let mut dot = 0f64;
                    for e in 0..c.d {
                        dot += q[idx(qv, [b, h, i, e])] as f64 * k[idx(kv, [b, h, j, e])] as f64;
                    }
                    let mut x = dot * scale as f64;
                    if let (Some(m), Some(mv)) = (m, mv) {
                        x += m[idx(mv, [b, h, i, j])] as f64;
                    }
                    *sj = x;
                }
                let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if mx == f64::NEG_INFINITY {
                    continue;
                }
                let w: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
                let tot: f64 = w.iter().sum();
                for e in 0..c.d {
                    let acc: f64 = (0..c.sk).map(|j| w[j] * v[idx(vv, [b, h, j, e])] as f64).sum();
                    out[((b * c.h + h) * c.sq + i) * c.d + e] = (acc / tot) as f32;
                }
            }
        }
    }
    out
}

fn run_case(be: &crate::VulkanBackend, c: Case, seed: u32) {
    let (b, h, sq, sk, d) = (c.b, c.h, c.sq, c.sk, c.d);
    let qv = View::contiguous(&[b, h, sq, d]);
    let (kv, kv_elems) = if c.cache {
        let cap = sk + 13;
        // [B, cap, H, D] read as [B, H, len, D]: strides b = cap*H*D, h = D, s = H*D, d = 1
        (View { offset: 0, strides: vec![(cap * h * d) as isize, d as isize, (h * d) as isize, 1] }, b * cap * h * d)
    } else {
        (View::contiguous(&[b, h, sk, d]), b * h * sk * d)
    };
    let round = |x: Vec<f32>| if c.f16 { x.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect() } else { x };
    let q = round(rnd(b * h * sq * d, seed));
    let k = round(rnd(kv_elems, seed + 1));
    let v = round(rnd(kv_elems, seed + 2));
    // mask broadcast over batch and heads: [1, 1, sq, sk]
    let mv = View { offset: 0, strides: vec![0, 0, sk as isize, 1] };
    let m: Vec<f32> = (0..sq * sk).map(|i| if i % 5 == 3 { -1e9 } else { ((i * 7) % 11) as f32 * 0.1 - 0.5 }).collect();
    let scale = 1.0 / (d as f32).sqrt();
    let want = reference(&c, &q, &k, &v, c.mask.then_some(&m[..]), &qv, &kv, &kv, c.mask.then_some(&mv), scale);
    let dt = if c.f16 { DType::F16 } else { DType::F32 };
    let enc = |x: &[f32]| if c.f16 { f16b(x) } else { f32b(x) };
    let spec = Arc::new(AttentionSpec { dtype: dt, mask_dtype: c.mask.then_some(DType::F32), causal: c.causal, out_dtype: DType::F32 });
    let dims = AttentionDims {
        batch: b,
        heads: h,
        sq,
        sk,
        d,
        scale: scale.to_bits(),
        q: qv,
        k: kv.clone(),
        v: kv,
        mask: c.mask.then_some(mv),
    };
    let mut ins = vec![enc(&q), enc(&k), enc(&v)];
    if c.mask {
        ins.push(f32b(&m));
    }
    let mut bufs: Vec<DeviceData> = ins.iter().map(|x| be.upload(x).unwrap()).collect();
    bufs.push(be.alloc(want.len() * 4).unwrap());
    let n = ins.len();
    be.execute(&[Launch { kernel: Kernel::Attention(spec, dims), inputs: (0..n).collect(), output: n }], &bufs).unwrap();
    let got = from_f32b(&be.download(&bufs[n]).unwrap());
    let vmax = v.iter().fold(0f32, |a, x| a.max(x.abs()));
    // f16 storage (and the coopmat path, which rounds P to f16) gets the f16 tolerance
    let coop_like = be.shared.caps.coopmat && d == 64 && sq > 8;
    let tol = if c.f16 || coop_like { 2e-3 } else { 1e-5 } * vmax;
    let tol = tol.max(1e-6);
    let mut worst = (0f32, 0usize);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let e = (g - w).abs();
        if e > worst.0 || e.is_nan() {
            worst = (e, i);
        }
    }
    let ok = worst.0 <= tol * 4.0;
    assert!(
        ok,
        "attention b{b} h{h} sq{sq} sk{sk} d{d} causal={} mask={} f16={} cache={}: max err {} at {} (got {}, want {}), tol {tol}",
        c.causal, c.mask, c.f16, c.cache, worst.0, worst.1, got[worst.1], want[worst.1]
    );
}

#[test]
fn attention_regression_shapes() {
    let Some(be) = backend() else { return };
    let mut n = 0u32;
    for &d in &[32usize, 64, 80, 128] {
        for &(sq, sk) in &[(1usize, 1usize), (1, 77), (1, 1500), (64, 64), (37, 37), (13, 100), (100, 130), (9, 9)] {
            for &(causal, mask) in &[(false, false), (true, false), (false, true), (true, true)] {
                for &f16 in &[false, true] {
                    let h = [1usize, 3, 12][(n as usize) % 3];
                    let b = 1 + (n as usize % 2);
                    // keep the f64 reference affordable
                    if b * h * sq * sk * d > 6_000_000 {
                        n += 1;
                        continue;
                    }
                    run_case(&be, Case { b, h, sq, sk, d, causal, mask, f16, cache: false }, n);
                    n += 1;
                }
            }
        }
    }
}

#[test]
fn attention_single_head_causal_square() {
    // the shape that exposed a causal bug in the CPU backend: h = 1, sq = sk = 64, d = 64
    let Some(be) = backend() else { return };
    for f16 in [false, true] {
        run_case(&be, Case { b: 1, h: 1, sq: 64, sk: 64, d: 64, causal: true, mask: false, f16, cache: false }, 7);
    }
}

#[test]
fn attention_kv_cache_layout() {
    let Some(be) = backend() else { return };
    let mut n = 100;
    for &d in &[64usize, 80, 128] {
        for &(sq, sk) in &[(1usize, 1usize), (1, 33), (1, 200), (5, 70), (24, 90)] {
            for &f16 in &[false, true] {
                for causal in [true, false] {
                    run_case(&be, Case { b: 2, h: 6, sq, sk, d, causal, mask: false, f16, cache: true }, n);
                    n += 1;
                }
            }
        }
    }
}
