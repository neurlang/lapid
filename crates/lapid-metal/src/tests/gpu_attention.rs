//! GPU tests: attention (tile, vector direct / split + merge) against a naive reference.

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::DType;

use super::util::*;
use crate::MetalBackend;

pub(crate) struct Case {
    pub b: usize,
    pub h: usize,
    pub sq: usize,
    pub sk: usize,
    pub d: usize,
    pub causal: bool,
    pub mask: bool,
    pub dtype: DType,
    /// q/k/v stored as [B, S, H, D] (permuted views) instead of [B, H, S, D]
    pub bshd: bool,
    /// KV-cache layout: k / v read from a [B, cap, H, D] buffer (cap >= sk) as [B, H, sk, D];
    /// 0 = off.
    pub cap: usize,
}

fn view(c: &Case, s: usize, kv: bool) -> View {
    if c.cap > 0 && kv {
        let (h, d) = (c.h as isize, c.d as isize);
        View { offset: 0, strides: vec![c.cap as isize * h * d, d, h * d, 1] }
    } else if c.bshd {
        View { offset: 0, strides: vec![(s * c.h * c.d) as isize, c.d as isize, (c.h * c.d) as isize, 1] }
    } else {
        View::contiguous(&[c.b, c.h, s, c.d])
    }
}

pub(crate) fn check(be: &MetalBackend, c: &Case) {
    let (b, h, sq, sk, d) = (c.b, c.h, c.sq, c.sk, c.d);
    let q = round_to(c.dtype, &rand(b * h * sq * d, 21).iter().map(|v| v * 2.0).collect::<Vec<_>>());
    let kv_len = b * h * sk.max(c.cap) * d;
    let k = round_to(c.dtype, &rand(kv_len, 22).iter().map(|v| v * 2.0).collect::<Vec<_>>());
    let v = round_to(c.dtype, &rand(kv_len, 23));
    // Mask [1, 1, sq, sk] broadcast over batch / heads, with some -inf entries.
    let mut mask = rand(sq * sk, 24);
    for (i, m) in mask.iter_mut().enumerate() {
        if i % 7 == 3 {
            *m = f32::NEG_INFINITY;
        }
    }
    let (qv, kv, vv) = (view(c, sq, false), view(c, sk, true), view(c, sk, true));
    let mv = View { offset: 0, strides: vec![0, 0, sk as isize, 1] };
    let scale = 1.0 / (d as f32).sqrt();
    let spec = Arc::new(AttentionSpec { dtype: c.dtype, mask_dtype: c.mask.then_some(DType::F32), causal: c.causal, out_dtype: DType::F32 });
    let dims = AttentionDims {
        batch: b,
        heads: h,
        sq,
        sk,
        d,
        scale: scale.to_bits(),
        q: qv.clone(),
        k: kv.clone(),
        v: vv.clone(),
        mask: c.mask.then(|| mv.clone()),
    };
    let mut ins = vec![encode(c.dtype, &q), encode(c.dtype, &k), encode(c.dtype, &v)];
    if c.mask {
        ins.push(encode(DType::F32, &mask));
    }
    let n = b * h * sq * d;
    let got = decode(DType::F32, &run(be, Kernel::Attention(spec, dims), &ins, n * 4), n);
    let mut want = vec![0f32; n];
    for bi in 0..b {
        for hi in 0..h {
            for i in 0..sq {
                let mut sc = vec![f64::NEG_INFINITY; sk];
                for (j, s) in sc.iter_mut().enumerate() {
                    if c.causal && j as isize > i as isize + sk as isize - sq as isize {
                        continue;
                    }
                    let mut dot = 0f64;
                    for dd in 0..d {
                        dot += q[at(0, &qv.strides, &[bi, hi, i, dd])] as f64 * k[at(0, &kv.strides, &[bi, hi, j, dd])] as f64;
                    }
                    *s = dot * scale as f64 + if c.mask { mask[at(0, &mv.strides, &[bi, hi, i, j])] as f64 } else { 0.0 };
                }
                let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if mx == f64::NEG_INFINITY {
                    continue;
                }
                let e: Vec<f64> = sc.iter().map(|&x| (x - mx).exp()).collect();
                let sum: f64 = e.iter().sum();
                for dd in 0..d {
                    let mut acc = 0f64;
                    for j in 0..sk {
                        acc += e[j] * v[at(0, &vv.strides, &[bi, hi, j, dd])] as f64;
                    }
                    want[((bi * h + hi) * sq + i) * d + dd] = (acc / sum) as f32;
                }
            }
        }
    }
    let what = format!(
        "attention b{b} h{h} sq{sq} sk{sk} d{d} causal={} mask={} {:?} bshd={}",
        c.causal, c.mask, c.dtype, c.bshd
    );
    // spec: abs <= 1e-5 * max|v| (f32), 2e-3 * max|v| (f16 storage); |v| <= 1 here.
    let what = format!("{what} cap={}", c.cap);
    assert_close(&what, &got, &want, if c.dtype == DType::F16 { 2e-3 } else { 2e-5 });
}

#[test]
fn attention_tile() {
    let Some(be) = backend() else { return };
    for &(sq, sk, d) in &[(70usize, 50usize, 64usize), (33, 100, 128), (40, 40, 40), (100, 7, 64)] {
        for causal in [false, true] {
            for mask in [false, true] {
                let c = Case { b: 2, h: 3, sq, sk, d, causal, mask, dtype: DType::F32, bshd: sq == 70, cap: 0 };
                check(&be, &c);
            }
        }
    }
    let c = Case { b: 1, h: 2, sq: 64, sk: 64, d: 64, causal: false, mask: false, dtype: DType::F16, bshd: true, cap: 0 };
    check(&be, &c);
}

#[test]
fn attention_vec() {
    let Some(be) = backend() else { return };
    // direct (one split), split + merge, causal with sq > 1, odd head dim, d > 128
    for &(sq, sk, d) in &[(1usize, 100usize, 64usize), (1, 300, 64), (3, 5, 64), (2, 1500, 64), (1, 129, 40), (4, 200, 200), (6, 30, 20)] {
        for causal in [false, true] {
            for mask in [false, true] {
                let c = Case { b: 1, h: 2, sq, sk, d, causal, mask, dtype: DType::F32, bshd: sk == 300, cap: 0 };
                check(&be, &c);
            }
        }
    }
    let c = Case { b: 2, h: 3, sq: 1, sk: 448, d: 64, causal: false, mask: false, dtype: DType::F16, bshd: true, cap: 0 };
    check(&be, &c);
}

#[test]
fn attention_large_head_dim() {
    // Gemma 4: head dim 256 (sliding layers) and 512 (full layers); prefill and decode.
    let Some(be) = backend() else { return };
    for &(sq, sk, d) in &[(1usize, 4096usize, 512usize), (1, 100, 512), (3, 700, 512), (160, 160, 512), (1, 1000, 256), (70, 90, 256)] {
        for causal in [false, true] {
            let c = Case { b: 1, h: 2, sq, sk, d, causal, mask: sq == 3, dtype: DType::F32, bshd: true, cap: 0 };
            check(&be, &c);
        }
    }
    let c = Case { b: 1, h: 2, sq: 33, sk: 300, d: 512, causal: true, mask: false, dtype: DType::F16, bshd: false, cap: 0 };
    check(&be, &c);
}

#[test]
fn attention_decode_sequence_shares_scratch() {
    // Several split attentions in one execute (scratch reuse needs barriers between them).
    let Some(be) = backend() else { return };
    let (h, sk, d) = (4usize, 700usize, 64usize);
    let spec = Arc::new(AttentionSpec { dtype: DType::F32, mask_dtype: None, causal: false, out_dtype: DType::F32 });
    let mk = |sk: usize| AttentionDims {
        batch: 1,
        heads: h,
        sq: 1,
        sk,
        d,
        scale: 0.125f32.to_bits(),
        q: View::contiguous(&[1, h, 1, d]),
        k: View::contiguous(&[1, h, sk, d]),
        v: View::contiguous(&[1, h, sk, d]),
        mask: None,
    };
    let q = rand(h * d, 31);
    let k = rand(h * sk * d, 32);
    let v = rand(h * sk * d, 33);
    let bufs: Vec<_> = [encode(DType::F32, &q), encode(DType::F32, &k), encode(DType::F32, &v)]
        .iter()
        .map(|b| lapid_compiler::Backend::upload(&be, b).unwrap())
        .chain((0..3).map(|_| lapid_compiler::Backend::alloc(&be, h * d * 4).unwrap()))
        .collect();
    // out0 = attn(q, k, v); out1 = attn(out0, k, v); out2 = attn(q, k, v) again (WAR on scratch)
    let launches = vec![
        Launch { kernel: Kernel::Attention(spec.clone(), mk(sk)), inputs: vec![0, 1, 2], output: 3 },
        Launch { kernel: Kernel::Attention(spec.clone(), mk(sk)), inputs: vec![3, 1, 2], output: 4 },
        Launch { kernel: Kernel::Attention(spec.clone(), mk(sk)), inputs: vec![0, 1, 2], output: 5 },
    ];
    lapid_compiler::Backend::execute(&be, &launches, &bufs).unwrap();
    let out: Vec<Vec<f32>> = (3..6).map(|i| decode(DType::F32, &lapid_compiler::Backend::download(&be, &bufs[i]).unwrap(), h * d)).collect();
    // Same launch one by one.
    let single = |inp: usize, outb: usize| {
        let l = Launch { kernel: Kernel::Attention(spec.clone(), mk(sk)), inputs: vec![inp, 1, 2], output: outb };
        lapid_compiler::Backend::execute(&be, &[l], &bufs).unwrap();
        decode(DType::F32, &lapid_compiler::Backend::download(&be, &bufs[outb]).unwrap(), h * d)
    };
    let o0 = single(0, 3);
    let o1 = single(3, 4);
    assert_eq!(out[0], o0);
    assert_eq!(out[1], o1);
    assert_eq!(out[2], o0);
}
