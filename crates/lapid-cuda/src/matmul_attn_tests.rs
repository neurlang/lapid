//! Matmul (GEMV / tensor-core / scalar GEMM) and attention kernel tests on CUDA device 0,
//! against exact references (dequantized-B CPU matmul, naive f64 attention).

use std::sync::Arc;

use lapid_compiler::*;
use lapid_cpu::CpuBackend;
use lapid_tensor::DType;

use crate::CudaBackend;
use crate::tests::{bytes, check, compare, cuda, floats, gelu_plus, mat_spec, quant, rnd, run_on};

#[allow(clippy::too_many_arguments)]
fn matmul_case(
    be: &CudaBackend,
    a_t: DType,
    b_t: DType,
    bt: bool,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    tol: f32,
) {
    let epi = Some(gelu_plus(0, 1));
    let spec = mat_spec(a_t, b_t, bt, vec![DType::F32], epi);
    let is_q = b_t.is_quant();
    let bb = if is_q { 1 } else { batch };
    let a = rnd(batch * m * k, 7);
    let b: Vec<f32> = rnd(bb * n * k, 8).iter().map(|v| v * 0.2).collect();
    let bias = rnd(n, 9);
    let b_bytes = if is_q {
        quant(b_t, &b, n, k)
    } else {
        bytes(b_t, &b)
    };
    let dims = MatMulDims {
        batch,
        m,
        n,
        k,
        a: MatView {
            offset: 0,
            batch_stride: m * k,
            row_stride: k,
        },
        b: MatView {
            offset: 0,
            batch_stride: if is_q { 0 } else { n * k },
            row_stride: if bt { k } else { n },
        },
        extra_views: vec![View {
            offset: 0,
            strides: vec![0, 0, 1],
        }],
        out: MatView::contiguous(m, n),
    };
    let what = format!("matmul {a_t:?}x{b_t:?} bt={bt} {batch}x{m}x{n}x{k}");
    let inputs = [bytes(a_t, &a), b_bytes, bytes(DType::F32, &bias)];
    if !is_q {
        return compare(
            be,
            &what,
            Kernel::MatMul(spec, dims),
            &inputs,
            DType::F32,
            batch * m * n,
            tol,
        );
    }
    // quantized B: exact reference = the same matmul with B dequantized to f32 (the CPU
    // backend's own int8 path quantizes A, which is less exact than this backend)
    let f = if b_t == DType::Int8 {
        lapid_quant::Format::Int8
    } else {
        lapid_quant::Format::Int4
    };
    let deq = lapid_quant::dequantize(f, &inputs[1], n, k);
    let ref_spec = mat_spec(
        a_t,
        DType::F32,
        true,
        vec![DType::F32],
        spec.epilogue.clone(),
    );
    let ref_inputs = [
        inputs[0].clone(),
        bytes(DType::F32, &deq),
        inputs[2].clone(),
    ];
    let ob = batch * m * n * 4;
    let got = floats(
        DType::F32,
        &run_on(be, &Kernel::MatMul(spec, dims.clone()), &inputs, ob),
    );
    let want = floats(
        DType::F32,
        &run_on(
            &CpuBackend::new(),
            &Kernel::MatMul(ref_spec, dims),
            &ref_inputs,
            ob,
        ),
    );
    check(&what, &got, &want, tol);
}

#[test]
fn matmul_gemm_variants() {
    let Some(be) = cuda() else { return };
    // scalar f32 GEMM (exact path)
    matmul_case(&be, DType::F32, DType::F32, true, 2, 70, 130, 100, 1e-5);
    matmul_case(&be, DType::F32, DType::F32, false, 1, 65, 64, 33, 1e-5);
    // tensor-core GEMM (f16 operands)
    for b in [DType::F16, DType::Int8, DType::Int4] {
        matmul_case(&be, DType::F32, b, true, 1, 150, 200, 96, 3e-3);
        matmul_case(&be, DType::F16, b, true, 1, 77, 129, 64, 3e-3);
    }
    matmul_case(&be, DType::F32, DType::F16, true, 3, 33, 50, 40, 3e-3);
    matmul_case(&be, DType::F32, DType::F16, false, 2, 40, 136, 72, 3e-3);
    matmul_case(&be, DType::F32, DType::F16, true, 1, 17, 30, 13, 3e-3);
}

#[test]
fn matmul_gemv_variants() {
    let Some(be) = cuda() else { return };
    for m in [1usize, 3] {
        for b in [
            DType::F32,
            DType::F16,
            DType::Int8,
            DType::Int4,
            DType::BF16,
        ] {
            matmul_case(&be, DType::F32, b, true, 1, m, 300, 768, 2e-5);
            matmul_case(&be, DType::F16, b, true, 1, m, 37, 96, 2e-3);
        }
        matmul_case(&be, DType::F32, DType::F16, true, 2, m, 45, 100, 2e-5);
        matmul_case(&be, DType::F32, DType::F32, true, 1, m, 33, 13, 2e-5);
    }
}

/// Attention shape / layout of one test case.
#[derive(Clone, Copy)]
struct Att {
    dt: DType,
    mask: bool,
    causal: bool,
    b: usize,
    h: usize,
    sq: usize,
    sk: usize,
    d: usize,
    /// K/V read from a `[B, cap, H, D]` KV cache (first `sk` positions) instead of `[B, Sk, H, D]`.
    cap: Option<usize>,
}

#[allow(clippy::too_many_arguments)]
fn attention_case(be: &CudaBackend, dt: DType, mask: bool, causal: bool, h: usize, sq: usize, sk: usize, d: usize, tol: f32) {
    attention_run(be, Att { dt, mask, causal, b: 1, h, sq, sk, d, cap: None }, tol);
}

/// Run one attention case against a naive f64 reference over the stored values.
fn attention_run(be: &CudaBackend, c: Att, tol: f32) {
    let Att { dt, mask, causal, b, h, sq, sk, d, cap } = c;
    let kv_s = cap.unwrap_or(sk);
    // q stored [B, Sq, H, D] (as projections are), viewed as [B, H, Sq, D]; k/v [B, kv_s, H, D]
    let view = |s: usize| View {
        offset: 0,
        strides: vec![(s * h * d) as isize, d as isize, (h * d) as isize, 1],
    };
    let q = rnd(b * sq * h * d, 14);
    let k = rnd(b * kv_s * h * d, 15);
    let v = rnd(b * kv_s * h * d, 16);
    let m: Vec<f32> = rnd(sq * sk, 17)
        .iter()
        .enumerate()
        .map(|(i, x)| if i % 7 == 3 { f32::NEG_INFINITY } else { *x })
        .collect();
    let spec = Arc::new(AttentionSpec {
        dtype: dt,
        mask_dtype: mask.then_some(DType::F32),
        causal,
        out_dtype: DType::F32,
    });
    let dims = AttentionDims {
        batch: b,
        heads: h,
        sq,
        sk,
        d,
        scale: (1.0 / (d as f32).sqrt()).to_bits(),
        q: view(sq),
        k: view(kv_s),
        v: view(kv_s),
        mask: mask.then(|| View {
            offset: 0,
            strides: vec![0, 0, sk as isize, 1],
        }),
    };
    let mut inputs = vec![bytes(dt, &q), bytes(dt, &k), bytes(dt, &v)];
    if mask {
        inputs.push(bytes(DType::F32, &m));
    }
    let what = format!("attention {dt:?} mask={mask} causal={causal} b={b} h={h} sq={sq} sk={sk} d={d} cap={cap:?}");
    let got = floats(
        DType::F32,
        &run_on(be, &Kernel::Attention(spec, dims), &inputs, b * h * sq * d * 4),
    );
    // naive f64 reference over the values as stored (f16 rounding included)
    let (q, k, v) = (
        floats(dt, &inputs[0]),
        floats(dt, &inputs[1]),
        floats(dt, &inputs[2]),
    );
    let scale = 1.0 / (d as f64).sqrt();
    let mut want = vec![0f32; b * h * sq * d];
    for bb in 0..b {
        for hh in 0..h {
            for i in 0..sq {
                let at = |x: &[f32], ns: usize, s: usize, c: usize| x[((bb * ns + s) * h + hh) * d + c] as f64;
                let sc: Vec<f64> = (0..sk)
                    .map(|j| {
                        let hidden = causal && j as isize > i as isize + sk as isize - sq as isize;
                        let mv = if mask { m[i * sk + j] as f64 } else { 0.0 };
                        if hidden {
                            f64::NEG_INFINITY
                        } else {
                            (0..d).map(|c| at(&q, sq, i, c) * at(&k, kv_s, j, c)).sum::<f64>() * scale + mv
                        }
                    })
                    .collect();
                let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if mx == f64::NEG_INFINITY {
                    continue;
                }
                let ex: Vec<f64> = sc.iter().map(|x| (x - mx).exp()).collect();
                let l: f64 = ex.iter().sum();
                for c in 0..d {
                    want[((bb * h + hh) * sq + i) * d + c] =
                        ((0..sk).map(|j| ex[j] * at(&v, kv_s, j, c)).sum::<f64>() / l) as f32;
                }
            }
        }
    }
    check(&what, &got, &want, tol);
}

#[test]
fn attention_variants() {
    let Some(be) = cuda() else { return };
    for causal in [false, true] {
        for mask in [false, true] {
            // tensor cores (many queries)
            attention_case(&be, DType::F32, mask, causal, 3, 100, 130, 64, 4e-3);
            attention_case(&be, DType::F16, mask, causal, 2, 70, 70, 48, 4e-3);
            // split (few queries): one chunk (direct) and several (merge)
            attention_case(&be, DType::F32, mask, causal, 4, 1, 30, 64, 1e-5);
            attention_case(&be, DType::F32, mask, causal, 4, 3, 300, 64, 1e-5);
            attention_case(&be, DType::F16, mask, causal, 2, 2, 129, 80, 1e-5);
        }
    }
    attention_case(&be, DType::F32, false, false, 2, 40, 200, 128, 4e-3);
    attention_case(&be, DType::F32, false, true, 2, 20, 10, 32, 4e-3);
}

/// Shape sweep: causal / not, mask / none, sq = 1, sq = sk, sq < sk, sk not a multiple of
/// the tile sizes, d in {32, 64, 80, 128}, heads 1..12, batch 1..2, f32 and f16.
#[test]
fn attention_shape_sweep() {
    let Some(be) = cuda() else { return };
    // includes the CPU-backend causal regression shape (h=1, sq=sk=64, d=64)
    let shapes: &[(usize, usize, usize, usize, usize)] = &[
        (1, 1, 64, 64, 64),
        (1, 1, 1, 1, 32),
        (1, 12, 1, 1500, 64),
        (2, 6, 1, 77, 80),
        (1, 8, 4, 33, 128),
        (1, 5, 17, 17, 32),
        (2, 3, 64, 200, 64),
        (1, 2, 129, 129, 80),
        (1, 1, 65, 300, 128),
        (1, 12, 150, 150, 64),
    ];
    for &(b, h, sq, sk, d) in shapes {
        for dt in [DType::F32, DType::F16] {
            for (causal, mask) in [(false, false), (true, false), (false, true), (true, true)] {
                // split kernel is exact f32 math; the tensor-core path rounds P / Q to f16
                let tol = if sq <= 4 { if dt == DType::F32 { 1e-5 } else { 2e-5 } } else { 4e-3 };
                attention_run(&be, Att { dt, mask, causal, b, h, sq, sk, d, cap: None }, tol);
            }
        }
    }
}

/// K/V read straight out of a `[B, cap, H, D]` KV cache as `[B, H, len, D]` with len < cap
/// (strides b = cap*H*D, h = D, s = H*D, d = 1), as the decoder does.
#[test]
fn attention_kv_cache_views() {
    let Some(be) = cuda() else { return };
    for &(b, h, sq, len, cap, d) in &[
        (1usize, 6usize, 1usize, 1usize, 448usize, 64usize),
        (1, 6, 1, 37, 448, 64),
        (2, 8, 1, 200, 256, 128),
        (1, 4, 3, 100, 128, 80),
        (1, 6, 24, 24, 448, 64),
        (2, 2, 40, 90, 100, 32),
    ] {
        for dt in [DType::F32, DType::F16] {
            for causal in [false, true] {
                let tol = if sq <= 4 { 2e-5 } else { 4e-3 };
                attention_run(&be, Att { dt, mask: false, causal, b, h, sq, sk: len, d, cap: Some(cap) }, tol);
            }
        }
    }
}
