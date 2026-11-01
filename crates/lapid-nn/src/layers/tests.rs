use super::*;
use crate::testutil::{assert_close, eval, matmul_t, rand_tensor};

fn f32s(shape: &[usize]) -> Tensor {
    Tensor::zeros(shape, DType::F32)
}

/// Naive conv1d: x [B, Cin, L], w [Cout, Cin/g, K].
#[allow(clippy::too_many_arguments)]
fn conv_ref(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    (b, cin, l): (usize, usize, usize),
    (cout, k): (usize, usize),
    stride: usize,
    (left, right): (usize, usize),
    dilation: usize,
    groups: usize,
) -> (Vec<f32>, usize) {
    let lp = l + left + right;
    let lout = (lp - dilation * (k - 1) - 1) / stride + 1;
    let (cin_g, cout_g) = (cin / groups, cout / groups);
    let mut out = vec![0.0f32; b * cout * lout];
    for bi in 0..b {
        for co in 0..cout {
            let gi = co / cout_g;
            for o in 0..lout {
                let mut acc = bias.map_or(0.0, |bb| bb[co] as f64);
                for ci in 0..cin_g {
                    let c = gi * cin_g + ci;
                    for t in 0..k {
                        let p = (o * stride + t * dilation) as isize - left as isize;
                        if p >= 0 && (p as usize) < l {
                            acc += x[(bi * cin + c) * l + p as usize] as f64 * w[(co * cin_g + ci) * k + t] as f64;
                        }
                    }
                }
                out[(bi * cout + co) * lout + o] = acc as f32;
            }
        }
    }
    (out, lout)
}

#[test]
fn linear_shape_and_values() {
    let lin = Linear::new(f32s(&[16, 8]), Some(f32s(&[16])));
    assert_eq!(lin.forward(&f32s(&[2, 5, 8])).unwrap().shape(), &[2, 5, 16]);
    assert_eq!((lin.in_features(), lin.out_features()), (8, 16));

    let (x, xv) = rand_tensor(&[3, 8], 1);
    let (w, wv) = rand_tensor(&[5, 8], 2);
    let (bias, bv) = rand_tensor(&[5], 3);
    let y = eval(&Linear::new(w.clone(), Some(bias)).forward(&x).unwrap());
    let mut want = matmul_t(&xv, &wv, 3, 8, 5);
    for (i, v) in want.iter_mut().enumerate() {
        *v += bv[i % 5];
    }
    assert_close(&y, &want, 1e-5);
    let y = eval(&Linear::new(w, None).forward(&x).unwrap());
    assert_close(&y, &matmul_t(&xv, &wv, 3, 8, 5), 1e-5);
}

#[test]
fn linear_act_quant_is_close() {
    let (x, _) = rand_tensor(&[2, 64], 4);
    let (w, _) = rand_tensor(&[3, 64], 5);
    let plain = eval(&Linear::new(w.clone(), None).forward(&x).unwrap());
    let fq = eval(&Linear::new(w, None).with_act_quant(true).forward(&x).unwrap());
    assert_close(&fq, &plain, 5e-2);
}

#[test]
fn embedding_lookup_and_logits() {
    let (table, tv) = rand_tensor(&[10, 4], 6);
    let e = Embedding::new(table);
    assert_eq!((e.vocab_size(), e.dim()), (10, 4));
    let ids = Tensor::from_i32(&[1, 2, 3], &[1, 3]).unwrap();
    assert_eq!(e.forward(&ids).unwrap().shape(), &[1, 3, 4]);
    let got = eval(&e.lookup(&[7, 0, 7]).unwrap());
    let want: Vec<f32> = [7, 0, 7].iter().flat_map(|&i| tv[i * 4..i * 4 + 4].to_vec()).collect();
    assert_eq!(got, want);
    assert_eq!(e.lookup(&[10]).unwrap_err().code, "invalid_argument");
    assert!(e.lookup(&[]).is_err());

    let (x, xv) = rand_tensor(&[1, 2, 4], 7);
    let logits = eval(&e.logits(&x).unwrap());
    assert_close(&logits, &matmul_t(&xv, &tv, 2, 4, 10), 1e-5);
}

#[test]
fn layer_norm_values() {
    let (x, xv) = rand_tensor(&[3, 6], 8);
    let (w, wv) = rand_tensor(&[6], 9);
    let (b, bv) = rand_tensor(&[6], 10);
    let got = eval(&LayerNorm::new(w.clone(), b, 1e-5).forward(&x).unwrap());
    let got_nb = eval(&LayerNorm::no_bias(w, 1e-5).forward(&x).unwrap());
    let mut want = Vec::new();
    let mut want_nb = Vec::new();
    for row in xv.chunks(6) {
        let mean = row.iter().sum::<f32>() / 6.0;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 6.0;
        for (i, v) in row.iter().enumerate() {
            let n = (v - mean) / (var + 1e-5).sqrt();
            want.push(n * wv[i] + bv[i]);
            want_nb.push(n * wv[i]);
        }
    }
    assert_close(&got, &want, 1e-5);
    assert_close(&got_nb, &want_nb, 1e-5);
}

#[test]
fn rms_norm_values() {
    let (x, xv) = rand_tensor(&[2, 3, 8], 11);
    let (w, wv) = rand_tensor(&[8], 12);
    let norm = RmsNorm::new(w.clone(), 1e-6);
    let got = eval(&norm.forward(&x).unwrap());
    let got_off = eval(&RmsNorm::new(w, 1e-6).with_weight_offset(1.0).forward(&x).unwrap());
    let mut want = Vec::new();
    let mut want_off = Vec::new();
    for row in xv.chunks(8) {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / 8.0;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for (i, v) in row.iter().enumerate() {
            want.push(v * inv * wv[i]);
            want_off.push(v * inv * (1.0 + wv[i]));
        }
    }
    assert_close(&got, &want, 1e-5);
    assert_close(&got_off, &want_off, 1e-5);
    // f16 input comes back as f16
    let h = Tensor::zeros(&[2, 8], DType::F16);
    assert_eq!(norm.forward(&h).unwrap().dtype(), DType::F16);
    assert!(norm.forward(&f32s(&[2, 7])).is_err());
}

#[test]
fn conv1d_shapes() {
    // Whisper conv1: stride 1, conv2: stride 2.
    let c1 = Conv1d::new(Tensor::zeros(&[12, 4, 3], DType::F16), Some(f32s(&[12]))).with_padding(1);
    assert_eq!(c1.forward(&f32s(&[1, 4, 30])).unwrap().shape(), &[1, 12, 30]);
    let c2 = Conv1d::new(f32s(&[6, 12, 3]), Some(f32s(&[6]))).with_stride(2).with_padding(1);
    assert_eq!(c2.forward(&f32s(&[1, 12, 30])).unwrap().shape(), &[1, 6, 15]);
    assert_eq!(c2.forward(&f32s(&[2, 12, 31])).unwrap().shape(), &[2, 6, 16]);
    let c3 = Conv1d::new(f32s(&[2, 3, 5]), None).with_stride(3);
    assert_eq!(c3.forward(&f32s(&[1, 3, 11])).unwrap().shape(), &[1, 2, 3]);
    assert!(c3.forward(&f32s(&[1, 4, 11])).is_err());
    let dw = Conv1d::depthwise(f32s(&[8, 1, 3]), None).causal();
    assert_eq!(dw.groups, 8);
    assert_eq!(dw.padding, (2, 0));
    assert_eq!(dw.forward(&f32s(&[1, 8, 10])).unwrap().shape(), &[1, 8, 10]);
}

#[test]
fn conv1d_matches_naive() {
    // (B, Cin, L), (Cout, K), stride, padding, dilation, groups
    type Case = ((usize, usize, usize), (usize, usize), usize, (usize, usize), usize, usize);
    let cases: &[Case] = &[
        ((1, 4, 17), (6, 3), 1, (1, 1), 1, 1), // whisper conv1
        ((2, 3, 20), (5, 3), 2, (1, 1), 1, 1), // whisper conv2
        ((1, 3, 23), (4, 5), 3, (2, 0), 1, 1),
        ((1, 2, 16), (3, 3), 1, (4, 0), 2, 1), // dilated causal
        ((2, 6, 12), (6, 3), 1, (2, 0), 1, 6), // depthwise causal (short conv)
        ((1, 4, 15), (4, 9), 1, (4, 4), 1, 4), // depthwise "same" (conformer)
        ((1, 4, 14), (6, 3), 2, (1, 1), 1, 2), // grouped
    ];
    for (n, &(xs, (cout, k), stride, pad, dil, groups)) in cases.iter().enumerate() {
        let (b, cin, l) = xs;
        let (x, xv) = rand_tensor(&[b, cin, l], 100 + n as u64);
        let (w, wv) = rand_tensor(&[cout, cin / groups, k], 200 + n as u64);
        let (bias, bv) = rand_tensor(&[cout], 300 + n as u64);
        let conv = Conv1d::new(w, Some(bias))
            .with_stride(stride)
            .with_padding_lr(pad.0, pad.1)
            .with_dilation(dil)
            .with_groups(groups);
        let y = conv.forward(&x).unwrap();
        let (want, lout) = conv_ref(&xv, &wv, Some(&bv), xs, (cout, k), stride, pad, dil, groups);
        assert_eq!(y.shape(), &[b, cout, lout], "case {n}");
        assert_close(&eval(&y), &want, 1e-5);
    }
}

#[test]
fn conv1d_stream_equals_full() {
    for (groups, cout) in [(1usize, 5usize), (4, 4)] {
        let (x, _) = rand_tensor(&[1, 4, 13], 400);
        let (w, _) = rand_tensor(&[cout, 4 / groups, 3], 401);
        let conv = Conv1d::new(w, None).with_dilation(2).with_groups(groups).causal();
        let full = eval(&conv.forward(&x).unwrap());
        let mut state = None;
        let mut parts: Vec<Vec<f32>> = Vec::new();
        let mut lens = Vec::new();
        for (a, b) in [(0usize, 5usize), (5, 6), (6, 13)] {
            let y = conv.forward_stream(&x.slice(2, a, b).unwrap(), &mut state).unwrap();
            assert_eq!(state.as_ref().unwrap().shape(), &[1, 4, 4]);
            lens.push(b - a);
            parts.push(eval(&y));
        }
        // stitch [Cout, len] chunks back together along time
        let mut stitched = vec![0.0f32; cout * 13];
        let mut t0 = 0;
        for (p, &len) in parts.iter().zip(&lens) {
            for c in 0..cout {
                stitched[c * 13 + t0..c * 13 + t0 + len].copy_from_slice(&p[c * len..(c + 1) * len]);
            }
            t0 += len;
        }
        assert_close(&stitched, &full, 1e-5);
    }
    // Depthwise, dilation 1, single-frame steps (the decode fast path) with a bias.
    for k in [2usize, 4] {
        let (x, _) = rand_tensor(&[2, 3, 9], 402);
        let (w, _) = rand_tensor(&[3, 1, k], 403);
        let (bias, _) = rand_tensor(&[3], 404);
        let conv = Conv1d::depthwise(w, Some(bias)).causal();
        let full = eval(&conv.forward(&x).unwrap());
        let mut state = None;
        let mut t0 = 0;
        let mut stitched = vec![0.0f32; 2 * 3 * 9];
        for len in [3usize, 1, 1, 1, 3] {
            let y = eval(&conv.forward_stream(&x.slice(2, t0, t0 + len).unwrap(), &mut state).unwrap());
            assert_eq!(state.as_ref().unwrap().shape(), &[2, 3, k - 1]);
            for bc in 0..6 {
                stitched[bc * 9 + t0..bc * 9 + t0 + len].copy_from_slice(&y[bc * len..(bc + 1) * len]);
            }
            t0 += len;
        }
        assert_close(&stitched, &full, 1e-5);
    }
    let strided = Conv1d::new(f32s(&[2, 2, 3]), None).with_stride(2).causal();
    assert!(strided.forward_stream(&f32s(&[1, 2, 8]), &mut None).is_err());
}
