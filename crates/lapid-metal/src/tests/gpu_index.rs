//! GPU tests: GatherAxis, TopK (argmax and general k), out-of-range index errors and strided
//! matmul outputs, compared with the CPU backend.

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::DType;

use super::keys::mat_spec;
use super::util::*;

#[test]
fn gather_axis_matches_cpu() {
    let Some(be) = backend() else { return };
    let (outer, axis_len, n, inner) = (3usize, 11usize, 7usize, 5usize);
    for dt in [DType::F32, DType::F16, DType::BF16, DType::I32] {
        let xn = outer * axis_len * inner + 4;
        let x: Vec<u8> = (0..xn * dt.size()).map(|i| (i * 37 + 11) as u8).collect(); // arbitrary bits (NaNs too)
        let idx: Vec<f32> = (0..outer * n * inner).map(|i| ((i * 7 + 3) % axis_len) as f32).collect();
        // x: contiguous at offset 4, and a permuted view ([inner, axis, outer] storage);
        // indices: contiguous, and broadcast along inner.
        let cases = [
            (View { offset: 4, strides: vec![(axis_len * inner) as isize, inner as isize, 1] }, View::contiguous(&[outer, n, inner])),
            (View { offset: 0, strides: vec![1, outer as isize, (outer * axis_len) as isize] }, View { offset: 2, strides: vec![n as isize, 1, 0] }),
        ];
        for (xv, iv) in cases {
            let dims = GatherAxisDims { outer, n, inner, axis_len, x: xv.clone(), idx: iv.clone() };
            let k = Kernel::GatherAxis(Arc::new(GatherAxisSpec { dtype: dt }), dims);
            let (m, c) = both(&be, k, &[x.clone(), encode(DType::I32, &idx)], &vec![0xCD; outer * n * inner * dt.size()]);
            assert_eq!(m, c, "gather_axis {dt:?} x {xv:?} idx {iv:?}");
        }
    }
}

#[test]
fn index_out_of_range_is_reported() {
    let Some(be) = backend() else { return };
    let x = encode(DType::F32, &rand(4 * 6, 1));
    let dims = |len: usize| GatherAxisDims { outer: 1, n: 3, inner: 6, axis_len: len, x: View::contiguous(&[1, 4, 6]), idx: View { offset: 0, strides: vec![0, 1, 0] } };
    let spec = Arc::new(GatherAxisSpec { dtype: DType::F32 });
    for bad in [-1.0f32, 4.0, 1e6] {
        let idx = encode(DType::I32, &[0.0, bad, 2.0]);
        let e = run_on(&be, Kernel::GatherAxis(spec.clone(), dims(4)), &[x.clone(), idx], &[0; 72]).unwrap_err();
        assert_eq!(e.code, "index_out_of_range");
    }
    // The flag is cleared: a valid launch afterwards succeeds.
    run_on(&be, Kernel::GatherAxis(spec, dims(4)), &[x.clone(), encode(DType::I32, &[0.0, 3.0, 2.0])], &[0; 72]).unwrap();
    // Row gather (embedding lookup).
    let gs = Arc::new(GatherSpec { table_dtype: DType::F32, out_dtype: DType::F32 });
    let gd = GatherDims { rows: 4, cols: 6, count: 2 };
    let e = run_on(&be, Kernel::Gather(gs.clone(), gd.clone()), &[x.clone(), encode(DType::I32, &[1.0, 4.0])], &[0; 48]).unwrap_err();
    assert_eq!(e.code, "index_out_of_range");
    run_on(&be, Kernel::Gather(gs, gd), &[x, encode(DType::I32, &[1.0, 3.0])], &[0; 48]).unwrap();
}

/// Indices of the k largest per row: value descending (-0 == +0), ties -> lower index.
fn topk_ref(v: &[f32], rows: usize, cols: usize, k: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let row = &v[r * cols..(r + 1) * cols];
        let mut idx: Vec<usize> = (0..cols).collect();
        idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
        out.extend(idx[..k].iter().map(|&i| i as u32));
    }
    out
}

#[test]
fn topk_matches_cpu() {
    let Some(be) = backend() else { return };
    let cases: &[(usize, usize, usize)] =
        &[(1, 51865, 1), (3, 1000, 1), (2, 7, 1), (1, 51865, 5), (4, 1000, 2), (2, 300, 37), (1, 5000, 300), (2, 2048, 1024), (3, 33, 33), (1, 150000, 50)];
    for &(rows, cols, k) in cases {
        for dt in [DType::F32, DType::F16, DType::I32] {
            // Coarse values (many ties, also across rows) plus a few exact duplicates of the max.
            let mut v: Vec<f32> = rand(rows * cols, (cols + k) as u64).iter().map(|x| (x * 40.0).round()).collect();
            if cols > 10 {
                v[3] = 1e3;
                v[cols - 2] = 1e3;
                v[1] = -0.0;
                v[2] = 0.0;
            }
            let spec = Arc::new(TopKSpec { in_dtype: dt });
            let (m, c) = both(&be, Kernel::TopK(spec, TopKDims { rows, cols, k }), &[encode(dt, &v)], &vec![0xEE; rows * k * 4]);
            let want = topk_ref(&round_to(dt, &v), rows, cols, k);
            let (wm, wc) = (words(&m), words(&c));
            if let Some(i) = wm.iter().zip(&want).position(|(a, b)| a != b) {
                panic!("topk rows {rows} cols {cols} k {k} {dt:?}: first mismatch at {i}: metal {} want {} (cpu {})", wm[i], want[i], wc[i]);
            }
            assert_eq!(wc, want, "cpu topk rows {rows} cols {cols} k {k} {dt:?}");
        }
    }
    // Distinct random values.
    let (rows, cols, k) = (2usize, 30000usize, 100usize);
    let v = rand(rows * cols, 99);
    let (m, c) = both(&be, Kernel::TopK(Arc::new(TopKSpec { in_dtype: DType::F32 }), TopKDims { rows, cols, k }), &[encode(DType::F32, &v)], &vec![0; rows * k * 4]);
    assert_eq!(words(&m), words(&c));
}

/// Matmul writing into a slice of a larger buffer: [batch, m, n] at `out` inside a
/// zero-filled buffer; elements outside the slice must stay untouched.
fn strided_out_case(be: &crate::MetalBackend, b_dtype: DType, batch: usize, m: usize, n: usize, k: usize) {
    // e.g. a K/V projection into a [B, cap, H, D] cache: row stride > n, batch stride > m * row.
    let out = MatView { offset: 5, batch_stride: (m + 3) * (n + 6), row_stride: n + 6 };
    let total = out.offset + batch * out.batch_stride + 7;
    let a = rand(batch * m * k, 11);
    let bt = rand(n * k, 12);
    let (b_bytes, b_ref) = match b_dtype {
        DType::Int8 => {
            let q = lapid_quant::quantize(lapid_quant::Format::Int8, &bt, n, k);
            let d = lapid_quant::dequantize(lapid_quant::Format::Int8, &q, n, k);
            (q, d)
        }
        _ => (encode(b_dtype, &bt), round_to(b_dtype, &bt)),
    };
    let bias = rand(n, 13);
    let epi = Expr::Binary(lapid_tensor::BinaryOp::Add, Box::new(Expr::Load(0)), Box::new(Expr::Load(1)));
    let spec = mat_spec(DType::F32, b_dtype, true, vec![DType::F32], Some(epi), DType::F32);
    let dims = MatMulDims {
        batch,
        m,
        n,
        k,
        a: MatView::contiguous(m, k),
        b: MatView { offset: 0, batch_stride: 0, row_stride: k },
        extra_views: vec![View { offset: 0, strides: vec![0, 0, 1] }],
        out: out.clone(),
    };
    let init = vec![-7.0f32; total];
    let got = decode(DType::F32, &run_with(be, Kernel::MatMul(spec, dims), &[encode(DType::F32, &a), b_bytes, encode(DType::F32, &bias)], encode(DType::F32, &init)), total);
    let mut want = init.clone();
    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let s: f64 = (0..k).map(|kk| a[(bi * m + i) * k + kk] as f64 * b_ref[j * k + kk] as f64).sum();
                want[out.offset + bi * out.batch_stride + i * out.row_stride + j] = (s + bias[j] as f64) as f32;
            }
        }
    }
    let tol = if b_dtype == DType::F32 { 1e-5 } else { 2e-3 };
    assert_close(&format!("strided out {b_dtype:?} batch {batch} m {m} n {n} k {k}"), &got, &want, tol);
}

#[test]
fn matmul_strided_output() {
    let Some(be) = backend() else { return };
    for b in [DType::F32, DType::F16, DType::Int8] {
        for (batch, m) in [(1usize, 1usize), (2, 3), (1, 4), (2, 70), (1, 130)] {
            strided_out_case(&be, b, batch, m, 72, 64);
        }
    }
}

