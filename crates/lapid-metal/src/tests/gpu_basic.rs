//! GPU tests: every key compiles; elementwise, reduce, softmax, layer norm and gather
//! against naive references.

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::{BinaryOp, DType, ReduceOp};

use super::keys::{all_keys, gelu_plus, load};
use super::util::*;

#[test]
fn every_key_compiles() {
    let Some(be) = backend() else { return };
    let mut failures = Vec::new();
    for key in all_keys() {
        if let Err(e) = be.compile_key(&key) {
            failures.push(format!("{key:?}:\n{}", e.message));
        }
    }
    assert!(failures.is_empty(), "{} kernels failed to compile:\n{}", failures.len(), failures.join("\n\n"));
}

#[test]
fn devices_list() {
    let list = crate::devices().unwrap();
    assert!(!list.is_empty());
    for (i, d) in list.iter().enumerate() {
        assert_eq!(d.index, i);
        assert!(!d.name.is_empty());
    }
}

#[test]
fn upload_download_roundtrip() {
    let Some(be) = backend() else { return };
    let bytes: Vec<u8> = (0..1003u32).map(|i| (i * 7) as u8).collect();
    let b = be.upload(&bytes).unwrap();
    assert_eq!(b.len(), 1003);
    assert_eq!(b.backend(), "metal");
    assert_eq!(be.download(&b).unwrap(), bytes);
    let e = be.upload(&[]).unwrap();
    assert_eq!(be.download(&e).unwrap(), Vec::<u8>::new());
}

#[test]
fn elementwise_strided_broadcast_dtypes() {
    let Some(be) = backend() else { return };
    let shape = [3usize, 5, 7];
    let n: usize = shape.iter().product();
    for &(d0, d1, out) in &[(DType::F16, DType::F32, DType::F32), (DType::F32, DType::BF16, DType::BF16), (DType::F32, DType::F32, DType::F16)] {
        // in0: transposed storage [7, 5, 3]; in1: broadcast along dim 1 (stride 0) of [3, 1, 7].
        let a = round_to(d0, &rand(n, 1));
        let b = round_to(d1, &rand(3 * 7, 2));
        let v0 = View { offset: 0, strides: vec![1, 3, 15] };
        let v1 = View { offset: 0, strides: vec![7, 0, 1] };
        // output written transposed as well (strided store) at offset 2
        let vo = View { offset: 2, strides: vec![1, 3, 15] };
        let spec = Arc::new(EwSpec { rank: 3, in_dtypes: vec![d0, d1], out_dtype: out, expr: gelu_plus(0, 1) });
        let dims = EwDims { shape: shape.to_vec(), in_views: vec![v0.clone(), v1.clone()], out_view: vo.clone() };
        let got = run(&be, Kernel::Elementwise(spec, dims), &[encode(d0, &a), encode(d1, &b)], (n + 2) * out.size());
        let got = decode(out, &got, n + 2);
        let mut want = vec![f32::NAN; n + 2];
        let mut gotv = vec![0.0; n + 2];
        for idx in indices(&shape) {
            let x0 = a[at(0, &v0.strides, &idx)] as f64;
            let x1 = b[at(0, &v1.strides, &idx)] as f64;
            let o = at(2, &vo.strides, &idx);
            want[o] = round_to(out, &[(gelu_ref(x0) + 2.0 * x1) as f32])[0];
            gotv[o] = got[o];
        }
        let tol = if out == DType::F32 { 1e-6 } else { 8e-3 };
        assert_close(&format!("ew {d0:?} {d1:?} -> {out:?}"), &gotv[2..], &want[2..], tol);
    }
}

#[test]
fn elementwise_fill_and_i32() {
    let Some(be) = backend() else { return };
    // fill (no inputs), rank 0
    let spec = Arc::new(EwSpec { rank: 0, in_dtypes: vec![], out_dtype: DType::F32, expr: Expr::konst(f32::NEG_INFINITY) });
    let dims = EwDims { shape: vec![], in_views: vec![], out_view: View { offset: 0, strides: vec![] } };
    let got = decode(DType::F32, &run(&be, Kernel::Elementwise(spec, dims), &[], 4), 1);
    assert_eq!(got, [f32::NEG_INFINITY]);
    // i32 in / out, contiguous: x * 3 - 1, with comparisons
    let n = 1000;
    let a: Vec<f32> = (0..n).map(|i| i as f32 - 500.0).collect();
    let expr = Expr::Binary(
        BinaryOp::Add,
        Box::new(Expr::Binary(BinaryOp::Mul, load(0), Box::new(Expr::konst(3.0)))),
        Box::new(Expr::Binary(BinaryOp::CmpLt, load(0), Box::new(Expr::konst(0.0)))),
    );
    let spec = Arc::new(EwSpec { rank: 1, in_dtypes: vec![DType::I32], out_dtype: DType::I32, expr });
    let dims = EwDims { shape: vec![n], in_views: vec![View::contiguous(&[n])], out_view: View::contiguous(&[n]) };
    let got = decode(DType::I32, &run(&be, Kernel::Elementwise(spec, dims), &[encode(DType::I32, &a)], n * 4), n);
    let want: Vec<f32> = a.iter().map(|&x| x * 3.0 + if x < 0.0 { 1.0 } else { 0.0 }).collect();
    assert_eq!(got, want);
}

#[test]
fn reduce_sum_max_both_variants() {
    let Some(be) = backend() else { return };
    // (shape, row variant expected)
    for shape in [vec![4usize, 3, 300], vec![5000, 8], vec![7], vec![2, 1, 1]] {
        let rank = shape.len();
        let n: usize = shape.iter().product();
        let len = shape[rank - 1];
        let outer: usize = n / len;
        // in0: storage with the reduced axis first (strided), f16; in1: contiguous f32.
        let mut st = vec![0isize; rank];
        let mut s = len as isize;
        for d in (0..rank - 1).rev() {
            st[d] = s;
            s *= shape[d] as isize;
        }
        st[rank - 1] = 1;
        // swap so the reduced axis is the slowest in storage
        let mut strided = vec![0isize; rank];
        let mut acc = 1isize;
        for d in (0..rank - 1).rev() {
            strided[d] = acc;
            acc *= shape[d] as isize;
        }
        strided[rank - 1] = acc;
        let a = round_to(DType::F16, &rand(n, 3));
        let b = rand(n, 4);
        for op in [ReduceOp::Sum, ReduceOp::Max] {
            let spec = Arc::new(ReduceSpec {
                op,
                rank,
                in_dtypes: vec![DType::F16, DType::F32],
                out_dtype: DType::F32,
                expr: Expr::Binary(BinaryOp::Mul, load(0), load(1)),
            });
            let dims = ReduceDims {
                shape: shape.clone(),
                in_views: vec![View { offset: 0, strides: strided.clone() }, View { offset: 0, strides: st.clone() }],
            };
            let got = decode(DType::F32, &run(&be, Kernel::Reduce(spec, dims), &[encode(DType::F16, &a), encode(DType::F32, &b)], outer * 4), outer);
            let mut want = vec![0f32; outer];
            for (o, w) in want.iter_mut().enumerate() {
                let mut r = if op == ReduceOp::Sum { 0f64 } else { f64::NEG_INFINITY };
                for j in 0..len {
                    let mut idx = vec![0; rank];
                    let mut rem = o;
                    for d in (0..rank - 1).rev() {
                        idx[d] = rem % shape[d];
                        rem /= shape[d];
                    }
                    idx[rank - 1] = j;
                    let v = a[at(0, &strided, &idx)] as f64 * b[at(0, &st, &idx)] as f64;
                    r = if op == ReduceOp::Sum { r + v } else { r.max(v) };
                }
                *w = r as f32;
            }
            assert_close(&format!("reduce {op:?} {shape:?}"), &got, &want, 1e-5);
        }
    }
}

#[test]
fn softmax_rows_with_masked_entries() {
    let Some(be) = backend() else { return };
    for &(rows, cols) in &[(5usize, 1500usize), (3, 7), (1, 51865)] {
        let mut x = round_to(DType::F16, &rand(rows * cols, 5).iter().map(|v| v * 8.0).collect::<Vec<_>>());
        x[..cols].fill(f32::NEG_INFINITY); // fully masked first row
        if rows > 1 {
            x[cols + 1] = f32::NEG_INFINITY;
        }
        let spec = Arc::new(SoftmaxSpec { in_dtype: DType::F16, out_dtype: DType::F32 });
        let got = decode(DType::F32, &run(&be, Kernel::Softmax(spec, RowDims { rows, cols }), &[encode(DType::F16, &x)], rows * cols * 4), rows * cols);
        let mut want = vec![0f32; rows * cols];
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let m = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64));
            if m == f64::NEG_INFINITY {
                continue;
            }
            let s: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
            for c in 0..cols {
                want[r * cols + c] = ((row[c] as f64 - m).exp() / s) as f32;
            }
        }
        assert_close(&format!("softmax {rows}x{cols}"), &got, &want, 1e-5);
    }
}

#[test]
fn layer_norm_rows() {
    let Some(be) = backend() else { return };
    for &(rows, cols, out) in &[(7usize, 768usize, DType::F16), (3, 50, DType::F32), (2, 3000, DType::F32)] {
        let x: Vec<f32> = rand(rows * cols, 6).iter().map(|v| v * 3.0 + 1.5).collect();
        let w = rand(cols, 7);
        let b = rand(cols, 8);
        let eps = 1e-5f32;
        let spec = Arc::new(LayerNormSpec { in_dtype: DType::F32, param_dtype: DType::F32, out_dtype: out });
        let dims = LayerNormDims { rows, cols, eps: eps.to_bits() };
        let ins = [encode(DType::F32, &x), encode(DType::F32, &w), encode(DType::F32, &b)];
        let got = decode(out, &run(&be, Kernel::LayerNorm(spec, dims), &ins, rows * cols * out.size()), rows * cols);
        let mut want = vec![0f32; rows * cols];
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / cols as f64;
            for c in 0..cols {
                want[r * cols + c] = ((row[c] as f64 - mean) / (var + eps as f64).sqrt() * w[c] as f64 + b[c] as f64) as f32;
            }
        }
        let tol = if out == DType::F32 { 1e-5 } else { 2e-3 };
        assert_close(&format!("layer_norm {rows}x{cols}"), &got, &round_to(out, &want), tol);
    }
}

#[test]
fn gather_all_tables() {
    let Some(be) = backend() else { return };
    let (rows, cols) = (50usize, 96usize);
    let table = rand(rows * cols, 9);
    let idx: Vec<f32> = [3.0, 0.0, 49.0, 17.0, 3.0].to_vec();
    let count = idx.len();
    for t in [DType::F32, DType::F16, DType::BF16, DType::Int8, DType::Int4] {
        let (bytes, deq) = match t {
            DType::Int8 | DType::Int4 => {
                let f = if t == DType::Int8 { lapid_quant::Format::Int8 } else { lapid_quant::Format::Int4 };
                let q = lapid_quant::quantize(f, &table, rows, cols);
                let d = lapid_quant::dequantize(f, &q, rows, cols);
                (q, d)
            }
            _ => (encode(t, &table), round_to(t, &table)),
        };
        let spec = Arc::new(GatherSpec { table_dtype: t, out_dtype: DType::F32 });
        let dims = GatherDims { rows, cols, count };
        let got = decode(DType::F32, &run(&be, Kernel::Gather(spec, dims), &[bytes, encode(DType::I32, &idx)], count * cols * 4), count * cols);
        let want: Vec<f32> = idx.iter().flat_map(|&i| deq[i as usize * cols..(i as usize + 1) * cols].to_vec()).collect();
        assert_close(&format!("gather {t:?}"), &got, &want, 0.0);
    }
}

#[test]
fn long_dependent_chain_split_command_buffers() {
    // > SPLIT_MIN dependent launches: the execute is split into two command buffers ordered
    // by an event; ping-pong x -> y -> x ... each adding 1.
    let Some(be) = backend() else { return };
    let n = 4096;
    let spec = Arc::new(EwSpec { rank: 1, in_dtypes: vec![DType::F32], out_dtype: DType::F32, expr: Expr::Binary(BinaryOp::Add, load(0), Box::new(Expr::konst(1.0))) });
    let dims = EwDims { shape: vec![n], in_views: vec![View::contiguous(&[n])], out_view: View::contiguous(&[n]) };
    let steps = 301;
    let launches: Vec<Launch> =
        (0..steps).map(|i| Launch { kernel: Kernel::Elementwise(spec.clone(), dims.clone()), inputs: vec![i % 2], output: 1 - i % 2 }).collect();
    for _ in 0..3 {
        let bufs = vec![be.upload(&encode(DType::F32, &vec![0.0; n])).unwrap(), be.alloc(n * 4).unwrap()];
        be.execute(&launches, &bufs).unwrap();
        let got = decode(DType::F32, &be.download(&bufs[1]).unwrap(), n);
        assert!(got.iter().all(|&v| v == steps as f32), "got {}", got[0]);
    }
}
