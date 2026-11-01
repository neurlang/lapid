//! GPU tests: typed integer expressions (elementwise raw stores, f32 roots reading ints,
//! reduce prologues, matmul epilogues), compared exactly with the CPU backend.

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::{BinaryOp, DType, ReduceOp};

use super::keys::{int_all_ops, mat_spec};
use super::util::*;

/// Interesting words: 0, ±1, extremes, small values, random bits.
fn int_words(n: usize, seed: u64) -> Vec<u32> {
    let special = [0u32, 1, u32::MAX, 0x8000_0000, 0x7fff_ffff, 2, 31, 32, 33, 0xffff_fffe, 7, 100, 0x8000_0001];
    let r = rand(n, seed);
    (0..n)
        .map(|i| if i < special.len() * 3 { special[i % special.len()] } else { (r[i] * 2.0e9) as i32 as u32 ^ (i as u32).wrapping_mul(2654435761) })
        .collect()
}

fn f32_values(n: usize, seed: u64) -> Vec<f32> {
    let special = [0.0f32, -0.0, 0.5, -0.5, 1.5, -1.5, 3e9, -3e9, 5e9, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2147483520.0, -2147483648.0, 4294967040.0];
    let r = rand(n, seed);
    (0..n).map(|i| if i < special.len() * 2 { special[(i * 7) % special.len()] } else { r[i] * 1e5 }).collect()
}

#[test]
fn int_elementwise_matches_cpu() {
    let Some(be) = backend() else { return };
    let (r0, r1) = (37usize, 29usize);
    let n = r0 * r1;
    let a = from_words(&int_words(n, 1));
    let f = encode(DType::F32, &f32_values(n, 2));
    // contiguous, and a transposed (strided) input with a broadcast second operand
    let views = [
        (View::contiguous(&[r0, r1]), View::contiguous(&[r0, r1])),
        (View { offset: 0, strides: vec![1, r0 as isize] }, View { offset: 3, strides: vec![0, 1] }),
    ];
    for (v0, v1) in &views {
        let dims = EwDims { shape: vec![r0, r1], in_views: vec![v0.clone(), v1.clone()], out_view: View::contiguous(&[r0, r1]) };
        let spec = Arc::new(EwSpec { rank: 2, in_dtypes: vec![DType::I32, DType::F32], out_dtype: DType::I32, expr: int_all_ops() });
        let (m, c) = both(&be, Kernel::Elementwise(spec, dims.clone()), &[a.clone(), f.clone()], &vec![0xAB; n * 4]);
        assert_eq!(words(&m), words(&c), "int root, views {v0:?} {v1:?}");
        // f32 root converted from the integer result (i32 → f32 rounding), stored as f16 and f32
        for out in [DType::F32, DType::F16] {
            let expr = Expr::Convert(Conv::I32ToF32, Box::new(int_all_ops()));
            let spec = Arc::new(EwSpec { rank: 2, in_dtypes: vec![DType::I32, DType::F32], out_dtype: out, expr });
            let (m, c) = both(&be, Kernel::Elementwise(spec, dims.clone()), &[a.clone(), f.clone()], &vec![0xAB; n * out.size()]);
            assert_eq!(m, c, "f32 root -> {out:?}");
        }
    }
}

#[test]
fn f32_to_i32_store_saturates() {
    let Some(be) = backend() else { return };
    let x = [3e9f32, -3e9, f32::NAN, 2147483520.0, -2147483648.0, 1.9, -1.9, f32::INFINITY, f32::NEG_INFINITY, 2147483648.0];
    let n = x.len();
    let spec = Arc::new(EwSpec { rank: 1, in_dtypes: vec![DType::F32], out_dtype: DType::I32, expr: Expr::Load(0) });
    let dims = EwDims { shape: vec![n], in_views: vec![View::contiguous(&[n])], out_view: View::contiguous(&[n]) };
    let (m, c) = both(&be, Kernel::Elementwise(spec, dims), &[encode(DType::F32, &x)], &vec![0; n * 4]);
    let want: Vec<u32> = x.iter().map(|&v| v as i32 as u32).collect();
    assert_eq!(words(&m), want);
    assert_eq!(words(&c), want);
}

#[test]
fn int_reduce_prologue_matches_cpu() {
    let Some(be) = backend() else { return };
    for shape in [vec![300usize, 9], vec![3, 700]] {
        let n: usize = shape.iter().product();
        let a = from_words(&int_words(n, 3));
        let f = encode(DType::F32, &f32_values(n, 4));
        // Max of an integer expression converted to f32 (exact regardless of order).
        let expr = Expr::Convert(Conv::I32ToF32, Box::new(int_all_ops()));
        let spec = Arc::new(ReduceSpec { op: ReduceOp::Max, rank: 2, in_dtypes: vec![DType::I32, DType::F32], out_dtype: DType::F32, expr });
        let dims = ReduceDims { shape: shape.clone(), in_views: vec![View::contiguous(&shape), View::contiguous(&shape)] };
        let (m, c) = both(&be, Kernel::Reduce(spec, dims), &[a.clone(), f.clone()], &vec![0; shape[0] * 4]);
        // Reference: the elementwise kernel (checked against the CPU above), then a row max.
        let ew = Arc::new(EwSpec {
            rank: 2,
            in_dtypes: vec![DType::I32, DType::F32],
            out_dtype: DType::F32,
            expr: Expr::Convert(Conv::I32ToF32, Box::new(int_all_ops())),
        });
        let ed = EwDims { shape: shape.clone(), in_views: vec![View::contiguous(&shape), View::contiguous(&shape)], out_view: View::contiguous(&shape) };
        let (em, ec) = both(&be, Kernel::Elementwise(ew, ed), &[a.clone(), f.clone()], &vec![0; n * 4]);
        assert_eq!(em, ec);
        let vals = decode(DType::F32, &em, n);
        let want: Vec<f32> = vals.chunks(shape[1]).map(|r| r.iter().cloned().fold(f32::NEG_INFINITY, f32::max)).collect();
        assert_eq!(decode(DType::F32, &m, shape[0]), want, "metal reduce {shape:?}");
        assert_eq!(decode(DType::F32, &c, shape[0]), want, "cpu reduce {shape:?}");
    }
}

#[test]
fn int_matmul_epilogue_matches_cpu() {
    let Some(be) = backend() else { return };
    // epilogue: e0 < 0 (signed word compare) ? -acc (sign bit flipped) : acc + float(e0 & 7)
    let epi = Expr::Where(
        Box::new(Expr::IntBinary(IntOp::CmpLtS, Box::new(Expr::LoadBits(1)), Box::new(Expr::ConstBits(0)))),
        Box::new(Expr::Convert(
            Conv::BitsF32,
            Box::new(Expr::IntBinary(IntOp::Xor, Box::new(Expr::LoadBits(0)), Box::new(Expr::ConstBits(0x8000_0000)))),
        )),
        Box::new(Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::Load(0)),
            Box::new(Expr::Convert(Conv::I32ToF32, Box::new(Expr::IntBinary(IntOp::And, Box::new(Expr::LoadBits(1)), Box::new(Expr::ConstBits(7)))))),
        )),
    );
    for m in [2usize, 70] {
        let (n, k) = (40usize, 64usize);
        let a = rand(m * k, 5);
        let b = round_to(DType::F16, &rand(n * k, 6));
        let e = from_words(&int_words(m * n, 7));
        let spec = mat_spec(DType::F32, DType::F16, true, vec![DType::I32], Some(epi.clone()), DType::F32);
        let dims = MatMulDims {
            batch: 1,
            m,
            n,
            k,
            a: MatView::contiguous(m, k),
            b: MatView::contiguous(n, k),
            extra_views: vec![View::contiguous(&[1, m, n])],
            out: MatView::contiguous(m, n),
        };
        let ew = int_words(m * n, 7);
        let (gm, gc) = both(&be, Kernel::MatMul(spec, dims), &[encode(DType::F32, &a), encode(DType::F16, &b), e], &vec![0; m * n * 4]);
        let mut want = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let acc: f64 = (0..k).map(|kk| a[i * k + kk] as f64 * b[j * k + kk] as f64).sum();
                let w = ew[i * n + j];
                want[i * n + j] = if (w as i32) < 0 { -(acc as f32) } else { acc as f32 + (w & 7) as f32 };
            }
        }
        assert_close(&format!("metal int epilogue m={m}"), &decode(DType::F32, &gm, m * n), &want, 1e-5);
        assert_close(&format!("cpu int epilogue m={m}"), &decode(DType::F32, &gc, m * n), &want, 1e-5);
    }
}
