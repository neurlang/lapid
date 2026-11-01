//! Tests for the core-ops round: typed integer expressions, GatherAxis, TopK, the
//! out-of-range error flag, strided matmul outputs and command-buffer replay. GPU results
//! are compared with the CPU backend (exact for integers / gathers / top-k).

use std::sync::Arc;

use lapid_compiler::*;
use lapid_cpu::CpuBackend;
use lapid_tensor::{BinaryOp, DType, DeviceData, UnaryOp};

use crate::codegen::{self, IdxMode, Key};
use crate::pipeline::compile_wgsl;
use crate::tests::{assert_close, backend, caps, f32b, from_f32b, load, mat_spec, rnd};
use crate::VulkanBackend;

fn bits(i: usize) -> Box<Expr> {
    Box::new(Expr::LoadBits(i))
}
fn ib(op: IntOp, a: Box<Expr>, b: Box<Expr>) -> Box<Expr> {
    Box::new(Expr::IntBinary(op, a, b))
}
fn ci(v: i32) -> Box<Expr> {
    Box::new(Expr::konst_int(v))
}

/// Run one launch on `be`; the last buffer is the output, initialised with `out_init`.
fn exec(be: &dyn Backend, kernel: &Kernel, inputs: &[Vec<u8>], out_init: &[u8]) -> lapid_tensor::Result<Vec<u8>> {
    let mut bufs: Vec<DeviceData> = inputs.iter().map(|b| be.upload(b)).collect::<lapid_tensor::Result<_>>()?;
    bufs.push(be.upload(out_init)?);
    let n = inputs.len();
    be.execute(&[Launch { kernel: kernel.clone(), inputs: (0..n).collect(), output: n }], &bufs)?;
    be.download(&bufs[n])
}

/// Vulkan and CPU results of the same launch.
fn both(vk: &VulkanBackend, kernel: Kernel, inputs: &[Vec<u8>], out_init: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let cpu = CpuBackend::with_threads(2);
    let g = exec(vk, &kernel, inputs, out_init).unwrap_or_else(|e| panic!("vulkan {}: {e:?}", kernel.kind()));
    let c = exec(&cpu, &kernel, inputs, out_init).unwrap_or_else(|e| panic!("cpu {}: {e:?}", kernel.kind()));
    (g, c)
}

fn i32b(v: &[i32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

/// Integer test values: edge cases plus pseudo-random words.
fn int_vals(n: usize, seed: u32) -> Vec<i32> {
    let edge = [0, 1, -1, 2, -2, 7, -7, 31, 32, 33, i32::MAX, i32::MIN, i32::MIN + 1, 1 << 24, -(1 << 24) - 1, 12345, -98765];
    let mut s = seed.wrapping_mul(2654435761) | 1;
    (0..n)
        .map(|i| {
            if i < edge.len() {
                return edge[i];
            }
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            if i % 3 == 0 { (s % 200) as i32 - 100 } else { s as i32 }
        })
        .collect()
}

fn int_exprs() -> Vec<(&'static str, Expr)> {
    use IntOp::*;
    let mut v: Vec<(&'static str, Expr)> = Vec::new();
    let names = [
        ("add", Add),
        ("sub", Sub),
        ("mul", Mul),
        ("divs", DivS),
        ("divu", DivU),
        ("rems", RemS),
        ("remu", RemU),
        ("mins", MinS),
        ("minu", MinU),
        ("maxs", MaxS),
        ("maxu", MaxU),
        ("and", And),
        ("or", Or),
        ("xor", Xor),
        ("shl", Shl),
        ("shrs", ShrS),
        ("shru", ShrU),
        ("cmplts", CmpLtS),
        ("cmpltu", CmpLtU),
        ("cmpeq", CmpEq),
    ];
    for (n, op) in names {
        v.push((n, Expr::IntBinary(op, bits(0), bits(1))));
    }
    v.push(("neg", Expr::IntUnary(IntUnaryOp::Neg, bits(0))));
    v.push(("abs", Expr::IntUnary(IntUnaryOp::Abs, bits(0))));
    v.push(("not", Expr::IntUnary(IntUnaryOp::Not, bits(0))));
    // where with an int condition; mixed int / float through conversions
    v.push(("where", Expr::Where(ib(CmpLtS, bits(0), bits(1)), bits(1), ib(Add, bits(0), ci(3)))));
    v.push((
        "f2i",
        Expr::Convert(
            Conv::F32ToI32,
            Box::new(Expr::Binary(BinaryOp::Mul, Box::new(Expr::Convert(Conv::I32ToF32, bits(0))), Box::new(Expr::konst(0.75)))),
        ),
    ));
    v.push(("f2u", Expr::Convert(Conv::F32ToU32, Box::new(Expr::Binary(BinaryOp::Mul, load(1), Box::new(Expr::konst(3.5)))))));
    v.push(("u2f_bits", Expr::Convert(Conv::F32Bits, Box::new(Expr::Convert(Conv::U32ToF32, bits(0))))));
    v
}

#[test]
fn integer_expressions_match_cpu() {
    let Some(be) = backend() else { return };
    let n = 1000;
    let a = int_vals(n, 1);
    let b = int_vals(n, 2).into_iter().rev().collect::<Vec<_>>();
    for (name, e) in int_exprs() {
        let spec = Arc::new(EwSpec { rank: 1, in_dtypes: vec![DType::I32, DType::I32], out_dtype: DType::I32, expr: e });
        let dims = EwDims { shape: vec![n], in_views: vec![View::contiguous(&[n]); 2], out_view: View::contiguous(&[n]) };
        let (g, c) = both(&be, Kernel::Elementwise(spec, dims), &[i32b(&a), i32b(&b)], &vec![0u8; n * 4]);
        let (g, c): (&[i32], &[i32]) = (bytemuck::cast_slice(&g), bytemuck::cast_slice(&c));
        for i in 0..n {
            // division / remainder by zero and MIN / -1 are unspecified
            let unspecified = matches!(name, "divs" | "divu" | "rems" | "remu") && (b[i] == 0 || (a[i] == i32::MIN && b[i] == -1));
            if !unspecified {
                assert_eq!(g[i], c[i], "{name}: a = {} b = {} (element {i})", a[i], b[i]);
            }
        }
    }
}

#[test]
fn float_to_int_store_saturates_and_float_roots() {
    let Some(be) = backend() else { return };
    let x = [0.0f32, -0.0, 1.9, -1.9, 3e9, -3e9, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2147483520.0, -2147483648.0, 0.5];
    let n = x.len();
    // f32 root stored to i32 (truncating, saturating, NaN -> 0)
    let spec = Arc::new(EwSpec { rank: 1, in_dtypes: vec![DType::F32], out_dtype: DType::I32, expr: Expr::Load(0) });
    let dims = EwDims { shape: vec![n], in_views: vec![View::contiguous(&[n])], out_view: View::contiguous(&[n]) };
    let (g, c) = both(&be, Kernel::Elementwise(spec, dims.clone()), &[f32b(&x)], &vec![0u8; n * 4]);
    assert_eq!(g, c, "f32 -> i32 store");
    // f32 output of an expression mixing bit tricks: |x| via bit masking
    let e = Expr::Convert(Conv::BitsF32, ib(IntOp::And, bits(0), ci(0x7fff_ffff)));
    let spec = Arc::new(EwSpec { rank: 1, in_dtypes: vec![DType::F32], out_dtype: DType::F32, expr: e });
    let (g, c) = both(&be, Kernel::Elementwise(spec, dims), &[f32b(&x)], &vec![0u8; n * 4]);
    assert_eq!(g, c, "bit-mask abs");
}

#[test]
fn integer_reduce_prologue_and_matmul_epilogue() {
    let Some(be) = backend() else { return };
    // sum over rows of (a & 0xff) as f32
    let (rows, cols) = (7usize, 300usize);
    let a = int_vals(rows * cols, 5);
    let e = Expr::Convert(Conv::I32ToF32, ib(IntOp::And, bits(0), ci(0xff)));
    let spec = Arc::new(ReduceSpec { op: lapid_tensor::ReduceOp::Sum, rank: 2, in_dtypes: vec![DType::I32], out_dtype: DType::F32, expr: e });
    let dims = ReduceDims { shape: vec![rows, cols], in_views: vec![View::contiguous(&[rows, cols])] };
    let (g, c) = both(&be, Kernel::Reduce(spec, dims), &[i32b(&a)], &vec![0u8; rows * 4]);
    let want: Vec<f32> = a.chunks(cols).map(|r| r.iter().map(|v| (v & 0xff) as f32).sum()).collect();
    assert_eq!(from_f32b(&g), want, "int reduce prologue (vulkan)");
    assert_eq!(from_f32b(&c), want, "int reduce prologue (cpu)");
    // matmul epilogue: acc + f32(extra_i32 >> 1)
    let (m, n, k) = (5usize, 9usize, 16usize);
    let x = rnd(m * k, 3);
    let w = rnd(n * k, 4);
    let extra: Vec<i32> = (0..n as i32).map(|j| j * 5 - 7).collect();
    let epi = Expr::Binary(BinaryOp::Add, load(0), Box::new(Expr::Convert(Conv::I32ToF32, ib(IntOp::ShrS, bits(1), ci(1)))));
    let spec = mat_spec(DType::F32, DType::F32, true, vec![DType::I32], Some(epi));
    let dims = MatMulDims {
        batch: 1,
        m,
        n,
        k,
        a: MatView::contiguous(m, k),
        b: MatView::contiguous(n, k),
        extra_views: vec![View { offset: 0, strides: vec![0, 0, 1] }],
        out: MatView::contiguous(m, n),
    };
    let (g, c) = both(&be, Kernel::MatMul(spec, dims), &[f32b(&x), f32b(&w), i32b(&extra)], &vec![0u8; m * n * 4]);
    assert_close(&from_f32b(&g), &from_f32b(&c), 1e-5, "int epilogue");
}

#[test]
fn gather_axis_all_dtypes() {
    let Some(be) = backend() else { return };
    // x [3, 8, 5] (inner dim strided by 2 in a bigger buffer), take along axis 1 with n = 6
    let (outer, len, inner, n) = (3usize, 8usize, 5usize, 6usize);
    let x_elems = outer * len * inner * 2 + 3;
    let idx: Vec<i32> = (0..outer * n * inner).map(|i| ((i * 7 + 3) % len) as i32).collect();
    let dims = GatherAxisDims {
        outer,
        n,
        inner,
        axis_len: len,
        x: View { offset: 3, strides: vec![(len * inner * 2) as isize, (inner * 2) as isize, 2] },
        idx: View::contiguous(&[outer, n, inner]),
    };
    for dt in [DType::F32, DType::I32, DType::F16, DType::BF16] {
        let xb: Vec<u8> = (0..x_elems * dt.size()).map(|i| (i * 37 % 251) as u8).collect();
        let total = outer * n * inner;
        let spec = Arc::new(GatherAxisSpec { dtype: dt });
        let (g, c) = both(&be, Kernel::GatherAxis(spec, dims.clone()), &[xb, i32b(&idx)], &vec![0u8; total * dt.size()]);
        assert_eq!(g[..total * dt.size()], c[..total * dt.size()], "gather_axis {dt:?}");
    }
}

#[test]
fn out_of_range_indices_raise_error_and_clear() {
    let Some(be) = backend() else { return };
    let table = rnd(4 * 8, 1);
    let spec = Arc::new(GatherSpec { table_dtype: DType::F32, out_dtype: DType::F32 });
    let k = Kernel::Gather(spec.clone(), GatherDims { rows: 4, cols: 8, count: 3 });
    for bad in [4i32, -1, i32::MAX] {
        let err = exec(&*be, &k, &[f32b(&table), i32b(&[0, bad, 2])], &[0u8; 96]).unwrap_err();
        assert_eq!(err.code, "index_out_of_range");
    }
    // the flag was cleared: a valid launch succeeds afterwards
    exec(&*be, &k, &[f32b(&table), i32b(&[0, 3, 2])], &[0u8; 96]).unwrap();
    let gspec = Arc::new(GatherAxisSpec { dtype: DType::F32 });
    let gd = GatherAxisDims { outer: 1, n: 2, inner: 1, axis_len: 4, x: View::contiguous(&[1, 4, 1]), idx: View::contiguous(&[1, 2, 1]) };
    let err = exec(&*be, &Kernel::GatherAxis(gspec.clone(), gd.clone()), &[f32b(&table[..4]), i32b(&[1, 9])], &[0u8; 8]).unwrap_err();
    assert_eq!(err.code, "index_out_of_range");
    exec(&*be, &Kernel::GatherAxis(gspec, gd), &[f32b(&table[..4]), i32b(&[1, 3])], &[0u8; 8]).unwrap();
}

fn topk_ref(x: &[f32], cols: usize, k: usize) -> Vec<i32> {
    x.chunks(cols)
        .flat_map(|row| {
            let mut ix: Vec<usize> = (0..cols).collect();
            ix.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
            ix.into_iter().take(k).map(|i| i as i32)
        })
        .collect()
}

#[test]
fn topk_and_argmax() {
    let Some(be) = backend() else { return };
    for &(rows, cols, k) in &[(3usize, 1000usize, 1usize), (2, 51865, 1), (1, 300, 300), (4, 4097, 50), (2, 37, 5), (1, 151_936, 64), (5, 2000, 1024)] {
        let mut x = rnd(rows * cols, (cols + k) as u32);
        // ties (also with -0.0 vs 0.0) and repeated maxima
        for (i, v) in x.iter_mut().enumerate() {
            if i % 7 == 0 {
                *v = 0.25;
            } else if i % 11 == 0 {
                *v = if i % 2 == 0 { 0.0 } else { -0.0 };
            } else if i % 13 == 0 {
                *v = f32::NEG_INFINITY;
            }
        }
        let want = topk_ref(&x, cols, k);
        let spec = Arc::new(TopKSpec { in_dtype: DType::F32 });
        let kern = Kernel::TopK(spec, TopKDims { rows, cols, k });
        let (g, c) = both(&be, kern, &[f32b(&x)], &vec![0u8; rows * k * 4]);
        let (g, c): (&[i32], &[i32]) = (bytemuck::cast_slice(&g), bytemuck::cast_slice(&c));
        assert_eq!(c, &want[..], "cpu topk {rows}x{cols} k={k}");
        assert_eq!(g, &want[..], "vulkan topk {rows}x{cols} k={k}");
    }
    // i32 input compared as f32, all equal -> 0..k
    let x = vec![5i32; 600];
    let spec = Arc::new(TopKSpec { in_dtype: DType::I32 });
    let (g, _) = both(&be, Kernel::TopK(spec, TopKDims { rows: 1, cols: 600, k: 9 }), &[i32b(&x)], &[0u8; 36]);
    assert_eq!(bytemuck::cast_slice::<u8, i32>(&g), &(0..9).collect::<Vec<i32>>()[..]);
}

#[test]
fn strided_matmul_out_all_paths() {
    let Some(be) = backend() else { return };
    // write [batch, m, n] into a [batch, cap, n + pad] buffer at row offset `start`
    for (b_dt, m) in [(DType::F32, 70usize), (DType::F16, 70), (DType::F16, 1), (DType::Int8, 3), (DType::Int8, 40), (DType::Int4, 65), (DType::F32, 2)] {
        let (batch, n, k, cap, pad, start) = (2usize, 72usize, 64usize, m + 9, 8usize, 5usize);
        let rs = n + pad;
        let a = rnd(batch * m * k, 21);
        let w = rnd(n * k, 22);
        let wb = match b_dt {
            DType::F32 => f32b(&w),
            DType::F16 => crate::tests::f16b(&w),
            DType::Int8 => lapid_quant::quantize(lapid_quant::Format::Int8, &w, n, k),
            _ => lapid_quant::quantize(lapid_quant::Format::Int4, &w, n, k),
        };
        let spec = mat_spec(DType::F32, b_dt, true, vec![], None);
        let dims = MatMulDims {
            batch,
            m,
            n,
            k,
            a: MatView::contiguous(m, k),
            b: MatView { offset: 0, batch_stride: 0, row_stride: k },
            extra_views: vec![],
            out: MatView { offset: start * rs + 1, batch_stride: cap * rs, row_stride: rs },
        };
        let total = batch * cap * rs;
        let init: Vec<f32> = (0..total).map(|i| i as f32 * -0.5).collect();
        let (g, c) = both(&be, Kernel::MatMul(spec, dims), &[f32b(&a), wb], &f32b(&init));
        let (g, c) = (from_f32b(&g), from_f32b(&c));
        // untouched elements are bit-identical, written ones close
        let mut wg = Vec::new();
        let mut wc = Vec::new();
        for i in 0..total {
            if g[i] == init[i] && c[i] == init[i] {
                continue;
            }
            wg.push(g[i]);
            wc.push(c[i]);
        }
        assert_eq!(wg.len(), batch * m * n, "{b_dt:?} m={m}: written element count");
        let tol = if b_dt.is_quant() { 3e-2 } else { 2e-3 };
        assert_close(&wg, &wc, tol, &format!("strided out {b_dt:?} m={m}"));
    }
}

#[test]
fn replay_reuses_recorded_sequences() {
    let Some(be) = backend() else { return };
    // a decode-like chain whose params change each step and whose output buffer is new
    let n = 4096usize;
    let spec = Arc::new(EwSpec {
        rank: 1,
        in_dtypes: vec![DType::F32, DType::F32],
        out_dtype: DType::F32,
        expr: Expr::Binary(BinaryOp::Add, load(0), Box::new(Expr::Unary(UnaryOp::Abs, load(1)))),
    });
    let x = rnd(n, 3);
    let mut bufs: Vec<DeviceData> = vec![be.upload(&f32b(&x)).unwrap(), be.alloc(n * 4).unwrap(), be.alloc(n * 4).unwrap()];
    for step in 0..6usize {
        bufs[2] = be.alloc(n * 4).unwrap();
        let len = n - step * 7;
        let ew = |i: usize, o: usize| Launch {
            kernel: Kernel::Elementwise(
                spec.clone(),
                EwDims { shape: vec![len], in_views: vec![View::contiguous(&[len]); 2], out_view: View::contiguous(&[len]) },
            ),
            inputs: vec![i, 0],
            output: o,
        };
        // 40 dependent launches: 0 -> 1 -> 1 ... -> 2
        let mut ls = vec![ew(0, 1)];
        for _ in 0..38 {
            ls.push(ew(1, 1));
        }
        ls.push(ew(1, 2));
        be.execute(&ls, &bufs).unwrap();
        let got = from_f32b(&be.download(&bufs[2]).unwrap());
        for i in (0..len).step_by(97) {
            let want = x[i] + 40.0 * x[i].abs();
            assert!((got[i] - want).abs() <= 1e-5 * want.abs().max(1.0), "step {step} elem {i}: {} vs {want}", got[i]);
        }
    }
    let s = be.replay_stats().unwrap_or_default();
    if be.shared.caps.push_descriptor && std::env::var_os("LAPID_VK_NO_REPLAY").is_none() {
        assert!(!s.starts_with("replay: 0 full hits, 0 partial"), "replay never hit: {s}");
    }
}

#[test]
fn new_kernels_compile_on_all_caps() {
    let int_spec = Arc::new(EwSpec { rank: 2, in_dtypes: vec![DType::I32, DType::F32], out_dtype: DType::I32, expr: int_exprs().remove(3).1 });
    let mut keys = vec![Key::Ew(int_spec, vec![IdxMode::Strided, IdxMode::Flat], IdxMode::Flat)];
    for (_, e) in int_exprs() {
        let s = Arc::new(EwSpec { rank: 1, in_dtypes: vec![DType::I32, DType::F32], out_dtype: DType::I32, expr: e });
        keys.push(Key::Ew(s, vec![IdxMode::Flat, IdxMode::Flat], IdxMode::Flat));
    }
    for dt in [DType::F32, DType::I32, DType::F16, DType::BF16] {
        keys.push(Key::GatherAxis(Arc::new(GatherAxisSpec { dtype: dt })));
        keys.push(Key::TopK(Arc::new(TopKSpec { in_dtype: dt }), true));
        keys.push(Key::TopK(Arc::new(TopKSpec { in_dtype: dt }), false));
    }
    for c in [caps(true, true, true), caps(false, false, false), caps(true, false, true)] {
        for key in &keys {
            let src = codegen::source(key, &c);
            if let Err(e) = compile_wgsl(key.label(), &src, &c) {
                panic!("{key:?}: {}", e.message);
            }
        }
    }
}
