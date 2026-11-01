//! Native kernel tests: every launch runs on adapter 0 and on the CPU backend; results are
//! compared (exact for integer / copy kernels). Skipped when no wgpu adapter exists.

use std::sync::Arc;

use lapid_compiler::*;
use lapid_cpu::CpuBackend;
use lapid_tensor::{BinaryOp, DType, DeviceData, ReduceOp, UnaryOp};
use half::{bf16, f16};

use crate::{WebGpuBackend, adapters};

struct TestBackend {
    be: WebGpuBackend,
    _gpu: std::sync::MutexGuard<'static, ()>,
}

impl std::ops::Deref for TestBackend {
    type Target = WebGpuBackend;
    fn deref(&self) -> &WebGpuBackend {
        &self.be
    }
}

/// GPU tests run one at a time (shared driver, see lapid-vulkan tests).
fn backend() -> Option<TestBackend> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let list = adapters().ok()?;
    let Some(a) = list.iter().find(|a| a.kind != "cpu") else {
        eprintln!("no wgpu GPU adapter, skipping");
        return None;
    };
    let gpu = GPU.lock().unwrap_or_else(|p| p.into_inner());
    let be = WebGpuBackend::new(a.index).expect("adapter must open");
    eprintln!("webgpu: {} [{}]", be.device_name(), be.features());
    Some(TestBackend { be, _gpu: gpu })
}

fn ld(i: usize) -> Box<Expr> {
    Box::new(Expr::Load(i))
}
fn f32b(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}
fn i32b(v: &[i32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}
fn f16b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| f16::from_f32(*x).to_le_bytes()).collect()
}
fn bf16b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| bf16::from_f32(*x).to_le_bytes()).collect()
}
fn as_f32(b: &[u8], d: DType) -> Vec<f32> {
    match d {
        DType::F16 => b.as_chunks::<2>().0.iter().map(|c| f16::from_le_bytes(*c).to_f32()).collect(),
        DType::BF16 => b.as_chunks::<2>().0.iter().map(|c| bf16::from_le_bytes(*c).to_f32()).collect(),
        DType::I32 => b.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c) as f32).collect(),
        _ => b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
    }
}

/// Deterministic pseudo-random values in [-1, 1).
fn rnd(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn exec(be: &dyn Backend, kernel: &Kernel, inputs: &[Vec<u8>], out_init: &[u8]) -> lapid_tensor::Result<Vec<u8>> {
    let mut bufs: Vec<DeviceData> = inputs.iter().map(|b| be.upload(b)).collect::<lapid_tensor::Result<_>>()?;
    bufs.push(be.upload(out_init)?);
    let n = inputs.len();
    be.execute(&[Launch { kernel: kernel.clone(), inputs: (0..n).collect(), output: n }], &bufs)?;
    be.download(&bufs[n])
}

/// WebGPU and CPU results of the same launch.
fn both(be: &WebGpuBackend, kernel: Kernel, inputs: &[Vec<u8>], out_bytes: usize) -> (Vec<u8>, Vec<u8>) {
    let cpu = CpuBackend::with_threads(2);
    let init = vec![0u8; out_bytes];
    let g = exec(be, &kernel, inputs, &init).unwrap_or_else(|e| panic!("webgpu {}: {e:?}", kernel.kind()));
    let c = exec(&cpu, &kernel, inputs, &init).unwrap_or_else(|e| panic!("cpu {}: {e:?}", kernel.kind()));
    (g, c)
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let num: f64 = got.iter().zip(want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().max(1e-30);
    let rel = (num / den).sqrt() as f32;
    assert!(rel <= tol, "{what}: rel err {rel} > {tol}");
}

fn check(be: &WebGpuBackend, kernel: Kernel, inputs: &[Vec<u8>], out: DType, n: usize, tol: f32, what: &str) {
    let (g, c) = both(be, kernel, inputs, n * out.size());
    assert_close(&as_f32(&g, out), &as_f32(&c, out), tol, what);
}

fn gelu_plus() -> Expr {
    let g = Expr::Binary(
        BinaryOp::Mul,
        Box::new(Expr::Binary(BinaryOp::Mul, ld(0), Box::new(Expr::konst(0.5)))),
        Box::new(Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::konst(1.0)),
            Box::new(Expr::Unary(UnaryOp::Erf, Box::new(Expr::Binary(BinaryOp::Mul, ld(0), Box::new(Expr::konst(std::f32::consts::FRAC_1_SQRT_2)))))),
        )),
    );
    Expr::Binary(BinaryOp::Add, Box::new(g), Box::new(Expr::Binary(BinaryOp::Mul, ld(1), Box::new(Expr::konst(2.0)))))
}

#[test]
fn adapters_never_panic() {
    match adapters() {
        Ok(list) => {
            for a in list {
                eprintln!("{a:?}");
            }
        }
        Err(e) => assert_eq!(e.code, "webgpu_unavailable"),
    }
}

#[test]
fn elementwise_strided_and_half() {
    let Some(be) = backend() else { return };
    let (d0, d1, d2) = (3usize, 5usize, 7usize);
    let a = rnd(d0 * d1 * d2, 1);
    let b = rnd(d1, 2);
    let spec = Arc::new(EwSpec { rank: 3, in_dtypes: vec![DType::F32, DType::F32], out_dtype: DType::F32, expr: gelu_plus() });
    let dims = EwDims {
        shape: vec![d0, d1, d2],
        in_views: vec![View { offset: 0, strides: vec![1, d0 as isize, (d0 * d1) as isize] }, View { offset: 0, strides: vec![0, 1, 0] }],
        out_view: View::contiguous(&[d0, d1, d2]),
    };
    check(&be, Kernel::Elementwise(spec, dims), &[f32b(&a), f32b(&b)], DType::F32, d0 * d1 * d2, 1e-5, "ew lapid");
    for out in [DType::F16, DType::BF16] {
        let n = 35usize;
        let spec = Arc::new(EwSpec { rank: 1, in_dtypes: vec![DType::F16, DType::BF16], out_dtype: out, expr: Expr::Binary(BinaryOp::Sub, ld(0), ld(1)) });
        let dims = EwDims { shape: vec![n], in_views: vec![View::contiguous(&[n]); 2], out_view: View::contiguous(&[n]) };
        check(&be, Kernel::Elementwise(spec, dims), &[f16b(&rnd(n, 3)), bf16b(&rnd(n, 4))], out, n, 1e-3, "ew half");
    }
}

#[test]
fn reduce_softmax_layer_norm() {
    let Some(be) = backend() else { return };
    for &(outer, inner, op) in &[(6usize, 300usize, ReduceOp::Sum), (1000, 5, ReduceOp::Max), (3, 1000, ReduceOp::Max)] {
        let x = rnd(outer * inner, 5);
        let spec = Arc::new(ReduceSpec { op, rank: 2, in_dtypes: vec![DType::F32], out_dtype: DType::F32, expr: Expr::Binary(BinaryOp::Mul, ld(0), ld(0)) });
        let dims = ReduceDims { shape: vec![outer, inner], in_views: vec![View::contiguous(&[outer, inner])] };
        check(&be, Kernel::Reduce(spec, dims), &[f32b(&x)], DType::F32, outer, 1e-5, "reduce");
    }
    let (rows, cols) = (5usize, 777usize);
    let x: Vec<f32> = rnd(rows * cols, 13).iter().map(|v| v * 8.0).collect();
    let spec = Arc::new(SoftmaxSpec { in_dtype: DType::F32, out_dtype: DType::F32 });
    check(&be, Kernel::Softmax(spec, RowDims { rows, cols }), &[f32b(&x)], DType::F32, rows * cols, 1e-5, "softmax");
    for (rows, cols) in [(5usize, 777usize), (2, 1500)] {
        let x = rnd(rows * cols, 13);
        let spec = Arc::new(LayerNormSpec { in_dtype: DType::F32, param_dtype: DType::F32, out_dtype: DType::F32 });
        let dims = LayerNormDims { rows, cols, eps: 1e-5f32.to_bits() };
        check(&be, Kernel::LayerNorm(spec, dims), &[f32b(&x), f32b(&rnd(cols, 14)), f32b(&rnd(cols, 15))], DType::F32, rows * cols, 1e-5, "layer_norm");
    }
}

fn mm_dims(m: usize, n: usize, k: usize, bt: bool, extras: Vec<View>) -> MatMulDims {
    MatMulDims {
        batch: 1,
        m,
        n,
        k,
        a: MatView { offset: 0, batch_stride: m * k, row_stride: k },
        b: MatView { offset: 0, batch_stride: 0, row_stride: if bt { k } else { n } },
        extra_views: extras,
        out: MatView::contiguous(m, n),
    }
}

#[test]
fn matmul_float_and_quantized() {
    let Some(be) = backend() else { return };
    let bias_epi = Some(Expr::Binary(BinaryOp::Add, ld(0), ld(1)));
    for &(m, n, k, bt) in &[(70usize, 67usize, 45usize, true), (33, 20, 64, false), (1, 40, 96, true), (3, 130, 64, true)] {
        for bd in [DType::F32, DType::F16] {
            let a = rnd(m * k, 7);
            let b = rnd(n * k, 8);
            let bias = rnd(n, 9);
            let spec = Arc::new(MatMulSpec { a_dtype: DType::F32, b_dtype: bd, b_transposed: bt, extra_dtypes: vec![DType::F32], out_dtype: DType::F32, epilogue: bias_epi.clone() });
            let dims = mm_dims(m, n, k, bt, vec![View { offset: 0, strides: vec![0, 0, 1] }]);
            let bb = if bd == DType::F16 { f16b(&b) } else { f32b(&b) };
            check(&be, Kernel::MatMul(spec, dims), &[f32b(&a), bb, f32b(&bias)], DType::F32, m * n, 1e-4, &format!("matmul {m}x{n}x{k} {bd:?}"));
        }
    }
    for fmt in [lapid_quant::Format::Int8, lapid_quant::Format::Int4] {
        let dt = if fmt == lapid_quant::Format::Int8 { DType::Int8 } else { DType::Int4 };
        for &(m, n, k) in &[(1usize, 40usize, 96usize), (2, 9, 4160), (37, 70, 64)] {
            let q = lapid_quant::quantize(fmt, &rnd(n * k, 11), n, k);
            let spec = Arc::new(MatMulSpec { a_dtype: DType::F32, b_dtype: dt, b_transposed: true, extra_dtypes: vec![], out_dtype: DType::F32, epilogue: None });
            check(&be, Kernel::MatMul(spec, mm_dims(m, n, k, true, vec![])), &[f32b(&rnd(m * k, 10)), q], DType::F32, m * n, 2e-2, &format!("quant {dt:?} {m}x{n}x{k}"));
        }
    }
}

#[test]
fn gather_and_out_of_range() {
    let Some(be) = backend() else { return };
    let (rows, cols) = (10usize, 64usize);
    let table = rnd(rows * cols, 12);
    let idx = i32b(&[3, 0, 9, 3, 7]);
    let dims = GatherDims { rows, cols, count: 5 };
    for dt in [DType::F32, DType::Int8, DType::Int4] {
        let t = match dt {
            DType::Int8 => lapid_quant::quantize(lapid_quant::Format::Int8, &table, rows, cols),
            DType::Int4 => lapid_quant::quantize(lapid_quant::Format::Int4, &table, rows, cols),
            _ => f32b(&table),
        };
        let spec = Arc::new(GatherSpec { table_dtype: dt, out_dtype: DType::F32 });
        let (g, c) = both(&be, Kernel::Gather(spec, dims.clone()), &[t, idx.clone()], 5 * cols * 4);
        assert_eq!(g, c, "gather {dt:?}");
    }
    let spec = Arc::new(GatherSpec { table_dtype: DType::F32, out_dtype: DType::F32 });
    let k = Kernel::Gather(spec, GatherDims { rows: 4, cols: 8, count: 3 });
    let t4 = f32b(&table[..32]);
    for bad in [4i32, -1] {
        let err = exec(&*be, &k, &[t4.clone(), i32b(&[0, bad, 2])], &[0u8; 96]).unwrap_err();
        assert_eq!(err.code, "index_out_of_range");
    }
    exec(&*be, &k, &[t4, i32b(&[0, 3, 2])], &[0u8; 96]).expect("flag cleared");
    // gather_axis is an exact copy
    let (outer, len, inner, n) = (3usize, 8usize, 5usize, 6usize);
    let gidx: Vec<i32> = (0..outer * n * inner).map(|i| ((i * 7 + 3) % len) as i32).collect();
    let gd = GatherAxisDims { outer, n, inner, axis_len: len, x: View::contiguous(&[outer, len, inner]), idx: View::contiguous(&[outer, n, inner]) };
    for dt in [DType::F32, DType::F16] {
        let xb: Vec<u8> = (0..outer * len * inner * dt.size()).map(|i| (i * 37 % 251) as u8).collect();
        let (g, c) = both(&be, Kernel::GatherAxis(Arc::new(GatherAxisSpec { dtype: dt }), gd.clone()), &[xb, i32b(&gidx)], outer * n * inner * dt.size());
        assert_eq!(g, c, "gather_axis {dt:?}");
    }
}

#[test]
fn topk_and_argmax() {
    let Some(be) = backend() else { return };
    for &(rows, cols, k) in &[(3usize, 1000usize, 1usize), (2, 51865, 1), (4, 4097, 50), (2, 37, 5)] {
        let x = rnd(rows * cols, (cols + k) as u32);
        let kern = Kernel::TopK(Arc::new(TopKSpec { in_dtype: DType::F32 }), TopKDims { rows, cols, k });
        let (g, c) = both(&be, kern, &[f32b(&x)], rows * k * 4);
        assert_eq!(g, c, "topk {rows}x{cols} k={k}");
    }
}

#[test]
fn attention_variants() {
    let Some(be) = backend() else { return };
    let h = 2usize;
    for &(sq, sk, d, causal, with_mask) in &[
        (20usize, 30usize, 64usize, true, true),
        (70, 70, 64, false, false),
        (20, 33, 40, true, false),
        (1, 45, 64, true, true),
        (3, 50, 50, false, true),
        (2, 300, 64, true, true),
        (1, 1500, 64, false, false),
    ] {
        let spec = Arc::new(AttentionSpec { dtype: DType::F32, mask_dtype: with_mask.then_some(DType::F32), causal, out_dtype: DType::F32 });
        let view = |s: usize| View { offset: 0, strides: vec![(h * s * d) as isize, (s * d) as isize, d as isize, 1] };
        let dims = AttentionDims {
            batch: 1,
            heads: h,
            sq,
            sk,
            d,
            scale: (1.0 / (d as f32).sqrt()).to_bits(),
            q: view(sq),
            k: view(sk),
            v: view(sk),
            mask: with_mask.then(|| View { offset: 0, strides: vec![0, 0, sk as isize, 1] }),
        };
        let mask: Vec<f32> = rnd(sq * sk, 19).iter().map(|x| if *x > 0.8 { f32::NEG_INFINITY } else { *x }).collect();
        let mut inputs = vec![f32b(&rnd(h * sq * d, 16)), f32b(&rnd(h * sk * d, 17)), f32b(&rnd(h * sk * d, 18))];
        if with_mask {
            inputs.push(f32b(&mask));
        }
        check(&be, Kernel::Attention(spec, dims), &inputs, DType::F32, h * sq * d, 1e-4, &format!("attention sq={sq} sk={sk} d={d}"));
    }
}

#[test]
fn chained_launches_and_aliasing() {
    let Some(be) = backend() else { return };
    let n = 1000usize;
    let x = rnd(n, 20);
    let add = Arc::new(EwSpec { rank: 1, in_dtypes: vec![DType::F32], out_dtype: DType::F32, expr: Expr::Binary(BinaryOp::Add, ld(0), Box::new(Expr::konst(1.0))) });
    let dims = EwDims { shape: vec![n], in_views: vec![View::contiguous(&[n])], out_view: View::contiguous(&[n]) };
    let bufs: Vec<DeviceData> = vec![be.upload(&f32b(&x)).unwrap(), be.alloc(n * 4).unwrap()];
    // 150 ping-pong launches, then 3 in place (input buffer == output buffer)
    let mut launches: Vec<Launch> = (0..150).map(|i| Launch { kernel: Kernel::Elementwise(add.clone(), dims.clone()), inputs: vec![i % 2], output: 1 - i % 2 }).collect();
    for _ in 0..3 {
        launches.push(Launch { kernel: Kernel::Elementwise(add.clone(), dims.clone()), inputs: vec![0], output: 0 });
    }
    be.prepare(&launches).unwrap();
    be.execute(&launches, &bufs).unwrap();
    let out = as_f32(&be.download(&bufs[0]).unwrap(), DType::F32);
    let want: Vec<f32> = x.iter().map(|v| v + 153.0).collect();
    assert_close(&out, &want, 1e-6, "chain");
    // empty and odd-sized buffers round-trip
    assert_eq!(be.download(&be.upload(&[]).unwrap()).unwrap(), Vec::<u8>::new());
    assert_eq!(be.download(&be.upload(&[1, 2, 3, 4, 5]).unwrap()).unwrap(), vec![1, 2, 3, 4, 5]);
}
