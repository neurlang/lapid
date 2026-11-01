//! Tests. `shaders_compile_*` only need naga (no GPU). The kernel tests run on adapter 0
//! against naive CPU references and are skipped when no Vulkan adapter is available.

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::{BinaryOp, DType, DeviceData, ReduceOp, UnaryOp};
use half::{bf16, f16};

use crate::codegen::{self, IdxMode, Key};
use crate::context::Caps;
use crate::pipeline::compile_wgsl;
use crate::{VulkanBackend, adapters};

// ------------------------------------------------------------------ helpers

pub(crate) fn caps(f16: bool, sg: bool, dot: bool) -> Caps {
    Caps {
        f16,
        int8: true,
        int16: false,
        int_dot: dot,
        subgroup: sg,
        subgroup_ballot: sg,
        subgroup_shuffle: sg,
        subgroup_vote: sg,
        subgroup_size: 32,
        full_subgroups: false,
        max_groups: [65535; 3],
        max_shared: 32768,
        max_invocations: 1024,
        storage_align: 64,
        max_storage_range: u32::MAX as u64,
        push_descriptor: false,
        max_push_descriptors: 0,
        coopmat: false,
        timestamp_period: 0.0,
    }
}

pub(crate) fn load(i: usize) -> Box<Expr> {
    Box::new(Expr::Load(i))
}

fn gelu_plus(a: usize, b: usize) -> Expr {
    // gelu(x_a) + 2 * x_b
    let x = || load(a);
    let g = Expr::Binary(
        BinaryOp::Mul,
        Box::new(Expr::Binary(BinaryOp::Mul, x(), Box::new(Expr::konst(0.5)))),
        Box::new(Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::konst(1.0)),
            Box::new(Expr::Unary(UnaryOp::Erf, Box::new(Expr::Binary(BinaryOp::Mul, x(), Box::new(Expr::konst(std::f32::consts::FRAC_1_SQRT_2)))))),
        )),
    );
    Expr::Binary(BinaryOp::Add, Box::new(g), Box::new(Expr::Binary(BinaryOp::Mul, load(b), Box::new(Expr::konst(2.0)))))
}

fn erf_ref(x: f32) -> f32 {
    // Numerical Recipes erfc approximation (fractional error < 1.2e-7), in f64
    let x = x as f64;
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let y = 1.0
        - t * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    (if x >= 0.0 { y } else { -y }) as f32
}

fn gelu_ref(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_ref(x * std::f32::consts::FRAC_1_SQRT_2))
}

pub(crate) fn mat_spec(a: DType, b: DType, bt: bool, extras: Vec<DType>, epi: Option<Expr>) -> Arc<MatMulSpec> {
    Arc::new(MatMulSpec { a_dtype: a, b_dtype: b, b_transposed: bt, extra_dtypes: extras, out_dtype: DType::F32, epilogue: epi })
}

/// Every kernel key the planner can produce, over a representative dtype mix.
fn all_keys() -> Vec<Key> {
    let mut keys = Vec::new();
    for &(ina, inb, out) in &[
        (DType::F32, DType::F32, DType::F32),
        (DType::F16, DType::BF16, DType::F16),
        (DType::I32, DType::F32, DType::I32),
        (DType::F32, DType::F16, DType::BF16),
    ] {
        let spec = Arc::new(EwSpec { rank: 3, in_dtypes: vec![ina, inb], out_dtype: out, expr: gelu_plus(0, 1) });
        keys.push(Key::Ew(spec.clone(), vec![IdxMode::Flat, IdxMode::Scalar], IdxMode::Flat));
        keys.push(Key::Ew(spec, vec![IdxMode::Strided, IdxMode::Strided], IdxMode::Strided));
    }
    let fill = Arc::new(EwSpec { rank: 0, in_dtypes: vec![], out_dtype: DType::F32, expr: Expr::konst(f32::NEG_INFINITY) });
    keys.push(Key::Ew(fill, vec![], IdxMode::Flat));
    for op in [ReduceOp::Sum, ReduceOp::Max] {
        for rank in [1, 3] {
            let spec = Arc::new(ReduceSpec {
                op,
                rank,
                in_dtypes: vec![DType::F16, DType::F32],
                out_dtype: DType::F32,
                expr: Expr::Binary(BinaryOp::Mul, load(0), load(1)),
            });
            keys.push(Key::Reduce(spec.clone(), true));
            keys.push(Key::Reduce(spec, false));
        }
    }
    let bias = Some(Expr::Binary(BinaryOp::Add, load(0), load(1)));
    for b in [DType::F32, DType::F16, DType::Int8, DType::Int4] {
        for a in [DType::F32, DType::F16] {
            let s = mat_spec(a, b, true, vec![DType::F32], bias.clone());
            keys.push(Key::Gemm(s.clone()));
            keys.push(Key::Gemv(s.clone(), false, false));
            keys.push(Key::Gemv(s.clone(), true, b == DType::F16));
            if caps_coop() {
                keys.push(Key::GemmCoop(s));
            }
        }
    }
    keys.push(Key::Gemm(mat_spec(DType::F32, DType::F16, false, vec![], None)));
    for t in [DType::F32, DType::F16, DType::Int8, DType::Int4] {
        keys.push(Key::Gather(Arc::new(GatherSpec { table_dtype: t, out_dtype: DType::F32 })));
    }
    keys.push(Key::Softmax(Arc::new(SoftmaxSpec { in_dtype: DType::F16, out_dtype: DType::F32 })));
    keys.push(Key::LayerNorm(Arc::new(LayerNormSpec { in_dtype: DType::F32, param_dtype: DType::F32, out_dtype: DType::F16 })));
    keys.push(Key::RmsNorm(Arc::new(LayerNormSpec { in_dtype: DType::F16, param_dtype: DType::F32, out_dtype: DType::F16 })));
    for causal in [false, true] {
        for mask in [None, Some(DType::F32)] {
            let s = Arc::new(AttentionSpec { dtype: DType::F16, mask_dtype: mask, causal, out_dtype: DType::F32 });
            keys.push(Key::AttnTile(s.clone(), 64));
            keys.push(Key::AttnTile(s.clone(), 128));
            keys.push(Key::AttnSplit(s.clone(), true, 64));
            keys.push(Key::AttnSplit(s.clone(), false, 50));
            keys.push(Key::AttnMerge(s.clone()));
            if caps_coop() {
                keys.push(Key::AttnCoop(s));
            }
        }
    }
    keys
}

/// GemmCoop keys are compiled only by `shaders_compile_coopmat` (needs the coopmat caps).
fn caps_coop() -> bool {
    COOP.with(|c| c.get())
}

thread_local! {
    static COOP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[test]
fn shaders_compile_coopmat() {
    let mut c = caps(true, true, true);
    c.coopmat = true;
    c.full_subgroups = true;
    COOP.with(|x| x.set(true));
    compile_all(&c);
}

fn compile_all(c: &Caps) {
    for key in all_keys() {
        let src = codegen::source(&key, c);
        if let Err(e) = compile_wgsl(key.label(), &src, c) {
            panic!("{key:?} failed (caps {c:?}):\n{}", e.message);
        }
    }
}

#[test]
fn shaders_compile_full_features() {
    compile_all(&caps(true, true, true));
}

#[test]
fn shaders_compile_fallbacks() {
    compile_all(&caps(false, false, false));
}

#[test]
fn shaders_compile_mixed() {
    compile_all(&caps(true, false, true));
    compile_all(&caps(false, true, false));
}

#[test]
fn adapters_never_panic() {
    match adapters() {
        Ok(list) => {
            for (i, a) in list.iter().enumerate() {
                assert_eq!(a.index, i);
            }
        }
        Err(e) => assert_eq!(e.code, "vulkan_unavailable"),
    }
}

/// A test backend holding the process-wide GPU test lock. GPU tests run one at a time:
/// with 6+ test threads submitting concurrently the NVIDIA driver intermittently killed the
/// shared context (Xid 109 "CTX SWITCH TIMEOUT" -> device lost) although every kernel
/// involved is short (< 0.2 ms) and the same tests pass serially and with 2 threads.
pub(crate) struct TestBackend {
    be: VulkanBackend,
    _gpu: std::sync::MutexGuard<'static, ()>,
}

impl std::ops::Deref for TestBackend {
    type Target = VulkanBackend;
    fn deref(&self) -> &VulkanBackend {
        &self.be
    }
}

pub(crate) fn backend() -> Option<TestBackend> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let list = adapters().ok()?;
    if list.is_empty() {
        eprintln!("no vulkan adapter, skipping");
        return None;
    }
    let gpu = GPU.lock().unwrap_or_else(|p| p.into_inner());
    Some(TestBackend { be: VulkanBackend::new(0).expect("adapter 0 must open"), _gpu: gpu })
}

pub(crate) fn f32b(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}
pub(crate) fn f16b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| f16::from_f32(*x).to_le_bytes()).collect()
}
fn bf16b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| bf16::from_f32(*x).to_le_bytes()).collect()
}
pub(crate) fn from_f32b(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
pub(crate) fn from_f16b(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<2>().0.iter().map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
}

/// Deterministic pseudo-random values in [-1, 1).
pub(crate) fn rnd(n: usize, seed: u32) -> Vec<f32> {
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

pub(crate) fn run(be: &VulkanBackend, kernel: Kernel, inputs: &[Vec<u8>], out_bytes: usize) -> Vec<u8> {
    let mut bufs: Vec<DeviceData> = inputs.iter().map(|b| be.upload(b).unwrap()).collect();
    bufs.push(be.alloc(out_bytes).unwrap());
    let n = inputs.len();
    let launch = Launch { kernel, inputs: (0..n).collect(), output: n };
    be.execute(&[launch], &bufs).unwrap();
    be.download(&bufs[n]).unwrap()
}

pub(crate) fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let num: f64 = got.iter().zip(want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().max(1e-30);
    let rel = (num / den).sqrt() as f32;
    let max_abs = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(rel <= tol, "{what}: rel err {rel} > {tol} (max abs {max_abs})");
}

// ------------------------------------------------------------------ kernel tests

#[test]
fn elementwise_broadcast_strided_gelu() {
    let Some(be) = backend() else { return };
    let (d0, d1, d2) = (3usize, 5usize, 7usize);
    // a stored transposed as [d2, d1, d0], viewed as [d0, d1, d2]
    let a_store = rnd(d0 * d1 * d2, 1);
    let b = rnd(d1, 2);
    let spec = Arc::new(EwSpec { rank: 3, in_dtypes: vec![DType::F32, DType::F32], out_dtype: DType::F32, expr: gelu_plus(0, 1) });
    let dims = EwDims {
        shape: vec![d0, d1, d2],
        in_views: vec![
            View { offset: 0, strides: vec![1, d0 as isize, (d0 * d1) as isize] },
            View { offset: 0, strides: vec![0, 1, 0] },
        ],
        out_view: View::contiguous(&[d0, d1, d2]),
    };
    let out = from_f32b(&run(&be, Kernel::Elementwise(spec, dims), &[f32b(&a_store), f32b(&b)], d0 * d1 * d2 * 4));
    let mut want = Vec::new();
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                want.push(gelu_ref(a_store[i + j * d0 + k * d0 * d1]) + 2.0 * b[j]);
            }
        }
    }
    assert_close(&out, &want, 1e-5, "ew lapid");
}

#[test]
fn elementwise_half_types_odd_length() {
    let Some(be) = backend() else { return };
    let n = 35usize;
    let a = rnd(n, 3);
    let b = rnd(n, 4);
    let expr = Expr::Binary(BinaryOp::Sub, load(0), load(1));
    let spec = Arc::new(EwSpec { rank: 1, in_dtypes: vec![DType::F16, DType::BF16], out_dtype: DType::F16, expr });
    let dims = EwDims { shape: vec![n], in_views: vec![View::contiguous(&[n]); 2], out_view: View::contiguous(&[n]) };
    let out = from_f16b(&run(&be, Kernel::Elementwise(spec, dims), &[f16b(&a), bf16b(&b)], n * 2));
    let want: Vec<f32> = (0..n)
        .map(|i| f16::from_f32(f16::from_f32(a[i]).to_f32() - bf16::from_f32(b[i]).to_f32()).to_f32())
        .collect();
    assert_close(&out, &want, 1e-3, "ew half");
}

#[test]
fn reduce_row_and_thread_variants() {
    let Some(be) = backend() else { return };
    for &(outer, inner, op) in &[(6usize, 300usize, ReduceOp::Sum), (1000, 5, ReduceOp::Max), (3, 1000, ReduceOp::Max)] {
        let x = rnd(outer * inner, 5);
        let spec = Arc::new(ReduceSpec {
            op,
            rank: 2,
            in_dtypes: vec![DType::F32],
            out_dtype: DType::F32,
            expr: Expr::Binary(BinaryOp::Mul, load(0), load(0)),
        });
        let dims = ReduceDims { shape: vec![outer, inner], in_views: vec![View::contiguous(&[outer, inner])] };
        let out = from_f32b(&run(&be, Kernel::Reduce(spec, dims), &[f32b(&x)], outer * 4));
        let want: Vec<f32> = (0..outer)
            .map(|o| {
                let it = x[o * inner..(o + 1) * inner].iter().map(|v| v * v);
                match op {
                    ReduceOp::Sum => it.sum(),
                    ReduceOp::Max => it.fold(f32::NEG_INFINITY, f32::max),
                }
            })
            .collect();
        assert_close(&out, &want, 1e-5, "reduce");
    }
}

#[allow(clippy::too_many_arguments)]
fn matmul_ref(a: &[f32], b: &[f32], batch: usize, m: usize, n: usize, k: usize, bt: bool, b_batch: bool) -> Vec<f32> {
    let mut out = vec![0f32; batch * m * n];
    for t in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f64;
                for kk in 0..k {
                    let bo = if b_batch { t * n * k } else { 0 };
                    let bv = if bt { b[bo + j * k + kk] } else { b[bo + kk * n + j] };
                    s += a[t * m * k + i * k + kk] as f64 * bv as f64;
                }
                out[(t * m + i) * n + j] = s as f32;
            }
        }
    }
    out
}

fn mm_dims(batch: usize, m: usize, n: usize, k: usize, bt: bool, b_batch: bool, extras: Vec<View>) -> MatMulDims {
    MatMulDims {
        batch,
        m,
        n,
        k,
        a: MatView { offset: 0, batch_stride: m * k, row_stride: k },
        b: MatView { offset: 0, batch_stride: if b_batch { n * k } else { 0 }, row_stride: if bt { k } else { n } },
        extra_views: extras,
        out: MatView::contiguous(m, n),
    }
}

#[test]
fn matmul_float_gemm_and_gemv() {
    let Some(be) = backend() else { return };
    for &(batch, m, n, k, bt) in &[(2usize, 70usize, 67usize, 45usize, true), (1, 33, 20, 64, false), (2, 1, 40, 96, true), (1, 3, 130, 50, true)] {
        let a = rnd(batch * m * k, 7);
        let b = rnd(batch * n * k, 8);
        let bias = rnd(n, 9);
        let spec = mat_spec(DType::F32, DType::F16, bt, vec![DType::F32], Some(Expr::Binary(BinaryOp::Add, load(0), load(1))));
        let dims = mm_dims(batch, m, n, k, bt, true, vec![View { offset: 0, strides: vec![0, 0, 1] }]);
        let b16: Vec<f32> = b.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
        let out = from_f32b(&run(&be, Kernel::MatMul(spec, dims), &[f32b(&a), f16b(&b), f32b(&bias)], batch * m * n * 4));
        let mut want = matmul_ref(&a, &b16, batch, m, n, k, bt, true);
        for (i, w) in want.iter_mut().enumerate() {
            *w += bias[i % n];
        }
        // f16 weights may run on cooperative matrices, which round A to f16 as well
        assert_close(&out, &want, 1e-3, &format!("matmul {batch}x{m}x{n}x{k} bt={bt}"));
    }
}

#[test]
fn matmul_quantized() {
    let Some(be) = backend() else { return };
    for fmt in [lapid_quant::Format::Int8, lapid_quant::Format::Int4] {
        let dt = if fmt == lapid_quant::Format::Int8 { DType::Int8 } else { DType::Int4 };
        for &(m, n, k) in &[(1usize, 40usize, 96usize), (2, 9, 4160), (37, 70, 64)] {
            let a = rnd(m * k, 10);
            let w = rnd(n * k, 11);
            let q = lapid_quant::quantize(fmt, &w, n, k);
            let wd = lapid_quant::dequantize(fmt, &q, n, k);
            let spec = mat_spec(DType::F32, dt, true, vec![], None);
            let dims = mm_dims(1, m, n, k, true, false, vec![]);
            let out = from_f32b(&run(&be, Kernel::MatMul(spec, dims), &[f32b(&a), q], m * n * 4));
            let want = matmul_ref(&a, &wd, 1, m, n, k, true, false);
            // the int8-dot path quantizes A per 32-block
            assert_close(&out, &want, 2e-2, &format!("quant {dt:?} {m}x{n}x{k}"));
        }
    }
}

#[test]
fn gather_plain_and_quantized() {
    let Some(be) = backend() else { return };
    let (rows, cols) = (10usize, 64usize);
    let table = rnd(rows * cols, 12);
    let idx: Vec<i32> = vec![3, 0, 9, 3, 7];
    let idx_b: Vec<u8> = bytemuck::cast_slice(&idx).to_vec();
    let dims = GatherDims { rows, cols, count: idx.len() };
    let want: Vec<f32> = idx.iter().flat_map(|&r| table[r as usize * cols..(r as usize + 1) * cols].to_vec()).collect();
    let spec = Arc::new(GatherSpec { table_dtype: DType::F32, out_dtype: DType::F32 });
    let out = from_f32b(&run(&be, Kernel::Gather(spec, dims.clone()), &[f32b(&table), idx_b.clone()], want.len() * 4));
    assert_close(&out, &want, 1e-7, "gather f32");
    for fmt in [lapid_quant::Format::Int8, lapid_quant::Format::Int4] {
        let dt = if fmt == lapid_quant::Format::Int8 { DType::Int8 } else { DType::Int4 };
        let q = lapid_quant::quantize(fmt, &table, rows, cols);
        let deq = lapid_quant::dequantize(fmt, &q, rows, cols);
        let want: Vec<f32> = idx.iter().flat_map(|&r| deq[r as usize * cols..(r as usize + 1) * cols].to_vec()).collect();
        let spec = Arc::new(GatherSpec { table_dtype: dt, out_dtype: DType::F32 });
        let out = from_f32b(&run(&be, Kernel::Gather(spec, dims.clone()), &[q, idx_b.clone()], want.len() * 4));
        assert_close(&out, &want, 1e-6, "gather quant");
    }
}

fn dims_rms(rows: usize, cols: usize, eps: f32) -> LayerNormDims {
    LayerNormDims { rows, cols, eps: eps.to_bits() }
}

#[test]
fn softmax_and_layer_norm() {
    let Some(be) = backend() else { return };
    let (rows, cols) = (5usize, 777usize);
    let x: Vec<f32> = rnd(rows * cols, 13).iter().map(|v| v * 8.0).collect();
    let spec = Arc::new(SoftmaxSpec { in_dtype: DType::F32, out_dtype: DType::F32 });
    let out = from_f32b(&run(&be, Kernel::Softmax(spec, RowDims { rows, cols }), &[f32b(&x)], rows * cols * 4));
    let mut want = Vec::new();
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let s: f32 = row.iter().map(|v| (v - m).exp()).sum();
        want.extend(row.iter().map(|v| (v - m).exp() / s));
    }
    assert_close(&out, &want, 1e-5, "softmax");

    // 777: row cached in registers; 1500: re-read path
    for (rows, cols) in [(rows, cols), (2usize, 1500usize)] {
    let x: Vec<f32> = rnd(rows * cols, 13).iter().map(|v| v * 8.0).collect();
    let w = rnd(cols, 14);
    let bias = rnd(cols, 15);
    let eps = 1e-5f32;
    let spec = Arc::new(LayerNormSpec { in_dtype: DType::F32, param_dtype: DType::F32, out_dtype: DType::F32 });
    let dims = LayerNormDims { rows, cols, eps: eps.to_bits() };
    let out = from_f32b(&run(&be, Kernel::LayerNorm(spec, dims), &[f32b(&x), f32b(&w), f32b(&bias)], rows * cols * 4));
    let mut want = Vec::new();
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        want.extend(row.iter().enumerate().map(|(j, v)| (v - mean) * rstd * w[j] + bias[j]));
    }
    assert_close(&out, &want, 1e-5, "layer_norm");
    let spec = Arc::new(LayerNormSpec { in_dtype: DType::F32, param_dtype: DType::F32, out_dtype: DType::F32 });
    let out = from_f32b(&run(&be, Kernel::RmsNorm(spec, dims_rms(rows, cols, eps)), &[f32b(&x), f32b(&w)], rows * cols * 4));
    let mut want = Vec::new();
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let rstd = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / cols as f32 + eps).sqrt();
        want.extend(row.iter().enumerate().map(|(j, v)| v * rstd * w[j]));
    }
    assert_close(&out, &want, 1e-5, "rms_norm");
    }
}

#[allow(clippy::too_many_arguments)]
fn attention_ref(q: &[f32], k: &[f32], v: &[f32], mask: Option<&[f32]>, h: usize, sq: usize, sk: usize, d: usize, scale: f32, causal: bool) -> Vec<f32> {
    let mut out = vec![0f32; h * sq * d];
    for hh in 0..h {
        for i in 0..sq {
            let mut s: Vec<f32> = (0..sk)
                .map(|j| {
                    let mut x = 0f32;
                    for t in 0..d {
                        x += q[(hh * sq + i) * d + t] * k[(hh * sk + j) * d + t];
                    }
                    x *= scale;
                    if let Some(m) = mask {
                        x += m[i * sk + j];
                    }
                    if causal && j > i + sk - sq {
                        x = f32::NEG_INFINITY;
                    }
                    x
                })
                .collect();
            let mx = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for x in s.iter_mut() {
                *x = (*x - mx).exp();
                sum += *x;
            }
            for t in 0..d {
                let mut o = 0f32;
                for j in 0..sk {
                    o += s[j] / sum * v[(hh * sk + j) * d + t];
                }
                out[(hh * sq + i) * d + t] = o;
            }
        }
    }
    out
}

#[test]
fn attention_tile_and_row() {
    let Some(be) = backend() else { return };
    let h = 2usize;
    // (sq, sk, d, causal, mask): tile (sq > 8, d % 4 == 0) and row variants
    for &(sq, sk, d, causal, with_mask) in &[
        (20usize, 30usize, 64usize, true, true),
        (70, 70, 64, false, false),
        (20, 33, 40, true, false),
        (1, 45, 64, true, true),
        (3, 50, 50, false, true),
        (2, 300, 64, true, true),
        (100, 100, 64, true, true),
        (70, 75, 64, false, false),
        (1, 1500, 64, false, false),
    ] {
        let q = rnd(h * sq * d, 16);
        let k = rnd(h * sk * d, 17);
        let v = rnd(h * sk * d, 18);
        let mask: Vec<f32> = rnd(sq * sk, 19).iter().map(|x| if *x > 0.8 { f32::NEG_INFINITY } else { *x }).collect();
        let scale = 1.0 / (d as f32).sqrt();
        let spec = Arc::new(AttentionSpec {
            dtype: DType::F32,
            mask_dtype: with_mask.then_some(DType::F32),
            causal,
            out_dtype: DType::F32,
        });
        let view = |s: usize| View { offset: 0, strides: vec![(h * s * d) as isize, (s * d) as isize, d as isize, 1] };
        let dims = AttentionDims {
            batch: 1,
            heads: h,
            sq,
            sk,
            d,
            scale: scale.to_bits(),
            q: view(sq),
            k: view(sk),
            v: view(sk),
            mask: with_mask.then(|| View { offset: 0, strides: vec![0, 0, sk as isize, 1] }),
        };
        let mut inputs = vec![f32b(&q), f32b(&k), f32b(&v)];
        if with_mask {
            inputs.push(f32b(&mask));
        }
        let out = from_f32b(&run(&be, Kernel::Attention(spec, dims), &inputs, h * sq * d * 4));
        let want = attention_ref(&q, &k, &v, with_mask.then_some(mask.as_slice()), h, sq, sk, d, scale, causal);
        // many-query d=64 attention may run on cooperative matrices (f16 Q/K/V/P)
        let tol = if sq > 8 && d == 64 { 2e-3 } else { 1e-4 };
        assert_close(&out, &want, tol, &format!("attention sq={sq} sk={sk} d={d}"));
    }
}

#[test]
fn chained_launches_with_barriers() {
    let Some(be) = backend() else { return };
    // y = x + 1 ; z = y * y  (z depends on y written by the previous launch)
    let n = 1000usize;
    let x = rnd(n, 20);
    let add = Arc::new(EwSpec {
        rank: 1,
        in_dtypes: vec![DType::F32],
        out_dtype: DType::F32,
        expr: Expr::Binary(BinaryOp::Add, load(0), Box::new(Expr::konst(1.0))),
    });
    let sq = Arc::new(EwSpec { rank: 1, in_dtypes: vec![DType::F32], out_dtype: DType::F32, expr: Expr::Binary(BinaryOp::Mul, load(0), load(0)) });
    let dims = EwDims { shape: vec![n], in_views: vec![View::contiguous(&[n])], out_view: View::contiguous(&[n]) };
    let bufs: Vec<DeviceData> = vec![be.upload(&f32b(&x)).unwrap(), be.alloc(n * 4).unwrap(), be.alloc(n * 4).unwrap()];
    let launches = vec![
        Launch { kernel: Kernel::Elementwise(add, dims.clone()), inputs: vec![0], output: 1 },
        Launch { kernel: Kernel::Elementwise(sq, dims), inputs: vec![1], output: 2 },
    ];
    be.prepare(&launches).unwrap();
    be.execute(&launches, &bufs).unwrap();
    let out = from_f32b(&be.download(&bufs[2]).unwrap());
    let want: Vec<f32> = x.iter().map(|v| (v + 1.0) * (v + 1.0)).collect();
    assert_close(&out, &want, 1e-6, "chain");
}

#[test]
fn long_dependent_chain_across_submission_chunks() {
    let Some(be) = backend() else { return };
    // 150 launches ping-ponging between two buffers: spans several command-buffer chunks
    let n = 300usize;
    let x = rnd(n, 21);
    let add = Arc::new(EwSpec {
        rank: 1,
        in_dtypes: vec![DType::F32],
        out_dtype: DType::F32,
        expr: Expr::Binary(BinaryOp::Add, load(0), Box::new(Expr::konst(1.0))),
    });
    let dims = EwDims { shape: vec![n], in_views: vec![View::contiguous(&[n])], out_view: View::contiguous(&[n]) };
    let bufs: Vec<DeviceData> = vec![be.upload(&f32b(&x)).unwrap(), be.alloc(n * 4).unwrap()];
    let steps = 150usize;
    let launches: Vec<Launch> = (0..steps)
        .map(|i| Launch { kernel: Kernel::Elementwise(add.clone(), dims.clone()), inputs: vec![i % 2], output: 1 - i % 2 })
        .collect();
    be.execute(&launches, &bufs).unwrap();
    let out = from_f32b(&be.download(&bufs[steps % 2]).unwrap());
    let want: Vec<f32> = x.iter().map(|v| v + steps as f32).collect();
    assert_close(&out, &want, 1e-6, "chain");
}
