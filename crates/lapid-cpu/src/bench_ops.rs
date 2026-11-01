//! Decode-step micro-benchmarks of the non-matmul kernels (ignored tests): per-launch cost of
//! the small elementwise / reduce / norm / attention / delta-net launches of a Qwen3.5 / Qwen3
//! token. `cargo test --release -p lapid-cpu bench_ops -- --ignored --nocapture --test-threads=1`

use std::sync::Arc;
use std::time::Instant;

use lapid_compiler::*;
use lapid_tensor::{BinaryOp, DType, DeviceData, ReduceOp, UnaryOp};
use half::f16;

use super::*;

fn rnd(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn up(be: &CpuBackend, v: &[f32]) -> DeviceData {
    be.upload(bytemuck::cast_slice(v)).unwrap()
}

fn up_f16(be: &CpuBackend, v: &[f32]) -> DeviceData {
    let h: Vec<f16> = v.iter().map(|&x| f16::from_f32(x)).collect();
    be.upload(bytemuck::cast_slice(&h)).unwrap()
}

/// Median us per launch: `l` repeated 64 times per execute.
fn time(be: &CpuBackend, name: &str, l: Launch, bufs: &[DeviceData]) {
    let launches = vec![l; 64];
    be.prepare(&launches).unwrap();
    for _ in 0..5 {
        be.execute(&launches, bufs).unwrap();
    }
    let mut v = Vec::new();
    for _ in 0..40 {
        let t = Instant::now();
        be.execute(&launches, bufs).unwrap();
        v.push(t.elapsed().as_secs_f64() * 1e6 / launches.len() as f64);
    }
    v.sort_by(|a, b| a.total_cmp(b));
    eprintln!("{name:<48} {:>8.2} us (min {:.2})", v[v.len() / 2], v[0]);
}

fn ld(i: usize) -> Expr {
    Expr::Load(i)
}
fn bin(op: BinaryOp, a: Expr, b: Expr) -> Expr {
    Expr::Binary(op, Box::new(a), Box::new(b))
}
fn un(op: UnaryOp, a: Expr) -> Expr {
    Expr::Unary(op, Box::new(a))
}
fn view(offset: usize, strides: &[isize]) -> View {
    View { offset, strides: strides.to_vec() }
}

fn ew(shape: &[usize], ins: Vec<(DType, View)>, out: View, expr: Expr) -> Kernel {
    Kernel::Elementwise(
        Arc::new(EwSpec { rank: shape.len(), in_dtypes: ins.iter().map(|x| x.0).collect(), out_dtype: DType::F32, expr }),
        EwDims { shape: shape.to_vec(), in_views: ins.into_iter().map(|x| x.1).collect(), out_view: out },
    )
}

fn sigmoid(x: Expr) -> Expr {
    un(UnaryOp::Recip, bin(BinaryOp::Add, un(UnaryOp::Exp, un(UnaryOp::Neg, x)), Expr::konst(1.0)))
}

#[test]
#[ignore]
fn bench_ops() {
    let be = CpuBackend::new();
    let f = DType::F32;
    let big = up(&be, &rnd(16 * 16512, 1));
    let small = up(&be, &rnd(8192, 2));
    let small2 = up(&be, &rnd(8192, 3));
    let w16 = up_f16(&be, &rnd(6144 * 4, 4));
    let out = be.alloc(16 * 16512 * 4).unwrap();
    let bufs = [big.clone(), small.clone(), small2.clone(), w16.clone(), out.clone()];

    // State copy [16, 16384] out of a [16, 16512] buffer.
    let k = ew(&[16, 16384], vec![(f, view(128, &[16512, 1]))], View::contiguous(&[16, 16384]), ld(0));
    time(&be, "copy [16,16384] from stride 16512", Launch { kernel: k, inputs: vec![0], output: 4 }, &bufs);
    let k = ew(&[2048], vec![(f, view(0, &[1]))], View::contiguous(&[2048]), ld(0));
    time(&be, "copy [2048]", Launch { kernel: k, inputs: vec![1], output: 4 }, &bufs);
    let k = ew(&[6144, 2], vec![(f, view(1, &[3, 1]))], view(0, &[3, 1]), ld(0));
    time(&be, "shift copy [6144,2] stride 3", Launch { kernel: k, inputs: vec![0], output: 4 }, &bufs);
    let k = ew(&[6144], vec![(f, view(0, &[1]))], view(2, &[3]), ld(0));
    time(&be, "copy [6144] -> stride 3", Launch { kernel: k, inputs: vec![1], output: 4 }, &bufs);

    // Conv tail: silu(x0 + x1 * w[3]) with an f16 stride-4 input.
    let s = bin(BinaryOp::Add, ld(0), bin(BinaryOp::Mul, ld(1), ld(2)));
    let k = ew(
        &[6144],
        vec![(f, view(0, &[1])), (f, view(0, &[1])), (DType::F16, view(3, &[4]))],
        View::contiguous(&[6144]),
        bin(BinaryOp::Mul, s.clone(), sigmoid(s)),
    );
    time(&be, "silu(x0+x1*w16) [6144] f16 stride 4", Launch { kernel: k, inputs: vec![0, 1, 3], output: 4 }, &bufs);

    for (name, w) in [("f32 contig w", (f, view(0, &[1]))), ("f16 contig w", (DType::F16, view(0, &[1]))), ("f32 stride4 w", (f, view(3, &[4])))] {
        let s = bin(BinaryOp::Add, ld(0), bin(BinaryOp::Mul, ld(1), ld(2)));
        let k = ew(&[6144], vec![(f, view(0, &[1])), (f, view(0, &[1])), w.clone()], View::contiguous(&[6144]), bin(BinaryOp::Mul, s.clone(), sigmoid(s)));
        let wi = if w.0 == f { 0 } else { 3 };
        time(&be, &format!("silu [6144] {name}"), Launch { kernel: k, inputs: vec![0, 1, wi], output: 4 }, &bufs);
        let k = ew(&[6144], vec![(f, view(0, &[1])), (f, view(0, &[1])), w], View::contiguous(&[6144]), bin(BinaryOp::Add, ld(0), bin(BinaryOp::Mul, ld(1), ld(2))));
        time(&be, &format!("fma only [6144] {name}"), Launch { kernel: k, inputs: vec![0, 1, wi], output: 4 }, &bufs);
    }
    // l2norm pieces.
    let k = ew(
        &[16, 128],
        vec![(f, view(0, &[128, 1])), (f, view(0, &[1, 0]))],
        View::contiguous(&[16, 128]),
        bin(BinaryOp::Mul, ld(0), un(UnaryOp::Rsqrt, bin(BinaryOp::Add, ld(1), Expr::konst(1e-6)))),
    );
    time(&be, "x*rsqrt(s+eps) [16,128] bcast", Launch { kernel: k, inputs: vec![1, 2], output: 4 }, &bufs);
    // Rope half.
    let k = ew(
        &[8, 64],
        vec![(f, view(0, &[128, 1])), (f, view(0, &[0, 1])), (f, view(64, &[128, 1])), (f, view(0, &[0, 1]))],
        view(0, &[128, 1]),
        bin(BinaryOp::Sub, bin(BinaryOp::Mul, ld(0), ld(1)), bin(BinaryOp::Mul, ld(2), ld(3))),
    );
    time(&be, "rope half [8,64]", Launch { kernel: k, inputs: vec![1, 2, 1, 2], output: 4 }, &bufs);

    // Reduces.
    let red = |shape: &[usize], dts: Vec<DType>, views: Vec<View>, expr: Expr| {
        Kernel::Reduce(
            Arc::new(ReduceSpec { op: ReduceOp::Sum, rank: shape.len(), in_dtypes: dts, out_dtype: f, expr }),
            ReduceDims { shape: shape.to_vec(), in_views: views },
        )
    };
    let k = red(&[6144, 3], vec![f, DType::F16], vec![view(0, &[3, 1]), view(0, &[4, 1])], bin(BinaryOp::Mul, ld(0), ld(1)));
    time(&be, "reduce conv [6144,3] x f16", Launch { kernel: k, inputs: vec![0, 3], output: 4 }, &bufs);
    let k = red(&[16, 128], vec![f], vec![view(0, &[128, 1])], bin(BinaryOp::Mul, ld(0), ld(0)));
    time(&be, "reduce sumsq [16,128]", Launch { kernel: k, inputs: vec![1], output: 4 }, &bufs);
    let k = red(&[1, 1024], vec![f], vec![view(0, &[1024, 1])], bin(BinaryOp::Mul, ld(0), ld(0)));
    time(&be, "reduce sumsq [1,1024]", Launch { kernel: k, inputs: vec![1], output: 4 }, &bufs);
    let k = red(&[64, 4096], vec![f], vec![view(0, &[128, 1])], ld(0));
    let k = match k {
        Kernel::Reduce(s, mut d) => {
            d.shape = vec![32, 4096];
            d.in_views = vec![view(0, &[4096, 1])];
            Kernel::Reduce(s, d)
        }
        k => k,
    };
    time(&be, "reduce sum [32,4096]", Launch { kernel: k, inputs: vec![0], output: 4 }, &bufs);

    // Norms.
    let ln = Arc::new(LayerNormSpec { in_dtype: f, param_dtype: f, out_dtype: f });
    for (rows, cols) in [(1usize, 1024usize), (16, 128), (8, 128)] {
        let k = Kernel::RmsNorm(ln.clone(), LayerNormDims { rows, cols, eps: 1e-6f32.to_bits() });
        time(&be, &format!("rms_norm {rows}x{cols}"), Launch { kernel: k, inputs: vec![1, 2], output: 4 }, &bufs);
    }

    // Delta net decode step.
    let (h, dk, dv) = (16usize, 128usize, 128usize);
    let q = up(&be, &rnd(h * dk, 5));
    let kk = up(&be, &rnd(h * dk, 6));
    let v = up(&be, &rnd(h * dv, 7));
    let g = up(&be, &rnd(h, 8).iter().map(|x| -x.abs()).collect::<Vec<_>>());
    let beta = up(&be, &rnd(h, 9).iter().map(|x| x.abs()).collect::<Vec<_>>());
    let st = up(&be, &rnd(h * dk * dv, 10).iter().map(|x| x * 0.1).collect::<Vec<_>>());
    let o = be.alloc(h * (1 + dk) * dv * 4).unwrap();
    let k = Kernel::GatedDeltaNet(
        Arc::new(GatedDeltaNetSpec { out_dtype: f }),
        GatedDeltaNetDims { b: 1, t: 1, h, dk, dv, scale: (1.0 / (dk as f32).sqrt()).to_bits() },
    );
    time(&be, "gated_delta_net t=1 h=16 128x128", Launch { kernel: k, inputs: vec![0, 1, 2, 3, 4, 5], output: 6 }, &[q, kk, v, g, beta, st, o]);

    // Decode attention, Qwen3-0.6B GQA layout (8 kv heads, 2 q heads each folded into sq).
    for sk in [16usize, 72, 512] {
        let d = 128;
        let q = up(&be, &rnd(16 * d, 11));
        let kc = up(&be, &rnd(8 * 1024 * d, 12));
        let vc = up(&be, &rnd(8 * 1024 * d, 13));
        let o = be.alloc(16 * d * 4).unwrap();
        let kv = view(0, &[(8 * 1024 * d) as isize, d as isize, (8 * d) as isize, 1]);
        let k = Kernel::Attention(
            Arc::new(AttentionSpec { dtype: f, mask_dtype: None, causal: false, out_dtype: f }),
            AttentionDims {
                batch: 1,
                heads: 8,
                sq: 2,
                sk,
                d,
                scale: 0.0884f32.to_bits(),
                q: view(0, &[0, 2 * d as isize, d as isize, 1]),
                k: kv.clone(),
                v: kv,
                mask: None,
            },
        );
        time(&be, &format!("attention h8 sq2 sk{sk} d128"), Launch { kernel: k, inputs: vec![0, 1, 2], output: 3 }, &[q, kc, vc, o]);
    }
}

#[test]
#[ignore]
fn bench_ops_program() {
    let s = bin(BinaryOp::Add, ld(0), bin(BinaryOp::Mul, ld(1), ld(2)));
    for (name, e) in [("fma", s.clone()), ("silu", bin(BinaryOp::Mul, s.clone(), sigmoid(s))), ("exp", un(UnaryOp::Exp, ld(0))), ("recip", un(UnaryOp::Recip, ld(0)))] {
        let p = crate::expr::Program::compile(&e);
        let mut regs = crate::expr::Regs::default();
        p.prepare(&mut regs);
        for i in 0..3 {
            if let Some(r) = p.input(&mut regs, i) {
                r.copy_from_slice(&rnd(crate::expr::CHUNK, i as u64 + 1));
            }
        }
        let t = Instant::now();
        let mut acc = 0.0;
        for _ in 0..20000 {
            acc += p.run(&mut regs, crate::expr::CHUNK)[7];
        }
        eprintln!("{name}: {:.3} ns/elem ({acc})", t.elapsed().as_secs_f64() * 1e9 / (20000.0 * crate::expr::CHUNK as f64));
    }
}

#[inline(always)]
fn exp_v2(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    const ROUND_MAGIC: f32 = 12_582_912.0;
    let xc = x.max(-104.0).min(88.722_84);
    let n = (xc * LOG2E + ROUND_MAGIC) - ROUND_MAGIC;
    let r = (xc - n * LN2_HI) - n * LN2_LO;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0)))))));
    let ni = n as i32;
    let n1 = ni >> 1;
    let n2 = ni - n1;
    let s1 = f32::from_bits(((n1 + 127) as u32) << 23);
    let s2 = f32::from_bits(((n2 + 127) as u32) << 23);
    let y = p * s1 * s2;
    let y = if x > 88.722_84 { f32::INFINITY } else { y };
    if x.is_nan() { x } else { y }
}

#[test]
#[ignore]
fn bench_ops_exp() {
    let a = rnd(512, 3);
    let mut d = vec![0f32; 512];
    for (name, f) in [("exp", crate::math::exp as fn(f32) -> f32), ("exp_neg", crate::math::exp_neg), ("v2", exp_v2)] {
        let t = Instant::now();
        let mut acc = 0.0;
        for _ in 0..20000 {
            for (o, &x) in d.iter_mut().zip(&a) {
                *o = f(x);
            }
            acc += d[7];
        }
        eprintln!("{name}: {:.3} ns/elem ({acc})", t.elapsed().as_secs_f64() * 1e9 / (20000.0 * 512.0));
    }
    let t = Instant::now();
    let mut acc = 0.0;
    for _ in 0..20000 {
        for (o, &x) in d.iter_mut().zip(&a) {
            *o = exp_v2(x);
        }
        acc += d[7];
    }
    eprintln!("v2 inline: {:.3} ns/elem ({acc})", t.elapsed().as_secs_f64() * 1e9 / (20000.0 * 512.0));
    let t = Instant::now();
    for _ in 0..20000 {
        for (o, &x) in d.iter_mut().zip(&a) {
            *o = crate::math::exp(x);
        }
        acc += d[7];
    }
    eprintln!("exp inline: {:.3} ns/elem ({acc})", t.elapsed().as_secs_f64() * 1e9 / (20000.0 * 512.0));
}

/// Fork-join cost of the pool for tiny jobs.
#[test]
#[ignore]
fn bench_ops_forkjoin() {
    let threads = std::env::var("LAPID_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    let pool = crate::par::Pool::new(threads);
    pool.install(|| {
        for n in [2usize, 10, 40, 160] {
            for _ in 0..1000 {
                crate::par::for_each(true, n, || (), |_, i| {
                    std::hint::black_box(i);
                });
            }
            let t = Instant::now();
            for _ in 0..20000 {
                crate::par::for_each(true, n, || (), |_, i| {
                    std::hint::black_box(i);
                });
            }
            eprintln!("for_each n={n}: {:.2} us", t.elapsed().as_secs_f64() * 1e6 / 20000.0);
        }
    });
}
