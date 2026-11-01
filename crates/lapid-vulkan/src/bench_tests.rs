//! Kernel micro-benchmarks on the Whisper-small launch shapes, timed with GPU timestamps.
//! Ignored by default; run with
//! `cargo test -p lapid-vulkan --release bench_kernels -- --ignored --nocapture`
//! (`LAPID_BENCH_FILTER=gemv` narrows the set).

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::{BinaryOp, DType, DeviceData, UnaryOp};

use crate::{VulkanBackend, adapters};

fn load(i: usize) -> Box<Expr> {
    Box::new(Expr::Load(i))
}

fn add01() -> Expr {
    Expr::Binary(BinaryOp::Add, load(0), load(1))
}

/// gelu(acc + bias)
fn gelu_bias() -> Expr {
    let x = || Box::new(add01());
    Expr::Binary(
        BinaryOp::Mul,
        x(),
        Box::new(Expr::Binary(
            BinaryOp::Mul,
            Box::new(Expr::Binary(
                BinaryOp::Add,
                Box::new(Expr::Unary(UnaryOp::Erf, Box::new(Expr::Binary(BinaryOp::Mul, x(), Box::new(Expr::konst(0.70710677)))))),
                Box::new(Expr::konst(1.0)),
            )),
            Box::new(Expr::konst(0.5)),
        )),
    )
}

fn bytes_of(dt: DType, rows: usize, cols: usize) -> usize {
    dt.storage_bytes(&[rows, cols])
}

struct Case {
    name: String,
    /// Inputs that are weights / caches: replicated so every launch reads them cold
    /// (from DRAM), as in a real model run.
    cold: Vec<usize>,
    launch: Launch,
    /// Buffer sizes in bytes (inputs..., output).
    sizes: Vec<usize>,
    flops: f64,
    bytes: f64,
}

fn matmul_case(m: usize, n: usize, k: usize, b: DType, epi: Option<Expr>) -> Case {
    let extras = if epi.is_some() { vec![DType::F32] } else { vec![] };
    let spec = Arc::new(MatMulSpec {
        a_dtype: DType::F32,
        b_dtype: b,
        b_transposed: true,
        extra_dtypes: extras.clone(),
        out_dtype: DType::F32,
        epilogue: epi,
    });
    let dims = MatMulDims {
        batch: 1,
        m,
        n,
        k,
        a: MatView { offset: 0, batch_stride: 0, row_stride: k },
        b: MatView { offset: 0, batch_stride: 0, row_stride: k },
        extra_views: extras.iter().map(|_| View { offset: 0, strides: vec![0, 0, 1] }).collect(),
        out: MatView::contiguous(m, n),
    };
    let mut sizes = vec![m * k * 4, bytes_of(b, n, k)];
    sizes.extend(extras.iter().map(|_| n * 4));
    sizes.push(m * n * 4);
    let ni = sizes.len() - 1;
    Case {
        name: format!("{} {m}x{n}x{k} {}", if m <= 4 { "gemv" } else { "matmul" }, b.name()),
        launch: Launch { kernel: Kernel::MatMul(spec, dims), inputs: (0..ni).collect(), output: ni },
        cold: vec![1],
        flops: 2.0 * (m * n * k) as f64,
        bytes: (sizes.iter().sum::<usize>()) as f64,
        sizes,
    }
}

fn attention_case(sq: usize, sk: usize, encoder: bool) -> Case {
    let (h, d) = (12usize, 64usize);
    let spec = Arc::new(AttentionSpec { dtype: DType::F32, mask_dtype: None, causal: false, out_dtype: DType::F32 });
    let (qv, kv) = if encoder {
        // [1500, 12, 64] activations viewed as [B, H, S, D]
        let v = View { offset: 0, strides: vec![0, 64, 768, 1] };
        (v.clone(), v)
    } else {
        (View { offset: 0, strides: vec![0, 64, 0, 1] }, View { offset: 0, strides: vec![(h * sk * d) as isize, (sk * d) as isize, 64, 1] })
    };
    let dims = AttentionDims { batch: 1, heads: h, sq, sk, d, scale: 0.125f32.to_bits(), q: qv, k: kv.clone(), v: kv, mask: None };
    let sizes = vec![h * sq * d * 4, h * sk * d * 4, h * sk * d * 4, h * sq * d * 4];
    Case {
        name: format!("attention sq={sq} sk={sk}"),
        launch: Launch { kernel: Kernel::Attention(spec, dims), inputs: vec![0, 1, 2], output: 3 },
        cold: if encoder { vec![] } else { vec![1, 2] },
        flops: 4.0 * (h * sq * sk * d) as f64,
        bytes: sizes.iter().sum::<usize>() as f64,
        sizes,
    }
}

/// Causal prefill attention with wide heads (Gemma 4: d 256 sliding / 512 full).
fn attention_wide_case(s: usize, h: usize, d: usize) -> Case {
    let spec = Arc::new(AttentionSpec { dtype: DType::F32, mask_dtype: None, causal: true, out_dtype: DType::F32 });
    let v = View::contiguous(&[1, h, s, d]);
    let dims = AttentionDims { batch: 1, heads: h, sq: s, sk: s, d, scale: 1f32.to_bits(), q: v.clone(), k: v.clone(), v: v.clone(), mask: None };
    let sizes = vec![h * s * d * 4; 4];
    Case {
        name: format!("attention causal s={s} h={h} d={d}"),
        launch: Launch { kernel: Kernel::Attention(spec, dims), inputs: vec![0, 1, 2], output: 3 },
        cold: vec![],
        flops: 2.0 * (h * s * s * d) as f64,
        bytes: sizes.iter().sum::<usize>() as f64,
        sizes,
    }
}

fn layer_norm_case(rows: usize) -> Case {
    let spec = Arc::new(LayerNormSpec { in_dtype: DType::F32, param_dtype: DType::F32, out_dtype: DType::F32 });
    let dims = LayerNormDims { rows, cols: 768, eps: 1e-5f32.to_bits() };
    let sizes = vec![rows * 768 * 4, 768 * 4, 768 * 4, rows * 768 * 4];
    Case {
        name: format!("layer_norm {rows}x768"),
        launch: Launch { kernel: Kernel::LayerNorm(spec, dims), inputs: vec![0, 1, 2], output: 3 },
        cold: vec![],
        flops: 0.0,
        bytes: sizes.iter().sum::<usize>() as f64,
        sizes,
    }
}

fn matmul_id_case(t: usize, r: usize, e: usize, n: usize, k: usize, w: DType) -> Case {
    let spec = Arc::new(MatMulIdSpec { x_dtype: DType::F32, w_dtype: w, out_dtype: DType::F32 });
    let dims = MatMulIdDims { t, r, e, n, k };
    let wbytes = e * w.storage_bytes(&[n, k]);
    // ids: input index 2, filled by `ids_for` below (spread over all experts)
    let sizes = vec![t * k * 4, wbytes, t * r * 4, t * r * n * 4];
    // bytes touched: experts actually used (all, for t * r >= e)
    let used = (t * r).min(e) as f64 / e as f64;
    Case {
        name: format!("matmul_id t{t} r{r} e{e} {n}x{k} {}", w.name()),
        launch: Launch { kernel: Kernel::MatMulId(spec, dims), inputs: vec![0, 1, 2], output: 3 },
        cold: vec![],
        flops: 2.0 * (t * r * n * k) as f64,
        bytes: wbytes as f64 * used,
        sizes,
    }
}

fn gdn_case(t: usize, h: usize, d: usize) -> Case {
    let spec = Arc::new(GatedDeltaNetSpec { out_dtype: DType::F32 });
    let dims = GatedDeltaNetDims { b: 1, t, h, dk: d, dv: d, scale: 0.088f32.to_bits() };
    let sizes = vec![t * h * d * 4, t * h * d * 4, t * h * d * 4, t * h * 4, t * h * 4, h * d * d * 4, h * (t + d) * d * 4];
    Case {
        name: format!("gated_delta_net t{t} h{h} d{d}"),
        launch: Launch { kernel: Kernel::GatedDeltaNet(spec, dims), inputs: vec![0, 1, 2, 3, 4, 5], output: 6 },
        cold: vec![],
        flops: 6.0 * (t * h * d * d) as f64,
        bytes: sizes.iter().sum::<usize>() as f64,
        sizes,
    }
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    // LLM prefill (Qwen3.5-0.8B: d 1024, ffn 3584, vocab 248320; Gemma 4 E2B: d 1536)
    for b in [DType::Q4K, DType::Q6K, DType::Int8, DType::F16] {
        v.push(matmul_case(512, 3584, 1024, b, None));
        v.push(matmul_case(512, 1024, 3584, b, None));
        v.push(matmul_case(512, 6144, 1536, b, None));
    }
    // MoE (Qwen3.5-35B-A3B: 256 experts top-8, d 2048, expert ffn 512)
    for w in [DType::Q4K, DType::Q5K, DType::Q6K, DType::Int8] {
        v.push(matmul_id_case(1, 8, 256, 512, 2048, w));
        v.push(matmul_id_case(512, 8, 256, 512, 2048, w));
        v.push(matmul_id_case(512, 8, 256, 2048, 512, w));
    }
    v.push(attention_wide_case(512, 8, 256));
    v.push(attention_wide_case(512, 8, 512));
    v.push(gdn_case(1, 32, 128));
    v.push(gdn_case(512, 32, 128));
    for b in [DType::Int8, DType::F16] {
        // encoder
        v.push(matmul_case(1500, 768, 768, b, Some(add01())));
        v.push(matmul_case(1500, 3072, 768, b, Some(gelu_bias())));
        v.push(matmul_case(1500, 768, 3072, b, Some(add01())));
        // decoder
        v.push(matmul_case(1, 768, 768, b, Some(add01())));
        v.push(matmul_case(1, 3072, 768, b, Some(gelu_bias())));
        v.push(matmul_case(1, 768, 3072, b, Some(add01())));
        v.push(matmul_case(1, 51865, 768, b, None));
        v.push(matmul_case(3, 768, 768, b, Some(add01())));
    }
    // LLM decode (LFM3-1.7B-like shapes: d 2048, ffn 8192, vocab 65536)
    for b in [DType::Int4, DType::Int8, DType::F16, DType::Q4K, DType::Q6K] {
        v.push(matmul_case(1, 2048, 2048, b, None));
        v.push(matmul_case(1, 8192, 2048, b, None));
        v.push(matmul_case(1, 2048, 8192, b, None));
    }
    v.push(matmul_case(1, 65536, 2048, DType::Int8, None));
    for b in [DType::Int4, DType::Q4K, DType::Q6K] {
        v.push(matmul_case(512, 8192, 2048, b, None));
    }
    v.push(attention_case(1500, 1500, true));
    v.push(attention_case(1, 1500, false));
    v.push(attention_case(1, 24, false));
    v.push(layer_norm_case(1));
    v.push(layer_norm_case(1500));
    v
}

#[test]
#[ignore]
fn bench_kernels() {
    let Ok(list) = adapters() else { return };
    if list.is_empty() {
        return;
    }
    let be = VulkanBackend::new(0).unwrap();
    let filter = std::env::var("LAPID_BENCH_FILTER").unwrap_or_default();
    let reps = 20usize;
    println!("{:<34} {:>10} {:>10} {:>9} {:>9}", "case", "median us", "min us", "TFLOP/s", "GB/s");
    for c in cases() {
        if !filter.is_empty() && !c.name.contains(&filter) {
            continue;
        }
        let mut bufs: Vec<DeviceData> = c.sizes.iter().enumerate().map(|(bi, &s)| {
            let data: Vec<u8> = match &c.launch.kernel {
                // expert ids spread over all experts
                Kernel::MatMulId(_, d) if bi == 2 => (0..s / 4).flat_map(|i| (((i * 7919 + i / d.r * 31) % d.e) as i32).to_le_bytes()).collect(),
                Kernel::GatedDeltaNet(..) => (0..s / 4).flat_map(|i| ((((i * 2654435761usize) >> 13) % 1000) as f32 * 1e-4 - 0.05).to_le_bytes()).collect(),
                _ => (0..s).map(|i| ((i * 2654435761usize) >> 13) as u8 & 0x3f).collect(),
            };
            be.upload(&data).unwrap()
        }).collect();
        // cold inputs: enough copies that one pass touches >= 96 MiB (beyond L2)
        let cold_bytes: usize = c.cold.iter().map(|&i| c.sizes[i]).sum();
        let n = if cold_bytes > 0 { reps.max((96usize << 20).div_ceil(cold_bytes)) } else { reps };
        let mut launches: Vec<Launch> = Vec::with_capacity(n);
        for r in 0..n {
            let mut l = c.launch.clone();
            if r > 0 {
                for &i in &c.cold {
                    bufs.push(be.upload(&vec![0x11u8; c.sizes[i]]).unwrap());
                    l.inputs[i] = bufs.len() - 1;
                }
            }
            launches.push(l);
        }
        be.prepare(&launches[..1]).unwrap();
        let _ = be.time_launches(&launches[..2], &bufs).unwrap();
        let _ = be.time_launches(&launches, &bufs).unwrap();
        let mut t = be.time_launches(&launches, &bufs).unwrap();
        t.sort_by(|a, b| a.total_cmp(b));
        let med = t[t.len() / 2];
        println!(
            "{:<34} {:>10.2} {:>10.2} {:>9.2} {:>9.1}",
            c.name,
            med * 1e3,
            t[0] * 1e3,
            c.flops / (med * 1e-3) / 1e12,
            c.bytes / (med * 1e-3) / 1e9
        );
    }
}

/// Host cost of recording + submitting a decode-step-sized execute (200 tiny dependent
/// launches): wall time per execute vs. GPU time.
#[test]
#[ignore]
fn bench_host_overhead() {
    let Ok(list) = adapters() else { return };
    if list.is_empty() {
        return;
    }
    let be = VulkanBackend::new(0).unwrap();
    let c = matmul_case(1, 64, 64, DType::Int8, Some(add01()));
    let bufs: Vec<DeviceData> = c.sizes.iter().map(|&s| be.upload(&vec![1u8; s]).unwrap()).collect();
    let launches: Vec<Launch> = (0..200).map(|_| c.launch.clone()).collect();
    be.execute(&launches, &bufs).unwrap();
    let n = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        be.execute(&launches, &bufs).unwrap();
    }
    let wall = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
    let gpu: f64 = be.time_launches(&launches, &bufs).unwrap().iter().sum();
    println!("200 launches: wall {wall:.3} ms / execute, gpu {gpu:.3} ms, host+submit ~{:.3} ms", wall - gpu);
}
