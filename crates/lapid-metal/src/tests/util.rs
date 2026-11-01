//! GPU test helpers: backend, dtype encoding, launch runner, comparisons.

use lapid_compiler::{Backend, Kernel, Launch};
use lapid_tensor::{DType, DeviceData};
use half::{bf16, f16};

use crate::MetalBackend;

/// Backend on device 0, or None (test skipped) when there is no Metal device.
pub(crate) fn backend() -> Option<MetalBackend> {
    match MetalBackend::new(0) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping: no Metal device ({e})");
            None
        }
    }
}

/// Deterministic pseudo-random values in [-1, 1).
pub(crate) fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Round values to what `d` can hold (so references see the stored inputs).
pub(crate) fn round_to(d: DType, v: &[f32]) -> Vec<f32> {
    match d {
        DType::F16 => v.iter().map(|&x| f16::from_f32(x).to_f32()).collect(),
        DType::BF16 => v.iter().map(|&x| bf16::from_f32(x).to_f32()).collect(),
        DType::I32 => v.iter().map(|&x| x.trunc()).collect(),
        _ => v.to_vec(),
    }
}

pub(crate) fn encode(d: DType, v: &[f32]) -> Vec<u8> {
    match d {
        DType::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DType::F16 => v.iter().flat_map(|&x| f16::from_f32(x).to_le_bytes()).collect(),
        DType::BF16 => v.iter().flat_map(|&x| bf16::from_f32(x).to_le_bytes()).collect(),
        DType::I32 => v.iter().flat_map(|&x| (x as i32).to_le_bytes()).collect(),
        _ => panic!("encode: quantized, use lapid_quant::quantize"),
    }
}

pub(crate) fn decode(d: DType, b: &[u8], n: usize) -> Vec<f32> {
    match d {
        DType::F32 => b.as_chunks::<4>().0.iter().take(n).map(|c| f32::from_le_bytes(*c)).collect(),
        DType::F16 => b.as_chunks::<2>().0.iter().take(n).map(|c| f16::from_le_bytes(*c).to_f32()).collect(),
        DType::BF16 => b.as_chunks::<2>().0.iter().take(n).map(|c| bf16::from_le_bytes(*c).to_f32()).collect(),
        DType::I32 => b.as_chunks::<4>().0.iter().take(n).map(|c| i32::from_le_bytes(*c) as f32).collect(),
        _ => panic!("decode: quantized"),
    }
}

/// Upload `inputs`, run `kernel` into a fresh output of `out_bytes` (pre-filled with `fill`
/// bytes so untouched elements are detectable) and download it.
pub(crate) fn run_with(be: &MetalBackend, kernel: Kernel, inputs: &[Vec<u8>], out_init: Vec<u8>) -> Vec<u8> {
    let mut bufs: Vec<DeviceData> = inputs.iter().map(|b| be.upload(b).unwrap()).collect();
    bufs.push(be.upload(&out_init).unwrap());
    let launch = Launch { kernel, inputs: (0..inputs.len()).collect(), output: inputs.len() };
    be.execute(&[launch], &bufs).unwrap();
    be.download(&bufs[inputs.len()]).unwrap()
}

pub(crate) fn run(be: &MetalBackend, kernel: Kernel, inputs: &[Vec<u8>], out_bytes: usize) -> Vec<u8> {
    run_with(be, kernel, inputs, vec![0xAB; out_bytes])
}

/// Max |got - want| / max(1, max |want|) must be <= tol.
#[track_caller]
pub(crate) fn assert_close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(1.0f32, |m, &x| if x.is_finite() { m.max(x.abs()) } else { m });
    let mut worst = (0usize, 0.0f32);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if w.is_nan() {
            assert!(g.is_nan(), "{what}[{i}]: got {g}, want NaN");
            continue;
        }
        if w.is_infinite() {
            assert_eq!(g, w, "{what}[{i}]");
            continue;
        }
        let e = (g - w).abs() / scale;
        let e = if e.is_nan() { f32::INFINITY } else { e };
        if e > worst.1 {
            worst = (i, e);
        }
    }
    let (i, e) = worst;
    assert!(e <= tol, "{what}: max scaled error {e} at {i} (got {}, want {}), tol {tol}", got[i], want[i]);
}

/// Element (idx) of a strided view.
pub(crate) fn at(offset: usize, strides: &[isize], idx: &[usize]) -> usize {
    (offset as isize + idx.iter().zip(strides).map(|(&i, &s)| i as isize * s).sum::<isize>()) as usize
}

/// All multi-indices of `shape` in row-major order.
pub(crate) fn indices(shape: &[usize]) -> Vec<Vec<usize>> {
    let n: usize = shape.iter().product();
    let mut out = Vec::with_capacity(n);
    for mut lin in 0..n {
        let mut idx = vec![0; shape.len()];
        for d in (0..shape.len()).rev() {
            idx[d] = lin % shape[d];
            lin /= shape[d];
        }
        out.push(idx);
    }
    out
}

pub(crate) fn erf_ref(x: f64) -> f64 {
    // erf via the everywhere-convergent series (x small) / continued fraction-free erfc.
    let a = x.abs();
    let y = if a < 6.0 {
        let x2 = a * a;
        let mut term = a;
        let mut sum = a;
        let mut n = 0.0;
        loop {
            n += 1.0;
            term *= 2.0 * x2 / (2.0 * n + 1.0);
            sum += term;
            if term < 1e-17 * sum {
                break;
            }
        }
        std::f64::consts::FRAC_2_SQRT_PI * (-x2).exp() * sum
    } else {
        1.0
    };
    if x < 0.0 { -y } else { y }
}

pub(crate) fn gelu_ref(x: f64) -> f64 {
    0.5 * x * (1.0 + erf_ref(x * std::f32::consts::FRAC_1_SQRT_2 as f64))
}

/// Run `kernel` on any backend (the CPU backend is the exact reference for integer / index
/// kernels): upload `inputs`, an output initialised to `out_init`, execute, download.
pub(crate) fn run_on(be: &dyn Backend, kernel: Kernel, inputs: &[Vec<u8>], out_init: &[u8]) -> lapid_tensor::Result<Vec<u8>> {
    let mut bufs: Vec<DeviceData> = inputs.iter().map(|b| be.upload(b)).collect::<lapid_tensor::Result<_>>()?;
    bufs.push(be.upload(out_init)?);
    let launch = Launch { kernel, inputs: (0..inputs.len()).collect(), output: inputs.len() };
    be.execute(&[launch], &bufs)?;
    be.download(&bufs[inputs.len()])
}

/// Same kernel on Metal and on the CPU backend; returns (metal, cpu) output bytes.
pub(crate) fn both(be: &MetalBackend, kernel: Kernel, inputs: &[Vec<u8>], out_init: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let cpu = lapid_cpu::CpuBackend::new();
    let m = run_on(be, kernel.clone(), inputs, out_init).unwrap();
    let c = run_on(&cpu, kernel, inputs, out_init).unwrap();
    (m, c)
}

pub(crate) fn words(b: &[u8]) -> Vec<u32> {
    b.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect()
}

pub(crate) fn from_words(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}
