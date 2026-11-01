//! Test helpers: CPU evaluation, deterministic data and naive reference math.

use lapid::{Device, Tensor};

/// Evaluate `t` on the CPU and read it back as f32.
pub fn eval(t: &Tensor) -> Vec<f32> {
    let dev = Device::cpu();
    let out = dev.eval(&[t]).expect("eval");
    dev.to_vec_f32(&out[0]).expect("read back")
}

/// Deterministic pseudo-random values in [-scale, scale).
pub fn rand_vec(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// f32 tensor of deterministic random values.
pub fn rand_tensor(shape: &[usize], seed: u64) -> (Tensor, Vec<f32>) {
    let n = shape.iter().product();
    let v = rand_vec(n, seed, 1.0);
    (Tensor::from_f32(&v, shape).expect("tensor"), v)
}

/// a [m, k] @ b^T where b is [n, k] -> [m, n].
pub fn matmul_t(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for t in 0..k {
                acc += a[i * k + t] as f64 * b[j * k + t] as f64;
            }
            out[i * n + j] = acc as f32;
        }
    }
    out
}

/// Assert element-wise |a - b| <= tol * max(1, |b|).
#[track_caller]
pub fn assert_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let ok = (g - w).abs() <= tol * w.abs().max(1.0) || (g.is_infinite() && g == w);
        assert!(ok, "index {i}: got {g}, want {w} (tol {tol})");
    }
}
