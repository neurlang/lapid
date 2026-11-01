//! End-to-end checks of the round-5 engine work through `Device` (compiler + backend) at real
//! model shapes: native `rms_norm`, attention with head dim 512 (Gemma 4 full-attention
//! layers), f32 x f32 matmul exactness, and MoE `matmul_id` at Qwen3.5-35B-A3B shapes.
//!
//! Runs on the devices in `LAPID_OPS_DEVICES` (comma separated, default `cpu`; e.g.
//! `cpu,cuda,vulkan`); unavailable devices are skipped.

use lapid::quant::{self, Format};
use lapid::{DType, Device, HostData, Tensor};

fn devices() -> Vec<Device> {
    let names = std::env::var("LAPID_OPS_DEVICES").unwrap_or_else(|_| "cpu".into());
    names.split(',').filter_map(|n| Device::from_name(n.trim()).ok()).collect()
}

fn noise(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

fn t(v: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(v, shape).unwrap()
}

/// Max |got - want| relative to max |want|.
fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    got.iter().zip(want).fold(0f32, |m, (g, w)| m.max((g - w).abs())) / scale
}

fn run(dev: &Device, y: &Tensor) -> Vec<f32> {
    let out = dev.eval(&[y]).unwrap();
    dev.to_vec_f32(&out[0]).unwrap()
}

#[test]
fn rms_norm_matches_reference() {
    for dev in devices() {
        for &(rows, cols) in &[(1usize, 1024usize), (37, 2048), (512, 1024), (3, 256), (5, 100)] {
            let x = noise(rows * cols, 3 + cols as u64, 2.0);
            let r = noise(rows * cols, 5, 1.0);
            let w = noise(cols, 7, 1.5);
            // Residual add as a lazy prologue (the usual `norm(h + attn)` shape).
            let h: Vec<f32> = x.iter().zip(&r).map(|(a, b)| a + b).collect();
            let mut want = vec![0f32; rows * cols];
            for i in 0..rows {
                let row = &h[i * cols..(i + 1) * cols];
                let ms = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / cols as f64;
                let inv = 1.0 / (ms + 1e-6).sqrt();
                for c in 0..cols {
                    want[i * cols + c] = (f64::from(row[c]) * inv * f64::from(w[c])) as f32;
                }
            }
            let xt = t(&x, &[rows, cols]).add(&t(&r, &[rows, cols])).unwrap();
            let y = xt.rms_norm(&t(&w, &[cols]), 1e-6).unwrap();
            let got = run(&dev, &y);
            let e = rel_err(&got, &want);
            assert!(e < 1e-5, "{}: rms_norm f32 [{rows}, {cols}] rel {e}", dev.name());
            // f16 activations and weight.
            let y = xt.cast(DType::F16).unwrap().rms_norm(&t(&w, &[cols]).cast(DType::F16).unwrap(), 1e-6).unwrap();
            assert_eq!(y.dtype(), DType::F16);
            let got = run(&dev, &y);
            let e = rel_err(&got, &want);
            assert!(e < 5e-3, "{}: rms_norm f16 [{rows}, {cols}] rel {e}", dev.name());
        }
    }
}

/// softmax(q k^T * scale [+ causal]) v on the host, [B, H, S, D] row-major.
fn attention_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    (b, h, sq, sk, d): (usize, usize, usize, usize, usize),
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut out = vec![0f32; b * h * sq * d];
    for bh in 0..b * h {
        for i in 0..sq {
            let qi = &q[(bh * sq + i) * d..(bh * sq + i + 1) * d];
            let lim = if causal { i + (sk - sq) + 1 } else { sk };
            let s: Vec<f64> = (0..lim)
                .map(|j| {
                    let kj = &k[(bh * sk + j) * d..(bh * sk + j + 1) * d];
                    qi.iter().zip(kj).map(|(a, c)| f64::from(*a) * f64::from(*c)).sum::<f64>() * f64::from(scale)
                })
                .collect();
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for c in 0..d {
                let acc: f64 = (0..lim).map(|j| e[j] * f64::from(v[(bh * sk + j) * d + c])).sum();
                out[(bh * sq + i) * d + c] = (acc / z) as f32;
            }
        }
    }
    out
}

#[test]
fn attention_head_dim_512() {
    for dev in devices() {
        // Prefill (causal), decode (sq = 1 over a long cache), and a non-causal block.
        for &(b, h, sq, sk, causal) in &[(1usize, 2usize, 37usize, 37usize, true), (1, 8, 1, 700, true), (2, 1, 5, 9, false)] {
            let d = 512;
            let q = noise(b * h * sq * d, 11, 1.0);
            let k = noise(b * h * sk * d, 12, 1.0);
            let v = noise(b * h * sk * d, 13, 1.0);
            let scale = 1.0;
            let want = attention_ref(&q, &k, &v, (b, h, sq, sk, d), scale, causal);
            let y = Tensor::attention(
                &t(&q, &[b, h, sq, d]),
                &t(&k, &[b, h, sk, d]),
                &t(&v, &[b, h, sk, d]),
                None,
                scale,
                causal,
            )
            .unwrap();
            let got = run(&dev, &y);
            let e = rel_err(&got, &want);
            assert!(e < 2e-3, "{}: attention d=512 b{b} h{h} sq{sq} sk{sk} rel {e}", dev.name());
            let y = Tensor::attention(
                &t(&q, &[b, h, sq, d]).cast(DType::F16).unwrap(),
                &t(&k, &[b, h, sk, d]).cast(DType::F16).unwrap(),
                &t(&v, &[b, h, sk, d]).cast(DType::F16).unwrap(),
                None,
                scale,
                causal,
            )
            .unwrap();
            let got = run(&dev, &y);
            let e = rel_err(&got, &want);
            assert!(e < 2e-2, "{}: attention f16 d=512 b{b} h{h} sq{sq} sk{sk} rel {e}", dev.name());
        }
    }
}

#[test]
fn f32_matmul_is_f32_exact() {
    for dev in devices() {
        for &(m, n, k) in &[(64usize, 96usize, 4096usize), (300, 256, 1024), (1, 512, 4096)] {
            let a = noise(m * k, 21, 1.0);
            let bw = noise(n * k, 22, 1.0);
            let mut want = vec![0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let s: f64 = (0..k).map(|c| f64::from(a[i * k + c]) * f64::from(bw[j * k + c])).sum();
                    want[i * n + j] = s as f32;
                }
            }
            let y = t(&a, &[m, k]).matmul_t(&t(&bw, &[n, k])).unwrap();
            let got = run(&dev, &y);
            let e = rel_err(&got, &want);
            assert!(e < 1e-4, "{}: f32 matmul {m}x{n}x{k} rel {e}", dev.name());
        }
    }
}

/// Qwen3.5-35B-A3B expert shapes: E = 256, top-8, hidden 2048, expert width 512.
#[test]
fn matmul_id_real_moe_shapes() {
    let (e, n, k, r) = (256usize, 512usize, 2048usize, 8usize);
    for (fmt, dt) in [(Format::Q4K, DType::Q4K), (Format::Q6K, DType::Q6K), (Format::Int8, DType::Int8)] {
        // Quantize 8 distinct experts once and tile them to E (quantizing 256 x 1M weights per
        // format would dominate the test); routing still spans the whole stack.
        let distinct = 8;
        let per = quant::layout(fmt, n, k).total;
        let mut uniq = Vec::with_capacity(distinct * per);
        let mut udeq = Vec::with_capacity(distinct * n * k);
        for i in 0..distinct {
            let w = noise(n * k, 31 + i as u64, 1.0);
            let q = quant::quantize(fmt, &w, n, k);
            udeq.extend(quant::dequantize(fmt, &q, n, k));
            uniq.extend(q);
        }
        let mut bytes = Vec::with_capacity(e * per);
        for i in 0..e {
            let u = i % distinct;
            bytes.extend_from_slice(&uniq[u * per..(u + 1) * per]);
        }
        for dev in devices() {
            let wt = dev.upload(&Tensor::from_host(HostData::from_vec(bytes.clone()), &[e, n, k], dt).unwrap()).unwrap();
            for tt in [1usize, 3, 40] {
                let x = noise(tt * k, 32 + tt as u64, 1.0);
                let ids: Vec<i32> = (0..tt * r).map(|i| ((i * 37 + i / r * 11) % e) as i32).collect();
                let mut want = vec![0f32; tt * r * n];
                for ti in 0..tt {
                    for ri in 0..r {
                        let ex = ids[ti * r + ri] as usize;
                        for j in 0..n {
                            let row = &udeq[((ex % distinct) * n + j) * k..((ex % distinct) * n + j + 1) * k];
                            want[(ti * r + ri) * n + j] =
                                row.iter().zip(&x[ti * k..(ti + 1) * k]).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>() as f32;
                        }
                    }
                }
                let y = t(&x, &[tt, k]).matmul_id(&wt, &Tensor::from_i32(&ids, &[tt, r]).unwrap()).unwrap();
                let got = run(&dev, &y);
                let e = rel_err(&got, &want);
                // Kernels may quantize activations to int8 (CPU) or stage f16 tiles (GPU GEMM).
                assert!(e < 2e-2, "{}: matmul_id {} t={tt} rel {e}", dev.name(), dt.name());
            }
        }
    }
}

#[test]
fn raw_bytes_device_list_and_log_hook() {
    for dev in devices() {
        let x = t(&[1.0, -2.5, 3.25, 0.5], &[2, 2]).cast(DType::F16).unwrap();
        let y = x.mul_scalar(2.0).unwrap();
        let bytes = dev.to_vec_bytes(&y).unwrap();
        let want: Vec<u8> = [2.0f32, -5.0, 6.5, 1.0].iter().flat_map(|v| half_bits(*v).to_le_bytes()).collect();
        assert_eq!(bytes, want, "{}", dev.name());
    }
    let list = Device::list_info();
    assert_eq!(list.first().map(|d| d.name.as_str()), Some("cpu"));
    assert!(list.iter().all(|d| Device::from_name(&d.name).is_ok()), "{list:?}");
    let seen = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let s2 = seen.clone();
    lapid::set_log_hook(Some(Box::new(move |_| {
        s2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    })));
    let dev = Device::cpu();
    dev.set_debug(true);
    let _ = dev.to_vec_f32(&t(&[1.0, 2.0], &[2]).add_scalar(1.0).unwrap()).unwrap();
    dev.set_debug(false);
    lapid::set_log_hook(None);
    assert!(seen.load(std::sync::atomic::Ordering::Relaxed) > 0);
}

/// f32 -> IEEE half bits (exact for the small values used here).
fn half_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    if v == 0.0 {
        return sign;
    }
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = ((b >> 13) & 0x3ff) as u16;
    sign | ((exp as u16) << 10) | mant
}
