//! End-to-end checks of the round-3 ops through `Device` (compiler + backend): `matmul_id`
//! (float and quantized expert stacks incl. ggml K-quants), K-quant `matmul_t` / `gather`,
//! and `gated_delta_net`, against host references and the reference decompositions.
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

const FORMATS: [(Format, DType); 5] =
    [(Format::Int8, DType::Int8), (Format::Int4, DType::Int4), (Format::Q4K, DType::Q4K), (Format::Q5K, DType::Q5K), (Format::Q6K, DType::Q6K)];

/// `e` quantized `[n, k]` matrices back to back + their dequantized values.
fn quant_stack(fmt: Format, e: usize, n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::new();
    let mut deq = Vec::new();
    for i in 0..e {
        let w = noise(n * k, seed + i as u64, 1.0);
        let q = quant::quantize(fmt, &w, n, k);
        deq.extend(quant::dequantize(fmt, &q, n, k));
        bytes.extend(q);
    }
    (bytes, deq)
}

/// out[t, r, :] = W[ids[t, r]] @ x[t] on the host.
fn matmul_id_ref(x: &[f32], w: &[f32], ids: &[i32], (tt, r, n, k): (usize, usize, usize, usize)) -> Vec<f32> {
    let mut out = vec![0f32; tt * r * n];
    for ti in 0..tt {
        for ri in 0..r {
            let e = ids[ti * r + ri] as usize;
            for j in 0..n {
                let row = &w[(e * n + j) * k..(e * n + j + 1) * k];
                out[(ti * r + ri) * n + j] = row.iter().zip(&x[ti * k..(ti + 1) * k]).map(|(a, b)| a * b).sum();
            }
        }
    }
    out
}

#[test]
fn matmul_id_float_and_decomposition() {
    for dev in devices() {
        let (tt, r, e, n, k) = (5, 2, 4, 24, 40);
        let x = noise(tt * k, 1, 1.0);
        let w = noise(e * n * k, 2, 0.5);
        let ids: Vec<i32> = (0..tt * r).map(|i| ((i * 3 + 1) % e) as i32).collect();
        let want = matmul_id_ref(&x, &w, &ids, (tt, r, n, k));
        let y = t(&x, &[tt, k]).matmul_id(&t(&w, &[e, n, k]), &Tensor::from_i32(&ids, &[tt, r]).unwrap()).unwrap();
        assert_eq!(y.shape(), [tt, r, n]);
        let d = lapid_tensor_decompose(&y);
        let outs = dev.eval(&[&y, &d]).unwrap();
        let got = dev.to_vec_f32(&outs[0]).unwrap();
        assert!(rel_err(&got, &want) < 1e-5, "{}: matmul_id f32 rel {}", dev.name(), rel_err(&got, &want));
        let got = dev.to_vec_f32(&outs[1]).unwrap();
        assert!(rel_err(&got, &want) < 1e-5, "{}: decomposition rel {}", dev.name(), rel_err(&got, &want));
        // f16 experts.
        let wh = t(&w, &[e, n, k]).cast(DType::F16).unwrap();
        let wh = dev.eval(&[&wh]).unwrap().pop().unwrap();
        let y = t(&x, &[tt, k]).matmul_id(&wh, &Tensor::from_i32(&ids, &[tt, r]).unwrap()).unwrap();
        let got = dev.to_vec_f32(&dev.eval(&[&y]).unwrap()[0]).unwrap();
        assert!(rel_err(&got, &want) < 2e-3, "{}: matmul_id f16 rel {}", dev.name(), rel_err(&got, &want));
    }
}

/// The reference decomposition of a composite node, as a plain primitive graph.
fn lapid_tensor_decompose(node: &Tensor) -> Tensor {
    lapid_tensor::decompose::decompose(node).unwrap().expect("composite node")
}

#[test]
fn matmul_id_quantized() {
    for dev in devices() {
        for (fmt, dt) in FORMATS {
            for &(tt, r) in &[(1usize, 4usize), (12, 2)] {
                let (e, n, k) = (6, 20, 512);
                let (bytes, deq) = quant_stack(fmt, e, n, k, 10);
                let w = Tensor::from_host(HostData::from_vec(bytes), &[e, n, k], dt).unwrap();
                let x = noise(tt * k, 3, 1.0);
                let ids: Vec<i32> = (0..tt * r).map(|i| ((i * 5 + 2) % e) as i32).collect();
                let want = matmul_id_ref(&x, &deq, &ids, (tt, r, n, k));
                let y = t(&x, &[tt, k]).matmul_id(&w, &Tensor::from_i32(&ids, &[tt, r]).unwrap()).unwrap();
                let got = dev.to_vec_f32(&dev.eval(&[&y]).unwrap()[0]).unwrap();
                let err = rel_err(&got, &want);
                assert!(err < 2e-2, "{} {fmt:?} t={tt}: matmul_id rel {err}", dev.name());
            }
        }
    }
}

#[test]
fn kquant_matmul_and_gather() {
    for dev in devices() {
        for (fmt, dt) in FORMATS {
            let (n, k) = (36, 768);
            let (bytes, deq) = quant_stack(fmt, 1, n, k, 20);
            let w = Tensor::from_host(HostData::from_vec(bytes), &[n, k], dt).unwrap();
            for m in [1usize, 3, 40] {
                let x = noise(m * k, 4, 1.0);
                let mut want = vec![0f32; m * n];
                for i in 0..m {
                    for j in 0..n {
                        want[i * n + j] = (0..k).map(|p| x[i * k + p] * deq[j * k + p]).sum();
                    }
                }
                let y = t(&x, &[m, k]).matmul_t(&w).unwrap();
                let got = dev.to_vec_f32(&dev.eval(&[&y]).unwrap()[0]).unwrap();
                let err = rel_err(&got, &want);
                assert!(err < 2e-2, "{} {fmt:?} m={m}: matmul rel {err}", dev.name());
            }
            let ids = [3i32, 0, 35, 3];
            let g = w.gather(&Tensor::from_i32(&ids, &[4]).unwrap()).unwrap();
            let got = dev.to_vec_f32(&dev.eval(&[&g]).unwrap()[0]).unwrap();
            let want: Vec<f32> = ids.iter().flat_map(|&i| deq[i as usize * k..(i as usize + 1) * k].to_vec()).collect();
            assert!(rel_err(&got, &want) < 1e-5, "{} {fmt:?}: gather", dev.name());
        }
    }
}

#[test]
fn matmul_id_index_out_of_range() {
    for dev in devices() {
        let w = t(&noise(2 * 4 * 8, 1, 1.0), &[2, 4, 8]);
        let y = t(&noise(8, 2, 1.0), &[1, 8]).matmul_id(&w, &Tensor::from_i32(&[2], &[1, 1]).unwrap()).unwrap();
        let err = dev.eval(&[&y]).err().map(|e| e.code);
        assert_eq!(err, Some("index_out_of_range"), "{}", dev.name());
    }
}

/// f64 reference of the gated delta rule (ggml semantics); output [B, H, T + Dk, Dv].
#[allow(clippy::too_many_arguments)]
fn gdn_ref(q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32], s0: &[f32], (b, tt, h, dk, dv): (usize, usize, usize, usize, usize), scale: f64) -> Vec<f32> {
    let mut out = vec![0f32; b * h * (tt + dk) * dv];
    for bi in 0..b {
        for hi in 0..h {
            let bh = bi * h + hi;
            let mut s: Vec<f64> = s0[bh * dk * dv..(bh + 1) * dk * dv].iter().map(|&x| x as f64).collect();
            for ti in 0..tt {
                let r = (bi * tt + ti) * h + hi;
                let dec = (g[r] as f64).exp();
                s.iter_mut().for_each(|x| *x *= dec);
                let u: Vec<f64> = (0..dv).map(|j| (0..dk).map(|i| s[i * dv + j] * k[r * dk + i] as f64).sum()).collect();
                for i in 0..dk {
                    for j in 0..dv {
                        s[i * dv + j] += k[r * dk + i] as f64 * beta[r] as f64 * (v[r * dv + j] as f64 - u[j]);
                    }
                }
                for j in 0..dv {
                    let o: f64 = (0..dk).map(|i| s[i * dv + j] * q[r * dk + i] as f64 * scale).sum();
                    out[(bh * (tt + dk) + ti) * dv + j] = o as f32;
                }
            }
            for (i, &x) in s.iter().enumerate() {
                out[(bh * (tt + dk) + tt) * dv + i] = x as f32;
            }
        }
    }
    out
}

#[test]
fn gated_delta_net_matches_reference_and_decomposition() {
    for dev in devices() {
        for &(b, tt, h, dk, dv) in &[(1usize, 1usize, 4usize, 16usize, 16usize), (2, 6, 3, 32, 24), (1, 3, 2, 128, 128)] {
            let q = noise(b * tt * h * dk, 1, 1.0);
            let k = noise(b * tt * h * dk, 2, 0.3);
            let v = noise(b * tt * h * dv, 3, 1.0);
            let g: Vec<f32> = noise(b * tt * h, 4, 1.0).iter().map(|x| -x.abs()).collect();
            let beta: Vec<f32> = noise(b * tt * h, 5, 1.0).iter().map(|x| x.abs()).collect();
            let s0 = noise(b * h * dk * dv, 6, 0.2);
            let scale = 1.0 / (dk as f32).sqrt();
            let want = gdn_ref(&q, &k, &v, &g, &beta, &s0, (b, tt, h, dk, dv), scale as f64);
            let y = Tensor::gated_delta_net(
                &t(&q, &[b, tt, h, dk]),
                &t(&k, &[b, tt, h, dk]),
                &t(&v, &[b, tt, h, dv]),
                &t(&g, &[b, tt, h]),
                &t(&beta, &[b, tt, h]),
                &t(&s0, &[b, h, dk, dv]),
                scale,
            )
            .unwrap();
            assert_eq!(y.shape(), [b, h, tt + dk, dv]);
            let d = lapid_tensor_decompose(&y);
            let outs = dev.eval(&[&y, &d]).unwrap();
            let got = dev.to_vec_f32(&outs[0]).unwrap();
            let err = rel_err(&got, &want);
            assert!(err < 1e-4, "{} {:?}: gdn rel {err}", dev.name(), (b, tt, h, dk, dv));
            let got = dev.to_vec_f32(&outs[1]).unwrap();
            let err = rel_err(&got, &want);
            assert!(err < 1e-4, "{} {:?}: gdn decomposition rel {err}", dev.name(), (b, tt, h, dk, dv));
        }
    }
}
