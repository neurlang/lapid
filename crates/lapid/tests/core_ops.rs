//! End-to-end checks of the round-2 core ops through `Device` (compiler + backend): in-place
//! `write_slice` (KV caches, fused matmul writes), attention over a prefix of a larger buffer,
//! `topk` / `argmax` / `gather_axis`, exact i32 arithmetic and index range errors.
//!
//! Runs on the devices in `LAPID_OPS_DEVICES` (comma separated, default `cpu`; e.g.
//! `cpu,cuda,vulkan`); unavailable devices are skipped.

use lapid::compiler::{self, CompileOptions, Kernel};
use lapid::{DType, Device, Op, Tensor};

fn devices() -> Vec<Device> {
    let names = std::env::var("LAPID_OPS_DEVICES").unwrap_or_else(|_| "cpu".into());
    names.split(',').filter_map(|n| Device::from_name(n.trim()).ok()).collect()
}

fn ramp(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..n).map(f).collect()
}

fn t(v: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(v, shape).unwrap()
}

/// A realized device tensor not owned by the upload cache.
fn realized(dev: &Device, v: &[f32], shape: &[usize]) -> Tensor {
    let x = t(v, shape).add_scalar(0.0).unwrap().mul_scalar(1.0).unwrap().contiguous().unwrap();
    dev.eval(&[&x]).unwrap().pop().unwrap()
}

fn same_buffer(a: &Tensor, b: &Tensor) -> bool {
    match (a.op(), b.op()) {
        (Op::Device(x), Op::Device(y)) => std::sync::Arc::ptr_eq(x, y),
        _ => false,
    }
}

fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol * (1.0 + w.abs()), "{what}[{i}]: {g} vs {w}");
    }
}

#[test]
fn write_slice_in_place_and_functional() {
    for dev in devices() {
        let (cap, c) = (6, 3);
        let base = ramp(cap * c, |i| i as f32);
        let dest = realized(&dev, &base, &[cap, c]);
        let src = t(&[100.0, 101.0, 102.0, 103.0, 104.0, 105.0], &[2, c]);
        let upd = dest.write_slice(0, 3, &src).unwrap();
        let out = dev.eval(&[&upd]).unwrap().pop().unwrap();
        assert!(same_buffer(&out, &dest), "{}: write_slice was not in place", dev.name());
        let mut want = base.clone();
        want[9..15].copy_from_slice(&[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);
        assert_eq!(dev.to_vec_f32(&out).unwrap(), want);
        assert_eq!(dev.to_vec_f32(&dest).unwrap(), want, "old handle sees the update");

        // dest also read elsewhere in the graph → functional copy; the old values stay.
        let d2 = realized(&dev, &base, &[cap, c]);
        let upd = d2.write_slice(0, 0, &src).unwrap();
        let sum = d2.sum(0, false).unwrap();
        let outs = dev.eval(&[&upd, &sum]).unwrap();
        assert!(!same_buffer(&outs[0], &d2));
        assert_eq!(dev.to_vec_f32(&d2).unwrap(), base);
        let mut want = base.clone();
        want[..6].copy_from_slice(&[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);
        assert_eq!(dev.to_vec_f32(&outs[0]).unwrap(), want);
        assert_eq!(dev.to_vec_f32(&outs[1]).unwrap(), [45.0, 51.0, 57.0]);

        // Constant dest (first KV allocation): zeros + slice.
        let z = Tensor::zeros(&[4, 2], DType::F32).write_slice(0, 1, &t(&[1.0, 2.0], &[1, 2])).unwrap();
        let z = dev.eval(&[&z]).unwrap().pop().unwrap();
        assert_eq!(dev.to_vec_f32(&z).unwrap(), [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0]);

        // Chained in-place writes along a middle axis with a strided (permuted) source.
        let d3 = realized(&dev, &vec![0.0; 2 * 5 * 3], &[2, 5, 3]);
        let a = t(&ramp(6, |i| i as f32 + 1.0), &[3, 2]).permute(&[1, 0]).unwrap().unsqueeze(1).unwrap(); // [2,1,3]
        let b = t(&ramp(6, |i| -(i as f32) - 1.0), &[2, 1, 3]);
        let u = d3.write_slice(1, 4, &a).unwrap().write_slice(1, 0, &b).unwrap();
        let out = dev.eval(&[&u]).unwrap().pop().unwrap();
        assert!(same_buffer(&out, &d3));
        let got = dev.to_vec_f32(&out).unwrap();
        assert_eq!(&got[12..15], &[1.0, 3.0, 5.0]);
        assert_eq!(&got[27..30], &[2.0, 4.0, 6.0]);
        assert_eq!(&got[0..3], &[-1.0, -2.0, -3.0]);
        assert_eq!(&got[15..18], &[-4.0, -5.0, -6.0]);
    }
}

/// K projection written straight into a `[B, cap, H, D]` cache: one matmul launch whose
/// output view is the cache slice, no copy.
#[test]
fn matmul_writes_into_cache_slice() {
    let (b, n, dm, h, d, cap, start) = (1, 3, 8, 2, 4, 10, 5);
    let x: Vec<f32> = ramp(b * n * dm, |i| ((i * 7) % 11) as f32 * 0.1 - 0.5);
    let w: Vec<f32> = ramp(h * d * dm, |i| ((i * 5) % 13) as f32 * 0.05 - 0.3);
    let bias: Vec<f32> = ramp(h * d, |i| i as f32 * 0.01);
    for dev in devices() {
        let cache = realized(&dev, &vec![0.5; b * cap * h * d], &[b, cap, h, d]);
        let xt = t(&x, &[b, n, dm]);
        let proj = xt.matmul_t(&t(&w, &[h * d, dm])).unwrap().add(&t(&bias, &[h * d])).unwrap();
        // The usual head split: [B, n, H, D] → [B, H, n, D] → back to cache order.
        let k = proj.reshape(&[b, n, h, d]).unwrap().permute(&[0, 2, 1, 3]).unwrap();
        let upd = cache.write_slice(1, start, &k.permute(&[0, 2, 1, 3]).unwrap()).unwrap();
        let sched = compiler::compile(std::slice::from_ref(&upd), &CompileOptions::default()).unwrap();
        assert_eq!(sched.launches.len(), 1, "{}", compiler::dump(&sched));
        let Kernel::MatMul(_, dims) = &sched.launches[0].kernel else { panic!("expected a matmul") };
        assert_eq!(dims.out.offset, start * h * d);
        let out = dev.eval(&[&upd]).unwrap().pop().unwrap();
        assert!(same_buffer(&out, &cache));
        let got = dev.to_vec_f32(&out).unwrap();
        for s in 0..cap {
            for j in 0..h * d {
                let want = if (start..start + n).contains(&s) {
                    let r = s - start;
                    bias[j] + (0..dm).map(|q| x[r * dm + q] * w[j * dm + q]).sum::<f32>()
                } else {
                    0.5
                };
                let g = got[s * h * d + j];
                assert!((g - want).abs() < 1e-4, "{}: s={s} j={j}: {g} vs {want}", dev.name());
            }
        }
    }
}

/// Attention over the valid prefix `[0, len)` of a larger `[B, cap, H, D]` buffer equals
/// attention over the same keys stored contiguously.
#[test]
fn attention_over_cache_prefix() {
    let (b, h, d, cap, len, sq) = (1, 3, 16, 40, 23, 2);
    let kv: Vec<f32> = ramp(b * cap * h * d, |i| ((i * 37) % 101) as f32 / 101.0 - 0.5);
    let vv: Vec<f32> = ramp(b * cap * h * d, |i| ((i * 53) % 97) as f32 / 97.0 - 0.5);
    let q: Vec<f32> = ramp(b * h * sq * d, |i| ((i * 29) % 89) as f32 / 89.0 - 0.5);
    for dev in devices() {
        let kc = realized(&dev, &kv, &[b, cap, h, d]);
        let vc = realized(&dev, &vv, &[b, cap, h, d]);
        let view = |c: &Tensor| c.slice(1, 0, len).unwrap().permute(&[0, 2, 1, 3]).unwrap();
        let qt = t(&q, &[b, h, sq, d]);
        let scale = 1.0 / (d as f32).sqrt();
        let a = Tensor::attention(&qt, &view(&kc), &view(&vc), None, scale, true).unwrap();
        let dense = |v: &[f32]| t(v, &[b, cap, h, d]).slice(1, 0, len).unwrap().permute(&[0, 2, 1, 3]).unwrap().contiguous().unwrap();
        let r = Tensor::attention(&qt, &dense(&kv), &dense(&vv), None, scale, true).unwrap();
        let outs = dev.eval(&[&a, &r]).unwrap();
        close(&dev.to_vec_f32(&outs[0]).unwrap(), &dev.to_vec_f32(&outs[1]).unwrap(), 1e-5, dev.name());
    }
}

#[test]
fn topk_argmax_gather_axis() {
    let (rows, cols) = (3, 37);
    let x: Vec<f32> = ramp(rows * cols, |i| (((i * 7919) % 61) as f32 - 30.0) * 0.5);
    for dev in devices() {
        let xt = t(&x, &[rows, cols]);
        let (vals, idx) = xt.topk(5, -1).unwrap();
        let am = xt.argmax(1).unwrap();
        // Along a non-last axis too.
        let idx0 = xt.topk_indices(2, 0).unwrap();
        let outs = dev.eval(&[&vals, &idx, &am, &idx0]).unwrap();
        let (gv, gi) = (dev.to_vec_f32(&outs[0]).unwrap(), dev.to_vec_i32(&outs[1]).unwrap());
        let gam = dev.to_vec_i32(&outs[2]).unwrap();
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let mut order: Vec<usize> = (0..cols).collect();
            order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
            for j in 0..5 {
                assert_eq!(gi[r * 5 + j] as usize, order[j], "{} topk row {r}", dev.name());
                assert_eq!(gv[r * 5 + j], row[order[j]]);
            }
            assert_eq!(gam[r] as usize, order[0]);
        }
        let g0 = dev.to_vec_i32(&outs[3]).unwrap();
        for c in 0..cols {
            let mut order: Vec<usize> = (0..rows).collect();
            order.sort_by(|&a, &b| x[b * cols + c].partial_cmp(&x[a * cols + c]).unwrap().then(a.cmp(&b)));
            assert_eq!([g0[c] as usize, g0[cols + c] as usize], [order[0], order[1]], "column {c}");
        }

        // take_along_axis on a middle axis with an i32 payload (exact copy).
        let payload = Tensor::from_i32(&(0..24).map(|i| i * 1_000_003).collect::<Vec<_>>(), &[2, 3, 4]).unwrap();
        let ids = Tensor::from_i32(&[2, 0, 1, 1, 0, 2, 2, 2], &[2, 1, 4]).unwrap();
        let g = dev.eval(&[&payload.gather_axis(1, &ids).unwrap()]).unwrap().pop().unwrap();
        let want: Vec<i32> =
            (0..8).map(|i| { let (o, c) = (i / 4, i % 4); let s = [2, 0, 1, 1, 0, 2, 2, 2][i]; (o * 12 + s * 4 + c) as i32 * 1_000_003 }).collect();
        assert_eq!(dev.to_vec_i32(&g).unwrap(), want);

        // Out-of-range indices are an error, for gather and gather_axis.
        let bad = Tensor::from_i32(&[0, 3, 1, 1], &[1, 1, 4]).unwrap();
        let e = dev.eval(&[&payload.slice(0, 0, 1).unwrap().gather_axis(1, &bad).unwrap()]).unwrap_err();
        assert_eq!(e.code, "index_out_of_range", "{}: {e:?}", dev.name());
        let table = t(&ramp(8, |i| i as f32), &[4, 2]);
        let e = dev.eval(&[&table.gather(&Tensor::from_i32(&[1, -1], &[2]).unwrap()).unwrap()]).unwrap_err();
        assert_eq!(e.code, "index_out_of_range", "{}: {e:?}", dev.name());
    }
}

#[test]
fn exact_integer_arithmetic() {
    let a: Vec<i32> = vec![16_777_217, -5, 7, i32::MAX, 123_456_789, -98_765_432, 0, 1 << 30];
    let b: Vec<i32> = vec![3, 2, -2, 1, 1000, 7, 5, 3];
    for dev in devices() {
        let (ta, tb) = (Tensor::from_i32(&a, &[8]).unwrap(), Tensor::from_i32(&b, &[8]).unwrap());
        let sum = ta.add(&tb).unwrap();
        let mixed = ta.mul(&tb).unwrap().rem(&Tensor::from_i32(&[1_000_007], &[1]).unwrap()).unwrap();
        let bits = ta.bitxor(&tb).unwrap().shr(&Tensor::from_i32(&[1], &[1]).unwrap()).unwrap().bitand(&ta.shl(&tb).unwrap()).unwrap();
        let div = ta.div(&tb).unwrap();
        // f32 → i32 cast fused with integer math; i32 → f32 consumer.
        let f = t(&[2.7, -2.7, 1e10, -0.5, 3.0, 4.9, 100.0, -1e10], &[8]);
        let fi = f.cast(DType::I32).unwrap().add_scalar(1.0).unwrap();
        let back = sum.cast(DType::F32).unwrap().mul_scalar(0.5).unwrap();
        let outs = dev.eval(&[&sum, &mixed, &bits, &div, &fi, &back]).unwrap();
        let get = |i: usize| dev.to_vec_i32(&outs[i]).unwrap();
        let n = a.len();
        let want = |f: &dyn Fn(i32, i32) -> i32| (0..n).map(|i| f(a[i], b[i])).collect::<Vec<i32>>();
        assert_eq!(get(0), want(&|x, y| x.wrapping_add(y)), "{}", dev.name());
        assert_eq!(get(1), want(&|x, y| x.wrapping_mul(y) % 1_000_007));
        assert_eq!(get(2), want(&|x, y| ((x ^ y) >> 1) & x.wrapping_shl((y & 31) as u32)));
        assert_eq!(get(3), want(&|x, y| x / y));
        assert_eq!(get(4), vec![3, -1, i32::MAX.wrapping_add(1), 1, 4, 5, 101, i32::MIN + 1]);
        let wb: Vec<f32> = (0..n).map(|i| a[i].wrapping_add(b[i]) as f32 * 0.5).collect();
        assert_eq!(dev.to_vec_f32(&outs[5]).unwrap(), wb);
    }
}

/// Integer prologue of a reduction on values beyond 2^24 (bit-exact i32 reads).
#[test]
fn integer_reduce_prologue() {
    let v: Vec<i32> = (0..7 * 300).map(|i: i32| i.wrapping_mul(2_654_435_761u32 as i32)).collect();
    for dev in devices() {
        let x = Tensor::from_i32(&v, &[7, 300]).unwrap();
        let low = x.bitand(&Tensor::from_i32(&[0xff], &[1]).unwrap()).unwrap();
        let s = low.cast(DType::F32).unwrap().sum(1, false).unwrap();
        let got = dev.to_vec_f32(&dev.eval(&[&s]).unwrap()[0]).unwrap();
        for r in 0..7 {
            let want: i32 = v[r * 300..(r + 1) * 300].iter().map(|x| x & 0xff).sum();
            assert_eq!(got[r], want as f32, "{} row {r}", dev.name());
        }
    }
}

/// Integer math on an i32 operand fused into a matmul epilogue (values beyond 2^24).
#[test]
fn integer_matmul_epilogue() {
    let (m, n, k) = (5, 6, 8);
    let a = ramp(m * k, |i| (i % 7) as f32 - 3.0);
    let w = ramp(n * k, |i| (i % 5) as f32 * 0.5);
    let e: Vec<i32> = (0..m * n).map(|i: usize| (i as i32).wrapping_mul(2_654_435_761u32 as i32)).collect();
    for dev in devices() {
        let mm = t(&a, &[m, k]).matmul_t(&t(&w, &[n, k])).unwrap();
        let bits = Tensor::from_i32(&e, &[m, n]).unwrap().bitand(&Tensor::from_i32(&[0x7ff], &[1]).unwrap()).unwrap();
        let y = mm.add(&bits.cast(DType::F32).unwrap()).unwrap();
        let got = dev.to_vec_f32(&dev.eval(&[&y]).unwrap()[0]).unwrap();
        for i in 0..m {
            for j in 0..n {
                let acc: f32 = (0..k).map(|q| a[i * k + q] * w[j * k + q]).sum();
                let want = acc + (e[i * n + j] & 0x7ff) as f32;
                assert!((got[i * n + j] - want).abs() < 1e-3, "{} ({i},{j}): {} vs {want}", dev.name(), got[i * n + j]);
            }
        }
    }
}
