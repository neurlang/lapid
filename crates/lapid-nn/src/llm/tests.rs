//! Real-shape kernel coverage of the LLM blocks at DeepSeek V4 sizes (random data, every
//! available device vs the CPU / host): top-k (k = 512 over a long compressed history, the
//! lightning indexer), head-dim-512 MQA logits + per-head sink softmax (64 heads), MatMulId
//! with 8 f16 "experts" of [1024, 4096] (grouped output projection) and the Sinkhorn
//! hyper-connection mixing at hc = 4.

use lapid::{DType, Device, Result, Tensor};

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn devices() -> Vec<Device> {
    ["cuda", "vulkan"].iter().filter_map(|d| Device::from_name(d).ok()).collect()
}

fn run(dev: &Device, t: &Tensor) -> Result<Vec<f32>> {
    let r = dev.eval(&[t])?;
    dev.to_vec_f32(&r[0])
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += f64::from(x - y).powi(2);
        den += f64::from(*y).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

#[test]
fn indexer_topk_512() -> Result<()> {
    let (n, t, k) = (3, 16384, 512);
    let s = rand(n * t, 1);
    let x = Tensor::from_f32(&s, &[n, t])?;
    for dev in std::iter::once(Device::cpu()).chain(devices()) {
        let top = x.topk_indices(k, 1)?;
        let got = {
            let r = dev.eval(&[&top])?;
            dev.to_vec_i32(&r[0])?
        };
        for i in 0..n {
            let mut want: Vec<usize> = (0..t).collect();
            want.sort_by(|&a, &b| s[i * t + b].total_cmp(&s[i * t + a]).then(a.cmp(&b)));
            let mut g: Vec<usize> = got[i * k..(i + 1) * k].iter().map(|&v| v as usize).collect();
            let mut w = want[..k].to_vec();
            g.sort_unstable();
            w.sort_unstable();
            assert_eq!(g, w, "{}: top-{k} set of row {i}", dev.name());
        }
    }
    Ok(())
}

#[test]
fn mqa_sink_attention_hd512() -> Result<()> {
    let (n, h, hd, s) = (4, 64, 512, 600);
    let q = Tensor::from_f32(&rand(n * h * hd, 2), &[n * h, hd])?;
    let keys = Tensor::from_f32(&rand(s * hd, 3), &[s, hd])?;
    let sink = Tensor::from_f32(&rand(h, 4), &[1, h, 1])?;
    let logits = q.matmul_t(&keys)?.reshape(&[n, h, s])?.mul_scalar(1.0 / (hd as f64).sqrt())?;
    let p = Tensor::cat(&[&logits, &sink.expand(&[n, h, 1])?], 2)?.softmax()?.slice(2, 0, s)?;
    let o = p.reshape(&[n * h, s])?.matmul(&keys)?;
    let want = run(&Device::cpu(), &o)?;
    for dev in devices() {
        let r = rel(&run(&dev, &o)?, &want);
        assert!(r < 1e-3, "{}: rel {r}", dev.name());
    }
    Ok(())
}

#[test]
fn grouped_output_matmul_id_f16() -> Result<()> {
    let (n, g, ol, inp) = (3, 8, 1024, 4096);
    let w = Tensor::from_f32(&rand(g * ol * inp, 5), &[g, ol, inp])?.cast(DType::F16)?;
    let x = Tensor::from_f32(&rand(n * g * inp, 6), &[n * g, inp])?;
    let ids: Vec<i32> = (0..n * g).map(|i| (i % g) as i32).collect();
    let y = x.matmul_id(&w, &Tensor::from_i32(&ids, &[n * g, 1])?)?;
    let want = run(&Device::cpu(), &y)?;
    for dev in devices() {
        let r = rel(&run(&dev, &y)?, &want);
        assert!(r < 2e-3, "{}: rel {r}", dev.name());
    }
    Ok(())
}

#[test]
fn sinkhorn_hc4_d4096() -> Result<()> {
    let (n, hc, d) = (5, 4, 4096);
    let x = Tensor::from_f32(&rand(n * hc * d, 7), &[n, hc, d])?;
    let mix = Tensor::from_f32(&rand(n * hc * hc, 8), &[n, hc, hc])?;
    let mut comb = mix.mul_scalar(3.0)?.softmax()?.add_scalar(1e-6)?;
    let norm = |c: &Tensor, axis: isize| -> Result<Tensor> { c.div(&c.sum(axis, true)?.add_scalar(1e-6)?) };
    comb = norm(&comb, 1)?;
    for _ in 1..20 {
        comb = norm(&norm(&comb, 2)?, 1)?;
    }
    let out = comb.unsqueeze(3)?.mul(&x.unsqueeze(2)?)?.sum(1, false)?;
    let want = run(&Device::cpu(), &out)?;
    let c = run(&Device::cpu(), &comb)?;
    for row in c.chunks(hc * hc) {
        for j in 0..hc {
            let col: f32 = (0..hc).map(|i| row[i * hc + j]).sum();
            assert!((col - 1.0).abs() < 1e-3, "column sum {col}");
        }
    }
    for dev in devices() {
        let r = rel(&run(&dev, &out)?, &want);
        assert!(r < 1e-4, "{}: rel {r}", dev.name());
    }
    Ok(())
}
