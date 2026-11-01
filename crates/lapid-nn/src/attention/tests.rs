use lapid::{DType, Device};

use super::*;
use crate::testutil::{assert_close, eval, rand_tensor, rand_vec};

/// Naive attention on host data. q [B, H, Sq, D], k/v [B, Hkv, Sk, D];
/// `visible(i, j)` decides masking; `add(b, i, j)` is an extra additive term.
fn attn_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    (b, h, hkv, sq, sk, d): (usize, usize, usize, usize, usize, usize),
    scale: f32,
    visible: &dyn Fn(usize, usize) -> bool,
    add: &dyn Fn(usize, usize, usize) -> f32,
) -> Vec<f32> {
    let rep = h / hkv;
    let mut out = vec![0.0f32; b * h * sq * d];
    for bi in 0..b {
        for hi in 0..h {
            let kh = hi / rep;
            for i in 0..sq {
                let qr = &q[((bi * h + hi) * sq + i) * d..][..d];
                let mut scores = vec![f64::NEG_INFINITY; sk];
                for (j, s) in scores.iter_mut().enumerate() {
                    if !visible(i, j) {
                        continue;
                    }
                    let kr = &k[((bi * hkv + kh) * sk + j) * d..][..d];
                    let dot: f64 = qr.iter().zip(kr).map(|(a, b)| *a as f64 * *b as f64).sum();
                    *s = dot * scale as f64 + add(bi, i, j) as f64;
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = e.iter().sum();
                let o = &mut out[((bi * h + hi) * sq + i) * d..][..d];
                for (j, w) in e.iter().enumerate() {
                    let vr = &v[((bi * hkv + kh) * sk + j) * d..][..d];
                    for (ov, vv) in o.iter_mut().zip(vr) {
                        *ov += (w / sum * *vv as f64) as f32;
                    }
                }
            }
        }
    }
    out
}

#[test]
fn sdpa_mha_gqa_mqa_causal() {
    let (b, sq, sk, d) = (2, 3, 5, 4);
    for (h, hkv) in [(4usize, 4usize), (4, 2), (4, 1)] {
        let (q, qv) = rand_tensor(&[b, h, sq, d], 1);
        let (k, kv) = rand_tensor(&[b, hkv, sk, d], 2);
        let (v, vv) = rand_tensor(&[b, hkv, sk, d], 3);
        let dims = (b, h, hkv, sq, sk, d);
        let none = eval(&scaled_dot_product_attention(&q, &k, &v, 0.5, &Mask::None).unwrap());
        assert_close(&none, &attn_ref(&qv, &kv, &vv, dims, 0.5, &|_, _| true, &|_, _, _| 0.0), 1e-5);
        let causal = eval(&scaled_dot_product_attention(&q, &k, &v, 0.5, &Mask::Causal).unwrap());
        let want = attn_ref(&qv, &kv, &vv, dims, 0.5, &|i, j| j <= i + sk - sq, &|_, _, _| 0.0);
        assert_close(&causal, &want, 1e-5);
    }
}

#[test]
fn sdpa_gqa_decode_folds_heads() {
    // Sq == 1: query heads of a kv group become rows of one attention head (no K/V repeat).
    let (b, sk, d) = (2, 7, 8);
    for (h, hkv) in [(8usize, 2usize), (6, 1)] {
        let (q, qv) = rand_tensor(&[b, h, 1, d], 4);
        let (k, kv) = rand_tensor(&[b, hkv, sk, d], 5);
        let (v, vv) = rand_tensor(&[b, hkv, sk, d], 6);
        let dims = (b, h, hkv, 1, sk, d);
        let want = attn_ref(&qv, &kv, &vv, dims, 0.3, &|_, _| true, &|_, _, _| 0.0);
        for mask in [Mask::None, Mask::Causal, Mask::Sliding(16)] {
            let got = eval(&scaled_dot_product_attention(&q, &k, &v, 0.3, &mask).unwrap());
            assert_close(&got, &want, 1e-5);
        }
        // A window shorter than the keys still masks (no fold).
        let got = eval(&scaled_dot_product_attention(&q, &k, &v, 0.3, &Mask::Sliding(3)).unwrap());
        assert_close(&got, &attn_ref(&qv, &kv, &vv, dims, 0.3, &|_, j| j + 3 >= sk, &|_, _, _| 0.0), 1e-5);
    }
}

#[test]
fn sdpa_sliding_window() {
    let (b, h, sq, sk, d) = (1, 2, 4, 7, 4);
    let (q, qv) = rand_tensor(&[b, h, sq, d], 4);
    let (k, kv) = rand_tensor(&[b, h, sk, d], 5);
    let (v, vv) = rand_tensor(&[b, h, sk, d], 6);
    for window in [1usize, 2, 3, 7, 100] {
        let got = eval(&scaled_dot_product_attention(&q, &k, &v, 0.5, &Mask::Sliding(window)).unwrap());
        let vis = |i: usize, j: usize| {
            let qpos = i + sk - sq;
            j <= qpos && qpos - j < window
        };
        let want = attn_ref(&qv, &kv, &vv, (b, h, h, sq, sk, d), 0.5, &vis, &|_, _, _| 0.0);
        assert_close(&got, &want, 1e-5);
    }
    assert!(scaled_dot_product_attention(&q, &k, &v, 0.5, &Mask::Sliding(0)).is_err());
}

#[test]
fn sdpa_custom_padding_mask() {
    let (b, h, s, d) = (2, 2, 4, 4);
    let (q, qv) = rand_tensor(&[b, h, s, d], 7);
    let (k, kv) = rand_tensor(&[b, h, s, d], 8);
    let (v, vv) = rand_tensor(&[b, h, s, d], 9);
    let lengths = [4usize, 2];
    let m = key_padding_mask(&lengths, s).unwrap();
    assert_eq!(m.shape(), &[2, 1, 1, 4]);
    let mask = Mask::Custom { additive: m, causal: false };
    let got = eval(&scaled_dot_product_attention(&q, &k, &v, 0.5, &mask).unwrap());
    let add = |bi: usize, _i: usize, j: usize| if j < lengths[bi] { 0.0 } else { f32::NEG_INFINITY };
    let want = attn_ref(&qv, &kv, &vv, (b, h, h, s, s, d), 0.5, &|_, _| true, &add);
    assert_close(&got, &want, 1e-5);
}

#[test]
fn masks_and_heads_helpers() {
    let m = eval(&sliding_window_mask(2, 4, 2).unwrap());
    let ni = f32::NEG_INFINITY;
    // query positions 2, 3 (window 2): keys 1..=2 and 2..=3 survive the window rule
    assert_eq!(m, vec![ni, 0.0, 0.0, 0.0, ni, ni, 0.0, 0.0]);

    let (x, xv) = rand_tensor(&[1, 3, 6], 10);
    let heads = split_heads(&x, 2).unwrap();
    assert_eq!(heads.shape(), &[1, 2, 3, 3]);
    let hv = eval(&heads);
    assert_eq!(hv[3], xv[6]); // head 0, row 1, col 0 == x[0, 1, 0]
    assert_eq!(hv[9], xv[3]); // head 1, row 0, col 0 == x[0, 0, 3]
    assert_eq!(eval(&merge_heads(&heads).unwrap()), xv);
    assert!(split_heads(&x, 4).is_err());

    let (kv, kvv) = rand_tensor(&[1, 2, 3, 2], 11);
    let r = eval(&repeat_kv(&kv, 3).unwrap());
    assert_eq!(r.len(), 36);
    for h in 0..6 {
        assert_eq!(r[h * 6..h * 6 + 6], kvv[(h / 3) * 6..(h / 3) * 6 + 6]);
    }
}

fn lin(out: usize, inp: usize, seed: u64, bias: bool) -> Linear {
    let w = Tensor::from_f32(&rand_vec(out * inp, seed, 0.5), &[out, inp]).unwrap();
    let b = bias.then(|| Tensor::from_f32(&rand_vec(out, seed + 1, 0.5), &[out]).unwrap());
    Linear::new(w, b)
}

fn block(n_head: usize, n_kv: usize, hd: usize, qk_norm: bool) -> Attention {
    let d = n_head * hd;
    let a = Attention::new(
        lin(d, d, 20, true),
        lin(n_kv * hd, d, 22, false),
        lin(n_kv * hd, d, 24, true),
        lin(d, d, 26, true),
        n_head,
        n_kv,
    )
    .unwrap();
    if qk_norm {
        let w = |s| Tensor::from_f32(&rand_vec(hd, s, 1.0), &[hd]).unwrap();
        a.with_qk_norm(RmsNorm::new(w(30), 1e-6), RmsNorm::new(w(31), 1e-6))
    } else {
        a
    }
}

/// Run x [1, S, D] through `attn` in chunks with a cache; return all output rows.
fn run_cached(attn: &Attention, rope: Option<&Rope>, x: &Tensor, chunks: &[usize], cache: &mut KvCache, mask: &Mask) -> Vec<f32> {
    let dev = Device::cpu();
    let mut out = Vec::new();
    let mut at = 0;
    for &n in chunks {
        let xs = x.slice(1, at, at + n).unwrap();
        let y = attn.forward(&xs, rope, Some((&mut *cache, 0)), mask).unwrap();
        let mut outs = vec![&y];
        outs.extend(cache.staged());
        let mut realized = dev.eval(&outs).unwrap();
        let rest = realized.split_off(1);
        out.extend(dev.to_vec_f32(&realized[0]).unwrap());
        cache.commit(rest).unwrap();
        at += n;
    }
    assert_eq!(cache.offset(), at);
    out
}

#[test]
fn cached_decoding_equals_full_pass() {
    let (n_head, n_kv, hd) = (4, 2, 8);
    let attn = block(n_head, n_kv, hd, true);
    assert_eq!((attn.head_dim, attn.n_kv_head), (8, 2));
    let rope = Rope::new(hd, 10_000.0);
    let (x, _) = rand_tensor(&[1, 6, n_head * hd], 40);
    let full = eval(&attn.forward(&x, Some(&rope), None, &Mask::Causal).unwrap());
    let mut cache = KvCache::new(1);
    let got = run_cached(&attn, Some(&rope), &x, &[3, 1, 2], &mut cache, &Mask::Causal);
    assert_close(&got, &full, 1e-4);
    assert_eq!(cache.len(0), 6);
}

#[test]
fn sliding_window_cache_equals_full_pass() {
    let attn = block(2, 2, 4, false);
    let (x, _) = rand_tensor(&[1, 7, 8], 41);
    let window = 3;
    let mask = Mask::Sliding(window);
    let full = eval(&attn.forward(&x, None, None, &mask).unwrap());
    let mut cache = KvCache::with_windows(vec![Some(window)]);
    let got = run_cached(&attn, None, &x, &[2, 1, 1, 3], &mut cache, &mask);
    assert_close(&got, &full, 1e-4);
    assert_eq!(cache.len(0), window);
}

#[test]
fn cross_attention_shapes() {
    let attn = block(2, 2, 4, false);
    let (x, _) = rand_tensor(&[1, 3, 8], 42);
    let (enc, _) = rand_tensor(&[1, 10, 8], 43);
    let (k, v) = attn.project_kv(&enc).unwrap();
    assert_eq!(k.shape(), &[1, 2, 10, 4]);
    let y = attn.attend(&x, &k, &v, &Mask::None).unwrap();
    assert_eq!(y.shape(), &[1, 3, 8]);
    assert!(eval(&y).iter().all(|v| v.is_finite()));
}

#[test]
fn kv_cache_contract() {
    let mut c = KvCache::new(2);
    assert!(c.is_empty() && c.get(0).is_none());
    let k = Tensor::zeros(&[1, 2, 3, 4], DType::F32);
    assert!(c.append(2, &k, &k).is_err());
    let (fk, _) = c.append(0, &k, &k).unwrap();
    assert_eq!(fk.shape(), &[1, 2, 3, 4]);
    assert!(c.append(0, &k, &k).is_err(), "double stage");
    let k2 = Tensor::zeros(&[1, 2, 2, 4], DType::F32);
    assert!(c.append(1, &k2, &k2).is_err(), "inconsistent length");
    assert_eq!(c.staged().len(), 2);
    assert!(c.commit(vec![]).is_err());
    c.discard();
    assert!(c.staged().is_empty() && c.offset() == 0);
    c.append(1, &k, &k).unwrap();
    let dev = Device::cpu();
    let realized = dev.eval(&c.staged()).unwrap();
    c.commit(realized).unwrap();
    assert_eq!((c.offset(), c.len(0), c.len(1)), (3, 0, 3));
    c.clear();
    assert_eq!((c.offset(), c.n_layers()), (0, 2));
    assert!(Attention::new(lin(8, 8, 1, false), lin(8, 8, 1, false), lin(8, 8, 1, false), lin(8, 8, 1, false), 4, 3).is_err());
}
