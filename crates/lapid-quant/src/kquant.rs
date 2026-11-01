//! ggml k-quant super-blocks (Q4_K, Q5_K, Q6_K): reference quantizers and dequantizers,
//! bit-exact ports of `quantize_row_q{4,5,6}_K_ref` / `dequantize_row_q{4,5,6}_K` from
//! ggml `ggml-quants.c` (compiled without floating-point contraction). Every f32
//! operation is written in the C evaluation order; do not "simplify" the arithmetic.
//!
//! Block layouts (ggml-common.h, little endian, 256 elements):
//! ```text
//! Q4_K (144 B): f16 d | f16 dmin | scales[12] (8 x 6-bit scale + 8 x 6-bit min) | qs[128]
//! Q5_K (176 B): f16 d | f16 dmin | scales[12] | qh[32] | qs[128]
//! Q6_K (210 B): ql[128] | qh[64] | i8 scales[16] | f16 d
//! ```

use half::f16;

/// Elements per super-block.
pub const QK_K: usize = 256;
pub const Q4K_BYTES: usize = 144;
pub const Q5K_BYTES: usize = 176;
pub const Q6K_BYTES: usize = 210;

const GROUP_MAX_EPS: f32 = 1e-15;

/// ggml `nearest_int`: round half to even via the 1.5 * 2^23 trick (|x| <= 4194303).
#[inline]
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12_582_912.0;
    ((val.to_bits() & 0x007f_ffff) as i32) - 0x0040_0000
}

#[inline]
fn h(b: &[u8], o: usize) -> f32 {
    f16::from_le_bytes([b[o], b[o + 1]]).to_f32()
}

#[inline]
fn put_h(b: &mut [u8], o: usize, v: f32) {
    b[o..o + 2].copy_from_slice(&f16::from_f32(v).to_le_bytes());
}

/// ggml `make_qkx2_quants` (asymmetric scale + min search). Returns (scale, the_min).
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    x: &[f32],
    weights: &[f32],
    nmax: i32,
    l_out: &mut [u8],
    l_aux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
) -> (f32, f32) {
    let n = x.len();
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l_out[..n].fill(0);
        return (0.0, -min);
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let l = nearest_int(iscale * (x[i] - min));
        l_out[i] = l.clamp(0, nmax) as u8;
        let diff = scale * l_out[i] as f32 + min - x[i];
        best_error += weights[i] * (diff * diff);
    }
    if nstep < 1 {
        return (scale, -min);
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let l = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            l_aux[i] = l as u8;
            let w = weights[i];
            let lf = l as f32;
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let diff = this_scale * l_aux[i] as f32 + this_min - x[i];
                cur_error += weights[i] * (diff * diff);
            }
            if cur_error < best_error {
                l_out[..n].copy_from_slice(&l_aux[..n]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// ggml `make_qx_quants` with `rmse_type = 1` and no quant weights (as used by Q6_K).
fn make_qx_quants_rmse1(x: &[f32], nmax: i32, l_out: &mut [i8]) -> f32 {
    let n = x.len();
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        l_out[..n].fill(0);
        return 0.0;
    }
    let mut iscale = -(nmax as f32) / max;
    let (mut sumlx, mut suml2) = (0.0f32, 0.0f32);
    for i in 0..n {
        let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l_out[i] = (l + nmax) as i8;
        let w = x[i] * x[i];
        let lf = l as f32;
        sumlx += w * x[i] * lf;
        suml2 += w * lf * lf;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;
    for is in -9i32..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1f32 * is as f32) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for &v in x {
            let lf = nearest_int(iscale * v).clamp(-nmax, nmax - 1) as f32;
            let w = v * v;
            sumlx += w * v * lf;
            suml2 += w * lf * lf;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
                l_out[i] = (nmax + l) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// ggml `get_scale_min_k4`: 6-bit scale and min `j` (0..8) of a Q4_K / Q5_K block.
#[inline]
pub fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Shared first half of the Q4_K / Q5_K quantizers: per-32 scales/mins, the packed 6-bit
/// scales, d / dmin, and the final quant values `l` (0..=nmax) per element.
fn quantize_k45(x: &[f32], nmax: i32, rmin: f32, nstep: i32, head: &mut [u8], l: &mut [u8; QK_K]) {
    let mut l_aux = [0u8; 32];
    let mut weights = [0f32; 32];
    let mut scales = [0f32; 8];
    let mut mins = [0f32; 8];
    let mut max_scale = 0.0f32;
    let mut max_min = 0.0f32;
    for j in 0..8 {
        let xs = &x[32 * j..32 * j + 32];
        let mut sum_x2 = 0.0f32;
        for &v in xs {
            sum_x2 += v * v;
        }
        let av_x = (sum_x2 / 32.0).sqrt();
        for (w, &v) in weights.iter_mut().zip(xs) {
            *w = av_x + v.abs();
        }
        let (s, m) = make_qkx2_quants(xs, &weights, nmax, &mut l[32 * j..32 * j + 32], &mut l_aux, rmin, 0.1, nstep);
        scales[j] = s;
        mins[j] = m;
        if s > max_scale {
            max_scale = s;
        }
        if m > max_min {
            max_min = m;
        }
    }
    let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let sc = &mut head[4..16];
    sc.fill(0);
    for j in 0..8 {
        // C: `uint8_t ls = nearest_int(..)` truncates to 8 bits, then MIN(63, ls).
        let ls = (nearest_int(inv_scale * scales[j]) as u8).min(63);
        let lm = (nearest_int(inv_min * mins[j]) as u8).min(63);
        if j < 4 {
            sc[j] = ls;
            sc[j + 4] = lm;
        } else {
            sc[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            sc[j - 4] |= (ls >> 4) << 6;
            sc[j] |= (lm >> 4) << 6;
        }
    }
    put_h(head, 0, max_scale / 63.0);
    put_h(head, 2, max_min / 63.0);
    let (d_all, dmin_all) = (h(head, 0), h(head, 2));
    for j in 0..8 {
        let (s, m) = scale_min_k4(j, &head[4..16]);
        let d = d_all * s as f32;
        if d == 0.0 {
            continue;
        }
        let dm = dmin_all * m as f32;
        for ii in 0..32 {
            let q = nearest_int((x[32 * j + ii] + dm) / d);
            l[32 * j + ii] = q.clamp(0, nmax) as u8;
        }
    }
}

/// Quantize 256 values into one Q4_K block (`quantize_row_q4_K_ref`).
pub fn quantize_q4k(x: &[f32], out: &mut [u8]) {
    assert!(x.len() == QK_K && out.len() == Q4K_BYTES);
    let mut l = [0u8; QK_K];
    quantize_k45(x, 15, -1.0, 20, &mut out[..16], &mut l);
    let qs = &mut out[16..144];
    for (k, j) in (0..QK_K).step_by(64).enumerate() {
        for i in 0..32 {
            qs[32 * k + i] = l[j + i] | (l[j + i + 32] << 4);
        }
    }
}

/// Quantize 256 values into one Q5_K block (`quantize_row_q5_K_ref`).
pub fn quantize_q5k(x: &[f32], out: &mut [u8]) {
    assert!(x.len() == QK_K && out.len() == Q5K_BYTES);
    let mut l = [0u8; QK_K];
    quantize_k45(x, 31, -0.5, 15, &mut out[..16], &mut l);
    let (qh, ql) = out[16..176].split_at_mut(32);
    qh.fill(0);
    let (mut m1, mut m2) = (1u8, 2u8);
    for (k, n) in (0..QK_K).step_by(64).enumerate() {
        for j in 0..32 {
            let mut l1 = l[n + j];
            if l1 > 15 {
                l1 -= 16;
                qh[j] |= m1;
            }
            let mut l2 = l[n + j + 32];
            if l2 > 15 {
                l2 -= 16;
                qh[j] |= m2;
            }
            ql[32 * k + j] = l1 | (l2 << 4);
        }
        m1 <<= 2;
        m2 <<= 2;
    }
}

/// Quantize 256 values into one Q6_K block (`quantize_row_q6_K_ref`).
pub fn quantize_q6k(x: &[f32], out: &mut [u8]) {
    assert!(x.len() == QK_K && out.len() == Q6K_BYTES);
    let mut l = [0i8; QK_K];
    let mut scales = [0f32; 16];
    let mut max_scale = 0.0f32;
    let mut max_abs_scale = 0.0f32;
    for ib in 0..16 {
        let s = make_qx_quants_rmse1(&x[16 * ib..16 * ib + 16], 32, &mut l[16 * ib..16 * ib + 16]);
        scales[ib] = s;
        if s.abs() > max_abs_scale {
            max_abs_scale = s.abs();
            max_scale = s;
        }
    }
    out.fill(0);
    if max_abs_scale < GROUP_MAX_EPS {
        return; // all zero, d = +0
    }
    let iscale = -128.0f32 / max_scale;
    put_h(out, 208, 1.0 / iscale);
    for ib in 0..16 {
        out[192 + ib] = nearest_int(iscale * scales[ib]).min(127) as i8 as u8;
    }
    let d_all = h(out, 208);
    for j in 0..16 {
        let d = d_all * (out[192 + j] as i8) as f32;
        if d == 0.0 {
            continue;
        }
        for ii in 0..16 {
            let q = nearest_int(x[16 * j + ii] / d).clamp(-32, 31);
            l[16 * j + ii] = (q + 32) as i8;
        }
    }
    let (ql, rest) = out.split_at_mut(128);
    let qh = &mut rest[..64];
    for (k, j) in (0..QK_K).step_by(128).enumerate() {
        for i in 0..32 {
            let lv = |o: usize| l[j + i + o] as u8;
            ql[64 * k + i] = (lv(0) & 0xF) | ((lv(64) & 0xF) << 4);
            ql[64 * k + i + 32] = (lv(32) & 0xF) | ((lv(96) & 0xF) << 4);
            qh[32 * k + i] = (lv(0) >> 4) | ((lv(32) >> 4) << 2) | ((lv(64) >> 4) << 4) | ((lv(96) >> 4) << 6);
        }
    }
}

/// Dequantize one Q4_K block into 256 values (`dequantize_row_q4_K`).
pub fn dequantize_q4k(b: &[u8], y: &mut [f32]) {
    let (d, min) = (h(b, 0), h(b, 2));
    let (scales, qs) = (&b[4..16], &b[16..144]);
    for j in 0..4 {
        let q = &qs[32 * j..32 * j + 32];
        let (s1, m1) = scale_min_k4(2 * j, scales);
        let (s2, m2) = scale_min_k4(2 * j + 1, scales);
        let (d1, m1) = (d * s1 as f32, min * m1 as f32);
        let (d2, m2) = (d * s2 as f32, min * m2 as f32);
        let o = &mut y[64 * j..64 * j + 64];
        for l in 0..32 {
            o[l] = d1 * (q[l] & 0xF) as f32 - m1;
            o[l + 32] = d2 * (q[l] >> 4) as f32 - m2;
        }
    }
}

/// Dequantize one Q5_K block into 256 values (`dequantize_row_q5_K`).
pub fn dequantize_q5k(b: &[u8], y: &mut [f32]) {
    let (d, min) = (h(b, 0), h(b, 2));
    let (scales, qh, qs) = (&b[4..16], &b[16..48], &b[48..176]);
    for j in 0..4 {
        let ql = &qs[32 * j..32 * j + 32];
        let (u1, u2) = (1u8 << (2 * j), 2u8 << (2 * j));
        let (s1, m1) = scale_min_k4(2 * j, scales);
        let (s2, m2) = scale_min_k4(2 * j + 1, scales);
        let (d1, m1) = (d * s1 as f32, min * m1 as f32);
        let (d2, m2) = (d * s2 as f32, min * m2 as f32);
        let o = &mut y[64 * j..64 * j + 64];
        for l in 0..32 {
            let h1 = if qh[l] & u1 != 0 { 16 } else { 0 };
            let h2 = if qh[l] & u2 != 0 { 16 } else { 0 };
            o[l] = d1 * ((ql[l] & 0xF) + h1) as f32 - m1;
            o[l + 32] = d2 * ((ql[l] >> 4) + h2) as f32 - m2;
        }
    }
}

/// Dequantize one Q6_K block into 256 values (`dequantize_row_q6_K`).
pub fn dequantize_q6k(b: &[u8], y: &mut [f32]) {
    let d = h(b, 208);
    for n in 0..2 {
        let ql = &b[64 * n..64 * n + 64];
        let qh = &b[128 + 32 * n..128 + 32 * n + 32];
        let sc = &b[192 + 8 * n..192 + 8 * n + 8];
        let o = &mut y[128 * n..128 * n + 128];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | (qh[l] & 3) << 4) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | ((qh[l] >> 2) & 3) << 4) as i32 - 32;
            let q3 = ((ql[l] >> 4) | ((qh[l] >> 4) & 3) << 4) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | ((qh[l] >> 6) & 3) << 4) as i32 - 32;
            let s = |k: usize| d * (sc[is + k] as i8) as f32;
            o[l] = s(0) * q1 as f32;
            o[l + 32] = s(2) * q2 as f32;
            o[l + 64] = s(4) * q3 as f32;
            o[l + 96] = s(6) * q4 as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    type Q = fn(&[f32], &mut [u8]);
    type D = fn(&[u8], &mut [f32]);
    const KINDS: [(Q, D, usize, f32); 3] =
        [(quantize_q4k, dequantize_q4k, Q4K_BYTES, 0.08), (quantize_q5k, dequantize_q5k, Q5K_BYTES, 0.04), (quantize_q6k, dequantize_q6k, Q6K_BYTES, 0.02)];

    #[test]
    fn nearest_int_rounds_half_even() {
        assert_eq!([nearest_int(0.5), nearest_int(1.5), nearest_int(-0.5), nearest_int(-1.5), nearest_int(2.4)], [0, 2, 0, -2, 2]);
    }

    #[test]
    fn round_trip_relative_error() {
        let mut x = noise(QK_K, 3);
        x[17] = 4.0; // an outlier
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / QK_K as f32).sqrt();
        for (q, d, nb, tol) in KINDS {
            let mut b = vec![0u8; nb];
            q(&x, &mut b);
            let mut y = vec![0f32; QK_K];
            d(&b, &mut y);
            let err = (x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / QK_K as f32).sqrt();
            assert!(err / rms < tol, "{nb}-byte blocks: rel rms {}", err / rms);
        }
    }

    #[test]
    fn zeros_and_constants() {
        for (q, d, nb, _) in KINDS {
            for c in [0.0f32, 1.0, -2.5] {
                let mut b = vec![0u8; nb];
                q(&[c; QK_K], &mut b);
                let mut y = vec![9f32; QK_K];
                d(&b, &mut y);
                for v in y {
                    assert!((v - c).abs() <= c.abs() * 0.01 + 1e-6, "{nb}: {c} -> {v}");
                }
            }
        }
    }

    /// Bit-exact cross-check against ggml's C code: `LAPID_GGML_KQ_REF=<file>` written by
    /// `plans/lapid-v1/ggml-ref/kq_ref` (records `u32 ggml_type, u32 n, n f32 x, blocks of
    /// quantize_row_*_ref(x), n f32 dequantize_row_*`). Skipped when unset.
    #[test]
    fn matches_ggml_c_reference() {
        let Some(path) = std::env::var_os("LAPID_GGML_KQ_REF") else { return };
        let b = std::fs::read(path).expect("read reference");
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().expect("4"));
        let f32s = |o: usize, n: usize| -> Vec<f32> {
            b[o..o + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4"))).collect()
        };
        let (mut o, mut records) = (0, 0);
        while o < b.len() {
            let (ty, n) = (u32_at(o), u32_at(o + 4) as usize);
            let (q, d, nb, _) = match ty {
                12 => KINDS[0],
                13 => KINDS[1],
                14 => KINDS[2],
                t => panic!("unexpected ggml type {t}"),
            };
            let x = f32s(o + 8, n);
            let blocks = &b[o + 8 + 4 * n..o + 8 + 4 * n + n / QK_K * nb];
            let want = f32s(o + 8 + 4 * n + blocks.len(), n);
            let mut got = vec![0u8; nb];
            let mut y = vec![0f32; QK_K];
            for (k, (xb, wb)) in x.chunks_exact(QK_K).zip(blocks.chunks_exact(nb)).enumerate() {
                q(xb, &mut got);
                assert_eq!(&got[..], wb, "type {ty} record {records} block {k}: quantized bytes differ");
                d(wb, &mut y);
                let w = &want[k * QK_K..(k + 1) * QK_K];
                assert!(y.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()), "type {ty} record {records} block {k}: dequant differs");
            }
            o += 8 + 8 * n + blocks.len();
            records += 1;
        }
        assert_eq!(records, 24);
    }
}
