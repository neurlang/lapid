//! GPTQ for the int4 format (group 32, x = (q - 8) * d), in its LDLQ form (QuIP, Chee et
//! al. 2023 — shown there to equal OPTQ/GPTQ, Frantar et al. 2022).
//!
//! For a weight row `w` and the input Gram matrix `H = Σ x xᵀ`, the proxy loss
//! `(ŵ - w) H (ŵ - w)ᵀ` is minimized greedily column by column: with `H = R Rᵀ`, `R`
//! upper triangular, column `k` is quantized as
//! `ŵ_k = Q(w_k + Σ_{i<k} (w_i - ŵ_i) R[i,k] / R[k,k])`.
//! `R = J M J` where `M` is the Cholesky factor of `J H J` (`J` reverses the index order),
//! so one Cholesky per weight suffices. Group scales are chosen at each group start from the
//! error-corrected values (as GPTQ does), searched over a small grid weighted by `diag(H)`.
//! Rows are independent: they are processed in tiles across threads, each tile streaming
//! the factor once.

use lapid_tensor::{Error, Result};
use half::f16;

/// Elements per int4 group.
const G: usize = lapid_quant::GROUP;
/// Rows per tile (one pass over the factor per tile).
const TILE: usize = 16;

/// GPTQ settings.
#[derive(Copy, Clone, Debug)]
pub struct GptqOptions {
    /// Diagonal damping as a fraction of mean(diag H) (GPTQ's `percdamp`, default 0.01).
    pub damp: f32,
    /// Search the group scale over a grid (else ggml q4_0's `max / -8`).
    pub scale_search: bool,
    /// GPTQ "act order" with static groups: quantize columns by decreasing `diag(H)`, group
    /// scales fixed up front from the original weights.
    pub act_order: bool,
}

impl Default for GptqOptions {
    fn default() -> Self {
        GptqOptions { damp: 0.01, scale_search: true, act_order: false }
    }
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

/// Run `f(i, item)` over `items` on all cores (contiguous chunks).
fn par_chunks<T: Send>(items: &mut [T], f: impl Fn(usize, &mut T) + Sync) {
    let n = items.len();
    if n == 0 {
        return;
    }
    let per = n.div_ceil(threads().min(n));
    std::thread::scope(|s| {
        for (ci, chunk) in items.chunks_mut(per).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (j, it) in chunk.iter_mut().enumerate() {
                    f(ci * per + j, it);
                }
            });
        }
    });
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// In-place lower Cholesky factor of the symmetric `n x n` row-major matrix `a` (only the
/// lower triangle is read; the strict upper triangle is zeroed). Blocked, rows in parallel.
/// Errors "numerical" when `a` is not positive definite.
fn cholesky(a: &mut [f32], n: usize) -> Result<()> {
    const B: usize = 64;
    let mut jb = 0;
    while jb < n {
        let je = (jb + B).min(n);
        // Panel update: A[i, jb..je] -= L[i, 0..jb] · L[j, 0..jb] for i >= jb.
        if jb > 0 {
            let panel: Vec<f32> = (jb..je).flat_map(|j| a[j * n..j * n + jb].to_vec()).collect();
            let mut rows: Vec<&mut [f32]> = a[jb * n..].chunks_mut(n).collect();
            par_chunks(&mut rows, |r, row| {
                let i = jb + r;
                for j in jb..je.min(i + 1) {
                    let lj = &panel[(j - jb) * jb..(j - jb + 1) * jb];
                    row[j] -= dot(&row[..jb], lj);
                }
            });
        }
        // Factor the panel column by column.
        for j in jb..je {
            let d = a[j * n + j] - dot(&a[j * n + jb..j * n + j], &a[j * n + jb..j * n + j]);
            if !(d > 0.0 && d.is_finite()) {
                return Err(Error::new("numerical", format!("cholesky: pivot {j} is {d}")));
            }
            let d = d.sqrt();
            a[j * n + j] = d;
            // Serial: at most B-long dots per row (n·B²/2 per panel, small next to the update).
            let lj: Vec<f32> = a[j * n + jb..j * n + j].to_vec();
            for row in a[(j + 1) * n..].chunks_mut(n) {
                row[j] = (row[j] - dot(&row[jb..j], &lj)) / d;
            }
        }
        jb = je;
    }
    for i in 0..n {
        a[i * n + i + 1..(i + 1) * n].fill(0.0);
    }
    Ok(())
}

/// Feedback matrix `T` (row-major, strictly lower): `T[k][i] = R[i,k] / R[k,k]` for i < k,
/// from the Gram matrix `h` (damped in place). Errors "numerical" if not positive definite
/// even after raising the damping.
fn feedback(h: &[f32], n: usize, damp: f32) -> Result<Vec<f32>> {
    let mean_diag = (0..n).map(|i| h[i * n + i] as f64).sum::<f64>() / n as f64;
    let mut lambda = damp.max(1e-6) as f64 * mean_diag.max(1e-12);
    for _ in 0..6 {
        // J H J, damped; dead channels (zero diagonal) get unit weight.
        let mut m = vec![0f32; n * n];
        for (r, row) in m.chunks_exact_mut(n).enumerate() {
            let hr = &h[(n - 1 - r) * n..(n - r) * n];
            for (c, v) in row.iter_mut().enumerate().take(r + 1) {
                *v = hr[n - 1 - c];
            }
            if row[r] <= 0.0 {
                row[r] = 1.0;
            }
            row[r] += lambda as f32;
        }
        if cholesky(&mut m, n).is_ok() {
            // R[i,k] = M[n-1-i, n-1-k]; T[k][i] = M[n-1-i][n-1-k] / M[n-1-k][n-1-k].
            let mut t = vec![0f32; n * n];
            par_chunks(&mut t.chunks_mut(n).collect::<Vec<_>>(), |k, row| {
                let mk = n - 1 - k;
                let d = m[mk * n + mk];
                for (i, v) in row.iter_mut().enumerate().take(k) {
                    *v = m[(n - 1 - i) * n + mk] / d;
                }
            });
            return Ok(t);
        }
        lambda *= 10.0;
    }
    Err(Error::new("numerical", "gptq: Gram matrix is not positive definite"))
}

/// int4 code of `v` at scale `d` (q4_0 rounding: floor(v/d + 8.5), clamped to 0..=15).
#[inline]
fn code(v: f32, inv_d: f32) -> f32 {
    (v * inv_d + 8.5).floor().clamp(0.0, 15.0)
}

/// Group scale for values `v` (diag weights `hw`): q4_0's `max / -8` (signed max), or the
/// best of a grid of multiples of it under the weighted squared error. Rounded to f16.
fn group_scale(v: &[f32], hw: &[f32], search: bool) -> f32 {
    let (mut amax, mut max) = (0f32, 0f32);
    for &x in v {
        if x.abs() > amax {
            amax = x.abs();
            max = x;
        }
    }
    if amax == 0.0 {
        return 0.0;
    }
    let base = max / -8.0;
    let round = |d: f32| f16::from_f32(d).to_f32();
    if !search {
        return round(base);
    }
    let err = |d: f32| -> f32 {
        if d == 0.0 {
            return f32::INFINITY;
        }
        let inv = 1.0 / d;
        v.iter().zip(hw).map(|(&x, &w)| {
            let e = x - (code(x, inv) - 8.0) * d;
            w * e * e
        }).sum()
    };
    let mut best = (err(round(base)), round(base));
    for k in 0..16 {
        let d = round(base * (0.80 + 0.02 * k as f32));
        let e = err(d);
        if e < best.0 {
            best = (e, d);
        }
    }
    best.1
}

/// GPTQ-quantize `w` (`[rows, cols]`, row-major) with Gram matrix `h` (`[cols, cols]`) to
/// int4 quant codes (0..16) and group scales. Errors "numerical" for a degenerate `h`.
pub fn quantize_int4(w: &[f32], rows: usize, cols: usize, h: &[f32], opts: &GptqOptions) -> Result<(Vec<u8>, Vec<f32>)> {
    assert!(cols % G == 0 && w.len() == rows * cols && h.len() == cols * cols);
    if opts.act_order {
        return quantize_int4_act_order(w, rows, cols, h, opts);
    }
    let t = feedback(h, cols, opts.damp)?;
    let hw: Vec<f32> = (0..cols).map(|i| h[i * cols + i].max(0.0) + 1e-12).collect();
    let mut q = vec![0u8; rows * cols];
    let mut scales = vec![0f32; rows * cols / G];
    let mut tiles: Vec<(&[f32], &mut [u8], &mut [f32])> = w
        .chunks(TILE * cols)
        .zip(q.chunks_mut(TILE * cols))
        .zip(scales.chunks_mut(TILE * cols / G))
        .map(|((a, b), c)| (a, b, c))
        .collect();
    par_chunks(&mut tiles, |_, (wt, qt, st)| {
        let nr = wt.len() / cols;
        // e[r][i] = w_i - ŵ_i for the columns done so far.
        let mut e = vec![0f32; nr * cols];
        let mut v = vec![0f32; nr * G];
        for g0 in (0..cols).step_by(G) {
            // Error-corrected group values from the columns before the group.
            for k in g0..g0 + G {
                let tk = &t[k * cols..k * cols + g0];
                for r in 0..nr {
                    v[r * G + k - g0] = wt[r * cols + k] + dot(&e[r * cols..r * cols + g0], tk);
                }
            }
            for r in 0..nr {
                let d = group_scale(&v[r * G..(r + 1) * G], &hw[g0..g0 + G], opts.scale_search);
                st[r * cols / G + g0 / G] = d;
                let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
                for k in g0..g0 + G {
                    let tk = &t[k * cols + g0..k * cols + k];
                    let val = v[r * G + k - g0] + dot(&e[r * cols + g0..r * cols + k], tk);
                    let c = if d != 0.0 { code(val, inv) } else { 8.0 };
                    qt[r * cols + k] = c as u8;
                    e[r * cols + k] = wt[r * cols + k] - (c - 8.0) * d;
                }
            }
        }
    });
    Ok((q, scales))
}

/// [`quantize_int4`] in act order (static groups).
fn quantize_int4_act_order(w: &[f32], rows: usize, cols: usize, h: &[f32], opts: &GptqOptions) -> Result<(Vec<u8>, Vec<f32>)> {
    let hw: Vec<f32> = (0..cols).map(|i| h[i * cols + i].max(0.0) + 1e-12).collect();
    let mut perm: Vec<usize> = (0..cols).collect();
    perm.sort_by(|&a, &b| hw[b].total_cmp(&hw[a]));
    let mut hp = vec![0f32; cols * cols];
    for (i, row) in hp.chunks_exact_mut(cols).enumerate() {
        let src = &h[perm[i] * cols..(perm[i] + 1) * cols];
        for (j, v) in row.iter_mut().enumerate() {
            *v = src[perm[j]];
        }
    }
    let t = feedback(&hp, cols, opts.damp)?;
    drop(hp);
    let mut q = vec![0u8; rows * cols];
    let mut scales = vec![0f32; rows * cols / G];
    let mut tiles: Vec<(&[f32], &mut [u8], &mut [f32])> = w
        .chunks(TILE * cols)
        .zip(q.chunks_mut(TILE * cols))
        .zip(scales.chunks_mut(TILE * cols / G))
        .map(|((a, b), c)| (a, b, c))
        .collect();
    par_chunks(&mut tiles, |_, (wt, qt, st)| {
        let nr = wt.len() / cols;
        for r in 0..nr {
            for g0 in (0..cols).step_by(G) {
                st[r * cols / G + g0 / G] =
                    group_scale(&wt[r * cols + g0..r * cols + g0 + G], &hw[g0..g0 + G], opts.scale_search);
            }
        }
        // e[r][k] = w - ŵ of the k-th column in permuted order.
        let mut e = vec![0f32; nr * cols];
        for (k, &c) in perm.iter().enumerate() {
            let tk = &t[k * cols..k * cols + k];
            for r in 0..nr {
                let d = st[r * cols / G + c / G];
                let x = wt[r * cols + c];
                let val = x + dot(&e[r * cols..r * cols + k], tk);
                let code = if d != 0.0 { code(val, 1.0 / d) } else { 8.0 };
                qt[r * cols + c] = code as u8;
                e[r * cols + k] = x - (code - 8.0) * d;
            }
        }
    });
    Ok((q, scales))
}

/// Proxy loss `Σ_rows (ŵ - w) H (ŵ - w)ᵀ` (tests).
#[cfg(test)]
pub fn proxy_loss(w: &[f32], wq: &[f32], rows: usize, cols: usize, h: &[f32]) -> f64 {
    let mut total = 0f64;
    for r in 0..rows {
        let d: Vec<f32> = (0..cols).map(|c| wq[r * cols + c] - w[r * cols + c]).collect();
        for i in 0..cols {
            total += d[i] as f64 * dot(&h[i * cols..(i + 1) * cols], &d) as f64;
        }
    }
    total
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

    #[test]
    fn cholesky_reconstructs() {
        let n = 150;
        let x = noise(n * 200, 5);
        let mut a = vec![0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (0..200).map(|t| x[t * n + i] * x[t * n + j]).sum::<f32>() + if i == j { 1.0 } else { 0.0 };
            }
        }
        let orig = a.clone();
        cholesky(&mut a, n).expect("pd");
        for i in 0..n {
            for j in 0..=i {
                let s: f32 = (0..=j).map(|k| a[i * n + k] * a[j * n + k]).sum();
                assert!((s - orig[i * n + j]).abs() < 1e-3 * orig[i * n + i].abs().max(1.0), "({i},{j})");
            }
        }
        let mut bad = vec![1.0, 2.0, 2.0, 1.0];
        assert!(cholesky(&mut bad, 2).is_err());
    }

    /// GPTQ beats RTN on the proxy loss for correlated inputs.
    #[test]
    fn beats_rtn() {
        let (rows, cols, tokens) = (24, 128, 512);
        let w = noise(rows * cols, 1);
        // Correlated activations with a few large channels.
        let base = noise(tokens * 8, 2);
        let mix = noise(8 * cols, 3);
        let mut x = vec![0f32; tokens * cols];
        for t in 0..tokens {
            for c in 0..cols {
                let s: f32 = (0..8).map(|k| base[t * 8 + k] * mix[k * cols + c]).sum();
                x[t * cols + c] = s * if c % 17 == 0 { 8.0 } else { 1.0 };
            }
        }
        let mut h = vec![0f32; cols * cols];
        for t in 0..tokens {
            let xt = &x[t * cols..(t + 1) * cols];
            for i in 0..cols {
                for j in 0..cols {
                    h[i * cols + j] += xt[i] * xt[j];
                }
            }
        }
        let rtn = lapid_quant::dequantize(
            lapid_quant::Format::Int4,
            &lapid_quant::quantize(lapid_quant::Format::Int4, &w, rows, cols),
            rows,
            cols,
        );
        let (q, s) = quantize_int4(&w, rows, cols, &h, &GptqOptions::default()).expect("gptq");
        let packed = lapid_quant::pack_int4(&q, &s, rows, cols);
        let gq = lapid_quant::dequantize(lapid_quant::Format::Int4, &packed, rows, cols);
        let (l_rtn, l_gptq) = (proxy_loss(&w, &rtn, rows, cols, &h), proxy_loss(&w, &gq, rows, cols, &h));
        assert!(l_gptq < 0.7 * l_rtn, "gptq {l_gptq} vs rtn {l_rtn}");
        let ao = GptqOptions { act_order: true, ..Default::default() };
        let (q, s) = quantize_int4(&w, rows, cols, &h, &ao).expect("act order");
        let aq = lapid_quant::dequantize(lapid_quant::Format::Int4, &lapid_quant::pack_int4(&q, &s, rows, cols), rows, cols);
        let l_ao = proxy_loss(&w, &aq, rows, cols, &h);
        assert!(l_ao < 0.7 * l_rtn, "act order {l_ao} vs rtn {l_rtn}");
        // Identity Gram: no feedback, plain (searched) rounding — never worse than RTN.
        let eye: Vec<f32> = (0..cols * cols).map(|i| if i % (cols + 1) == 0 { 1.0 } else { 0.0 }).collect();
        let (q, s) = quantize_int4(&w, rows, cols, &eye, &GptqOptions::default()).expect("gptq");
        let gq = lapid_quant::dequantize(lapid_quant::Format::Int4, &lapid_quant::pack_int4(&q, &s, rows, cols), rows, cols);
        assert!(proxy_loss(&w, &gq, rows, cols, &eye) <= proxy_loss(&w, &rtn, rows, cols, &eye) * 1.01);
    }
}
