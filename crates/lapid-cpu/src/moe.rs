//! Round-3 kernels: `MatMulId` (mixture-of-experts matmul) and `GatedDeltaNet`.
//!
//! MatMulId groups the (token, slot) pairs by expert and runs one ordinary matmul per used
//! expert (`matmul::run`: GEMV / int8-dot / K-quant / GEMM paths by row count) on the gathered
//! token rows, then scatters the result rows.
//!
//! GatedDeltaNet runs one task per (batch, head); inside, tokens are sequential. The state
//! `S [Dk, Dv]` stays in a task-local f32 buffer; per token two passes over it:
//! `S *= exp(g); u = Sᵀk` then `S += k ⊗ beta (v - u); o = Sᵀq * scale`.

use lapid_compiler::{GatedDeltaNetDims, MatMulDims, MatMulIdDims, MatMulIdSpec, MatMulSpec, MatView};
use lapid_tensor::{DType, Error, Result};

use crate::Ctx;
use crate::buffer::CpuBuffer;
use crate::matmul::{self, BInput, Workspace};
use crate::par;
use crate::qmat::{self, QMat};
use crate::typed::{Dst, Src};

fn err(code: &'static str, msg: String) -> Error {
    Error::new(code, msg)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_id(
    ctx: &Ctx,
    ws: &mut Workspace,
    spec: &MatMulIdSpec,
    d: &MatMulIdDims,
    x: &Src,
    w: &CpuBuffer,
    ids: &Src,
    out: &Dst,
) -> Result<()> {
    let (t, r, e, n, k) = (d.t, d.r, d.e, d.n, d.k);
    if t * r * n == 0 {
        return Ok(());
    }
    let Src::I32(ids) = ids else {
        return Err(err("unsupported_dtype", "cpu matmul_id: ids must be i32".into()));
    };
    if ids.len() < t * r || x.len() < t * k || out.len() < t * r * n {
        return Err(err("out_of_bounds", format!("cpu matmul_id: buffers too small for t={t} r={r} n={n} k={k}")));
    }
    // Expert -> list of flat (token, slot) indices.
    let mut by_expert: Vec<Vec<usize>> = vec![Vec::new(); e];
    for (i, &id) in ids[..t * r].iter().enumerate() {
        if id < 0 || id as usize >= e {
            return Err(err(
                "index_out_of_range",
                format!("cpu matmul_id: expert id {id} at position {i} out of range for {e} experts"),
            ));
        }
        by_expert[id as usize].push(i);
    }
    let quant = qmat::format_of(spec.w_dtype);
    // SAFETY: launches run sequentially and `w` is an input (never the output buffer).
    let wbytes = unsafe { w.bytes() };
    let one = match quant {
        Some(f) => lapid_quant::layout(f, n, k).total,
        None => n * k,
    };
    if quant.is_some() && wbytes.len() < e * one {
        return Err(err("out_of_bounds", format!("cpu matmul_id: expert stack holds {} < {} bytes", wbytes.len(), e * one)));
    }
    // SAFETY: as above.
    let wsrc = if quant.is_none() { Some(unsafe { w.src(spec.w_dtype)? }) } else { None };
    if let Some(s) = &wsrc
        && s.len() < e * n * k
    {
        return Err(err("out_of_bounds", format!("cpu matmul_id: expert stack holds {} < {} elements", s.len(), e * n * k)));
    }

    let mm_spec = MatMulSpec {
        a_dtype: DType::F32,
        b_dtype: spec.w_dtype,
        b_transposed: true,
        extra_dtypes: Vec::new(),
        out_dtype: DType::F32,
        epilogue: None,
    };
    let mut a_rows: Vec<f32> = Vec::new();
    let mut res: Vec<f32> = Vec::new();
    for (ex, list) in by_expert.iter().enumerate() {
        let m = list.len();
        if m == 0 {
            continue;
        }
        a_rows.resize(m * k, 0.0);
        res.resize(m * n, 0.0);
        for (row, &i) in list.iter().enumerate() {
            x.load_contig((i / r) * k, &mut a_rows[row * k..(row + 1) * k]);
        }
        let (b, b_off) = match (quant, &wsrc) {
            (Some(f), _) => (BInput::Quant(QMat::new(f, &wbytes[ex * one..(ex + 1) * one], n, k)?), 0),
            (None, Some(s)) => (BInput::Plain(*s), ex * n * k),
            (None, None) => return Err(err("internal", "cpu matmul_id: no expert source".into())),
        };
        let dims = MatMulDims {
            batch: 1,
            m,
            n,
            k,
            a: MatView::contiguous(m, k),
            b: MatView { offset: b_off, batch_stride: 0, row_stride: k },
            extra_views: Vec::new(),
            out: MatView::contiguous(m, n),
        };
        let a = Src::F32(&a_rows);
        let dst = Dst::new(res.as_mut_ptr() as *mut u8, res.len(), DType::F32);
        // Prefill-sized launches keep float activations for every expert (as dense
        // matmuls with many rows do), however few rows an expert got: one numeric path.
        ws.float_act = t > 8;
        let r = matmul::run(ctx, ws, &mm_spec, &dims, &a, &b, &[], &dst);
        ws.float_act = false;
        r?;
        for (row, &i) in list.iter().enumerate() {
            // SAFETY: every (token, slot) index appears in exactly one expert list, so the
            // output rows written here are disjoint; single-threaded.
            unsafe { out.store(i * n, 1, &res[row * n..(row + 1) * n]) };
        }
    }
    Ok(())
}

/// One (batch, head) recurrence. `s` is the `[dk, dv]` state (in/out), `o` receives the
/// `t` output rows `[t, dv]`.
#[allow(clippy::too_many_arguments)]
fn gdn_head(
    q: impl Fn(usize) -> usize,
    qs: &[f32],
    ks: &[f32],
    vs: &[f32],
    gb: impl Fn(usize) -> usize,
    gs: &[f32],
    betas: &[f32],
    (t, dk, dv): (usize, usize, usize),
    scale: f32,
    s: &mut [f32],
    o: &mut [f32],
    u: &mut [f32],
) {
    for ti in 0..t {
        let (qo, go) = (q(ti), gb(ti));
        let (qv, kv) = (&qs[qo * dk..(qo + 1) * dk], &ks[qo * dk..(qo + 1) * dk]);
        let vv = &vs[qo * dv..(qo + 1) * dv];
        let decay = gs[go].exp();
        let beta = betas[go];
        u.fill(0.0);
        for i in 0..dk {
            let row = &mut s[i * dv..(i + 1) * dv];
            let ki = kv[i];
            for (x, acc) in row.iter_mut().zip(u.iter_mut()) {
                *x *= decay;
                *acc += *x * ki;
            }
        }
        for (uj, &vj) in u.iter_mut().zip(vv) {
            *uj = beta * (vj - *uj);
        }
        let orow = &mut o[ti * dv..(ti + 1) * dv];
        orow.fill(0.0);
        for i in 0..dk {
            let row = &mut s[i * dv..(i + 1) * dv];
            let (ki, qi) = (kv[i], qv[i] * scale);
            for ((x, &dj), oj) in row.iter_mut().zip(u.iter()).zip(orow.iter_mut()) {
                *x += ki * dj;
                *oj += *x * qi;
            }
        }
    }
}

pub(crate) fn gated_delta_net(ctx: &Ctx, d: &GatedDeltaNetDims, ins: &[Src], out: &Dst) -> Result<()> {
    let (b, t, h, dk, dv) = (d.b, d.t, d.h, d.dk, d.dv);
    let scale = f32::from_bits(d.scale);
    if b * h * dv * (t + dk) == 0 {
        return Ok(());
    }
    let f = |i: usize, want: usize| -> Result<&[f32]> {
        let s = ins.get(i).and_then(Src::as_f32).ok_or_else(|| {
            err("unsupported_dtype", format!("cpu gated_delta_net: input {i} must be f32"))
        })?;
        if s.len() < want {
            return Err(err("out_of_bounds", format!("cpu gated_delta_net: input {i} holds {} < {want}", s.len())));
        }
        Ok(s)
    };
    let (qs, ks) = (f(0, b * t * h * dk)?, f(1, b * t * h * dk)?);
    let vs = f(2, b * t * h * dv)?;
    let (gs, betas) = (f(3, b * t * h)?, f(4, b * t * h)?);
    let st = f(5, b * h * dk * dv)?;
    let rows = t + dk;
    if out.len() < b * h * rows * dv {
        return Err(err("out_of_bounds", "cpu gated_delta_net: output too small".into()));
    }
    let parallel = ctx.threads > 1 && b * h > 1 && t * dk * dv * b * h >= 1 << 15;
    let init = || (vec![0f32; dk * dv], vec![0f32; t * dv], vec![0f32; dv]);
    par::for_each(parallel, b * h, init, |(s, o, u), bh| {
        let (bi, hi) = (bh / h, bh % h);
        s.copy_from_slice(&st[bh * dk * dv..(bh + 1) * dk * dv]);
        // Row of token ti in q/k/v ([B, T, H, D]) and in g/beta ([B, T, H]).
        let row = |ti: usize| (bi * t + ti) * h + hi;
        gdn_head(row, qs, ks, vs, row, gs, betas, (t, dk, dv), scale, s, o, u);
        let base = bh * rows * dv;
        // SAFETY: each task writes its own (batch, head) block of the output.
        unsafe {
            out.store(base, 1, o);
            out.store(base + t * dv, 1, s);
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f64 reference of the recurrence (ggml semantics).
    fn reference(q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32], s0: &[f32], dims: (usize, usize, usize, usize, usize), scale: f64) -> Vec<f64> {
        let (b, t, h, dk, dv) = dims;
        let mut out = vec![0f64; b * h * (t + dk) * dv];
        for bi in 0..b {
            for hi in 0..h {
                let bh = bi * h + hi;
                let mut s: Vec<f64> = s0[bh * dk * dv..(bh + 1) * dk * dv].iter().map(|&x| x as f64).collect();
                for ti in 0..t {
                    let r = (bi * t + ti) * h + hi;
                    let dec = (g[r] as f64).exp();
                    s.iter_mut().for_each(|x| *x *= dec);
                    let mut u = vec![0f64; dv];
                    for i in 0..dk {
                        for j in 0..dv {
                            u[j] += s[i * dv + j] * k[r * dk + i] as f64;
                        }
                    }
                    for i in 0..dk {
                        for j in 0..dv {
                            s[i * dv + j] += k[r * dk + i] as f64 * beta[r] as f64 * (v[r * dv + j] as f64 - u[j]);
                        }
                    }
                    for j in 0..dv {
                        let o: f64 = (0..dk).map(|i| s[i * dv + j] * q[r * dk + i] as f64 * scale).sum();
                        out[(bh * (t + dk) + ti) * dv + j] = o;
                    }
                }
                out[(bh * (t + dk) + t) * dv..(bh + 1) * (t + dk) * dv].copy_from_slice(&s);
            }
        }
        out
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

    #[test]
    fn gated_delta_net_matches_reference() {
        let programs = crate::ProgramCache::default();
        let ctx = Ctx { threads: 4, isa: crate::Isa::detect(), programs: &programs };
        for &(b, t, h, dk, dv) in &[(1, 1, 2, 8, 8), (2, 5, 3, 16, 12), (1, 7, 4, 32, 64)] {
            let q = noise(b * t * h * dk, 1, 1.0);
            let k = noise(b * t * h * dk, 2, 0.5);
            let v = noise(b * t * h * dv, 3, 1.0);
            let g = noise(b * t * h, 4, 1.0).iter().map(|x| -x.abs()).collect::<Vec<_>>();
            let beta = noise(b * t * h, 5, 0.5).iter().map(|x| x.abs()).collect::<Vec<_>>();
            let s0 = noise(b * h * dk * dv, 6, 0.3);
            let scale = 1.0 / (dk as f32).sqrt();
            let want = reference(&q, &k, &v, &g, &beta, &s0, (b, t, h, dk, dv), scale as f64);
            let mut got = vec![0f32; want.len()];
            let dst = Dst::new(got.as_mut_ptr() as *mut u8, got.len(), DType::F32);
            let ins = [Src::F32(&q), Src::F32(&k), Src::F32(&v), Src::F32(&g), Src::F32(&beta), Src::F32(&s0)];
            let d = GatedDeltaNetDims { b, t, h, dk, dv, scale: scale.to_bits() };
            gated_delta_net(&ctx, &d, &ins, &dst).unwrap();
            for (i, (&gv, &wv)) in got.iter().zip(&want).enumerate() {
                assert!((gv as f64 - wv).abs() <= 1e-4 * wv.abs().max(1.0), "{:?} @{i}: {gv} vs {wv}", (b, t, h, dk, dv));
            }
        }
    }
}
