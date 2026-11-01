//! `GatedDeltaNet`: the gated delta rule recurrence (see `Op::GatedDeltaNet`).
//!
//! Per (batch, head) the state `S [Dk, Dv]` is updated token by token:
//! `S *= exp(g); u = Sᵀk; S += k ⊗ beta (v - u); o = Sᵀq * scale`. Every value is a per-column
//! (`Dv`) quantity accumulated over the `Dk` rows, so columns are independent: tasks are
//! (batch, head, block of `JB` columns), which gives enough parallelism for a single decode
//! token (16 heads -> 64 tasks) and keeps each task's slice of the state in L1 across tokens.
//! The state is read from the input on the first token and then updated in place in its
//! final location in the output (no task-local copy, no copy-out). The arithmetic per
//! element is the same sequence as the straightforward loop (unfused mul / add, rows in
//! order), so results do not depend on the blocking.

use lapid_compiler::GatedDeltaNetDims;
use lapid_tensor::{Error, Result};

use crate::Ctx;
use crate::par::{self, PAR_MIN_WORK};
use crate::typed::{Dst, Src};

/// Columns per task.
const JB: usize = 32;

fn err(code: &'static str, msg: String) -> Error {
    Error::new(code, msg)
}

/// Extra destinations (see `fuse`): the states as contiguous `[b*h, dk*dv]`, the outputs as
/// contiguous `[b*h, t*dv]`.
#[derive(Default)]
pub(crate) struct Extra {
    pub state: Option<Dst>,
    pub o: Option<Dst>,
}

pub(crate) fn gated_delta_net(ctx: &Ctx, d: &GatedDeltaNetDims, ins: &[Src], out: &Dst, extra: &Extra) -> Result<()> {
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
    if out.len() < b * h * rows * dv || out.dtype() != lapid_tensor::DType::F32 {
        return Err(err("out_of_bounds", "cpu gated_delta_net: output too small or not f32".into()));
    }
    let optr = crate::par::SyncPtr(out.f32_ptr());
    let xdst = |x: &Option<Dst>, n: usize| -> Result<Option<crate::par::SyncPtr<f32>>> {
        match x {
            Some(x) if x.dtype() != lapid_tensor::DType::F32 || x.len() < n => {
                Err(err("out_of_bounds", "cpu gated_delta_net: extra destination too small or not f32".into()))
            }
            Some(x) => Ok(Some(crate::par::SyncPtr(x.f32_ptr()))),
            None => Ok(None),
        }
    };
    let xs = xdst(&extra.state, b * h * dk * dv)?;
    let xo = xdst(&extra.o, b * h * t * dv)?;
    let jblocks = dv.div_ceil(JB);
    let tasks = b * h * jblocks;
    let parallel = ctx.threads > 1 && tasks > 1 && b * h * t * dk * dv * 6 >= PAR_MIN_WORK;
    par::for_each(parallel, tasks, || (), |_, u| {
        let p = optr;
        let (bh, jb) = (u / jblocks, u % jblocks);
        let (bi, hi) = (bh / h, bh % h);
        let j0 = jb * JB;
        let w = JB.min(dv - j0);
        let obase = bh * rows * dv;
        let sbase = obase + t * dv;
        let s_in = &st[bh * dk * dv..(bh + 1) * dk * dv];
        let mut uacc = [0f32; JB];
        let mut dd = [0f32; JB];
        let mut oacc = [0f32; JB];
        for ti in 0..t {
            // Row of token ti in q/k/v ([B, T, H, D]) and in g/beta ([B, T, H]).
            let r = (bi * t + ti) * h + hi;
            let kv = &ks[r * dk..(r + 1) * dk];
            let qv = &qs[r * dk..(r + 1) * dk];
            let vv = &vs[r * dv + j0..r * dv + j0 + w];
            let decay = gs[r].exp();
            let beta = betas[r];
            let (uacc, dd, oacc) = (&mut uacc[..w], &mut dd[..w], &mut oacc[..w]);
            uacc.fill(0.0);
            for (i, &ki) in kv.iter().enumerate() {
                // SAFETY: row i, columns [j0, j0 + w) of this head's state block in the
                // output; bounds checked above, and only this task touches these columns.
                let srow = unsafe { std::slice::from_raw_parts_mut(p.0.add(sbase + i * dv + j0), w) };
                if ti == 0 {
                    let src = &s_in[i * dv + j0..i * dv + j0 + w];
                    for ((x, &s0), acc) in srow.iter_mut().zip(src).zip(uacc.iter_mut()) {
                        *x = s0 * decay;
                        *acc += *x * ki;
                    }
                } else {
                    for (x, acc) in srow.iter_mut().zip(uacc.iter_mut()) {
                        *x *= decay;
                        *acc += *x * ki;
                    }
                }
            }
            for ((dj, &uj), &vj) in dd.iter_mut().zip(uacc.iter()).zip(vv) {
                *dj = beta * (vj - uj);
            }
            oacc.fill(0.0);
            for (i, (&ki, &qi)) in kv.iter().zip(qv).enumerate() {
                let qi = qi * scale;
                // SAFETY: as above.
                let srow = unsafe { std::slice::from_raw_parts_mut(p.0.add(sbase + i * dv + j0), w) };
                for ((x, &dj), oj) in srow.iter_mut().zip(dd.iter()).zip(oacc.iter_mut()) {
                    *x += ki * dj;
                    *oj += *x * qi;
                }
            }
            // SAFETY: output row ti, columns [j0, j0 + w) of this head; this task only.
            let orow = unsafe { std::slice::from_raw_parts_mut(p.0.add(obase + ti * dv + j0), w) };
            orow.copy_from_slice(oacc);
            if let Some(xo) = xo {
                // SAFETY: row (bh, ti), columns [j0, j0 + w) of the [b*h, t*dv] destination
                // (length checked above); this task only.
                let x = unsafe { std::slice::from_raw_parts_mut(xo.0.add((bh * t + ti) * dv + j0), w) };
                x.copy_from_slice(oacc);
            }
        }
        if t == 0 {
            for i in 0..dk {
                // SAFETY: as above.
                let srow = unsafe { std::slice::from_raw_parts_mut(p.0.add(sbase + i * dv + j0), w) };
                srow.copy_from_slice(&s_in[i * dv + j0..i * dv + j0 + w]);
            }
        }
        if let Some(xs) = xs {
            for i in 0..dk {
                // SAFETY: state row (bh, i), columns [j0, j0 + w) in both buffers (lengths
                // checked above); this task only.
                unsafe { std::ptr::copy_nonoverlapping(p.0.add(sbase + i * dv + j0), xs.0.add((bh * dk + i) * dv + j0), w) };
            }
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use lapid_tensor::DType;

    use super::*;

    /// f64 reference of the recurrence (ggml semantics).
    #[allow(clippy::type_complexity)]
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
    fn delta_net_matches_reference() {
        let programs = crate::ProgramCache::default();
        let ctx = Ctx { threads: 4, isa: crate::Isa::detect(), programs: &programs };
        for &(b, t, h, dk, dv) in &[(1, 1, 2, 8, 8), (2, 5, 3, 16, 12), (1, 7, 4, 32, 64), (1, 1, 16, 128, 128), (1, 3, 2, 16, 80), (1, 0, 2, 4, 4)] {
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
            let mut xs = vec![0f32; b * h * dk * dv];
            let mut xo = vec![0f32; b * h * t * dv];
            let extra = Extra {
                state: Some(Dst::new(xs.as_mut_ptr() as *mut u8, xs.len(), DType::F32)),
                o: Some(Dst::new(xo.as_mut_ptr() as *mut u8, xo.len(), DType::F32)),
            };
            gated_delta_net(&ctx, &d, &ins, &dst, &extra).unwrap();
            for bh in 0..b * h {
                let base = bh * (t + dk) * dv;
                assert_eq!(&got[base..base + t * dv], &xo[bh * t * dv..(bh + 1) * t * dv]);
                assert_eq!(&got[base + t * dv..base + (t + dk) * dv], &xs[bh * dk * dv..(bh + 1) * dk * dv]);
            }
            for (i, (&gv, &wv)) in got.iter().zip(&want).enumerate() {
                assert!((gv as f64 - wv).abs() <= 1e-4 * wv.abs().max(1.0), "{:?} @{i}: {gv} vs {wv}", (b, t, h, dk, dv));
            }
        }
    }
}
