//! Launch-sequence peepholes the CPU backend applies while executing a schedule.
//!
//! `GatedDeltaNet` writes `[o; S]` per head into one buffer, and the graph then copies the new
//! state `S` (and often `o`) out of it with plain elementwise copies: for a decode step that
//! is a 1 MB copy per layer, from memory just written by other cores into a cold output.
//! When such a copy follows the delta-net launch (with nothing in between touching its
//! destination), the delta-net kernel writes the copied region straight into the copy's
//! destination too, from its L1-resident state block, and the copy launch is skipped.

use lapid_compiler::{BufId, Expr, GatedDeltaNetDims, Kernel, Launch, View};
use lapid_tensor::DType;

use crate::buffer::CpuBuffer;
use crate::view::collapse;

/// Copies of a delta-net output taken over by the kernel: (launch index, destination buffer).
#[derive(Default, Debug)]
pub(crate) struct GdnCopies {
    /// Destination receiving the states contiguously as `[b*h, dk*dv]`.
    pub state: Option<(usize, BufId)>,
    /// Destination receiving the outputs contiguously as `[b*h, t*dv]`.
    pub o: Option<(usize, BufId)>,
}

/// How far after the delta-net launch a copy is looked for.
const LOOKAHEAD: usize = 4;

/// Copies after launch `li` (a `GatedDeltaNet`) that the kernel can perform itself.
pub(crate) fn gdn_copies(launches: &[Launch], li: usize, bufs: &[&CpuBuffer]) -> GdnCopies {
    let mut r = GdnCopies::default();
    let Some(l) = launches.get(li) else { return r };
    let Kernel::GatedDeltaNet(_, d) = &l.kernel else { return r };
    let src = l.output;
    let ptr = |id: BufId| bufs.get(id).map(|b| b.as_ptr());
    for j in li + 1..(li + 1 + LOOKAHEAD).min(launches.len()) {
        let c = &launches[j];
        let dst = c.output;
        // Every launch strictly between li and j must leave the destination memory alone
        // (temps share slots) and must not write the delta-net output.
        let clear = launches[li + 1..j].iter().all(|m| {
            ptr(m.output) != ptr(dst)
                && ptr(m.output) != ptr(src)
                && m.inputs.iter().all(|&i| ptr(i) != ptr(dst))
        });
        if !clear {
            break;
        }
        // The kernel writes the destination while it still reads its operands (temps of
        // later launches may reuse an operand's slot). Only the state destination may be the
        // state input itself: each element is read before it is overwritten in place.
        let clobbers = |may: Option<usize>| {
            ptr(dst) == ptr(src) || l.inputs.iter().enumerate().any(|(k, &i)| ptr(i) == ptr(dst) && Some(k) != may)
        };
        match region(c, src, d) {
            Some(Region::State) if r.state.is_none() && !clobbers(Some(5)) => r.state = Some((j, dst)),
            Some(Region::O) if r.o.is_none() && !clobbers(None) => r.o = Some((j, dst)),
            _ => {}
        }
    }
    r
}

enum Region {
    State,
    O,
}

/// Which region of the delta-net output `src` the launch `c` copies out contiguously.
fn region(c: &Launch, src: BufId, d: &GatedDeltaNetDims) -> Option<Region> {
    let Kernel::Elementwise(spec, dims) = &c.kernel else { return None };
    if spec.expr != Expr::Load(0) || spec.in_dtypes != [DType::F32] || spec.out_dtype != DType::F32 || c.inputs != [src] {
        return None;
    }
    let (bh, t, dk, dv) = ((d.b * d.h) as isize, d.t as isize, d.dk as isize, d.dv as isize);
    let got = canon(&dims.shape, &dims.out_view, &dims.in_views[0]);
    let want = |rows: isize, at: isize| canon(&[bh as usize, (rows * dv) as usize], &cview(rows * dv), &View { offset: (at * dv) as usize, strides: vec![(t + dk) * dv, 1] });
    if dk > 0 && got == want(dk, t) {
        return Some(Region::State);
    }
    if t > 0 && got == want(t, 0) {
        return Some(Region::O);
    }
    None
}

fn cview(cols: isize) -> View {
    View { offset: 0, strides: vec![cols, 1] }
}

/// Collapsed (shape, out offset/strides, in offset/strides) of a copy.
#[allow(clippy::type_complexity)]
fn canon(shape: &[usize], out: &View, inp: &View) -> Option<(Vec<usize>, usize, Vec<isize>, usize, Vec<isize>)> {
    if out.strides.len() != shape.len() || inp.strides.len() != shape.len() {
        return None;
    }
    let c = collapse(shape, &[&out.strides, &inp.strides]);
    Some((c.shape, out.offset, c.strides[0].clone(), inp.offset, c.strides[1].clone()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lapid_compiler::{Backend, EwDims, EwSpec, GatedDeltaNetSpec};

    use super::*;
    use crate::CpuBackend;

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// GDN + state copy + o copy executed as one sequence (fused) and launch by launch.
    #[test]
    fn fused_copies_match() {
        let be = CpuBackend::with_threads(4);
        for t in [1usize, 3] {
            let (b, h, dk, dv) = (1usize, 3usize, 16usize, 40usize);
            let up = |v: &[f32]| be.upload(bytemuck::cast_slice(v)).unwrap();
            let g: Vec<f32> = noise(b * t * h, 4).iter().map(|x| -x.abs()).collect();
            let mut bufs = vec![
                up(&noise(b * t * h * dk, 1)),
                up(&noise(b * t * h * dk, 2)),
                up(&noise(b * t * h * dv, 3)),
                up(&g),
                up(&noise(b * t * h, 5)),
                up(&noise(b * h * dk * dv, 6)),
            ];
            let rows = t + dk;
            bufs.push(be.alloc(b * h * rows * dv * 4).unwrap()); // 6: gdn out
            bufs.push(be.alloc(b * h * dk * dv * 4).unwrap()); // 7: state copy
            bufs.push(be.alloc(b * h * t * dv * 4).unwrap()); // 8: o copy
            let gdn = Launch {
                kernel: Kernel::GatedDeltaNet(
                    Arc::new(GatedDeltaNetSpec { out_dtype: DType::F32 }),
                    GatedDeltaNetDims { b, t, h, dk, dv, scale: 0.25f32.to_bits() },
                ),
                inputs: vec![0, 1, 2, 3, 4, 5],
                output: 6,
            };
            let copy = |at: usize, n: usize, out: BufId| Launch {
                kernel: Kernel::Elementwise(
                    Arc::new(EwSpec { rank: 3, in_dtypes: vec![DType::F32], out_dtype: DType::F32, expr: Expr::Load(0) }),
                    EwDims {
                        shape: vec![b * h, n, dv],
                        in_views: vec![View { offset: at * dv, strides: vec![(rows * dv) as isize, dv as isize, 1] }],
                        out_view: View::contiguous(&[b * h, n, dv]),
                    },
                ),
                inputs: vec![6],
                output: out,
            };
            let launches = [gdn, copy(t, dk, 7), copy(0, t, 8)];
            let ptrs: Vec<&CpuBuffer> = bufs.iter().map(|b| crate::as_cpu(b).unwrap()).collect();
            let c = gdn_copies(&launches, 0, &ptrs);
            assert_eq!(c.state.map(|x| x.0), Some(1));
            assert_eq!(c.o.map(|x| x.0), Some(2));
            be.execute(&launches, &bufs).unwrap();
            let fused = [be.download(&bufs[7]).unwrap(), be.download(&bufs[8]).unwrap()];
            for l in &launches {
                be.execute(std::slice::from_ref(l), &bufs).unwrap();
            }
            assert_eq!(fused[0], be.download(&bufs[7]).unwrap());
            assert_eq!(fused[1], be.download(&bufs[8]).unwrap());
        }
    }
}
