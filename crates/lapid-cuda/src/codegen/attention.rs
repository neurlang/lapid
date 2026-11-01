//! Fused attention: softmax(scale * q k^T + mask, causal) v without materializing scores.
//!
//! Many queries (encoder) on sm_80+: tensor-core flash attention (`flash.rs`).
//! Few queries (decoder) or no tensor cores: split over 64-key chunks
//! (`kernels/attn_split.ptx`) plus a merge pass (`kernels/attn_merge.ptx`) when there is
//! more than one chunk.
//!
//! Params: `[B, H, sq, sk, d, scale_bits, nsplit | q_blocks, q(off, s0..s3), k(..), v(..), mask(..)?]`

use std::sync::Arc;

use lapid_compiler::{AttentionDims, AttentionSpec, View};
use lapid_tensor::{DType, Error, Result};

use super::ptx::{self, Gen};
use super::{Key, Params, Plan, bad, check_plain, flash, grid_threads, grid1, template};
use crate::context::Caps;

/// Keys per block of the split variant.
const SPLIT_KEYS: usize = 64;
/// Largest head dim of the split variant (one thread per dim).
const SPLIT_MAX_D: usize = 256;
/// Fewest queries for the tensor-core kernel (64-query tiles).
const MMA_MIN_Q: usize = 16;

const SPLIT: &str = include_str!("../kernels/attn_split.ptx");
const MERGE: &str = include_str!("../kernels/attn_merge.ptx");

pub(super) fn plan(spec: &Arc<AttentionSpec>, d: &AttentionDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.dtype, "attention q/k/v")?;
    check_plain(spec.out_dtype, "attention output")?;
    if let Some(m) = spec.mask_dtype {
        check_plain(m, "attention mask")?;
    }
    if spec.mask_dtype.is_some() != d.mask.is_some() {
        return Err(bad("attention: mask dtype / view mismatch"));
    }
    let mut p = Params::default();
    p.u(d.batch)?
        .u(d.heads)?
        .u(d.sq)?
        .u(d.sk)?
        .u(d.d)?
        .bits(d.scale)
        .u(0)?;
    p.view(&d.q, 4)?.view(&d.k, 4)?.view(&d.v, 4)?;
    if let Some(m) = &d.mask {
        p.view(m, 4)?;
    }
    let rows = d.batch * d.heads * d.sq;
    if rows == 0 || d.d == 0 {
        return Ok(Plan::new(
            Key::AttnSplit(spec.clone(), true, d.d.max(1) as u32),
            p.0,
            [0, 0, 0],
            256,
        ));
    }
    let mma = caps.mma() && d.d.is_multiple_of(16) && d.d <= flash::MAX_D && d.sq >= MMA_MIN_Q;
    if mma {
        let q_blocks = d.sq.div_ceil(flash::BQ);
        let bh = d.batch * d.heads;
        if bh > caps.max_grid[1] as usize {
            return Err(Error::new(
                "too_large",
                format!("attention: {bh} batch x heads exceeds the grid limit"),
            ));
        }
        p.0[6] = q_blocks as u32;
        let vec = kv_vec(spec.dtype, &d.k, d.d) && kv_vec(spec.dtype, &d.v, d.d);
        return Ok(Plan::new(
            Key::AttnMma(spec.clone(), d.d as u32, vec),
            p.0,
            [q_blocks as u32, bh as u32, 1],
            flash::THREADS,
        ));
    }
    if d.d > SPLIT_MAX_D {
        return Err(Error::new(
            "unsupported",
            format!("attention head dim {} > {SPLIT_MAX_D}", d.d),
        ));
    }
    let nsplit = d.sk.div_ceil(SPLIT_KEYS).max(1);
    p.0[6] = u32::try_from(nsplit).map_err(|_| bad("attention: too many keys"))?;
    let grid = grid1((rows * nsplit) as u64, caps)?;
    if nsplit == 1 {
        return Ok(Plan::new(
            Key::AttnSplit(spec.clone(), true, d.d as u32),
            p.0,
            grid,
            256,
        ));
    }
    let mut mp = Params::default();
    mp.u(rows)?.u(d.d)?.u(nsplit)?;
    let merge = Plan::new(
        Key::AttnMerge(spec.clone()),
        mp.0,
        grid_threads((rows * d.d) as u64, caps)?,
        256,
    );
    let mut plan = Plan::new(
        Key::AttnSplit(spec.clone(), false, d.d as u32),
        p.0,
        grid,
        256,
    );
    plan.scratch = rows * nsplit * (d.d + 2) * 4;
    plan.next = Some(Box::new(merge));
    Ok(plan)
}

/// K / V rows can be read as 16-byte vectors: contiguous head dim, 16-byte aligned rows.
fn kv_vec(t: DType, v: &View, d: usize) -> bool {
    let align = match t {
        DType::F16 => 8,
        DType::F32 => 4,
        _ => return false,
    };
    d.is_multiple_of(8)
        && v.strides.len() == 4
        && v.strides[3] == 1
        && v.offset.is_multiple_of(align)
        && v.strides[..3]
            .iter()
            .all(|&s| s.rem_euclid(align as isize) == 0)
}

/// Buffer names: q, k, v, [msk], out, [scr].
pub(super) fn names(spec: &AttentionSpec, scratch: bool) -> Vec<&'static str> {
    let mut v = vec!["q", "k", "v"];
    if spec.mask_dtype.is_some() {
        v.push("msk");
    }
    v.push("out");
    if scratch {
        v.push("scr");
    }
    v
}

pub(super) fn flags(spec: &AttentionSpec) -> Vec<&'static str> {
    let mut f = Vec::new();
    if spec.mask_dtype.is_some() {
        f.push("MASK");
    }
    if spec.causal {
        f.push("CAUSAL");
    }
    f
}

fn snippet(f: impl FnOnce(&mut Gen)) -> String {
    let mut g = Gen::new();
    f(&mut g);
    g.finish()
}

pub(super) fn source_split(spec: &AttentionSpec, direct: bool, d: u32) -> String {
    let (args, ptrs) = ptx::args(&names(spec, !direct));
    let t = spec.dtype;
    let load_q = snippet(|g| ptx::load(g, t, "%g_q", "%t", "%qv", true));
    let load_k = snippet(|g| ptx::load(g, t, "%g_k", "%t", "%kv", true));
    let load_v = snippet(|g| ptx::load(g, t, "%g_v", "%t", "%vv", true));
    let load_m = match spec.mask_dtype {
        Some(m) => snippet(|g| ptx::load(g, m, "%g_msk", "%t", "%mv", true)),
        None => String::new(),
    };
    let store = snippet(|g| ptx::store(g, spec.out_dtype, "%g_out", "%t", "%x"));
    let d = d.clamp(1, SPLIT_MAX_D as u32);
    let g = 256 / d;
    let (ds, gs, dc, vt, d2) = (
        d.to_string(),
        g.to_string(),
        d.div_ceil(32).to_string(),
        (SPLIT_KEYS as u32).div_ceil(g).to_string(),
        (d + 2).to_string(),
    );
    let mut fl = flags(spec);
    if direct {
        fl.push("DIRECT");
    }
    template::fill(
        SPLIT,
        &[
            ("ARGS", &args),
            ("PTRS", &ptrs),
            ("LOAD_Q", &load_q),
            ("LOAD_K", &load_k),
            ("LOAD_V", &load_v),
            ("LOAD_M", &load_m),
            ("STORE", &store),
            ("D", &ds),
            ("G", &gs),
            ("DC", &dc),
            ("VT", &vt),
            ("D2", &d2),
        ],
        &fl,
    )
}

pub(super) fn source_merge(spec: &AttentionSpec) -> String {
    let (args, ptrs) = ptx::args(&names(spec, true));
    let store = snippet(|g| ptx::store(g, spec.out_dtype, "%g_out", "%e", "%x"));
    template::fill(
        MERGE,
        &[("ARGS", &args), ("PTRS", &ptrs), ("STORE", &store)],
        &[],
    )
}
