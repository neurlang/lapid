//! Attention: flash tile kernel on simdgroup matrices (`kernels/attn_tile.metal`) for many
//! queries, and the split-key vector kernel (`kernels/attn_vec.metal`) + merge
//! (`kernels/attn_merge.metal`) for decoding.

use std::sync::Arc;

use lapid_compiler::{AttentionDims, AttentionSpec, View};
use lapid_tensor::{DType, Result};

use super::msl::bindings;
use super::template::{self, ATTN_MERGE, ATTN_TILE, ATTN_VEC};
use super::{Bind, Key, Params, Plan, Stage, bad, check_plain, dim, grid_threads};

/// Row blocks (of 8 queries) per simdgroup of the tile kernel. More than one spills the
/// Q / O fragments out of registers (measured 7x slower on M4 Pro at d = 64).
fn row_blocks(_d: usize) -> usize {
    1
}

/// Simdgroups per tile threadgroup: 8 for d <= 64 (the K/V tile is shared by 64 queries),
/// else 4 (threadgroup memory: Q is staged through the K/V tile buffer).
fn tile_simdgroups(d: usize) -> usize {
    if d <= 64 { 8 } else { 4 }
}
/// Keys per vector-kernel threadgroup (one per thread).
const KS: usize = 128;
/// Up to this many queries use the vector kernel.
const VEC_MAX_SQ: usize = 4;

/// Whether the tile kernel supports head dim `d`.
fn tile_ok(d: usize) -> bool {
    d > 0 && d.is_multiple_of(8) && d <= 128
}

pub(super) fn plan(s: &Arc<AttentionSpec>, d: &AttentionDims) -> Result<Plan> {
    check_plain(s.dtype, "attention q/k/v")?;
    check_plain(s.out_dtype, "attention output")?;
    if let Some(m) = s.mask_dtype {
        check_plain(m, "attention mask")?;
    }
    if s.mask_dtype.is_some() != d.mask.is_some() {
        return Err(bad("attention: mask dtype / mask view mismatch"));
    }
    let mut p = Params::default();
    p.u(d.sq)?.u(d.sk)?.bits(d.scale).u(d.heads)?;
    p.view(&d.q, 4)?.view(&d.k, 4)?.view(&d.v, 4)?;
    match &d.mask {
        Some(m) => p.view(m, 4)?,
        None => p.view(&View { offset: 0, strides: vec![0; 4] }, 4)?,
    };
    let nin = 3 + d.mask.is_some() as usize;
    let empty = d.batch == 0 || d.heads == 0 || d.sq == 0 || d.d == 0;
    let hd = u32::try_from(d.d).map_err(|_| bad("attention: head dim too large"))?;

    if tile_ok(d.d) && d.sq > VEC_MAX_SQ {
        let mut binds: Vec<Bind> = (0..nin).map(Bind::Input).collect();
        binds.push(Bind::Output);
        let bq = 8 * row_blocks(d.d) * tile_simdgroups(d.d);
        let groups = [dim(d.sq.div_ceil(bq))?, dim(d.heads)?, dim(d.batch)?];
        let stage = Stage {
            key: Key::AttnTile(s.clone(), hd),
            params: p.0,
            groups: if empty { [0; 3] } else { groups },
            threads: 32 * tile_simdgroups(d.d) as u32,
            binds,
        };
        return Ok(Plan { stages: vec![stage], scratch: 0 });
    }

    let splits = d.sk.div_ceil(KS).max(1);
    let direct = splits == 1;
    p.u(splits)?;
    let mut binds: Vec<Bind> = (0..nin).map(Bind::Input).collect();
    binds.push(if direct { Bind::Output } else { Bind::Scratch });
    let bh = d.batch.checked_mul(d.heads).ok_or_else(|| bad("attention: size overflow"))?;
    let groups = [dim(splits)?, dim(d.sq)?, dim(bh)?];
    let vec = Stage {
        key: Key::AttnVec(s.clone(), hd, direct),
        params: p.0,
        groups: if empty { [0; 3] } else { groups },
        threads: KS as u32,
        binds,
    };
    if direct {
        return Ok(Plan { stages: vec![vec], scratch: 0 });
    }
    let rows = bh * d.sq;
    let mut mp = Params::default();
    mp.u(rows)?.u(d.d)?.u(splits)?;
    let n = rows * d.d;
    let merge = Stage {
        key: Key::AttnMerge(s.out_dtype),
        params: mp.0,
        groups: if empty { [0; 3] } else { grid_threads(n, 256) },
        threads: 256,
        binds: vec![Bind::Scratch, Bind::Output],
    };
    let scratch = rows * splits * (d.d + 2) * 4;
    Ok(Plan { stages: vec![vec, merge], scratch })
}

fn inputs(s: &AttentionSpec) -> Vec<(String, DType)> {
    let mut v = vec![("q".to_string(), s.dtype), ("k".to_string(), s.dtype), ("v".to_string(), s.dtype)];
    if let Some(m) = s.mask_dtype {
        v.push(("mask".to_string(), m));
    }
    v
}

fn flags(s: &AttentionSpec) -> Vec<&'static str> {
    let mut f = Vec::new();
    if s.mask_dtype.is_some() {
        f.push("MASK");
    }
    if s.causal {
        f.push("CAUSAL");
    }
    f
}

pub(super) fn source_tile(s: &AttentionSpec, d: u32) -> String {
    let binds = bindings(&inputs(s), Some(s.out_dtype), &[]);
    let bkv = if d <= 64 { "32" } else { "16" };
    let (ds, rb, nsg) = (d.to_string(), row_blocks(d as usize).to_string(), tile_simdgroups(d as usize).to_string());
    template::module(ATTN_TILE, &[("BINDINGS", &binds), ("D", &ds), ("RB", &rb), ("NSG", &nsg), ("BKV", bkv)], &flags(s))
}

pub(super) fn source_vec(s: &AttentionSpec, d: u32, direct: bool) -> String {
    let binds = if direct {
        bindings(&inputs(s), Some(s.out_dtype), &[])
    } else {
        bindings(&inputs(s), None, &["device float* scratch"])
    };
    let g = if (d as usize) <= KS { KS / d as usize } else { 1 };
    let lk = if d.is_multiple_of(32) { 8 } else { 1 };
    let wide = if (d as usize) >= KS && (d as usize).is_multiple_of(KS) { d as usize / KS } else { 0 };
    let (ds, gs, lks, ws) = (d.to_string(), g.to_string(), lk.to_string(), wide.to_string());
    let mut f = flags(s);
    if direct {
        f.push("DIRECT");
    }
    if wide > 0 {
        f.push("WIDE");
    }
    template::module(ATTN_VEC, &[("BINDINGS", &binds), ("D", &ds), ("G", &gs), ("LK", &lks), ("WIDE", &ws)], &f)
}

pub(super) fn source_merge(out: DType) -> String {
    let binds = format!(
        "    device const float* scratch [[buffer(1)]],\n{}",
        bindings(&[], Some(out), &[]).replace("[[buffer(1)]]", "[[buffer(2)]]")
    );
    template::module(ATTN_MERGE, &[("BINDINGS", &binds)], &[])
}
