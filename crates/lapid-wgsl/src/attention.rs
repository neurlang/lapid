//! Fused attention: softmax(scale * q k^T + mask, causal) v without materializing scores.
//!
//! Tile variant (many queries): 64 queries per workgroup, one thread per query holding q
//! and the output accumulator in registers (head dim baked in, vec4 granularity); K/V tiles
//! staged through shared memory; per-key online softmax.
//! Row variant (few queries, e.g. the decoder): one workgroup per query; scores for a chunk
//! of keys computed in parallel into shared memory, workgroup max/sum reductions, then the
//! weighted V sum split over (key group, head-dim lane).
//!
//! Params: `[B, H, sq, sk, d, scale_bits, q_blocks, q(off, s0..s3), k(..), v(..), mask(..)?]`

use std::sync::Arc;

use lapid_compiler::{AttentionDims, AttentionSpec};
use lapid_tensor::{Error, Result};

use super::common;
use super::{Key, Params, Plan, bad, check_plain, grid, grid_threads, template};
use crate::Caps;

const TILE_Q: usize = 64;
/// Head dims above 256 run DC = ceil(D / 256) passes per thread in the split kernel.
const ROW_MAX_D: usize = 1024;

pub(super) fn plan(spec: &Arc<AttentionSpec>, d: &AttentionDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.dtype, "attention q/k/v")?;
    check_plain(spec.out_dtype, "attention output")?;
    if let Some(m) = spec.mask_dtype {
        check_plain(m, "attention mask")?;
    }
    if spec.mask_dtype.is_some() != d.mask.is_some() {
        return Err(bad("attention: mask dtype / view mismatch"));
    }
    let tile_d = d.d.is_multiple_of(4) && d.d <= 128 && d.d > 0;
    let tile = tile_d && d.sq > 8 && tile_shared_bytes(d.d) <= caps.max_shared as usize;
    if !tile && d.d > ROW_MAX_D {
        return Err(Error::new("unsupported", format!("attention head dim {} > {ROW_MAX_D}", d.d)));
    }
    let q_blocks = d.sq.div_ceil(TILE_Q);
    let mut p = Params::default();
    p.u(d.batch)?.u(d.heads)?.u(d.sq)?.u(d.sk)?.u(d.d)?.bits(d.scale).u(q_blocks)?;
    p.view(&d.q, 4)?.view(&d.k, 4)?.view(&d.v, 4)?;
    if let Some(m) = &d.mask {
        p.view(m, 4)?;
    }
    if tile && caps.coopmat && d.d == 64 {
        let items = (d.batch * d.heads * q_blocks) as u64;
        return Ok(Plan { scratch: 0, next: None, key: Key::AttnCoop(spec.clone()), params: p.0, groups: grid(items, caps) });
    }
    if tile {
        let items = (d.batch * d.heads * q_blocks) as u64;
        let groups = if d.d == 0 { [0, 0, 0] } else { grid(items, caps) };
        return Ok(Plan { scratch: 0, next: None, key: Key::AttnTile(spec.clone(), d.d as u32), params: p.0, groups });
    }
    // split over key chunks; `nsplit` sits at params[6] (the tile variant's q_blocks slot)
    let nsplit = d.sk.div_ceil(SPLIT_KEYS).max(1);
    p.0[6] = u32::try_from(nsplit).map_err(|_| bad("attention: too many keys"))?;
    let rows = d.batch * d.heads * d.sq;
    let items = (rows * nsplit) as u64;
    let groups = if d.d == 0 || rows == 0 { [0, 0, 0] } else { grid(items, caps) };
    if nsplit == 1 {
        return Ok(Plan { scratch: 0, next: None, key: Key::AttnSplit(spec.clone(), true, d.d as u32), params: p.0, groups });
    }
    let mut mp = Params::default();
    mp.u(rows)?.u(d.d)?.u(nsplit)?;
    let merge = Plan {
        key: Key::AttnMerge(spec.clone()),
        params: mp.0,
        groups: grid_threads((rows * d.d) as u64, caps),
        scratch: 0,
        next: None,
    };
    Ok(Plan {
        key: Key::AttnSplit(spec.clone(), false, d.d as u32),
        params: p.0,
        groups,
        scratch: rows * nsplit * (d.d + 2) * 4,
        next: Some(Box::new(merge)),
    })
}

/// Keys per workgroup of the split variant (`CH` in attn_split.wgsl).
const SPLIT_KEYS: usize = 64;
const SPLIT: &str = include_str!("kernels/attn_split.wgsl");
const MERGE: &str = include_str!("kernels/attn_merge.wgsl");

fn split_flags(spec: &AttentionSpec, direct: bool) -> Vec<&'static str> {
    let mut f = Vec::new();
    if spec.mask_dtype.is_some() {
        f.push("MASK");
    }
    if spec.causal {
        f.push("CAUSAL");
    }
    if direct {
        f.push("DIRECT");
    }
    f
}

pub(super) fn source_split(spec: &AttentionSpec, direct: bool, d: u32, caps: &Caps) -> String {
    let mut b = bindings(spec, caps);
    b.push_str(template::WG_REDUCE);
    let sb = (5 + spec.mask_dtype.is_some() as usize).to_string();
    let (ch, ds) = (SPLIT_KEYS.to_string(), d.to_string());
    template::module(caps, SPLIT, &[("BINDINGS", &b), ("SCRATCH_BINDING", &sb), ("CH", &ch), ("D", &ds)], &split_flags(spec, direct), 256, true)
}

const COOP: &str = include_str!("kernels/attn_coop.wgsl");

pub(super) fn source_coop(spec: &AttentionSpec, caps: &Caps) -> String {
    let b = bindings(spec, caps);
    let mut flags = split_flags(spec, false);
    flags.push("COOPMAT");
    template::module(caps, COOP, &[("BINDINGS", &b)], &flags, 256, true)
}

pub(super) fn source_merge(spec: &AttentionSpec, caps: &Caps) -> String {
    let n_in = 3 + spec.mask_dtype.is_some() as usize;
    let b = common::output(n_in + 1, spec.out_dtype, caps);
    let sb = (n_in + 2).to_string();
    template::module(caps, MERGE, &[("BINDINGS", &b), ("SCRATCH_BINDING", &sb)], &[], 256, false)
}

fn tile_keys(d: usize) -> usize {
    if d <= 64 { 32 } else { 16 }
}

fn tile_shared_bytes(d: usize) -> usize {
    2 * tile_keys(d) * d * 4
}

fn bindings(spec: &AttentionSpec, caps: &Caps) -> String {
    let mut s = String::new();
    s.push_str(&common::input(1, "q", spec.dtype, caps));
    s.push_str(&common::input(2, "k", spec.dtype, caps));
    s.push_str(&common::input(3, "v", spec.dtype, caps));
    let out_binding = match spec.mask_dtype {
        Some(m) => {
            s.push_str(&common::input(4, "msk", m, caps));
            5
        }
        None => 4,
    };
    s.push_str(&common::output(out_binding, spec.out_dtype, caps));
    s
}

const TILE: &str = include_str!("kernels/attn_tile.wgsl");

pub(super) fn source_tile(spec: &AttentionSpec, d: u32, caps: &Caps) -> String {
    let d4 = d / 4;
    let bk = tile_keys(d as usize) as u32;
    let b = bindings(spec, caps);
    let (d4s, bks, kv) = (d4.to_string(), bk.to_string(), (bk * d4).to_string());
    template::module(
        caps,
        TILE,
        &[("BINDINGS", &b), ("D4", &d4s), ("BK", &bks), ("KV_SIZE", &kv)],
        &split_flags(spec, false),
        64,
        false,
    )
}
