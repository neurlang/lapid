//! Fused attention: softmax(scale * q k^T + mask, causal) v without materializing scores.
//!
//! Many queries (encoder) on sm_80+ with head dim <= 128: tensor-core flash attention
//! (`flash.rs`; f32 inputs use its hi/lo-split precise variant). Few queries (decoder), wider
//! heads (up to 1024) or no tensor cores: split over 64-key chunks with up to 8 queries per
//! block sharing the K / V loads (`kernels/attn_split.ptx`) plus a merge pass
//! (`kernels/attn_merge.ptx`) when there is more than one chunk.
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
/// Largest head dim of the split variant (one thread per dim, blocks of up to 1024 threads).
const SPLIT_MAX_D: usize = 1024;
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
            Key::AttnSplit(spec.clone(), true, d.d.max(1) as u32, 1),
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
            Key::AttnMma(spec.clone(), d.d as u32, vec, spec.dtype != DType::F16 && !fast_attention()),
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
    let threads = split_threads(d.d as u32);
    let qb = split_queries(d.sq, d.d as u32);
    let groups = d.batch * d.heads * d.sq.div_ceil(qb as usize);
    let grid = grid1((groups * nsplit) as u64, caps)?;
    if nsplit == 1 {
        return Ok(Plan::new(
            Key::AttnSplit(spec.clone(), true, d.d as u32, qb),
            p.0,
            grid,
            threads,
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
        Key::AttnSplit(spec.clone(), false, d.d as u32, qb),
        p.0,
        grid,
        threads,
    );
    plan.scratch = rows * nsplit * (d.d + 2) * 4;
    plan.next = Some(Box::new(merge));
    Ok(plan)
}

/// `LAPID_CUDA_FAST_ATTENTION=1`: f32 inputs of the tensor-core flash kernel are rounded to
/// f16 (about 2.5x faster, ~1e-3 relative error) instead of the precise hi/lo variant.
fn fast_attention() -> bool {
    static FAST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FAST.get_or_init(|| std::env::var("LAPID_CUDA_FAST_ATTENTION").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// Threads per block of the split variant: 256, or one per head dim rounded to warps.
fn split_threads(d: u32) -> u32 {
    d.clamp(1, SPLIT_MAX_D as u32).div_ceil(32).max(8) * 32
}

/// Queries per block of the split variant: up to 8 (they share K / V loads), within the
/// shared memory budget.
fn split_queries(sq: usize, d: u32) -> u32 {
    let mut qb = match sq {
        0..=1 => 1,
        2..=3 => 2,
        4..=7 => 4,
        _ => 8,
    };
    while qb > 1 && split_smem(d, qb) > 44 * 1024 {
        qb /= 2;
    }
    qb
}

fn split_smem(d: u32, qb: u32) -> u32 {
    let d = d.clamp(1, SPLIT_MAX_D as u32);
    let t = split_threads(d);
    let nw = t / 32;
    let dots = nw * (SPLIT_KEYS as u32).div_ceil(nw);
    4 * qb * (d + dots + SPLIT_KEYS as u32 + t + 2)
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

pub(super) fn source_split(spec: &AttentionSpec, direct: bool, d: u32, qb: u32) -> String {
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
    let threads = split_threads(d);
    let nw = threads / 32;
    let g = threads / d;
    let (ds, gs, dc, vt, d2) = (
        d.to_string(),
        g.to_string(),
        d.div_ceil(32).to_string(),
        (SPLIT_KEYS as u32).div_ceil(g).to_string(),
        (d + 2).to_string(),
    );
    let scu = (SPLIT_KEYS as u32).div_ceil(nw);
    let dots = nw * scu;
    let (ts, nws, scus) = (threads.to_string(), nw.to_string(), scu.to_string());
    let qb = qb.max(1);
    let qbu = qb as usize;
    let nacc = scu as usize * qbu;
    let regs = format!("    .reg .f32 %xs<{nacc}>;\n    .reg .f32 %qr<{qb}>;\n    .reg .f32 %acc<{qb}>;\n");
    let (mut sc_zero, mut sc_body, mut sc_red) = (String::new(), String::new(), String::new());
    let (mut pv_zero, mut pv_fma, mut pv_st) = (String::new(), String::new(), String::new());
    for i in 0..nacc {
        sc_zero.push_str(&format!("    mov.f32 %xs{i}, 0f00000000;\n"));
    }
    for q in 0..qbu {
        sc_body.push_str(&format!("    ld.shared.f32 %qr{q}, [%a+{}];\n", 4 * q * d as usize));
    }
    sc_red.push_str("    setp.eq.u32 %pq, %lane, 0;\n");
    for u in 0..scu as usize {
        sc_body.push_str(&format!(
            "    add.u32 %kj, %warp, {};\n    setp.lt.u32 %pv, %kj, %kn;\n    selp.b32 %kc, %kj, 0, %pv;\n    add.u32 %kr, %k0, %kc;\n    mad.lo.u32 %kr, %kr, %ks, %kbase;\n    mad.lo.u32 %t, %dd, %kd, %kr;\n{load_k}    selp.f32 %kv, %kv, 0f00000000, %p;\n",
            u * nw as usize
        ));
        for q in 0..qbu {
            let i = u * qbu + q;
            sc_body.push_str(&format!("    fma.rn.f32 %xs{i}, %qr{q}, %kv, %xs{i};\n"));
            for sh in [16, 8, 4, 2, 1] {
                sc_red.push_str(&format!("    shfl.sync.bfly.b32 %sh, %xs{i}, {sh}, 31, -1;\n    add.f32 %xs{i}, %xs{i}, %sh;\n"));
            }
        }
        sc_red.push_str(&format!("    add.u32 %kj, %warp, {};\n    mov.u32 %a, dots;\n    mad.lo.u32 %a, %kj, 4, %a;\n", u * nw as usize));
        for q in 0..qbu {
            sc_red.push_str(&format!("    @%pq st.shared.f32 [%a+{}], %xs{};\n", 4 * q * dots as usize, u * qbu + q));
        }
    }
    for q in 0..qbu {
        pv_zero.push_str(&format!("    mov.f32 %acc{q}, 0f00000000;\n"));
        pv_fma.push_str(&format!(
            "    ld.shared.f32 %pt, [%a+{}];\n    fma.rn.f32 %acc{q}, %pt, %vv, %acc{q};\n",
            4 * q * SPLIT_KEYS
        ));
        pv_st.push_str(&format!("    st.shared.f32 [%a+{}], %acc{q};\n", 4 * q * threads as usize));
    }
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
            ("T", &ts),
            ("NW", &nws),
            ("SCU", &scus),
            ("QB", &qb.to_string()),
            ("QB1", &(qb - 1).to_string()),
            ("QD", &(qb * d).to_string()),
            ("QDOTS", &(qb * dots).to_string()),
            ("DOTS4", &(4 * dots).to_string()),
            ("QP", &(qb * SPLIT_KEYS as u32).to_string()),
            ("QT", &(qb * threads).to_string()),
            ("Q2", &(2 * qb).to_string()),
            ("REGS", &regs),
            ("SC_ZERO", &sc_zero),
            ("SC_BODY", &sc_body),
            ("SC_RED", &sc_red),
            ("PV_ZERO", &pv_zero),
            ("PV_FMA", &pv_fma),
            ("PV_ST", &pv_st),
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
