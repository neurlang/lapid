//! Gated delta rule (`Kernel::GatedDeltaNet`, template `kernels/gdn.ptx`): the state lives
//! in registers, 8 threads per state column, sequential over t. Dk is baked into the key
//! (register count per thread); every other size travels in the params.

use std::sync::Arc;

use lapid_compiler::{GatedDeltaNetDims, GatedDeltaNetSpec};
use lapid_tensor::{Error, Result};

use super::expr;
use super::ptx::{self, Gen};
use super::{Key, Params, Plan, check_plain, grid1, template};
use crate::context::Caps;

const KERNEL: &str = include_str!("../kernels/gdn.ptx");
/// Threads per state column.
const PARTS: usize = 8;
/// Columns per 128-thread block.
const COLS: usize = 16;
/// Largest Dk (rows per thread = Dk / 8 registers, times three).
const MAX_DK: usize = 512;

pub(super) fn plan(spec: &Arc<GatedDeltaNetSpec>, d: &GatedDeltaNetDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.out_dtype, "gated_delta_net output")?;
    if d.dk == 0 || d.dk > MAX_DK {
        return Err(Error::new(
            "unsupported",
            format!("gated_delta_net: dk = {} (supported 1..={MAX_DK})", d.dk),
        ));
    }
    let ncb = d.dv.div_ceil(COLS);
    let mut p = Params::default();
    p.u(d.b)?.u(d.t)?.u(d.h)?.u(d.dk)?.u(d.dv)?.bits(d.scale);
    p.u(ncb)?;
    // output must be addressable with 32-bit element indices
    let total = (d.b as u128) * (d.h as u128) * ((d.t + d.dk) as u128) * (d.dv as u128);
    if total > u32::MAX as u128 {
        return Err(Error::new("too_large", "gated_delta_net output exceeds 32-bit indexing"));
    }
    let blocks = (d.b * d.h * ncb) as u64;
    let grid = if blocks == 0 { [0, 0, 0] } else { grid1(blocks, caps)? };
    Ok(Plan::new(Key::Gdn(spec.clone(), d.dk as u32), p.0, grid, (COLS * PARTS) as u32))
}

pub(super) fn source(spec: &GatedDeltaNetSpec, dk: usize) -> String {
    let r = dk.div_ceil(PARTS);
    let full = dk == r * PARTS;
    let vec = full && r.is_multiple_of(4);
    let (args, ptrs) = ptx::args(&["q", "k", "v", "g", "beta", "state", "out"]);
    let regs = format!(
        "    .reg .f32 %s<{r}>;\n    .reg .f32 %k<{r}>;\n    .reg .f32 %q<{r}>;\n    .reg .f32 %kn<{r}>;\n    .reg .f32 %qn<{r}>;\n"
    );

    // valid-row predicate of row r (when Dk is not a multiple of 8 rows per part)
    let row_ok = |g: &mut Gen, i: usize| -> Option<String> {
        if full {
            return None;
        }
        let (t, p) = (g.reg("u32"), g.reg("pred"));
        g.op(format!("add.u32 {t}, %i0, {i}"));
        g.op(format!("setp.lt.u32 {p}, {t}, %DK"));
        Some(p)
    };

    let mut g = Gen::new();
    let idx = g.reg("u32");
    for i in 0..r {
        g.op(format!("mad.lo.u32 {idx}, {i}, %DV, %sbase"));
        let a = g.addr("%g_state", &idx, 4);
        match row_ok(&mut g, i) {
            None => g.op(format!("ld.global.nc.f32 %s{i}, [{a}]")),
            Some(p) => {
                g.op(format!("mov.f32 %s{i}, 0f00000000"));
                g.op(format!("@{p} ld.global.nc.f32 %s{i}, [{a}]"));
            }
        }
    }
    let load_state = g.finish();

    let mut g = Gen::new();
    if vec {
        let ak = g.addr("%g_k", "%x", 4);
        let aq = g.addr("%g_q", "%x", 4);
        for c in 0..r / 4 {
            let ks = (0..4).map(|t| format!("%kn{}", 4 * c + t)).collect::<Vec<_>>().join(", ");
            let qs = (0..4).map(|t| format!("%qn{}", 4 * c + t)).collect::<Vec<_>>().join(", ");
            g.op(format!("ld.global.nc.v4.f32 {{{ks}}}, [{ak}+{}]", 16 * c));
            g.op(format!("ld.global.nc.v4.f32 {{{qs}}}, [{aq}+{}]", 16 * c));
        }
    } else {
        let t = g.reg("u32");
        for i in 0..r {
            g.op(format!("add.u32 {t}, %x, {i}"));
            let ak = g.addr("%g_k", &t, 4);
            let aq = g.addr("%g_q", &t, 4);
            match row_ok(&mut g, i) {
                None => {
                    g.op(format!("ld.global.nc.f32 %kn{i}, [{ak}]"));
                    g.op(format!("ld.global.nc.f32 %qn{i}, [{aq}]"));
                }
                Some(p) => {
                    g.op(format!("mov.f32 %kn{i}, 0f00000000"));
                    g.op(format!("mov.f32 %qn{i}, 0f00000000"));
                    g.op(format!("@{p} ld.global.nc.f32 %kn{i}, [{ak}]"));
                    g.op(format!("@{p} ld.global.nc.f32 %qn{i}, [{aq}]"));
                }
            }
        }
    }
    let load_kq = g.finish();

    let mut take = String::from("    mov.f32 %gv, %gn;\n    mov.f32 %bv, %bn;\n    mov.f32 %vj, %vn;\n");
    for i in 0..r {
        take.push_str(&format!("    mov.f32 %k{i}, %kn{i};\n    mov.f32 %q{i}, %qn{i};\n"));
    }

    let mut g = Gen::new();
    let e = expr::exp(&mut g, "%gv");
    g.op(format!("mov.f32 %gt, {e}"));
    let exp = g.finish();

    // S *= exp(g); dk = S . k  (4 independent chains)
    let dot = |decay: bool, upd: bool, vecr: &str, acc: &str| -> String {
        let mut g = Gen::new();
        let chains = r.min(4);
        let c: Vec<String> = (0..chains).map(|_| g.reg("f32")).collect();
        for x in &c {
            g.op(format!("mov.f32 {x}, 0f00000000"));
        }
        for i in 0..r {
            if decay {
                g.op(format!("mul.f32 %s{i}, %s{i}, %gt"));
            }
            if upd {
                g.op(format!("fma.rn.f32 %s{i}, %k{i}, %delta, %s{i}"));
            }
            let x = &c[i % chains];
            g.op(format!("fma.rn.f32 {x}, %s{i}, %{vecr}{i}, {x}"));
        }
        g.op(format!("mov.f32 {acc}, {}", c[0]));
        for x in &c[1..] {
            g.op(format!("add.f32 {acc}, {acc}, {x}"));
        }
        g.finish()
    };
    let dot_k = dot(true, false, "k", "%dk");
    let dot_q = dot(false, true, "q", "%o");

    let mut g = Gen::new();
    ptx::store_if(&mut g, spec.out_dtype, "%g_out", "%x", "%o", Some("%p0"));
    let store_o = g.finish();

    let mut g = Gen::new();
    let idx = g.reg("u32");
    for i in 0..r {
        g.op(format!("mad.lo.u32 {idx}, {i}, %DV, %x"));
        let pred = match row_ok(&mut g, i) {
            None => "%cv".to_string(),
            Some(p) => {
                g.op(format!("and.pred {p}, {p}, %cv"));
                p
            }
        };
        ptx::store_if(&mut g, spec.out_dtype, "%g_out", &idx, &format!("%s{i}"), Some(&pred));
    }
    let store_state = g.finish();

    let rs = r.to_string();
    template::fill(
        KERNEL,
        &[
            ("ARGS", &args),
            ("PTRS", &ptrs),
            ("REGS", &regs),
            ("R", &rs),
            ("LOAD_STATE", &load_state),
            ("EXP", &exp),
            ("LOAD_KQ", &load_kq),
            ("TAKE", &take),
            ("DOT_K", &dot_k),
            ("DOT_Q", &dot_q),
            ("STORE_O", &store_o),
            ("STORE_STATE", &store_state),
        ],
        &["PREFETCH"],
    )
}
