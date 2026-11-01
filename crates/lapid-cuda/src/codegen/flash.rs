//! Tensor-core flash attention (template `kernels/attn_mma.ptx`). The template holds the
//! block structure (index setup, Q staging, the K/V tile loop); Rust fills the tile loaders
//! for the dtype / alignment variant and unrolls the register blocks whose size depends on
//! the head dim: Q fragments, S = Q K^T, the online softmax on the S fragments, O += P V
//! and the final normalize + store.
//!
//! Fragment layout (mma m16n8k16, lane l): accumulator register r of n-tile t sits at row
//! l / 4 + 8 (r / 2), column 8 t + 2 (l % 4) + r % 2.

use lapid_compiler::AttentionSpec;
use lapid_tensor::DType;

use super::attention::{flags, names};
use super::ptx::{self, Gen, fimm};
use super::template;

const KERNEL: &str = include_str!("../kernels/attn_mma.ptx");

/// Queries per block (16 per warp).
pub(crate) const BQ: usize = 64;
/// Keys per tile.
const BK: usize = 64;
pub(crate) const THREADS: u32 = 128;
pub(crate) const MAX_D: usize = 128;

const LOG2E: f32 = std::f32::consts::LOG2_E;

struct Dims {
    d: usize,
    /// shared row stride in halves / bytes
    ld: usize,
    rb: usize,
    /// bytes of one K or V tile
    kv: usize,
}

fn dims(d: u32) -> Dims {
    let d = (d as usize).clamp(16, MAX_D) / 16 * 16;
    let ld = d + 8;
    Dims {
        d,
        ld,
        rb: 2 * ld,
        kv: BK * 2 * ld,
    }
}

fn regs(dm: &Dims) -> String {
    format!(
        "    .reg .f32 %s<32>;\n    .reg .f32 %o<{}>;\n    .reg .b32 %qf<{}>;\n    .reg .b32 %pa<16>;\n    .reg .b32 %kb<4>;\n",
        dm.d / 2,
        dm.d / 4
    )
}

/// One Q element per thread-iteration `%e`: Q[q0 + e / D, e % D] → f16 smem[e / D][e % D].
fn load_q(spec: &AttentionSpec, dm: &Dims) -> String {
    let mut g = Gen::new();
    let (qq, dd, gq, idx, a) = (
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
    );
    let valid = g.reg("pred");
    let v = g.reg("f32");
    let h = g.reg("b16");
    g.op(format!("div.u32 {qq}, %e, {}", dm.d));
    g.op(format!("mul.lo.u32 {dd}, {qq}, {}", dm.d));
    g.op(format!("sub.u32 {dd}, %e, {dd}"));
    g.op(format!("add.u32 {gq}, %q0, {qq}"));
    g.op(format!("setp.lt.u32 {valid}, {gq}, %SQ"));
    g.op(format!("mad.lo.u32 {idx}, {gq}, %qs, %qbase"));
    g.op(format!("mad.lo.u32 {idx}, {dd}, %qd, {idx}"));
    g.op(format!("selp.b32 {idx}, {idx}, 0, {valid}"));
    ptx::load(&mut g, spec.dtype, "%g_q", &idx, &v, true);
    g.op(format!("selp.f32 {v}, {v}, 0f00000000, {valid}"));
    g.op(format!("cvt.rn.f16.f32 {h}, {v}"));
    g.op(format!("mad.lo.u32 {a}, {qq}, {}, {dd}", dm.ld));
    g.op(format!("mad.lo.u32 {a}, {a}, 2, %sm"));
    g.op(format!("st.shared.b16 [{a}], {h}"));
    g.finish()
}

/// K and V of tile `%kt0`, item `%e` (one element, or one 8-element chunk when `vec`).
fn load_kv(spec: &AttentionSpec, dm: &Dims, vec: bool) -> String {
    let mut g = Gen::new();
    let (key, c, gk, ik, iv, a) = (
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
    );
    let valid = g.reg("pred");
    let per = if vec { dm.d / 8 } else { dm.d };
    g.op(format!("div.u32 {key}, %e, {per}"));
    g.op(format!("mul.lo.u32 {c}, {key}, {per}"));
    g.op(format!("sub.u32 {c}, %e, {c}"));
    g.op(format!("add.u32 {gk}, %kt0, {key}"));
    g.op(format!("setp.lt.u32 {valid}, {gk}, %SK"));
    if vec {
        g.op(format!("mad.lo.u32 {ik}, {gk}, %ks, %kbase"));
        g.op(format!("mad.lo.u32 {ik}, {c}, 8, {ik}"));
        g.op(format!("mad.lo.u32 {iv}, {gk}, %vs, %vbase"));
        g.op(format!("mad.lo.u32 {iv}, {c}, 8, {iv}"));
    } else {
        g.op(format!("mad.lo.u32 {ik}, {gk}, %ks, %kbase"));
        g.op(format!("mad.lo.u32 {ik}, {c}, %kd, {ik}"));
        g.op(format!("mad.lo.u32 {iv}, {gk}, %vs, %vbase"));
        g.op(format!("mad.lo.u32 {iv}, {c}, %vd, {iv}"));
    }
    g.op(format!("selp.b32 {ik}, {ik}, 0, {valid}"));
    g.op(format!("selp.b32 {iv}, {iv}, 0, {valid}"));
    if vec {
        g.op(format!("mul.lo.u32 {a}, {key}, {}", dm.rb));
        g.op(format!("mad.lo.u32 {a}, {c}, 16, {a}"));
        g.op(format!("add.u32 {a}, {a}, %sm"));
        for (name, idx, off) in [("k", &ik, 0), ("v", &iv, dm.kv)] {
            let w: Vec<String> = (0..4).map(|_| g.reg("b32")).collect();
            if spec.dtype == DType::F16 {
                let ad = g.addr(&format!("%g_{name}"), idx, 2);
                g.op(format!(
                    "ld.global.nc.v4.b32 {{{}, {}, {}, {}}}, [{ad}]",
                    w[0], w[1], w[2], w[3]
                ));
            } else {
                let ad = g.addr(&format!("%g_{name}"), idx, 4);
                let f: Vec<String> = (0..8).map(|_| g.reg("f32")).collect();
                g.op(format!(
                    "ld.global.nc.v4.f32 {{{}, {}, {}, {}}}, [{ad}]",
                    f[0], f[1], f[2], f[3]
                ));
                g.op(format!(
                    "ld.global.nc.v4.f32 {{{}, {}, {}, {}}}, [{ad}+16]",
                    f[4], f[5], f[6], f[7]
                ));
                for i in 0..4 {
                    g.op(format!(
                        "cvt.rn.f16x2.f32 {}, {}, {}",
                        w[i],
                        f[2 * i + 1],
                        f[2 * i]
                    ));
                }
            }
            for wi in &w {
                g.op(format!("selp.b32 {wi}, {wi}, 0, {valid}"));
            }
            g.op(format!(
                "st.shared.v4.b32 [{a}+{off}], {{{}, {}, {}, {}}}",
                w[0], w[1], w[2], w[3]
            ));
        }
    } else {
        g.op(format!("mad.lo.u32 {a}, {key}, {}, {c}", dm.ld));
        g.op(format!("mad.lo.u32 {a}, {a}, 2, %sm"));
        for (name, idx, off) in [("k", &ik, 0), ("v", &iv, dm.kv)] {
            let v = g.reg("f32");
            let h = g.reg("b16");
            ptx::load(&mut g, spec.dtype, &format!("%g_{name}"), idx, &v, true);
            g.op(format!("selp.f32 {v}, {v}, 0f00000000, {valid}"));
            g.op(format!("cvt.rn.f16.f32 {h}, {v}"));
            g.op(format!("st.shared.b16 [{a}+{off}], {h}"));
        }
    }
    g.finish()
}

fn q_frags(dm: &Dims) -> String {
    let mut s = String::new();
    for ks in 0..dm.d / 16 {
        let r = 4 * ks;
        s.push_str(&format!(
            "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%qf{r}, %qf{}, %qf{}, %qf{}}}, [%q_lane+{}];\n",
            r + 1,
            r + 2,
            r + 3,
            32 * ks
        ));
    }
    s
}

fn init(dm: &Dims) -> String {
    let mut s = String::new();
    for i in 0..dm.d / 2 {
        s.push_str(&format!("    mov.f32 %o{i}, 0f00000000;\n"));
    }
    s.push_str("    mov.f32 %m0, 0fFF800000;\n    mov.f32 %m1, 0fFF800000;\n");
    s.push_str("    mov.f32 %l0, 0f00000000;\n    mov.f32 %l1, 0f00000000;\n");
    s
}

/// mma with accumulator registers `%<acc>{c}..{c+3}`, A fragments `%<a>{a0}..{a0+3}`.
fn mma_acc(acc: &str, c: usize, a: &str, a0: usize, b: [&str; 2]) -> String {
    let d = format!(
        "%{acc}{c}, %{acc}{}, %{acc}{}, %{acc}{}",
        c + 1,
        c + 2,
        c + 3
    );
    format!(
        "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{{d}}}, {{%{a}{a0}, %{a}{}, %{a}{}, %{a}{}}}, {{{}, {}}}, {{{d}}};\n",
        a0 + 1,
        a0 + 2,
        a0 + 3,
        b[0],
        b[1]
    )
}

fn qk(dm: &Dims) -> String {
    let mut s = String::new();
    for i in 0..32 {
        s.push_str(&format!("    mov.f32 %s{i}, 0f00000000;\n"));
    }
    for p in 0..BK / 16 {
        for ks in 0..dm.d / 16 {
            s.push_str(&format!(
                "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%kb0, %kb1, %kb2, %kb3}}, [%k_lane+{}];\n",
                p * 16 * dm.rb + 32 * ks
            ));
            s.push_str(&mma_acc("s", 8 * p, "qf", 4 * ks, ["%kb0", "%kb1"]));
            s.push_str(&mma_acc("s", 8 * p + 4, "qf", 4 * ks, ["%kb2", "%kb3"]));
        }
    }
    s
}

fn softmax(spec: &AttentionSpec, dm: &Dims) -> String {
    let mut g = Gen::new();
    let ninf = fimm(f32::NEG_INFINITY);
    let key = g.reg("u32");
    let kb = g.reg("u32");
    let pk = g.reg("pred");
    let skm1 = g.reg("u32");
    g.op(format!("add.u32 {kb}, %kt0, %kl2"));
    g.op(format!("sub.u32 {skm1}, %SK, 1"));
    let (kc, mv) = (g.reg("u32"), g.reg("f32"));
    for nt in 0..BK / 8 {
        for r in 0..4 {
            let s = format!("%s{}", nt * 4 + r);
            let row = r / 2;
            g.op(format!("mul.f32 {s}, {s}, %scl"));
            g.op(format!("add.u32 {key}, {kb}, {}", nt * 8 + r % 2));
            if let Some(mt) = spec.mask_dtype {
                g.op(format!("min.u32 {kc}, {key}, {skm1}"));
                g.op(format!("mad.lo.u32 {kc}, {kc}, %mstep, %mrow{row}"));
                ptx::load(&mut g, mt, "%g_msk", &kc, &mv, true);
                g.op(format!("fma.rn.f32 {s}, {mv}, {}, {s}", fimm(LOG2E)));
            }
            g.op(format!("setp.lt.u32 {pk}, {key}, %kend{row}"));
            g.op(format!("selp.f32 {s}, {s}, {ninf}, {pk}"));
        }
    }
    // per row: tile max, rescale factor, probabilities, running sums, O rescale
    let t = g.reg("f32");
    for row in 0..2 {
        let mx = g.reg("f32");
        g.op(format!("mov.f32 {mx}, {ninf}"));
        for nt in 0..BK / 8 {
            for c in 0..2 {
                g.op(format!("max.f32 {mx}, {mx}, %s{}", nt * 4 + 2 * row + c));
            }
        }
        for sh in [1, 2] {
            g.op(format!("shfl.sync.bfly.b32 {t}, {mx}, {sh}, 31, -1"));
            g.op(format!("max.f32 {mx}, {mx}, {t}"));
        }
        let (mn, mu, corr, sum) = (g.reg("f32"), g.reg("f32"), g.reg("f32"), g.reg("f32"));
        let pinf = g.reg("pred");
        g.op(format!("max.f32 {mn}, %m{row}, {mx}"));
        g.op(format!("setp.eq.f32 {pinf}, {mn}, {ninf}"));
        g.op(format!("selp.f32 {mu}, 0f00000000, {mn}, {pinf}"));
        g.op(format!("sub.f32 {corr}, %m{row}, {mu}"));
        g.op(format!("ex2.approx.f32 {corr}, {corr}"));
        g.op(format!("mov.f32 %m{row}, {mn}"));
        g.op(format!("mov.f32 {sum}, 0f00000000"));
        for nt in 0..BK / 8 {
            for c in 0..2 {
                let s = format!("%s{}", nt * 4 + 2 * row + c);
                g.op(format!("sub.f32 {s}, {s}, {mu}"));
                g.op(format!("ex2.approx.f32 {s}, {s}"));
                g.op(format!("add.f32 {sum}, {sum}, {s}"));
            }
        }
        g.op(format!("fma.rn.f32 %l{row}, %l{row}, {corr}, {sum}"));
        for nt in 0..dm.d / 8 {
            for c in 0..2 {
                let o = format!("%o{}", nt * 4 + 2 * row + c);
                g.op(format!("mul.f32 {o}, {o}, {corr}"));
            }
        }
    }
    // P (f32 accumulators) → f16 A fragments, one k16 step per pair of key n-tiles
    for j in 0..BK / 16 {
        let (s0, s1) = (8 * j, 8 * j + 4);
        g.op(format!(
            "cvt.rn.f16x2.f32 %pa{}, %s{}, %s{}",
            4 * j,
            s0 + 1,
            s0
        ));
        g.op(format!(
            "cvt.rn.f16x2.f32 %pa{}, %s{}, %s{}",
            4 * j + 1,
            s0 + 3,
            s0 + 2
        ));
        g.op(format!(
            "cvt.rn.f16x2.f32 %pa{}, %s{}, %s{}",
            4 * j + 2,
            s1 + 1,
            s1
        ));
        g.op(format!(
            "cvt.rn.f16x2.f32 %pa{}, %s{}, %s{}",
            4 * j + 3,
            s1 + 3,
            s1 + 2
        ));
    }
    g.finish()
}

fn pv(dm: &Dims) -> String {
    let mut s = String::new();
    for dp in 0..dm.d / 16 {
        for j in 0..BK / 16 {
            s.push_str(&format!(
                "    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%kb0, %kb1, %kb2, %kb3}}, [%v_lane+{}];\n",
                j * 16 * dm.rb + 32 * dp
            ));
            s.push_str(&mma_acc("o", 8 * dp, "pa", 4 * j, ["%kb0", "%kb1"]));
            s.push_str(&mma_acc("o", 8 * dp + 4, "pa", 4 * j, ["%kb2", "%kb3"]));
        }
    }
    s
}

fn finish(spec: &AttentionSpec, dm: &Dims) -> String {
    let mut g = Gen::new();
    let t = g.reg("f32");
    let mut inv = Vec::new();
    for row in 0..2 {
        for sh in [1, 2] {
            g.op(format!("shfl.sync.bfly.b32 {t}, %l{row}, {sh}, 31, -1"));
            g.op(format!("add.f32 %l{row}, %l{row}, {t}"));
        }
        let (i, p) = (g.reg("f32"), g.reg("pred"));
        g.op(format!("rcp.rn.f32 {i}, %l{row}"));
        g.op(format!("setp.gt.f32 {p}, %l{row}, 0f00000000"));
        g.op(format!("selp.f32 {i}, {i}, 0f00000000, {p}"));
        inv.push(i);
    }
    let (v, idx) = (g.reg("f32"), g.reg("u32"));
    for nt in 0..dm.d / 8 {
        for r in 0..4 {
            let row = r / 2;
            g.op(format!("mul.f32 {v}, %o{}, {}", nt * 4 + r, inv[row]));
            g.op(format!("add.u32 {idx}, %orow{row}, {}", nt * 8 + r % 2));
            ptx::store_if(
                &mut g,
                spec.out_dtype,
                "%g_out",
                &idx,
                &v,
                Some(&format!("%qv{row}")),
            );
        }
    }
    g.finish()
}

pub(super) fn source(spec: &AttentionSpec, d: u32, vec: bool) -> String {
    let dm = dims(d);
    let (args, ptrs) = ptx::args(&names(spec, false));
    let vec = vec && matches!(spec.dtype, DType::F16 | DType::F32);
    let kv_items = if vec { BK * dm.d / 8 } else { BK * dm.d };
    let s = |x: usize| x.to_string();
    template::fill(
        KERNEL,
        &[
            ("ARGS", &args),
            ("PTRS", &ptrs),
            ("SMEM", &s(2 * dm.kv)),
            ("REGS", &regs(&dm)),
            ("D", &s(dm.d)),
            ("LD", &s(dm.ld)),
            ("KV_BYTES", &s(dm.kv)),
            ("QD", &s(BQ * dm.d)),
            ("KV_ITEMS", &s(kv_items)),
            ("LOAD_Q", &load_q(spec, &dm)),
            ("Q_FRAGS", &q_frags(&dm)),
            ("INIT", &init(&dm)),
            ("LOAD_KV", &load_kv(spec, &dm, vec)),
            ("QK", &qk(&dm)),
            ("SOFTMAX", &softmax(spec, &dm)),
            ("PV", &pv(&dm)),
            ("FINAL", &finish(spec, &dm)),
        ],
        &flags(spec),
    )
}
