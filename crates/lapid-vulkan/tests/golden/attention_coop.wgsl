// lapid prelude: extensions, the params binding, builtin mirrors and math helpers.
enable f16;
enable wgpu_cooperative_matrix;

// Launch params (sizes, strides, offsets) as u32 words; layout documented per kernel.
@group(0) @binding(0) var<storage, read> P: array<u32>;

var<private> g_lid: u32;
var<private> g_sgi: u32;
var<private> g_sgid: u32;
var<private> g_nsg: u32;
var<private> g_sgs: u32;

// erf: Taylor series near 0, Abramowitz & Stegun 7.1.26 elsewhere.
fn erf_(x: f32) -> f32 {
    let a = abs(x);
    if (a < 0.5) {
        let x2 = x * x;
        return x * (1.1283791671 + x2 * (-0.3761263890 + x2 * (0.1128379167 + x2 * (-0.0268661706 + x2 * 0.0052239776))));
    }
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * exp(-a * a);
    return select(y, -y, x < 0.0);
}

fn tanh_(x: f32) -> f32 {
    return tanh(clamp(x, -15.0, 15.0));
}

// pow with negative bases for integer exponents.
fn pow_(a: f32, b: f32) -> f32 {
    if (b == 0.0) {
        return 1.0;
    }
    let r = exp2(b * log2(abs(a)));
    if (a < 0.0) {
        if (floor(b) != b) {
            return bitcast<f32>(0x7fc00000u);
        }
        let h = b * 0.5;
        if (floor(h) != h) {
            return -r;
        }
    }
    return r;
}

fn idiv_up(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}
// Attention for many queries on cooperative matrices (16x16x16 f16, f32 accumulate),
// head dim 64. One workgroup (8 subgroups) per (batch, head, block of 64 queries).
// Two passes over the keys in tiles of 32 (avoids rescaling the output accumulators):
//   pass 1: S = Q K^T per tile (one 16x16 tile per subgroup) -> per-row running max / sum
//           (each thread tracks 8 of the 32 columns; combined once after the pass)
//   pass 2: S again, P = exp(S - m) / l as f16, O += P V (two 16x16 output tiles per
//           subgroup, kept in accumulators for the whole pass).
// Q is pre-scaled; masks / causal limits are applied to S in the scalar phase.
// Params: [B, H, sq, sk, d, scale_bits, q_blocks, q(off, s0..s3), k(..), v(..), mask(..)?]
@group(0) @binding(1) var<storage, read> q: array<f32>;
fn ld_q(e: u32) -> f32 {
    return q[e];
}
@group(0) @binding(2) var<storage, read> k: array<f32>;
fn ld_k(e: u32) -> f32 {
    return k[e];
}
@group(0) @binding(3) var<storage, read> v: array<f32>;
fn ld_v(e: u32) -> f32 {
    return v[e];
}
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
fn st(e: u32, v: f32) {
    out[e] = v;
}

const BQ: u32 = 64u;
const BKV: u32 = 32u;
const D: u32 = 64u;
const LDQ: u32 = 72u;  // f16 pitch of Qs / Ks / Vs rows (D + 8)
const LDS: u32 = 36u;  // f32 pitch of Ss rows (BKV + 4)
const LDP: u32 = 40u;  // f16 pitch of Ps rows (BKV + 8)
const NEG_INF: f32 = -3.4028235e38;

var<workgroup> Qs: array<f16, 4608>;   // BQ x LDQ
var<workgroup> Ks: array<f16, 2304>;   // BKV x LDQ, [key][d]
var<workgroup> Vs: array<f16, 2304>;   // BKV x LDQ, [key][d]
var<workgroup> Ss: array<f32, 2304>;   // BQ x LDS; also the output staging area
var<workgroup> Ps: array<f16, 2560>;   // BQ x LDP
var<workgroup> rowm: array<f32, 256>;
var<workgroup> rowl: array<f32, 256>;

// Load key / value tile k0 into Ks / Vs (keys >= kend read as zero).
fn load_kv(kbase: u32, vbase: u32, k0: u32, kend: u32) {
    let ks = P[15u];
    let kd = P[16u];
    let vs = P[20u];
    let vd = P[21u];
    for (var t = 0u; t < 8u; t += 1u) {
        let e = g_lid + t * 256u;
        let key = e / D;
        let dd = e % D;
        var kv: f32 = 0.0;
        var vv: f32 = 0.0;
        if (k0 + key < kend) {
            kv = ld_k(kbase + (k0 + key) * ks + dd * kd);
            vv = ld_v(vbase + (k0 + key) * vs + dd * vd);
        }
        Ks[key * LDQ + dd] = f16(kv);
        Vs[key * LDQ + dd] = f16(vv);
    }
}

// Ss = Q K^T for the current tile (subgroup s computes rows (s/2)*16, cols (s%2)*16).
fn scores() {
    let rb = (g_sgid / 2u) * 16u;
    let cb = (g_sgid % 2u) * 16u;
    var acc = coop_mat16x16<f32, C>();
    for (var dk = 0u; dk < D; dk += 16u) {
        let a = coopLoadT<coop_mat16x16<f16, A>>(&Qs[rb * LDQ + dk], LDQ);
        let b = coopLoad<coop_mat16x16<f16, B>>(&Ks[cb * LDQ + dk], LDQ);
        acc = coopMultiplyAdd(a, b, acc);
    }
    coopStoreT(acc, &Ss[rb * LDS + cb], LDS);
}

// Score of (row, col) of the current tile with mask / causal / bounds applied.
fn masked(row: u32, col: u32, q0: u32, k0: u32, kend: u32, mrow0: u32, SQ: u32, SK: u32) -> f32 {
    let key = k0 + col;
    var s = Ss[row * LDS + col];
    if (key >= kend) {
        return NEG_INF;
    }
    return s;
}

fn kernel(wg_lin: u32, wg_total: u32) {
    let H = P[1];
    let SQ = P[2];
    let SK = P[3];
    let scale = bitcast<f32>(P[5]);
    let nqb = P[6];
    let items = P[0] * H * nqb;
    let row = g_lid / 4u;
    let part = g_lid % 4u;
    for (var item = wg_lin; item < items; item += wg_total) {
        let qb = item % nqb;
        let bh = item / nqb;
        let h = bh % H;
        let b = bh / H;
        let q0 = qb * BQ;
        let qbase = P[7u] + b * P[8u] + h * P[9u];
        let kbase = P[12u] + b * P[13u] + h * P[14u];
        let vbase = P[17u] + b * P[18u] + h * P[19u];
        let mrow0 = 0u;
        let kend = SK;
        workgroupBarrier();
        for (var t = 0u; t < 16u; t += 1u) {
            let e = g_lid + t * 256u;
            let r = e / D;
            let dd = e % D;
            var v: f32 = 0.0;
            if (q0 + r < SQ) {
                v = ld_q(qbase + (q0 + r) * P[10u] + dd * P[11u]) * scale;
            }
            Qs[r * LDQ + dd] = f16(v);
        }
        // pass 1: row max / sum
        var pm = NEG_INF;
        var pl: f32 = 0.0;
        for (var k0 = 0u; k0 < kend; k0 += BKV) {
            workgroupBarrier();
            load_kv(kbase, vbase, k0, kend);
            workgroupBarrier();
            scores();
            workgroupBarrier();
            var tm = NEG_INF;
            for (var c = 0u; c < 8u; c += 1u) {
                tm = max(tm, masked(row, part * 8u + c, q0, k0, kend, mrow0, SQ, SK));
            }
            let nm = max(pm, tm);
            if (nm > NEG_INF) {
                var ts: f32 = 0.0;
                for (var c = 0u; c < 8u; c += 1u) {
                    ts = ts + exp(masked(row, part * 8u + c, q0, k0, kend, mrow0, SQ, SK) - nm);
                }
                pl = pl * exp(pm - nm) + ts;
                pm = nm;
            }
        }
        rowm[g_lid] = pm;
        rowl[g_lid] = pl;
        workgroupBarrier();
        var m = NEG_INF;
        for (var i = 0u; i < 4u; i += 1u) {
            m = max(m, rowm[row * 4u + i]);
        }
        var l: f32 = 0.0;
        if (m > NEG_INF) {
            for (var i = 0u; i < 4u; i += 1u) {
                l = l + rowl[row * 4u + i] * exp(rowm[row * 4u + i] - m);
            }
        }
        let inv_l = select(0.0, 1.0 / l, l > 0.0);
        // pass 2: O = P V
        var o0 = coop_mat16x16<f32, C>();
        var o1 = coop_mat16x16<f32, C>();
        let orb = (g_sgid / 2u) * 16u;
        let ocb = (g_sgid % 2u) * 32u;
        for (var k0 = 0u; k0 < kend; k0 += BKV) {
            workgroupBarrier();
            load_kv(kbase, vbase, k0, kend);
            workgroupBarrier();
            scores();
            workgroupBarrier();
            for (var c = 0u; c < 8u; c += 1u) {
                let col = part * 8u + c;
                var p: f32 = 0.0;
                if (m > NEG_INF) {
                    p = exp(masked(row, col, q0, k0, kend, mrow0, SQ, SK) - m) * inv_l;
                }
                Ps[row * LDP + col] = f16(p);
            }
            workgroupBarrier();
            for (var kk = 0u; kk < BKV; kk += 16u) {
                let a = coopLoadT<coop_mat16x16<f16, A>>(&Ps[orb * LDP + kk], LDP);
                let v0 = coopLoadT<coop_mat16x16<f16, B>>(&Vs[kk * LDQ + ocb], LDQ);
                let v1 = coopLoadT<coop_mat16x16<f16, B>>(&Vs[kk * LDQ + ocb + 16u], LDQ);
                o0 = coopMultiplyAdd(a, v0, o0);
                o1 = coopMultiplyAdd(a, v1, o1);
            }
        }
        // output: stage each 16x16 tile in this subgroup's 256-float slice of Ss
        let ob = ((b * H + h) * SQ) * D;
        for (var t = 0u; t < 2u; t += 1u) {
            workgroupBarrier();
            if (t == 0u) {
                coopStoreT(o0, &Ss[g_sgid * 256u], 16u);
            } else {
                coopStoreT(o1, &Ss[g_sgid * 256u], 16u);
            }
            workgroupBarrier();
            for (var e = g_sgi; e < 256u; e += g_sgs) {
                let qi = q0 + orb + e / 16u;
                let dd = ocb + t * 16u + e % 16u;
                if (qi < SQ) {
                    st(ob + qi * D + dd, Ss[g_sgid * 256u + e]);
                }
            }
        }
    }
}

// Entry point: mirrors builtins into private globals and runs `kernel` with the
// linearized workgroup id / count (kernels grid-stride over their work items).
@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_index) lidx: u32,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(subgroup_invocation_id) sgi: u32,
    @builtin(subgroup_id) sgid: u32,
    @builtin(num_subgroups) nsg: u32,
    @builtin(subgroup_size) sgs: u32,
) {
    g_lid = lidx;
    g_sgi = sgi;
    g_sgid = sgid;
    g_nsg = nsg;
    g_sgs = sgs;
    let wg_lin = wgid.x + nwg.x * (wgid.y + nwg.y * wgid.z);
    let wg_total = nwg.x * nwg.y * nwg.z;
    kernel(wg_lin, wg_total);
}
