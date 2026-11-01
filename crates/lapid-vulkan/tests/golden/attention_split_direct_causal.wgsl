// lapid prelude: extensions, the params binding, builtin mirrors and math helpers.
enable f16;

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
// Attention for few queries (decoder), split over keys ("flash decoding").
// One workgroup per (batch, head, query, key chunk of CH keys):
//   scores: one subgroup per key (SG32), else TPK threads per key reduced in shared memory; chunk max / exp / sum (workgroup reductions); then the weighted V sum split over (key group, dim lane).
// DIRECT (one chunk covers all keys): normalize and write the output.
// Otherwise write the partial (m, l, o[D]) to scratch S; `attn_merge` combines the chunks.
// Head dim D is baked in (constant trip counts let the compiler unroll the dot loops).
// D > 256: each thread covers DC = ceil(D / 256) dims in the V phase (one key group).
// Params: [B, H, sq, sk, d, scale_bits, nsplit, q(off, s0..s3), k(..), v(..), mask(..)?]
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

// Workgroup-wide sum / max over 256 threads; every thread gets the result.
// Must be called in workgroup-uniform control flow.
var<workgroup> red: array<f32, 256>;

fn wg_sum(v: f32) -> f32 {
    let s = subgroupAdd(v);
    if (g_sgi == 0u) {
        red[g_sgid] = s;
    }
    workgroupBarrier();
    var r: f32 = 0.0;
    for (var i = 0u; i < g_nsg; i += 1u) {
        r = r + red[i];
    }
    workgroupBarrier();
    return r;
}

fn wg_max(v: f32) -> f32 {
    let s = subgroupMax(v);
    if (g_sgi == 0u) {
        red[g_sgid] = s;
    }
    workgroupBarrier();
    var r: f32 = bitcast<f32>(0xff800000u);
    for (var i = 0u; i < g_nsg; i += 1u) {
        r = max(r, red[i]);
    }
    workgroupBarrier();
    return r;
}


const CH: u32 = 64u;
const D: u32 = 64u;
// V-phase lane width (dims per pass) and passes per thread
const DW: u32 = min(D, 256u);
const DC: u32 = (D + DW - 1u) / DW;
// key groups in the V phase
const G: u32 = 256u / DW;
// threads per key in the score phase
const TPK: u32 = 256u / CH;
// lowest finite f32 ("no score yet"; real -inf mask values compare below it)
const NEG_INF: f32 = -3.4028235e38;

var<workgroup> qsh: array<f32, DC * 256u>;
var<workgroup> dots: array<f32, 256>;
var<workgroup> pp: array<f32, CH>;
var<workgroup> part: array<f32, DC * 256u>;

fn kernel(wg_lin: u32, wg_total: u32) {
    let H = P[1];
    let SQ = P[2];
    let SK = P[3];
    let scale = bitcast<f32>(P[5]);
    let NS = P[6];
    let items = P[0] * H * SQ * NS;
    let kl = g_lid / TPK;
    let quarter = g_lid % TPK;
    let dl = g_lid % DW;
    let grp = g_lid / DW;
    let lane_on = grp < G;
    for (var item = wg_lin; item < items; item += wg_total) {
        let split = item % NS;
        let bhq = item / NS;
        let qi = bhq % SQ;
        let bh = bhq / SQ;
        let h = bh % H;
        let b = bh / H;
        let qbase = P[7u] + b * P[8u] + h * P[9u] + qi * P[10u];
        let kbase = P[12u] + b * P[13u] + h * P[14u];
        let vbase = P[17u] + b * P[18u] + h * P[19u];
        let ks = P[15u];
        let kd = P[16u];
        let vs = P[20u];
        let vd = P[21u];
        let kend = u32(clamp(i32(qi) + 1 + i32(SK) - i32(SQ), 0, i32(SK)));
        let k0 = split * CH;
        let kn = select(0u, min(CH, kend - k0), kend > k0);
        workgroupBarrier();
        for (var c = 0u; c < DC; c += 1u) {
            let qd = g_lid + c * 256u;
            if (qd < D) {
                qsh[qd] = ld_q(qbase + qd * P[11u]) * scale;
            }
        }
        workgroupBarrier();
        // one 32-lane subgroup per key (8 subgroups): lane covers dims lane, lane + 32, ...
        // (coalesced 128-byte rows); all loads issued before the subgroup sums
        var xs: array<f32, CH / 8u>;
        for (var u = 0u; u < CH / 8u; u += 1u) {
            let kj = g_sgid + u * 8u;
            var x: f32 = 0.0;
            if (kj < kn) {
                let kr = kbase + (k0 + kj) * ks;
                for (var c = 0u; c < (D + 31u) / 32u; c += 1u) {
                    let dd = g_sgi + c * 32u;
                    if (dd < D) {
                        x = x + qsh[dd] * ld_k(kr + dd * kd);
                    }
                }
            }
            xs[u] = x;
        }
        for (var u = 0u; u < CH / 8u; u += 1u) {
            let tot = subgroupAdd(xs[u]);
            if (g_sgi == 0u) {
                dots[g_sgid + u * 8u] = tot;
            }
        }
        workgroupBarrier();
        // thread j < kn owns key k0 + j
        var sc = NEG_INF;
        if (g_lid < kn) {
            sc = dots[g_lid];
        }
        let m = wg_max(sc);
        var p: f32 = 0.0;
        if (g_lid < kn && m > NEG_INF) {
            p = exp(sc - m);
        }
        if (g_lid < CH) {
            pp[g_lid] = p;
        }
        // (wg_sum's barriers also publish pp)
        let l = wg_sum(p);
        for (var c = 0u; c < DC; c += 1u) {
            let dd = dl + c * DW;
            var acc: f32 = 0.0;
            if (lane_on && dd < D) {
                let vcol = vbase + k0 * vs + dd * vd;
                // constant trip count (unrolled: all loads in flight at once); rounded up so
                // every key is covered when G does not divide CH (e.g. D = 80 -> G = 3)
                for (var u = 0u; u < (CH + G - 1u) / G; u += 1u) {
                    let t = grp + u * G;
                    if (t < kn) {
                        acc = acc + pp[t] * ld_v(vcol + t * vs);
                    }
                }
            }
            part[c * 256u + g_lid] = select(0.0, acc, lane_on);
        }
        workgroupBarrier();
        if (g_lid < DW) {
            for (var c = 0u; c < DC; c += 1u) {
                let dd = g_lid + c * DW;
                if (dd < D) {
                    var tot: f32 = 0.0;
                    for (var g = 0u; g < G; g += 1u) {
                        tot = tot + part[c * 256u + g * DW + g_lid];
                    }
                    st(bhq * D + dd, select(0.0, tot / l, l > 0.0));
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
