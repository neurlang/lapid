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
// Flash attention without matrix cores (many queries, head dim D baked in, multiple of 4,
// <= 128): 64 queries per workgroup, one thread per query holding q and the output
// accumulator in vec4 registers; K/V tiles of BK keys staged through shared memory;
// per-key online softmax.
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
@group(0) @binding(4) var<storage, read> msk: array<f32>;
fn ld_msk(e: u32) -> f32 {
    return msk[e];
}
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
fn st(e: u32, v: f32) {
    out[e] = v;
}

const D4: u32 = 16u;
const BK: u32 = 32u;

var<workgroup> Ks: array<vec4<f32>, 512>;
var<workgroup> Vs: array<vec4<f32>, 512>;

fn kernel(wg_lin: u32, wg_total: u32) {
    let H = P[1];
    let SQ = P[2];
    let SK = P[3];
    let scale = bitcast<f32>(P[5]);
    let nqb = P[6];
    let items = P[0] * H * nqb;
    for (var item = wg_lin; item < items; item += wg_total) {
        let qb = item % nqb;
        let bh = item / nqb;
        let h = bh % H;
        let b = bh / H;
        let qstart = qb * 64u;
        let qi = qstart + g_lid;
        let valid = qi < SQ;
        let qbase = P[7u] + b * P[8u] + h * P[9u];
        let kbase = P[12u] + b * P[13u] + h * P[14u];
        let vbase = P[17u] + b * P[18u] + h * P[19u];
        let qrow = qbase + qi * P[10u];
        let qs = P[11u];
        var q: array<vec4<f32>, D4>;
        var o: array<vec4<f32>, D4>;
        for (var i = 0u; i < D4; i += 1u) {
            q[i] = vec4<f32>(0.0);
            o[i] = vec4<f32>(0.0);
            if (valid) {
                let e = qrow + i * 4u * qs;
                q[i] = vec4<f32>(ld_q(e), ld_q(e + qs), ld_q(e + 2u * qs), ld_q(e + 3u * qs)) * scale;
            }
        }
        let mrow = P[22u] + b * P[23u] + h * P[24u] + qi * P[25u];
        let mstep = P[26u];
        let last_q = min(qstart + 63u, SQ - 1u);
        let kend = u32(clamp(i32(last_q) + 1 + i32(SK) - i32(SQ), 0, i32(SK)));
        let ks = P[15u];
        let kd = P[16u];
        let vs = P[20u];
        let vd = P[21u];
        var m: f32 = -3.0e38;
        var l: f32 = 0.0;
        for (var k0 = 0u; k0 < kend; k0 += BK) {
            workgroupBarrier();
            for (var t = g_lid; t < BK * D4; t += 64u) {
                let kj = t / D4;
                let dd = (t % D4) * 4u;
                let key = k0 + kj;
                var kv = vec4<f32>(0.0);
                var vv = vec4<f32>(0.0);
                if (key < kend) {
                    let kr = kbase + key * ks + dd * kd;
                    let vr = vbase + key * vs + dd * vd;
                    kv = vec4<f32>(ld_k(kr), ld_k(kr + kd), ld_k(kr + 2u * kd), ld_k(kr + 3u * kd));
                    vv = vec4<f32>(ld_v(vr), ld_v(vr + vd), ld_v(vr + 2u * vd), ld_v(vr + 3u * vd));
                }
                Ks[t] = kv;
                Vs[t] = vv;
            }
            workgroupBarrier();
            if (valid) {
                for (var kj = 0u; kj < BK; kj += 1u) {
                    let key = k0 + kj;
                    let kb = kj * D4;
                    var x: f32 = 0.0;
                    for (var i = 0u; i < D4; i += 1u) {
                        x = x + dot(q[i], Ks[kb + i]);
                    }
                    if (key < kend) {
                        x = x + ld_msk(mrow + key * mstep);
                    }
                    let ok = key < kend && i32(key) <= i32(qi) + i32(SK) - i32(SQ);
                    x = select(bitcast<f32>(0xff800000u), x, ok);
                    if (x > m) {
                        let c = exp(m - x);
                        l = l * c;
                        for (var i = 0u; i < D4; i += 1u) {
                            o[i] = o[i] * c;
                        }
                        m = x;
                    }
                    let p = exp(x - m);
                    l = l + p;
                    for (var i = 0u; i < D4; i += 1u) {
                        o[i] = fma(vec4<f32>(p), Vs[kb + i], o[i]);
                    }
                }
            }
        }
        if (valid) {
            let inv = select(0.0, 1.0 / l, l > 0.0);
            let ob = ((b * H + h) * SQ + qi) * D4 * 4u;
            for (var i = 0u; i < D4; i += 1u) {
                st(ob + i * 4u, o[i].x * inv);
                st(ob + i * 4u + 1u, o[i].y * inv);
                st(ob + i * 4u + 2u, o[i].z * inv);
                st(ob + i * 4u + 3u, o[i].w * inv);
            }
        }
    }
}

// Entry point: mirrors builtins into private globals and runs `kernel` with the
// linearized workgroup id / count (kernels grid-stride over their work items).
@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) lidx: u32,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    g_lid = lidx;
    let wg_lin = wgid.x + nwg.x * (wgid.y + nwg.y * wgid.z);
    let wg_total = nwg.x * nwg.y * nwg.z;
    kernel(wg_lin, wg_total);
}
