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
// Mixture-of-experts matmul (ggml MUL_MAT_ID): out[t, r, n] = W[ids[t, r]][n, :] . x[t, :].
// One group of 32 lanes (a subgroup, or 32 threads reducing through shared memory) per
// output column. An expert id outside [0, E) raises the error flag and uses expert 0.
// Params: [T, R, N, K, scales_words, expert_stride (words for quantized, elements else),
//          col_blocks, E]
@group(0) @binding(1) var<storage, read> a4: array<vec4<f32>>;
fn lda4(e: u32) -> vec4<f32> {
    return a4[e >> 2u];
}
@group(0) @binding(2) var<storage, read> w: array<u32>;
fn w_b8(o: u32) -> u32 {
    return (w[o >> 2u] >> ((o & 3u) * 8u)) & 255u;
}
fn w_h(o: u32) -> f32 {
    return unpack2x16float(w[o >> 2u])[(o >> 1u) & 1u];
}
// 32-bit word at a 2-byte aligned offset
fn w_w(o: u32) -> u32 {
    let w0 = w[o >> 2u];
    if ((o & 2u) == 0u) {
        return w0;
    }
    return (w0 >> 16u) | (w[(o >> 2u) + 1u] << 16u);
}
fn w_smk(bb: u32, j: u32) -> vec2<f32> {
    let q = bb + 4u;
    if (j < 4u) {
        return vec2<f32>(f32(w_b8(q + j) & 63u), f32(w_b8(q + j + 4u) & 63u));
    }
    let sc = (w_b8(q + j + 4u) & 15u) | ((w_b8(q + j - 4u) >> 6u) << 4u);
    let m = (w_b8(q + j + 4u) >> 4u) | ((w_b8(q + j) >> 6u) << 4u);
    return vec2<f32>(f32(sc), f32(m));
}
fn w_kdeq(bb: u32, i: u32) -> f32 {
    let c = i >> 6u;
    let wi = i & 63u;
    let hi = wi >> 5u;
    let q = (w_b8(bb + 16u + c * 32u + (wi & 31u)) >> (hi * 4u)) & 15u;
    let sm = w_smk(bb, 2u * c + hi);
    let dm = unpack2x16float(w[bb >> 2u]);
    return dm.x * sm.x * f32(q) - dm.y * sm.y;
}
fn w_kdeq4(bb: u32, i: u32) -> vec4<f32> {
    let c = i >> 6u;
    let hi = (i >> 5u) & 1u;
    let l = i & 31u;
    let qw = w[(bb + 16u + c * 32u + l) >> 2u];
    var q = (vec4<u32>(qw, qw >> 8u, qw >> 16u, qw >> 24u) >> vec4<u32>(hi * 4u)) & vec4<u32>(15u);
    let sm = w_smk(bb, 2u * c + hi);
    let dm = unpack2x16float(w[bb >> 2u]);
    return vec4<f32>(q) * (dm.x * sm.x) - vec4<f32>(dm.y * sm.y);
}
fn w_kdeq16(bb: u32, i: u32) -> array<vec4<f32>, 4> {
    let c = i >> 6u;
    let hi = (i >> 5u) & 1u;
    let l = i & 31u;
    let sm = w_smk(bb, 2u * c + hi);
    let dm = unpack2x16float(w[bb >> 2u]);
    let sc = dm.x * sm.x;
    let mn = vec4<f32>(dm.y * sm.y);
    let w0 = (bb + 16u + c * 32u + l) >> 2u;
    var r: array<vec4<f32>, 4>;
    for (var u = 0u; u < 4u; u += 1u) {
        let qw = w[w0 + u];
        var q = (vec4<u32>(qw, qw >> 8u, qw >> 16u, qw >> 24u) >> vec4<u32>(hi * 4u)) & vec4<u32>(15u);
        r[u] = vec4<f32>(q) * sc - mn;
    }
    return r;
}
fn w_deq(so: u32, e: u32) -> f32 {
    return w_kdeq((e >> 8u) * 144u, e & 255u);
}
fn ld_w(e: u32) -> f32 {
    return w_deq(0u, e);
}
fn kq_sb(sw: vec3<u32>, i: u32) -> u32 {
    return (sw[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}
fn kq_smr(sw: vec3<u32>, j: u32) -> vec2<f32> {
    if (j < 4u) {
        return vec2<f32>(f32(kq_sb(sw, j) & 63u), f32(kq_sb(sw, j + 4u) & 63u));
    }
    let hi = kq_sb(sw, j + 4u);
    return vec2<f32>(f32((hi & 15u) | ((kq_sb(sw, j - 4u) >> 6u) << 4u)), f32((hi >> 4u) | ((kq_sb(sw, j) >> 6u) << 4u)));
}
fn w_kdot(bb: u32, s: u32, a0: u32) -> f32 {
    // lane s (0..8) of a block: qs bytes 16s..16s+15 -> elements c*64 + l0 + [0, 16)
    // (low nibbles, sub-block 2c) and c*64 + 32 + l0 + [0, 16) (high nibbles, 2c + 1)
    let c = s >> 1u;
    let l0 = (s & 1u) * 16u;
    let wb = bb >> 2u;
    let sw = vec3<u32>(w[wb + 1u], w[wb + 2u], w[wb + 3u]);
    let s1 = kq_smr(sw, 2u * c);
    let s2 = kq_smr(sw, 2u * c + 1u);
    let dm = unpack2x16float(w[wb]);
    let one = vec4<f32>(1.0);
    var dlo: f32 = 0.0;
    var dhi: f32 = 0.0;
    var alo_s: f32 = 0.0;
    var ahi_s: f32 = 0.0;
    for (var t = 0u; t < 4u; t += 1u) {
        let qw = w[wb + 4u + 4u * s + t];
        let qv = vec4<u32>(qw, qw >> 8u, qw >> 16u, qw >> 24u);
        var lo = vec4<f32>(qv & vec4<u32>(15u));
        var hi = vec4<f32>((qv >> vec4<u32>(4u)) & vec4<u32>(15u));
        let alo = lda4(a0 + c * 64u + l0 + 4u * t);
        let ahi = lda4(a0 + c * 64u + 32u + l0 + 4u * t);
        dlo = dlo + dot(lo, alo);
        dhi = dhi + dot(hi, ahi);
        alo_s = alo_s + dot(alo, one);
        ahi_s = ahi_s + dot(ahi, one);
    }
    return dm.x * (s1.x * dlo + s2.x * dhi) - dm.y * (s1.y * alo_s + s2.y * ahi_s);
}
fn wdeq(x: u32, e: u32) -> f32 {
    return w_kdeq(x * P[5] * 4u + (e >> 8u) * 144u, e & 255u);
}
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
fn st(e: u32, v: f32) {
    out[e] = v;
}

@group(0) @binding(3) var<storage, read> ids: array<i32>;
@group(0) @binding(5) var<storage, read_write> status: array<u32>;

fn kernel(wg_lin: u32, wg_total: u32) {
    let T = P[0];
    let R = P[1];
    let N = P[2];
    let K = P[3];
    let ncb = P[6];
    let E = P[7];
    let items = T * R * ncb;
    let slot = g_sgid;
    let lane = g_sgi;
    let lanes = g_sgs;
    let cols = g_nsg;
    for (var item = wg_lin; item < items; item += wg_total) {
        let cb = item % ncb;
        let tr = item / ncb;
        let t = tr / R;
        let raw = ids[tr];
        var x = 0u;
        if (raw < 0 || u32(raw) >= E) {
            if (lane == 0u) {
                status[0] = 1u;
            }
        } else {
            x = u32(raw);
        }
        let j = cb * cols + slot;
        let a_row = t * K;
        var part: f32 = 0.0;
        if (j < N) {
            let nb = K / 256u;
            let row_byte = x * P[5] * 4u + j * nb * 144u;
            for (var idx = lane; idx < nb * 8u; idx += lanes) {
                let blk = idx >> 3u;
                part = part + w_kdot(row_byte + blk * 144u, idx & 7u, a_row + blk * 256u);
            }
        }
        let tot = subgroupAdd(part);
        if (lane == 0u && j < N) {
            st(tr * N + j, tot);
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
