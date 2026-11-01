// lapid prelude: extensions, the params binding, builtin mirrors and math helpers.

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
// Matrix-vector product with K-quant (Q4_K / Q5_K / Q6_K) transposed B, small m.
// One group of 32 lanes (a subgroup, or 32 threads reducing through shared memory) per
// output column; each 256-element super-block is shared by 8 lanes (32 elements each,
// word loads), consecutive blocks of the row go to consecutive lanes.
// Params: same as gemv ([batch, m, n, k, a.off, a.bs, a.rs, b.off, b.bs, b.rs, -, col_blocks,
//          -, out(off, bs, rs), extra_e(off, s_batch, s_m, s_n)...]).
@group(0) @binding(1) var<storage, read> a: array<f32>;
fn ld_a(e: u32) -> f32 {
    return a[e];
}
fn lda4(e: u32) -> vec4<f32> {
    return vec4<f32>(ld_a(e), ld_a(e + 1u), ld_a(e + 2u), ld_a(e + 3u));
}
fn lda(e: u32) -> f32 {
    return ld_a(e);
}
@group(0) @binding(2) var<storage, read> b: array<u32>;
fn b_b8(o: u32) -> u32 {
    return (b[o >> 2u] >> ((o & 3u) * 8u)) & 255u;
}
fn b_h(o: u32) -> f32 {
    return unpack2x16float(b[o >> 2u])[(o >> 1u) & 1u];
}
// 32-bit word at a 2-byte aligned offset
fn b_w(o: u32) -> u32 {
    let w0 = b[o >> 2u];
    if ((o & 2u) == 0u) {
        return w0;
    }
    return (w0 >> 16u) | (b[(o >> 2u) + 1u] << 16u);
}
fn b_smk(bb: u32, j: u32) -> vec2<f32> {
    let q = bb + 4u;
    if (j < 4u) {
        return vec2<f32>(f32(b_b8(q + j) & 63u), f32(b_b8(q + j + 4u) & 63u));
    }
    let sc = (b_b8(q + j + 4u) & 15u) | ((b_b8(q + j - 4u) >> 6u) << 4u);
    let m = (b_b8(q + j + 4u) >> 4u) | ((b_b8(q + j) >> 6u) << 4u);
    return vec2<f32>(f32(sc), f32(m));
}
fn b_kdeq(bb: u32, i: u32) -> f32 {
    let c = i >> 6u;
    let wi = i & 63u;
    let hi = wi >> 5u;
    let l = wi & 31u;
    let q = ((b_b8(bb + 48u + c * 32u + l) >> (hi * 4u)) & 15u) | (((b_b8(bb + 16u + l) >> (2u * c + hi)) & 1u) << 4u);
    let sm = b_smk(bb, 2u * c + hi);
    let dm = unpack2x16float(b[bb >> 2u]);
    return dm.x * sm.x * f32(q) - dm.y * sm.y;
}
fn b_kdeq4(bb: u32, i: u32) -> vec4<f32> {
    let c = i >> 6u;
    let hi = (i >> 5u) & 1u;
    let l = i & 31u;
    let qw = b[(bb + 48u + c * 32u + l) >> 2u];
    var q = (vec4<u32>(qw, qw >> 8u, qw >> 16u, qw >> 24u) >> vec4<u32>(hi * 4u)) & vec4<u32>(15u);
    let qh = b[(bb >> 2u) + 4u + (l >> 2u)];
    q = q | (((vec4<u32>(qh, qh >> 8u, qh >> 16u, qh >> 24u) >> vec4<u32>(2u * c + hi)) & vec4<u32>(1u)) << vec4<u32>(4u));
    let sm = b_smk(bb, 2u * c + hi);
    let dm = unpack2x16float(b[bb >> 2u]);
    return vec4<f32>(q) * (dm.x * sm.x) - vec4<f32>(dm.y * sm.y);
}
fn b_deq(so: u32, e: u32) -> f32 {
    return b_kdeq((e >> 8u) * 176u, e & 255u);
}
fn ld_b(e: u32) -> f32 {
    return b_deq(0u, e);
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
fn b_kdot(bb: u32, s: u32, a0: u32) -> f32 {
    // lane s (0..8) of a block: qs bytes 16s..16s+15 -> elements c*64 + l0 + [0, 16)
    // (low nibbles, sub-block 2c) and c*64 + 32 + l0 + [0, 16) (high nibbles, 2c + 1)
    let c = s >> 1u;
    let l0 = (s & 1u) * 16u;
    let wb = bb >> 2u;
    let sw = vec3<u32>(b[wb + 1u], b[wb + 2u], b[wb + 3u]);
    let s1 = kq_smr(sw, 2u * c);
    let s2 = kq_smr(sw, 2u * c + 1u);
    let dm = unpack2x16float(b[wb]);
    let one = vec4<f32>(1.0);
    var dlo: f32 = 0.0;
    var dhi: f32 = 0.0;
    var alo_s: f32 = 0.0;
    var ahi_s: f32 = 0.0;
    for (var t = 0u; t < 4u; t += 1u) {
        let qw = b[wb + 12u + 4u * s + t];
        let qv = vec4<u32>(qw, qw >> 8u, qw >> 16u, qw >> 24u);
        var lo = vec4<f32>(qv & vec4<u32>(15u));
        var hi = vec4<f32>((qv >> vec4<u32>(4u)) & vec4<u32>(15u));
        let qh = b[wb + 4u + (l0 >> 2u) + t];
        let qhv = vec4<u32>(qh, qh >> 8u, qh >> 16u, qh >> 24u) >> vec4<u32>(2u * c);
        lo = lo + vec4<f32>((qhv & vec4<u32>(1u)) * 16u);
        hi = hi + vec4<f32>(((qhv >> vec4<u32>(1u)) & vec4<u32>(1u)) * 16u);
        let alo = lda4(a0 + c * 64u + l0 + 4u * t);
        let ahi = lda4(a0 + c * 64u + 32u + l0 + 4u * t);
        dlo = dlo + dot(lo, alo);
        dhi = dhi + dot(hi, ahi);
        alo_s = alo_s + dot(alo, one);
        ahi_s = ahi_s + dot(ahi, one);
    }
    return dm.x * (s1.x * dlo + s2.x * dhi) - dm.y * (s1.y * alo_s + s2.y * ahi_s);
}
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
fn st(e: u32, v: f32) {
    out[e] = v;
}

var<workgroup> gred: array<f32, 256>;

fn kernel(wg_lin: u32, wg_total: u32) {
    let batch = P[0];
    let M = P[1];
    let N = P[2];
    let K = P[3];
    let ncb = P[11];
    let items = batch * M * ncb;
    let nb = K / 256u;
    let slot = g_lid / 32u;
    let lane = g_lid % 32u;
    let lanes = 32u;
    let cols = 8u;
    for (var item = wg_lin; item < items; item += wg_total) {
        let cb = item % ncb;
        let rest = item / ncb;
        let i = rest % M;
        let bt = rest / M;
        let j = cb * cols + slot;
        let a_row = P[4] + bt * P[5] + i * P[6];
        var part: f32 = 0.0;
        if (j < N) {
            let row_block = j * nb;
            for (var idx = lane; idx < nb * 8u; idx += lanes) {
                let blk = idx >> 3u;
                part = part + b_kdot((row_block + blk) * 176u, idx & 7u, a_row + blk * 256u);
            }
        }
        workgroupBarrier();
        gred[g_lid] = part;
        workgroupBarrier();
        for (var s = 16u; s > 0u; s = s >> 1u) {
            if (lane < s) {
                gred[g_lid] = gred[g_lid] + gred[g_lid + s];
            }
            workgroupBarrier();
        }
        let tot = gred[slot * 32u];
        if (lane == 0u && j < N) {
            let acc = tot;
            st(P[13u] + bt * P[14u] + i * P[15u] + j, acc);
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
) {
    g_lid = lidx;
    let wg_lin = wgid.x + nwg.x * (wgid.y + nwg.y * wgid.z);
    let wg_total = nwg.x * nwg.y * nwg.z;
    kernel(wg_lin, wg_total);
}
