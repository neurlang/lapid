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
// Matrix-vector product with K-quant (Q4_K / Q5_K / Q6_K) transposed B, small m.
// One group of 32 lanes (a subgroup, or 32 threads reducing through shared memory) per
// output column; each 256-element super-block is shared by 8 lanes (32 elements each,
// word loads), consecutive blocks of the row go to consecutive lanes.
// Params: same as gemv ([batch, m, n, k, a.off, a.bs, a.rs, b.off, b.bs, b.rs, -, col_blocks,
//          -, out(off, bs, rs), extra_e(off, s_batch, s_m, s_n)...]).
@group(0) @binding(1) var<storage, read> a4: array<vec4<f32>>;
fn lda4(e: u32) -> vec4<f32> {
    return a4[e >> 2u];
}
fn lda(e: u32) -> f32 {
    return a4[e >> 2u][e & 3u];
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
fn b_kdeq(bb: u32, i: u32) -> f32 {
    let h = i >> 7u;
    let wi = i & 127u;
    let quad = wi >> 5u;
    let l = wi & 31u;
    let lo = (b_b8(bb + h * 64u + l + (quad & 1u) * 32u) >> ((quad >> 1u) * 4u)) & 15u;
    let hb = (b_b8(bb + 128u + h * 32u + l) >> (quad * 2u)) & 3u;
    let q = i32(lo | (hb << 4u)) - 32;
    let sc = extractBits(bitcast<i32>(b_b8(bb + 192u + h * 8u + (l >> 4u) + 2u * quad)), 0u, 8u);
    return b_h(bb + 208u) * f32(sc) * f32(q);
}
fn b_kdeq4(bb: u32, i: u32) -> vec4<f32> {
    let h = i >> 7u;
    let quad = (i >> 5u) & 3u;
    let l = i & 31u;
    let ql = b_w(bb + h * 64u + l + (quad & 1u) * 32u);
    let qh = b_w(bb + 128u + h * 32u + l);
    let lo = (vec4<u32>(ql, ql >> 8u, ql >> 16u, ql >> 24u) >> vec4<u32>((quad >> 1u) * 4u)) & vec4<u32>(15u);
    let hb = (vec4<u32>(qh, qh >> 8u, qh >> 16u, qh >> 24u) >> vec4<u32>(quad * 2u)) & vec4<u32>(3u);
    let q = vec4<f32>(lo | (hb << vec4<u32>(4u))) - vec4<f32>(32.0);
    let sc = extractBits(bitcast<i32>(b_b8(bb + 192u + h * 8u + (l >> 4u) + 2u * quad)), 0u, 8u);
    return q * (b_h(bb + 208u) * f32(sc));
}
fn b_kdeq16(bb: u32, i: u32) -> array<vec4<f32>, 4> {
    let h = i >> 7u;
    let quad = (i >> 5u) & 3u;
    let l = i & 31u;
    let sc = extractBits(bitcast<i32>(b_b8(bb + 192u + h * 8u + (l >> 4u) + 2u * quad)), 0u, 8u);
    let f = b_h(bb + 208u) * f32(sc);
    var r: array<vec4<f32>, 4>;
    for (var u = 0u; u < 4u; u += 1u) {
        let ql = b_w(bb + h * 64u + l + (quad & 1u) * 32u + 4u * u);
        let qh = b_w(bb + 128u + h * 32u + l + 4u * u);
        let lo = (vec4<u32>(ql, ql >> 8u, ql >> 16u, ql >> 24u) >> vec4<u32>((quad >> 1u) * 4u)) & vec4<u32>(15u);
        let hb = (vec4<u32>(qh, qh >> 8u, qh >> 16u, qh >> 24u) >> vec4<u32>(quad * 2u)) & vec4<u32>(3u);
        r[u] = (vec4<f32>(lo | (hb << vec4<u32>(4u))) - vec4<f32>(32.0)) * f;
    }
    return r;
}
fn b_deq(so: u32, e: u32) -> f32 {
    return b_kdeq((e >> 8u) * 210u, e & 255u);
}
fn ld_b(e: u32) -> f32 {
    return b_deq(0u, e);
}
fn b_kdot(bb: u32, s: u32, a0: u32) -> f32 {
    // lane s (0..8): half h, ql bytes h*64 + p*32 + l0 + [0, 16) -> quads p (low nibbles)
    // and p + 2 (high nibbles)
    let h = s >> 2u;
    let p = (s >> 1u) & 1u;
    let l0 = (s & 1u) * 16u;
    let so = bb + 192u + h * 8u + (s & 1u) + 2u * p;
    let sc_lo = f32(extractBits(bitcast<i32>(b_b8(so)), 0u, 8u));
    let sc_hi = f32(extractBits(bitcast<i32>(b_b8(so + 4u)), 0u, 8u));
    var dlo: f32 = 0.0;
    var dhi: f32 = 0.0;
    for (var t = 0u; t < 4u; t += 1u) {
        let ql = b_w(bb + h * 64u + p * 32u + l0 + 4u * t);
        let qh = b_w(bb + 128u + h * 32u + l0 + 4u * t);
        let qlv = vec4<u32>(ql, ql >> 8u, ql >> 16u, ql >> 24u);
        let qhv = vec4<u32>(qh, qh >> 8u, qh >> 16u, qh >> 24u);
        let qlo = vec4<f32>((qlv & vec4<u32>(15u)) | (((qhv >> vec4<u32>(2u * p)) & vec4<u32>(3u)) << vec4<u32>(4u))) - vec4<f32>(32.0);
        let qhi = vec4<f32>(((qlv >> vec4<u32>(4u)) & vec4<u32>(15u)) | (((qhv >> vec4<u32>(2u * p + 4u)) & vec4<u32>(3u)) << vec4<u32>(4u))) - vec4<f32>(32.0);
        dlo = dlo + dot(qlo, lda4(a0 + h * 128u + p * 32u + l0 + 4u * t));
        dhi = dhi + dot(qhi, lda4(a0 + h * 128u + (p + 2u) * 32u + l0 + 4u * t));
    }
    return b_h(bb + 208u) * (sc_lo * dlo + sc_hi * dhi);
}
@group(0) @binding(3) var<storage, read> x0: array<f32>;
fn ld_x0(e: u32) -> f32 {
    return x0[e];
}
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
fn st(e: u32, v: f32) {
    out[e] = v;
}


fn kernel(wg_lin: u32, wg_total: u32) {
    let batch = P[0];
    let M = P[1];
    let N = P[2];
    let K = P[3];
    let ncb = P[11];
    let items = batch * M * ncb;
    let nb = K / 256u;
    let slot = g_sgid;
    let lane = g_sgi;
    let lanes = g_sgs;
    let cols = g_nsg;
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
                part = part + b_kdot((row_block + blk) * 210u, idx & 7u, a_row + blk * 256u);
            }
        }
        let tot = subgroupAdd(part);
        if (lane == 0u && j < N) {
            let acc = tot;
            let e0 = ld_x0(P[16u] + bt * P[17u] + i * P[18u] + j * P[19u]);
            st(P[13u] + bt * P[14u] + i * P[15u] + j, ((acc + e0) * ((erf_(((acc + e0) * 0.70710677f)) + 1.0f) * 0.5f)));
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
