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
// Gather rows of a (possibly quantized) [rows, cols] table by i32 indices. An index outside
// [0, rows) raises the backend's error flag (`status[0] = 1`) and reads row 0 instead.
// Params: [rows, cols, count, scales_words, total = count * cols]
@group(0) @binding(1) var<storage, read> tab: array<u32>;
fn tab_b8(o: u32) -> u32 {
    return (tab[o >> 2u] >> ((o & 3u) * 8u)) & 255u;
}
fn tab_h(o: u32) -> f32 {
    return unpack2x16float(tab[o >> 2u])[(o >> 1u) & 1u];
}
// 32-bit word at a 2-byte aligned offset
fn tab_w(o: u32) -> u32 {
    let w0 = tab[o >> 2u];
    if ((o & 2u) == 0u) {
        return w0;
    }
    return (w0 >> 16u) | (tab[(o >> 2u) + 1u] << 16u);
}
fn tab_smk(bb: u32, j: u32) -> vec2<f32> {
    let q = bb + 4u;
    if (j < 4u) {
        return vec2<f32>(f32(tab_b8(q + j) & 63u), f32(tab_b8(q + j + 4u) & 63u));
    }
    let sc = (tab_b8(q + j + 4u) & 15u) | ((tab_b8(q + j - 4u) >> 6u) << 4u);
    let m = (tab_b8(q + j + 4u) >> 4u) | ((tab_b8(q + j) >> 6u) << 4u);
    return vec2<f32>(f32(sc), f32(m));
}
fn tab_kdeq(bb: u32, i: u32) -> f32 {
    let c = i >> 6u;
    let wi = i & 63u;
    let hi = wi >> 5u;
    let l = wi & 31u;
    let q = ((tab_b8(bb + 48u + c * 32u + l) >> (hi * 4u)) & 15u) | (((tab_b8(bb + 16u + l) >> (2u * c + hi)) & 1u) << 4u);
    let sm = tab_smk(bb, 2u * c + hi);
    let dm = unpack2x16float(tab[bb >> 2u]);
    return dm.x * sm.x * f32(q) - dm.y * sm.y;
}
fn tab_kdeq4(bb: u32, i: u32) -> vec4<f32> {
    let c = i >> 6u;
    let hi = (i >> 5u) & 1u;
    let l = i & 31u;
    let qw = tab[(bb + 48u + c * 32u + l) >> 2u];
    var q = (vec4<u32>(qw, qw >> 8u, qw >> 16u, qw >> 24u) >> vec4<u32>(hi * 4u)) & vec4<u32>(15u);
    let qh = tab[(bb >> 2u) + 4u + (l >> 2u)];
    q = q | (((vec4<u32>(qh, qh >> 8u, qh >> 16u, qh >> 24u) >> vec4<u32>(2u * c + hi)) & vec4<u32>(1u)) << vec4<u32>(4u));
    let sm = tab_smk(bb, 2u * c + hi);
    let dm = unpack2x16float(tab[bb >> 2u]);
    return vec4<f32>(q) * (dm.x * sm.x) - vec4<f32>(dm.y * sm.y);
}
fn tab_deq(so: u32, e: u32) -> f32 {
    return tab_kdeq((e >> 8u) * 176u, e & 255u);
}
fn ld_tab(e: u32) -> f32 {
    return tab_deq(0u, e);
}
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
fn st(e: u32, v: f32) {
    out[e] = v;
}

@group(0) @binding(2) var<storage, read> idx: array<i32>;
@group(0) @binding(4) var<storage, read_write> status: array<u32>;

fn kernel(wg_lin: u32, wg_total: u32) {
    let rows = P[0];
    let cols = P[1];
    let so = P[3];
    let total = P[4];
    let gstride = wg_total * 256u;
    for (var e = wg_lin * 256u + g_lid; e < total; e += gstride) {
        let i = e / cols;
        let c = e - i * cols;
        let raw = idx[i];
        var r = 0u;
        if (raw < 0 || u32(raw) >= rows) {
            status[0] = 1u;
        } else {
            r = u32(raw);
        }
        let src = r * cols + c;
        st(e, tab_deq(so, src));
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
