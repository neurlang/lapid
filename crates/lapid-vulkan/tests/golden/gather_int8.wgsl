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
// [0, all_rows) raises the backend's error flag (`status[0] = 1`) and reads row 0 instead.
// A split launch binds rows [base, base + rows) only and skips indices outside them.
// Params: [rows, cols, count, scales_words, total = count * cols, base, all_rows]
@group(0) @binding(1) var<storage, read> tab: array<u32>;
fn tab_q(e: u32) -> f32 {
    let w = tab[e >> 2u];
    return f32(extractBits(bitcast<i32>(w), (e & 3u) * 8u, 8u));
}
fn tab_scale(so: u32, gi: u32) -> f32 {
    return unpack2x16float(tab[so + (gi >> 1u)])[gi & 1u];
}
fn tab_deq(so: u32, e: u32) -> f32 {
    return tab_q(e) * tab_scale(so, e >> 5u);
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
    let base = P[5];
    let all_rows = P[6];
    let gstride = wg_total * 256u;
    for (var e = wg_lin * 256u + g_lid; e < total; e += gstride) {
        let i = e / cols;
        let c = e - i * cols;
        let raw = idx[i];
        var r = 0u;
        if (raw < 0 || u32(raw) >= all_rows) {
            status[0] = 1u;
            if (base != 0u) {
                continue;
            }
        } else {
            if (u32(raw) < base || u32(raw) - base >= rows) {
                continue;
            }
            r = u32(raw) - base;
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
