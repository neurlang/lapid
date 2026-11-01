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
// take_along_axis over the canonical [outer, n, inner] space: exact element copy of x at the
// index read from `idx`. Out-of-range indices raise the error flag and write 0 (x is not read).
// HALF: 16-bit elements, one thread per output word (two elements, no atomics).
// Params: [outer, n, inner, axis_len, total, x(off, s0, s1, s2), idx(off, s0, s1, s2)]
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read> idx: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
@group(0) @binding(4) var<storage, read_write> status: array<u32>;

// Source element of output element `e`, or 0xffffffff when its index is out of range.
fn src_of(e: u32) -> u32 {
    let inner = P[2];
    let i = e % inner;
    let rest = e / inner;
    let j = rest % P[1];
    let o = rest / P[1];
    let raw = idx[P[9] + o * P[10] + j * P[11] + i * P[12]];
    if (raw < 0 || u32(raw) >= P[3]) {
        status[0] = 1u;
        return 0xffffffffu;
    }
    return P[5] + o * P[6] + u32(raw) * P[7] + i * P[8];
}

fn half_at(s: u32) -> u32 {
    if (s == 0xffffffffu) {
        return 0u;
    }
    return (x[s >> 1u] >> ((s & 1u) * 16u)) & 0xffffu;
}

fn kernel(wg_lin: u32, wg_total: u32) {
    let total = P[4];
    let gstride = wg_total * 256u;
    let words = (total + 1u) / 2u;
    for (var w = wg_lin * 256u + g_lid; w < words; w += gstride) {
        var v = half_at(src_of(2u * w));
        if (2u * w + 1u < total) {
            v = v | (half_at(src_of(2u * w + 1u)) << 16u);
        }
        out[w] = v;
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
