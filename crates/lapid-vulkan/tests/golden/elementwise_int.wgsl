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
// Integer-word helpers for typed integer expressions (included when a kernel calls one).
// Words are u32; signedness is a property of the operation. Division / remainder by zero
// (and i32::MIN / -1) give 0 and never trap; f32 -> int conversions saturate, NaN -> 0.
fn gi_div_s(a: u32, b: u32) -> u32 {
    let x = bitcast<i32>(a);
    let y = bitcast<i32>(b);
    let bad = y == 0 || (x == bitcast<i32>(0x80000000u) && y == -1);
    return select(bitcast<u32>(x / select(y, 1, bad)), 0u, bad);
}

fn gi_rem_s(a: u32, b: u32) -> u32 {
    let x = bitcast<i32>(a);
    let y = bitcast<i32>(b);
    let bad = y == 0 || (x == bitcast<i32>(0x80000000u) && y == -1);
    return select(bitcast<u32>(x % select(y, 1, bad)), 0u, bad);
}

fn gi_div_u(a: u32, b: u32) -> u32 {
    return select(a / max(b, 1u), 0u, b == 0u);
}

fn gi_rem_u(a: u32, b: u32) -> u32 {
    return select(a % max(b, 1u), 0u, b == 0u);
}

fn gi_f2i(x: f32) -> u32 {
    // NaN test on the bits (drivers may fold `x != x`)
    if ((bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u) {
        return 0u;
    }
    if (x >= 2147483648.0) {
        return 0x7fffffffu;
    }
    if (x <= -2147483648.0) {
        return 0x80000000u;
    }
    return bitcast<u32>(i32(x));
}

fn gi_f2u(x: f32) -> u32 {
    if (!(x > 0.0)) {
        return 0u;
    }
    if (x >= 4294967296.0) {
        return 0xffffffffu;
    }
    return u32(x);
}

fn gi_abs(a: u32) -> u32 {
    return select(a, 0u - a, bitcast<i32>(a) < 0);
}
// Elementwise: out[out_view(idx)] = expr(in_i[in_view_i(idx)]) for every idx of a rank-R
// space, grid-stride over the linear index. Views are Flat (offset + lin), Scalar (offset)
// or Strided (offset + sum(c_d * stride_d)); the coordinates are only decoded when needed.
// Params: [numel, shape[R], out(off, strides[R]), in_i(off, strides[R])...]
@group(0) @binding(1) var<storage, read> in0: array<i32>;
fn ld_in0(e: u32) -> f32 {
    return f32(in0[e]);
}
fn ldb_in0(e: u32) -> u32 {
    return bitcast<u32>(in0[e]);
}
@group(0) @binding(2) var<storage, read> in1: array<i32>;
fn ld_in1(e: u32) -> f32 {
    return f32(in1[e]);
}
fn ldb_in1(e: u32) -> u32 {
    return bitcast<u32>(in1[e]);
}
@group(0) @binding(3) var<storage, read_write> out: array<i32>;
fn st(e: u32, v: f32) {
    out[e] = bitcast<i32>(gi_f2i(v));
}
fn st_bits(e: u32, v: u32) {
    out[e] = bitcast<i32>(v);
}

fn kernel(wg_lin: u32, wg_total: u32) {
    let numel = P[0];
    let gstride = wg_total * 256u;
    for (var lin = wg_lin * 256u + g_lid; lin < numel; lin += gstride) {
        var rem = lin;
        let c1 = rem % P[2u];
        rem = rem / P[2u];
        let c0 = rem;
        let xb0 = ldb_in0(P[6u] + lin);
        let xb1 = ldb_in1(P[9u] + c0 * P[10u] + c1 * P[11u]);
        st_bits(P[3u] + lin, (gi_div_s(xb0, xb1) ^ bitcast<u32>(bitcast<i32>(gi_f2i((f32(bitcast<i32>(xb1)) * 0.5f))) >> (0x3u & 31u))));
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
