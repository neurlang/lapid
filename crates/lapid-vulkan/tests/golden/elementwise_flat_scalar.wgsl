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
// Elementwise: out[out_view(idx)] = expr(in_i[in_view_i(idx)]) for every idx of a rank-R
// space, grid-stride over the linear index. Views are Flat (offset + lin), Scalar (offset)
// or Strided (offset + sum(c_d * stride_d)); the coordinates are only decoded when needed.
// Params: [numel, shape[R], out(off, strides[R]), in_i(off, strides[R])...]
@group(0) @binding(1) var<storage, read> in0: array<f32>;
fn ld_in0(e: u32) -> f32 {
    return in0[e];
}
@group(0) @binding(2) var<storage, read> in1: array<f32>;
fn ld_in1(e: u32) -> f32 {
    return in1[e];
}
@group(0) @binding(3) var<storage, read_write> out: array<f16>;
fn st(e: u32, v: f32) {
    out[e] = f16(v);
}

fn kernel(wg_lin: u32, wg_total: u32) {
    let numel = P[0];
    let gstride = wg_total * 256u;
    for (var lin = wg_lin * 256u + g_lid; lin < numel; lin += gstride) {
        let x0 = ld_in0(P[6u] + lin);
        let x1 = ld_in1(P[9u]);
        st(P[3u] + lin, ((x0 + x1) * ((erf_(((x0 + x1) * 0.70710677f)) + 1.0f) * 0.5f)));
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
