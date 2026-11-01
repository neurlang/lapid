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
// Layer norm over rows, one workgroup per row (two-pass mean / variance). Rows up to
// 4 * 256 columns are read once into registers; longer rows are re-read per pass.
// Params: [rows, cols, eps_bits]
@group(0) @binding(1) var<storage, read> x: array<f32>;
fn ld_x(e: u32) -> f32 {
    return x[e];
}
@group(0) @binding(2) var<storage, read> w: array<f32>;
fn ld_w(e: u32) -> f32 {
    return w[e];
}
@group(0) @binding(3) var<storage, read> bias: array<f32>;
fn ld_bias(e: u32) -> f32 {
    return bias[e];
}
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
fn st(e: u32, v: f32) {
    out[e] = v;
}

// Workgroup-wide sum / max over 256 threads; every thread gets the result.
// Must be called in workgroup-uniform control flow.
var<workgroup> red: array<f32, 256>;

fn wg_sum(v: f32) -> f32 {
    red[g_lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (g_lid < s) {
            red[g_lid] = red[g_lid] + red[g_lid + s];
        }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

fn wg_max(v: f32) -> f32 {
    red[g_lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (g_lid < s) {
            red[g_lid] = max(red[g_lid], red[g_lid + s]);
        }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

fn kernel(wg_lin: u32, wg_total: u32) {
    let rows = P[0];
    let cols = P[1];
    let eps = bitcast<f32>(P[2]);
    let fcols = f32(cols);
    let cached = cols <= 1024u;
    for (var r = wg_lin; r < rows; r += wg_total) {
        let base = r * cols;
        var xv: array<f32, 4>;
        var sum: f32 = 0.0;
        if (cached) {
            for (var u = 0u; u < 4u; u += 1u) {
                let j = g_lid + u * 256u;
                xv[u] = 0.0;
                if (j < cols) {
                    xv[u] = ld_x(base + j);
                }
                sum += xv[u];
            }
        } else {
            for (var j = g_lid; j < cols; j += 256u) {
                sum += ld_x(base + j);
            }
        }
        let mean = wg_sum(sum) / fcols;
        var sq: f32 = 0.0;
        if (cached) {
            for (var u = 0u; u < 4u; u += 1u) {
                if (g_lid + u * 256u < cols) {
                    let dlt = xv[u] - mean;
                    sq += dlt * dlt;
                }
            }
        } else {
            for (var j = g_lid; j < cols; j += 256u) {
                let dlt = ld_x(base + j) - mean;
                sq += dlt * dlt;
            }
        }
        let rstd = inverseSqrt(wg_sum(sq) / fcols + eps);
        if (cached) {
            for (var u = 0u; u < 4u; u += 1u) {
                let j = g_lid + u * 256u;
                if (j < cols) {
                    st(base + j, (xv[u] - mean) * rstd * ld_w(j) + ld_bias(j));
                }
            }
        } else {
            for (var j = g_lid; j < cols; j += 256u) {
                st(base + j, (ld_x(base + j) - mean) * rstd * ld_w(j) + ld_bias(j));
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
) {
    g_lid = lidx;
    let wg_lin = wgid.x + nwg.x * (wgid.y + nwg.y * wgid.z);
    let wg_total = nwg.x * nwg.y * nwg.z;
    kernel(wg_lin, wg_total);
}
