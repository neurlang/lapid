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
// Row softmax, one workgroup per row. Params: [rows, cols]
@group(0) @binding(1) var<storage, read> x: array<f32>;
fn ld_x(e: u32) -> f32 {
    return x[e];
}
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
fn st(e: u32, v: f32) {
    out[e] = v;
}

// Workgroup-wide sum / max over 256 threads; every thread gets the result.
// Must be called in workgroup-uniform control flow.
var<workgroup> red: array<f32, 256>;

fn wg_sum(v: f32) -> f32 {
    let s = subgroupAdd(v);
    if (g_sgi == 0u) {
        red[g_sgid] = s;
    }
    workgroupBarrier();
    var r: f32 = 0.0;
    for (var i = 0u; i < g_nsg; i += 1u) {
        r = r + red[i];
    }
    workgroupBarrier();
    return r;
}

fn wg_max(v: f32) -> f32 {
    let s = subgroupMax(v);
    if (g_sgi == 0u) {
        red[g_sgid] = s;
    }
    workgroupBarrier();
    var r: f32 = bitcast<f32>(0xff800000u);
    for (var i = 0u; i < g_nsg; i += 1u) {
        r = max(r, red[i]);
    }
    workgroupBarrier();
    return r;
}

fn kernel(wg_lin: u32, wg_total: u32) {
    let rows = P[0];
    let cols = P[1];
    for (var r = wg_lin; r < rows; r += wg_total) {
        let base = r * cols;
        var m: f32 = bitcast<f32>(0xff800000u);
        for (var j = g_lid; j < cols; j += 256u) {
            m = max(m, ld_x(base + j));
        }
        m = wg_max(m);
        var sum: f32 = 0.0;
        for (var j = g_lid; j < cols; j += 256u) {
            sum += exp(ld_x(base + j) - m);
        }
        sum = wg_sum(sum);
        let inv = 1.0 / sum;
        for (var j = g_lid; j < cols; j += 256u) {
            st(base + j, exp(ld_x(base + j) - m) * inv);
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
