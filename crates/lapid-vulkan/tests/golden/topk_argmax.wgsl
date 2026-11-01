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
// Top-k column indices of each row of x [rows, cols] (values compared as f32), descending,
// ties -> lower index first. One 256-thread workgroup per row.
// ARGMAX (k = 1): per-thread best, then a subgroup / shared-memory (value, index) reduction.
// Otherwise: radix select (4 x 8-bit passes over order-preserving keys) finds the k-th
// largest key T; if more elements equal T than are needed, a second radix select over
// their indices finds the cut-off; the k winners are appended to shared memory and sorted
// with a bitonic sort (k <= MAXK).
// Params: [rows, cols, k]
@group(0) @binding(1) var<storage, read> x: array<f32>;
fn ld_x(e: u32) -> f32 {
    return x[e];
}

@group(0) @binding(2) var<storage, read_write> out: array<i32>;

const MAXK: u32 = 1024u;

// Order-preserving u32 key of an f32 (larger value -> larger key; -0 == +0).
fn okey(v: f32) -> u32 {
    var b = bitcast<u32>(v);
    if (v == 0.0) {
        b = 0u;
    }
    return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

// (ka, ia) ranks before (kb, ib)
fn before(ka: u32, ia: u32, kb: u32, ib: u32) -> bool {
    return ka > kb || (ka == kb && ia < ib);
}

var<workgroup> rk: array<u32, 256>;
var<workgroup> ri: array<u32, 256>;

fn kernel(wg_lin: u32, wg_total: u32) {
    let rows = P[0];
    let cols = P[1];
    for (var r = wg_lin; r < rows; r += wg_total) {
        let base = r * cols;
        var bk = 0u;
        var bi = 0xffffffffu;
        for (var c = g_lid; c < cols; c += 256u) {
            let kk = okey(ld_x(base + c));
            if (bi == 0xffffffffu || kk > bk) {
                bk = kk;
                bi = c;
            }
        }
        let m = subgroupMax(select(0u, bk, bi != 0xffffffffu));
        let mi = subgroupMin(select(0xffffffffu, bi, bk == m && bi != 0xffffffffu));
        workgroupBarrier();
        if (g_sgi == 0u) {
            rk[g_sgid] = m;
            ri[g_sgid] = mi;
        }
        workgroupBarrier();
        if (g_lid == 0u) {
            var k0 = rk[0];
            var i0 = ri[0];
            for (var s = 1u; s < g_nsg; s += 1u) {
                if (ri[s] != 0xffffffffu && (i0 == 0xffffffffu || before(rk[s], ri[s], k0, i0))) {
                    k0 = rk[s];
                    i0 = ri[s];
                }
            }
            out[r] = i32(i0);
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
