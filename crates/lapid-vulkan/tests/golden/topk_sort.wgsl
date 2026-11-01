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
@group(0) @binding(1) var<storage, read> x: array<f16>;
fn ld_x(e: u32) -> f32 {
    return f32(x[e]);
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

var<workgroup> hist: array<atomic<u32>, 256>;
var<workgroup> sel: array<u32, 3>;
var<workgroup> cnt: atomic<u32>;
var<workgroup> ck: array<u32, MAXK>;
var<workgroup> ci: array<u32, MAXK>;

// One radix-select pass: among values v (key, or index for `by_index`) whose bits under
// `mask` equal `prefix`, find the 8-bit digit at `shift` holding the `need`-th element
// (largest first for keys, smallest first for indices). Result in sel[0] (new prefix),
// sel[1] (new need), sel[2] (count in the chosen bin).
fn radix_pass(base: u32, cols: u32, shift: u32, mask: u32, prefix: u32, need: u32, by_index: bool, tkey: u32) {
    atomicStore(&hist[g_lid], 0u);
    workgroupBarrier();
    for (var c = g_lid; c < cols; c += 256u) {
        let kk = okey(ld_x(base + c));
        var v = kk;
        var ok = true;
        if (by_index) {
            v = c;
            ok = kk == tkey;
        }
        if (ok && (v & mask) == prefix) {
            atomicAdd(&hist[(v >> shift) & 255u], 1u);
        }
    }
    workgroupBarrier();
    if (g_lid == 0u) {
        var acc = 0u;
        var b = 0u;
        var h = 0u;
        for (var t = 0u; t < 256u; t += 1u) {
            b = select(255u - t, t, by_index);
            h = atomicLoad(&hist[b]);
            if (acc + h >= need) {
                break;
            }
            acc += h;
        }
        sel[0] = prefix | (b << shift);
        sel[1] = need - acc;
        sel[2] = h;
    }
    workgroupBarrier();
}

fn kernel(wg_lin: u32, wg_total: u32) {
    let rows = P[0];
    let cols = P[1];
    let k = P[2];
    var n2 = 1u;
    while (n2 < k) {
        n2 = n2 * 2u;
    }
    for (var r = wg_lin; r < rows; r += wg_total) {
        let base = r * cols;
        // threshold key T = k-th largest key; `need` of the elements equal to T are taken
        var prefix = 0u;
        var mask = 0u;
        var need = k;
        var eq = 0u;
        for (var ps = 0u; ps < 4u; ps += 1u) {
            let shift = 24u - ps * 8u;
            radix_pass(base, cols, shift, mask, prefix, need, false, 0u);
            prefix = sel[0];
            need = sel[1];
            eq = sel[2];
            mask = mask | (255u << shift);
            workgroupBarrier();
        }
        let tkey = prefix;
        // index cut-off among the elements equal to T (all of them when no tie is cut)
        var ilim = 0xffffffffu;
        if (eq > need) {
            var ip = 0u;
            var im = 0u;
            var ineed = need;
            for (var ps = 0u; ps < 4u; ps += 1u) {
                let shift = 24u - ps * 8u;
                radix_pass(base, cols, shift, im, ip, ineed, true, tkey);
                ip = sel[0];
                ineed = sel[1];
                im = im | (255u << shift);
                workgroupBarrier();
            }
            ilim = ip;
        }
        if (g_lid == 0u) {
            atomicStore(&cnt, 0u);
        }
        workgroupBarrier();
        for (var c = g_lid; c < cols; c += 256u) {
            let kk = okey(ld_x(base + c));
            if (kk > tkey || (kk == tkey && c <= ilim)) {
                let s = atomicAdd(&cnt, 1u);
                if (s < MAXK) {
                    ck[s] = kk;
                    ci[s] = c;
                }
            }
        }
        workgroupBarrier();
        for (var t = k + g_lid; t < n2; t += 256u) {
            ck[t] = 0u;
            ci[t] = 0xffffffffu;
        }
        workgroupBarrier();
        // bitonic sort of n2 entries, "before" order first
        for (var size = 2u; size <= n2; size = size * 2u) {
            for (var stride = size / 2u; stride > 0u; stride = stride / 2u) {
                for (var t = g_lid; t < n2 / 2u; t += 256u) {
                    let lo = 2u * stride * (t / stride) + t % stride;
                    let hi = lo + stride;
                    let up = (lo & size) == 0u;
                    let swap = before(ck[hi], ci[hi], ck[lo], ci[lo]) == up;
                    if (swap) {
                        let tk = ck[lo];
                        let ti = ci[lo];
                        ck[lo] = ck[hi];
                        ci[lo] = ci[hi];
                        ck[hi] = tk;
                        ci[hi] = ti;
                    }
                }
                workgroupBarrier();
            }
        }
        for (var t = g_lid; t < k; t += 256u) {
            out[r * k + t] = i32(ci[t]);
        }
        workgroupBarrier();
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
