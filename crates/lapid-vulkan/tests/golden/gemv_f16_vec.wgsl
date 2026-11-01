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
// Matrix-vector product for small m with transposed B (the decoder hot path).
// One group of 32 lanes (a subgroup, or 32 threads reducing through shared memory) per
// output column, 256 / 32 columns per workgroup, no barriers on the subgroup path.
// Each lane group computes NC adjacent columns (A loaded once for all of them); lane l
// handles 8 consecutive k per 256-wide step (coalesced A and B reads); the column sums are
// subgroup adds and lane 0 runs the fused epilogue.
// Params: [batch, m, n, k, a.off, a.bs, a.rs, b.off, b.bs, b.rs, scales_words, col_blocks,
//          -, out(off, bs, rs), extra_e(off, s_batch, s_m, s_n)...]
@group(0) @binding(1) var<storage, read> a4: array<vec4<f32>>;
fn lda4(e: u32) -> vec4<f32> {
    return a4[e >> 2u];
}
fn lda(e: u32) -> f32 {
    return a4[e >> 2u][e & 3u];
}
@group(0) @binding(2) var<storage, read> bv: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
fn st(e: u32, v: f32) {
    out[e] = v;
}

const NC: u32 = 1u;

// sum_{t < 8} A[a_row + k0 + t] * B[j, k0 + t]  (k0 is a multiple of 8)
fn dot8(a_row: u32, b_base: u32, j: u32, k0: u32, K: u32) -> f32 {
    let w = bv[(b_base + j * P[9] + k0) >> 3u];
    let b0 = vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y));
    let b1 = vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w));
    return dot(b0, lda4(a_row + k0)) + dot(b1, lda4(a_row + k0 + 4u));
}

fn kernel(wg_lin: u32, wg_total: u32) {
    let batch = P[0];
    let M = P[1];
    let N = P[2];
    let K = P[3];
    let ncb = P[11];
    let items = batch * M * ncb;
    let slot = g_sgid;
    let lane = g_sgi;
    let lanes = g_sgs;
    let cols = g_nsg;
    let step = lanes * 8u;
    for (var item = wg_lin; item < items; item += wg_total) {
        let cb = item % ncb;
        let rest = item / ncb;
        let i = rest % M;
        let bt = rest / M;
        // NC adjacent columns per lane group: A is loaded once for all of them
        let j0 = (cb * cols + slot) * NC;
        let a_row = P[4] + bt * P[5] + i * P[6];
        let b_base = P[7] + bt * P[8];
        var part: array<f32, NC>;
        if (j0 < N) {
            for (var k0 = lane * 8u; k0 < K; k0 += step) {
                for (var c = 0u; c < NC; c += 1u) {
                    part[c] = part[c] + dot8(a_row, b_base, min(j0 + c, N - 1u), k0, K);
                }
            }
        }
        for (var c = 0u; c < NC; c += 1u) {
            let tot = subgroupAdd(part[c]);
            let j = j0 + c;
            if (lane == 0u && j < N) {
                let acc = tot;
                st(P[13u] + bt * P[14u] + i * P[15u] + j, acc);
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
