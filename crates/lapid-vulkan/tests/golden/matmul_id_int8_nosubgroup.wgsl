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
// Mixture-of-experts matmul (ggml MUL_MAT_ID): out[t, r, n] = W[ids[t, r]][n, :] . x[t, :].
// One group of 32 lanes (a subgroup, or 32 threads reducing through shared memory) per
// output column. An expert id outside [0, E) raises the error flag and uses expert 0.
// Params: [T, R, N, K, scales_words, expert_stride (words for quantized, elements else),
//          col_blocks, E]
@group(0) @binding(1) var<storage, read> a4: array<vec4<f32>>;
fn lda4(e: u32) -> vec4<f32> {
    return a4[e >> 2u];
}
@group(0) @binding(2) var<storage, read> w: array<u32>;
fn wdeq(x: u32, e: u32) -> f32 {
    let wb = x * P[5];
    let q = extractBits(bitcast<i32>(w[wb + (e >> 2u)]), (e & 3u) * 8u, 8u);
    return f32(q) * unpack2x16float(w[wb + P[4] + (e >> 6u)])[(e >> 5u) & 1u];
}
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
fn st(e: u32, v: f32) {
    out[e] = v;
}

@group(0) @binding(3) var<storage, read> ids: array<i32>;
@group(0) @binding(5) var<storage, read_write> status: array<u32>;
var<workgroup> gred: array<f32, 256>;

fn kernel(wg_lin: u32, wg_total: u32) {
    let T = P[0];
    let R = P[1];
    let N = P[2];
    let K = P[3];
    let ncb = P[6];
    let E = P[7];
    let items = T * R * ncb;
    let slot = g_lid / 32u;
    let lane = g_lid % 32u;
    let lanes = 32u;
    let cols = 8u;
    for (var item = wg_lin; item < items; item += wg_total) {
        let cb = item % ncb;
        let tr = item / ncb;
        let t = tr / R;
        let raw = ids[tr];
        var x = 0u;
        if (raw < 0 || u32(raw) >= E) {
            if (lane == 0u) {
                status[0] = 1u;
            }
        } else {
            x = u32(raw);
        }
        let j = cb * cols + slot;
        let a_row = t * K;
        var part: f32 = 0.0;
        if (j < N) {
            let wb = x * P[5];
            for (var k4 = lane; k4 < K / 4u; k4 += lanes) {
                let e = j * K + k4 * 4u;
                let sc = unpack2x16float(w[wb + P[4] + (e >> 6u)])[(e >> 5u) & 1u];
                part = part + dot(vec4<f32>(unpack4xI8(w[wb + (e >> 2u)])), lda4(a_row + k4 * 4u)) * sc;
            }
        }
        workgroupBarrier();
        gred[g_lid] = part;
        workgroupBarrier();
        for (var s = 16u; s > 0u; s = s >> 1u) {
            if (lane < s) {
                gred[g_lid] = gred[g_lid] + gred[g_lid + s];
            }
            workgroupBarrier();
        }
        let tot = gred[slot * 32u];
        if (lane == 0u && j < N) {
            st(tr * N + j, tot);
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
