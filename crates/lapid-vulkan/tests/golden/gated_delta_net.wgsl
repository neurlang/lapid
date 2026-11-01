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
// Gated delta rule (ggml GATED_DELTA_NET). Columns j of the state S[Dk, Dv] evolve
// independently, so one group of lanes (a subgroup, or 32 threads reducing through shared
// memory) owns column j of one (b, h) and keeps S[:, j] in registers (lane l holds rows
// l, l + lanes, ...). Per token t:
//   S *= exp(g_t);  d_j = beta_t * (v_tj - sum_i S_ij k_ti);  S_ij += k_ti d_j;
//   o_tj = scale * sum_i S_ij q_ti
// out [B, H, T + Dk, Dv]: rows [0, T) = o, rows [T, T + Dk) = final state.
// Params: [B, T, H, Dk, Dv, scale bits, col_blocks]
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> g: array<f32>;
@group(0) @binding(5) var<storage, read> beta: array<f32>;
@group(0) @binding(6) var<storage, read> state: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
fn st(e: u32, v: f32) {
    out[e] = v;
}

const MAXP: u32 = 16u;

fn colsum(x: f32, lane: u32, slot: u32) -> f32 {
    return subgroupAdd(x);
}

fn kernel(wg_lin: u32, wg_total: u32) {
    let B = P[0];
    let T = P[1];
    let H = P[2];
    let DK = P[3];
    let DV = P[4];
    let scale = bitcast<f32>(P[5]);
    let ncb = P[6];
    let items = B * H * ncb;
    let slot = g_sgid;
    let lane = g_sgi;
    let lanes = g_sgs;
    let cols = g_nsg;
    let np = (DK + lanes - 1u) / lanes;
    for (var item = wg_lin; item < items; item += wg_total) {
        let cb = item % ncb;
        let bh = item / ncb;
        let h = bh % H;
        let b = bh / H;
        let j = cb * cols + slot;
        let act = j < DV;
        var S: array<f32, MAXP>;
        let sbase = bh * DK * DV;
        for (var p = 0u; p < MAXP; p += 1u) {
            let i = lane + p * lanes;
            S[p] = 0.0;
            if (act && p < np && i < DK) {
                S[p] = state[sbase + i * DV + j];
            }
        }
        let obase = bh * (T + DK) * DV;
        for (var t = 0u; t < T; t += 1u) {
            let r = (b * T + t) * H + h;
            let eg = exp(g[r]);
            let bt = beta[r];
            var kv: f32 = 0.0;
            for (var p = 0u; p < np; p += 1u) {
                let i = lane + p * lanes;
                if (i < DK) {
                    S[p] = S[p] * eg;
                    kv = kv + S[p] * k[r * DK + i];
                }
            }
            let ks = colsum(kv, lane, slot);
            var vj: f32 = 0.0;
            if (act) {
                vj = v[r * DV + j];
            }
            let dl = bt * (vj - ks);
            var qv: f32 = 0.0;
            for (var p = 0u; p < np; p += 1u) {
                let i = lane + p * lanes;
                if (i < DK) {
                    S[p] = S[p] + k[r * DK + i] * dl;
                    qv = qv + S[p] * q[r * DK + i];
                }
            }
            let o = colsum(qv, lane, slot);
            if (act && lane == 0u) {
                st(obase + t * DV + j, o * scale);
            }
        }
        for (var p = 0u; p < np; p += 1u) {
            let i = lane + p * lanes;
            if (act && i < DK) {
                st(obase + (T + i) * DV + j, S[p]);
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
