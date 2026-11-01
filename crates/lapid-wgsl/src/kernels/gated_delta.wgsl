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
{{BINDINGS}}
// state rows per lane (ceil(Dk / lanes), baked in)
const NP: u32 = {{NP}}u;
// #if !SUBGROUP
var<workgroup> gred: array<f32, 256>;
// #endif

fn colsum(x: f32, lane: u32, slot: u32) -> f32 {
// #if SUBGROUP
    return subgroupAdd(x);
// #endif
// #if !SUBGROUP
    workgroupBarrier();
    gred[g_lid] = x;
    workgroupBarrier();
    for (var s = 16u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            gred[g_lid] = gred[g_lid] + gred[g_lid + s];
        }
        workgroupBarrier();
    }
    return gred[slot * 32u];
// #endif
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
// #if SUBGROUP
    let slot = g_sgid;
    let lane = g_sgi;
    let lanes = g_sgs;
    let cols = g_nsg;
// #endif
// #if !SUBGROUP
    let slot = g_lid / 32u;
    let lane = g_lid % 32u;
    let lanes = 32u;
    let cols = 8u;
// #endif
    for (var item = wg_lin; item < items; item += wg_total) {
        let cb = item % ncb;
        let bh = item / ncb;
        let h = bh % H;
        let b = bh / H;
        let j = cb * cols + slot;
        let act = j < DV;
        var S: array<f32, NP>;
        let sbase = bh * DK * DV;
        for (var p = 0u; p < NP; p += 1u) {
            let i = lane + p * lanes;
            S[p] = 0.0;
            if (act && i < DK) {
                S[p] = state[sbase + i * DV + j];
            }
        }
        let obase = bh * (T + DK) * DV;
        for (var t = 0u; t < T; t += 1u) {
            let r = (b * T + t) * H + h;
            let eg = exp(g[r]);
            let bt = beta[r];
            var kv: f32 = 0.0;
            for (var p = 0u; p < NP; p += 1u) {
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
            for (var p = 0u; p < NP; p += 1u) {
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
        for (var p = 0u; p < NP; p += 1u) {
            let i = lane + p * lanes;
            if (act && i < DK) {
                st(obase + (T + i) * DV + j, S[p]);
            }
        }
    }
}
