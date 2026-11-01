// Gated delta rule (ggml GATED_DELTA_NET), LPC lanes per state column (32-lane subgroups,
// CPS = 32 / LPC columns each): lane (c, s) keeps rows s, s + LPC, ... of column j in
// registers, so a workgroup covers 8 * CPS columns. Per token the two column dot products
// are one subgroupAdd of a CPS-vector (component c = this lane's column).
// The next token's q / k / v / g / beta loads are issued before the current token's math.
// out [B, H, T + Dk, Dv]: rows [0, T) = o, rows [T, T + Dk) = final state.
// Params: [B, T, H, Dk, Dv, scale bits, col_blocks]
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> g: array<f32>;
@group(0) @binding(5) var<storage, read> beta: array<f32>;
@group(0) @binding(6) var<storage, read> state: array<f32>;
{{BINDINGS}}
// state rows per lane (Dk / LPC, baked in)
const NP: u32 = {{NP}}u;
const LPC: u32 = {{LPC}}u;
const CPS: u32 = 32u / LPC;

fn colsum4(x: f32, c: u32) -> f32 {
    var v = vec{{CPS}}<f32>(0.0);
    v[c] = x;
    return subgroupAdd(v)[c];
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
    let c = g_sgi / LPC;
    let sl = g_sgi % LPC;
    for (var item = wg_lin; item < items; item += wg_total) {
        let cb = item % ncb;
        let bh = item / ncb;
        let h = bh % H;
        let b = bh / H;
        let j = cb * 8u * CPS + g_sgid * CPS + c;
        let act = j < DV;
        let jc = min(j, DV - 1u);
        var S: array<f32, NP>;
        let sbase = bh * DK * DV;
        for (var p = 0u; p < NP; p += 1u) {
            S[p] = state[sbase + (sl + LPC * p) * DV + jc];
        }
        let obase = bh * (T + DK) * DV;
        var kc: array<f32, NP>;
        var qc: array<f32, NP>;
        var gc: f32 = 0.0;
        var bc: f32 = 0.0;
        var vc: f32 = 0.0;
        if (T > 0u) {
            let r0 = b * T * H + h;
            for (var p = 0u; p < NP; p += 1u) {
                kc[p] = k[r0 * DK + sl + LPC * p];
                qc[p] = q[r0 * DK + sl + LPC * p];
            }
            gc = g[r0];
            bc = beta[r0];
            vc = v[r0 * DV + jc];
        }
        for (var t = 0u; t < T; t += 1u) {
            let rn = (b * T + min(t + 1u, T - 1u)) * H + h;
            var kn: array<f32, NP>;
            var qn: array<f32, NP>;
            for (var p = 0u; p < NP; p += 1u) {
                kn[p] = k[rn * DK + sl + LPC * p];
                qn[p] = q[rn * DK + sl + LPC * p];
            }
            let gn = g[rn];
            let bn = beta[rn];
            let vn = v[rn * DV + jc];
            let eg = exp(gc);
            var kv: f32 = 0.0;
            for (var p = 0u; p < NP; p += 1u) {
                S[p] = S[p] * eg;
                kv = kv + S[p] * kc[p];
            }
            let dl = bc * (vc - colsum4(kv, c));
            var qv: f32 = 0.0;
            for (var p = 0u; p < NP; p += 1u) {
                S[p] = S[p] + kc[p] * dl;
                qv = qv + S[p] * qc[p];
            }
            let o = colsum4(qv, c);
            if (act && sl == 0u) {
                st(obase + t * DV + j, o * scale);
            }
            kc = kn;
            qc = qn;
            gc = gn;
            bc = bn;
            vc = vn;
        }
        if (act) {
            for (var p = 0u; p < NP; p += 1u) {
                st(obase + (T + sl + LPC * p) * DV + j, S[p]);
            }
        }
    }
}
