// Matrix-vector product for small m with transposed B (the decoder hot path).
// One group of 32 lanes (a subgroup, or 32 threads reducing through shared memory) per
// output column, 256 / 32 columns per workgroup, no barriers on the subgroup path.
// Each lane group computes NC adjacent columns (A loaded once for all of them); lane l
// handles 8 consecutive k per 256-wide step (coalesced A and B reads); the column sums are
// subgroup adds and lane 0 runs the fused epilogue.
// Params: [batch, m, n, k, a.off, a.bs, a.rs, b.off, b.bs, b.rs, scales_words, col_blocks,
//          -, out(off, bs, rs), extra_e(off, s_batch, s_m, s_n)...]
{{BINDINGS}}
const NC: u32 = {{NC}}u;
// #if !SUBGROUP
var<workgroup> gred: array<f32, 256>;
// #endif

// sum_{t < 8} A[a_row + k0 + t] * B[j, k0 + t]  (k0 is a multiple of 8)
fn dot8(a_row: u32, b_base: u32, j: u32, k0: u32, K: u32) -> f32 {
{{DOT8}}}

fn kernel(wg_lin: u32, wg_total: u32) {
    let batch = P[0];
    let M = P[1];
    let N = P[2];
    let K = P[3];
    let ncb = P[11];
    let items = batch * M * ncb;
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
// #if SUBGROUP
            let tot = subgroupAdd(part[c]);
// #endif
// #if !SUBGROUP
            workgroupBarrier();
            gred[g_lid] = part[c];
            workgroupBarrier();
            for (var s = 16u; s > 0u; s = s >> 1u) {
                if (lane < s) {
                    gred[g_lid] = gred[g_lid] + gred[g_lid + s];
                }
                workgroupBarrier();
            }
            let tot = gred[slot * 32u];
// #endif
            let j = j0 + c;
            if (lane == 0u && j < N) {
                let acc = tot;
{{EPILOGUE}}            }
        }
    }
}
