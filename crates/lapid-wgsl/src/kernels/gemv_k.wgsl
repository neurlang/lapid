// Matrix-vector product with K-quant (Q4_K / Q5_K / Q6_K) transposed B, small m.
// One group of 32 lanes (a subgroup, or 32 threads reducing through shared memory) per
// output column; each 256-element super-block is shared by 8 lanes (32 elements each,
// word loads), consecutive blocks of the row go to consecutive lanes.
// Params: same as gemv ([batch, m, n, k, a.off, a.bs, a.rs, b.off, b.bs, b.rs, -, col_blocks,
//          -, out(off, bs, rs), extra_e(off, s_batch, s_m, s_n)...]).
{{BINDINGS}}
// #if !SUBGROUP
var<workgroup> gred: array<f32, 256>;
// #endif

fn kernel(wg_lin: u32, wg_total: u32) {
    let batch = P[0];
    let M = P[1];
    let N = P[2];
    let K = P[3];
    let ncb = P[11];
    let items = batch * M * ncb;
    let nb = K / 256u;
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
        let rest = item / ncb;
        let i = rest % M;
        let bt = rest / M;
        let j = cb * cols + slot;
        let a_row = P[4] + bt * P[5] + i * P[6];
        var part: f32 = 0.0;
        if (j < N) {
            let row_block = j * nb;
            for (var idx = lane; idx < nb * 8u; idx += lanes) {
                let blk = idx >> 3u;
                part = part + b_kdot((row_block + blk) * {{BB}}u, idx & 7u, a_row + blk * 256u);
            }
        }
// #if SUBGROUP
        let tot = subgroupAdd(part);
// #endif
// #if !SUBGROUP
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
// #endif
        if (lane == 0u && j < N) {
            let acc = tot;
{{EPILOGUE}}        }
    }
}
