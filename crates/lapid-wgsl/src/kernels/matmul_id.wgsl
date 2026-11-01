// Mixture-of-experts matmul (ggml MUL_MAT_ID): out[t, r, n] = W[ids[t, r]][n, :] . x[t, :].
// One group of 32 lanes (a subgroup, or 32 threads reducing through shared memory) per
// output column. An expert id outside [0, E) raises the error flag and uses expert 0.
// Params: [T, R, N, K, scales_words, expert_stride (words for quantized, elements else),
//          col_blocks, E]
{{BINDINGS}}
@group(0) @binding(3) var<storage, read> ids: array<i32>;
@group(0) @binding(5) var<storage, read_write> status: array<u32>;
// #if !SUBGROUP
var<workgroup> gred: array<f32, 256>;
// #endif

fn kernel(wg_lin: u32, wg_total: u32) {
    let T = P[0];
    let R = P[1];
    let N = P[2];
    let K = P[3];
    let ncb = P[6];
    let E = P[7];
    let items = T * R * ncb;
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
{{DOT}}        }
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
            st(tr * N + j, tot);
        }
    }
}
