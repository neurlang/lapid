// Top-1 (argmax) per row of contiguous [rows, cols]: one threadgroup per row (grid-stride over
// rows), each thread scans a strided slice, then a simdgroup butterfly and a threadgroup step.
// Order: value descending (as f32, -0 == +0), ties -> lower index. P: [0] rows, [1] cols.
kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
{{BINDINGS}}    uint tg [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint red_k[32];
    threadgroup uint red_i[32];
    const uint rows = P[0];
    const uint cols = P[1];
    const uint nsg = (tgsize + 31u) / 32u;
    for (uint r = tg; r < rows; r += ntg) {
        const uint base = r * cols;
        // (key, index) with the sentinel (0, ~0) losing to every real element.
        uint bk = 0u;
        uint bi = 0xffffffffu;
        for (uint c = tid; c < cols; c += tgsize) {
            const uint key = ord_key(ldv(x, base + c));
            if (key > bk || (key == bk && c < bi)) {
                bk = key;
                bi = c;
            }
        }
        argmax_simd(bk, bi);
        if (lane == 0) {
            red_k[sg] = bk;
            red_i[sg] = bi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            bk = lane < nsg ? red_k[lane] : 0u;
            bi = lane < nsg ? red_i[lane] : 0xffffffffu;
            argmax_simd(bk, bi);
            if (lane == 0) {
                out[r] = int(bi);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
