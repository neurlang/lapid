// Row softmax over contiguous [rows, cols]; one threadgroup per row (grid-stride over rows).
// Fully masked rows (all -inf) produce zeros. P: [0] rows, [1] cols.
kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
{{BINDINGS}}    uint tg [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[32];
    const uint rows = P[0];
    const uint cols = P[1];
    const uint nsg = (tgsize + 31u) / 32u;
    for (uint r = tg; r < rows; r += ntg) {
        const uint base = r * cols;
        float m = -INFINITY;
        for (uint j = tid; j < cols; j += tgsize) {
            m = max(m, ldv(x, base + j));
        }
        m = tg_max(m, red, lane, sg, nsg);
        if (m == -INFINITY) {
            for (uint j = tid; j < cols; j += tgsize) {
                stv(out, base + j, 0.0f);
            }
            continue;
        }
        float s = 0.0f;
        for (uint j = tid; j < cols; j += tgsize) {
            s += exp(ldv(x, base + j) - m);
        }
        s = tg_sum(s, red, lane, sg, nsg);
        const float inv = 1.0f / s;
        for (uint j = tid; j < cols; j += tgsize) {
            stv(out, base + j, exp(ldv(x, base + j) - m) * inv);
        }
    }
}
