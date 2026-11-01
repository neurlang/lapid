// Layer norm over contiguous [rows, cols] with weight / bias [cols]; one threadgroup per row.
// Two-pass (mean, then centered variance). P: [0] rows, [1] cols, [2] eps (f32 bits).
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
    const float eps = as_type<float>(P[2]);
    const uint nsg = (tgsize + 31u) / 32u;
    const float inv_n = 1.0f / float(cols);
    for (uint r = tg; r < rows; r += ntg) {
        const uint base = r * cols;
        float s = 0.0f;
        for (uint j = tid; j < cols; j += tgsize) {
            s += ldv(x, base + j);
        }
        const float mean = tg_sum(s, red, lane, sg, nsg) * inv_n;
        float s2 = 0.0f;
        for (uint j = tid; j < cols; j += tgsize) {
            const float d = ldv(x, base + j) - mean;
            s2 = fma(d, d, s2);
        }
        const float rstd = rsqrt(tg_sum(s2, red, lane, sg, nsg) * inv_n + eps);
        for (uint j = tid; j < cols; j += tgsize) {
            const float y = (ldv(x, base + j) - mean) * rstd;
            stv(out, base + j, fma(y, ldv(w, j), ldv(b, j)));
        }
    }
}
