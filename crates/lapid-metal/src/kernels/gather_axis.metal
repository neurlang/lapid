// take_along_axis over the canonical space [outer, n, inner], exact element copy (raw words):
// out[o, j, i] = x[x.off + o*xs0 + idx*xs1 + i*xs2], idx = indices[ix.off + o*is0 + j*is1 + i*is2].
// An index outside [0, axis_len) raises the status flag and writes 0 (never reads out of bounds).
// P: [0] total = outer*n*inner, [1] n, [2] inner, [3] axis_len, [4] x.offset, [5..8) x strides,
//    [8] indices.offset, [9..12) indices strides.
kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
{{BINDINGS}}    uint gid [[thread_position_in_grid]],
    uint gsize [[threads_per_grid]])
{
    const uint total = P[0];
    const uint n = P[1];
    const uint inner = P[2];
    const uint axis_len = P[3];
    for (uint e = gid; e < total; e += gsize) {
        const uint i = e % inner;
        const uint r = e / inner;
        const uint j = r % n;
        const uint o = r / n;
        const int idx = indices[P[8] + o * P[9] + j * P[10] + i * P[11]];
        const bool ok = idx >= 0 && uint(idx) < axis_len;
        if (!ok) {
            status[0] = 1u;
        }
        out[e] = ok ? x[P[4] + o * P[5] + uint(idx) * P[6] + i * P[7]] : {{T}}(0);
    }
}
