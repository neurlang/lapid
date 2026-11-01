// Gather rows: out[i, c] = dequant(table[indices[i], c]); an out-of-range index raises the
// status flag and gives 0.
// P: [0] rows, [1] cols, [2] count, [3] scales offset in halves (quantized tables).
kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
{{BINDINGS}}    uint gid [[thread_position_in_grid]],
    uint gsize [[threads_per_grid]])
{
    const uint rows = P[0];
    const uint cols = P[1];
    const uint n = P[2] * cols;
    for (uint i = gid; i < n; i += gsize) {
        const uint r = i / cols;
        const uint c = i - r * cols;
        const int idx = indices[r];
        float v = 0.0f;
        if (idx >= 0 && uint(idx) < rows) {
            v = {{LOAD}};
        } else {
            status[0] = 1u;
        }
        stv(out, i, v);
    }
}
