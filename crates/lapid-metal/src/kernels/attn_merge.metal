// Merge the key splits of `attn_vec`: per output element (row = (b, h, q), column c),
// out = sum_s o_s[c] e^(m_s - M) / sum_s l_s e^(m_s - M) with M = max_s m_s.
// Scratch row layout: splits x (m, l, o[d]). P: [0] rows, [1] d, [2] splits.
kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
{{BINDINGS}}    uint gid [[thread_position_in_grid]],
    uint gsize [[threads_per_grid]])
{
    const uint rows = P[0];
    const uint d = P[1];
    const uint splits = P[2];
    const uint n = rows * d;
    for (uint i = gid; i < n; i += gsize) {
        const uint r = i / d;
        const uint c = i - r * d;
        const uint base = r * splits * (d + 2);
        float mx = -INFINITY;
        for (uint s = 0; s < splits; s++) {
            mx = max(mx, scratch[base + s * (d + 2)]);
        }
        float o = 0.0f;
        float l = 0.0f;
        if (mx != -INFINITY) {
            for (uint s = 0; s < splits; s++) {
                const uint sb = base + s * (d + 2);
                const float ms = scratch[sb];
                const float w = ms == -INFINITY ? 0.0f : exp(ms - mx);
                l = fma(scratch[sb + 1], w, l);
                o = fma(scratch[sb + 2 + c], w, o);
            }
        }
        stv(out, i, l > 0.0f ? o / l : 0.0f);
    }
}
