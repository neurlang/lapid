// Reduce, one thread per output (short reductions / many rows). Reduced axis is last.
// P: [0] rows (outer numel), [1] len, [2..2+rank-1) outer shape,
//    then per input: offset, strides[rank] (last = reduced axis stride).
kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
{{BINDINGS}}    uint gid [[thread_position_in_grid]],
    uint gsize [[threads_per_grid]])
{
    const uint rows = P[0];
    const uint len = P[1];
    for (uint o = gid; o < rows; o += gsize) {
{{OUTER}}        float acc = {{IDENT}};
        for (uint j = 0; j < len; j++) {
{{LOADS}}            const float v = {{EXPR}};
            acc = {{COMBINE}};
        }
        stv(out, o, acc);
    }
}
