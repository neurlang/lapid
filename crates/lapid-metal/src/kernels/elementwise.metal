// Elementwise: out[out_view(idx)] = expr(inputs[i][in_view_i(idx)]) over `shape`.
// P: [0] numel, [1..=rank] shape, then per input and the output: offset, strides[rank].
// Grid-stride loop over the flat index; coordinates are decomposed only when some view is
// strided (contiguous views index with offset + i, broadcast scalars with offset).
kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
{{BINDINGS}}    uint gid [[thread_position_in_grid]],
    uint gsize [[threads_per_grid]])
{
    const uint n = P[0];
    for (uint i = gid; i < n; i += gsize) {
{{INDEX}}{{LOADS}}        const {{RTY}} r = {{EXPR}};
        {{STORE}}(out, {{OUT_INDEX}}, r);
    }
}
