// Reduce, one threadgroup per output row (long reductions). Reduced axis is last.
// P: [0] rows (outer numel), [1] len, [2..2+rank-1) outer shape,
//    then per input: offset, strides[rank] (last = reduced axis stride).
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
    const uint len = P[1];
    const uint nsg = (tgsize + 31u) / 32u;
    for (uint o = tg; o < rows; o += ntg) {
{{OUTER}}        float acc = {{IDENT}};
        for (uint j = tid; j < len; j += tgsize) {
{{LOADS}}            const float v = {{EXPR}};
            acc = {{COMBINE}};
        }
        acc = {{TG_REDUCE}}(acc, red, lane, sg, nsg);
        if (tid == 0) {
            stv(out, o, acc);
        }
    }
}
