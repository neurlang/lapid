// MoE routing for the MATMUL_ID GEMM: one threadgroup buckets the (t, r) pairs by expert.
// route (scratch) = counts[E], then E lists of `cap` pair indices (p = t * R + r). The order
// inside a list is unspecified (every output row is computed independently, so results are
// deterministic). An out-of-range expert id raises the status flag and the pair is dropped.
// P: [0] pairs (T * R), [1] E, [2] cap.
#define MAX_E {{MAX_E}}

kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
{{BINDINGS}}    uint tid [[thread_index_in_threadgroup]],
    uint nt [[threads_per_threadgroup]])
{
    threadgroup atomic_uint cnt[MAX_E];
    const uint pairs = P[0];
    const uint ne = P[1];
    const uint cap = P[2];
    for (uint e = tid; e < ne; e += nt) {
        atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < pairs; p += nt) {
        const int e = ids[p];
        if (e < 0 || uint(e) >= ne) {
            status[0] = 1u;
            continue;
        }
        const uint slot = atomic_fetch_add_explicit(&cnt[uint(e)], 1u, memory_order_relaxed);
        route[ne + uint(e) * cap + slot] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < ne; e += nt) {
        route[e] = atomic_load_explicit(&cnt[e], memory_order_relaxed);
    }
}
