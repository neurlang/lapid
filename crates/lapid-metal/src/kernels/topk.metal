// Top-k per row of contiguous [rows, cols] (1 < k <= KMAX): one threadgroup per row.
// Every element gets a unique 64-bit key (ordered value bits << 32 | ~index), so "value
// descending, ties -> lower index" is plain key order. A radix select (8 bits per pass, from the
// top, stopping as soon as the chosen bucket holds exactly the remaining count) finds the
// prefix of the k-th largest key; the k elements with (key & mask) >= prefix are collected in
// threadgroup memory and bitonic-sorted descending. P: [0] rows, [1] cols, [2] k.
#define KMAX {{KMAX}}

kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
{{BINDINGS}}    uint tg [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup atomic_uint hist[256];
    threadgroup ulong cand[KMAX];
    threadgroup ulong st_prefix;
    threadgroup ulong st_mask;
    threadgroup uint st_remaining;
    threadgroup uint st_done;
    threadgroup atomic_uint count;
    const uint rows = P[0];
    const uint cols = P[1];
    const uint k = P[2];
    uint p2 = 1u;
    while (p2 < k) {
        p2 <<= 1;
    }
    for (uint r = tg; r < rows; r += ntg) {
        const uint base = r * cols;
        if (tid == 0) {
            st_prefix = 0ul;
            st_mask = 0ul;
            st_remaining = k;
            st_done = 0u;
            atomic_store_explicit(&count, 0u, memory_order_relaxed);
        }
        for (int shift = 56; shift >= 0; shift -= 8) {
            for (uint b = tid; b < 256u; b += tgsize) {
                atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const ulong prefix = st_prefix;
            const ulong mask = st_mask;
            for (uint c = tid; c < cols; c += tgsize) {
                const ulong key = topk_key(ldv(x, base + c), c);
                if ((key & mask) == prefix) {
                    atomic_fetch_add_explicit(&hist[uint(key >> shift) & 255u], 1u, memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                const uint want = st_remaining;
                uint above = 0u;
                for (int b = 255; b >= 0; b--) {
                    const uint h = atomic_load_explicit(&hist[b], memory_order_relaxed);
                    if (above + h >= want) {
                        st_prefix = prefix | (ulong(b) << shift);
                        st_mask = mask | (0xfful << shift);
                        st_remaining = want - above;
                        st_done = h == want - above ? 1u : 0u;
                        break;
                    }
                    above += h;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (st_done != 0u) {
                break;
            }
        }
        // Collect the k selected keys (unordered), pad to a power of two, sort descending.
        const ulong prefix = st_prefix;
        const ulong mask = st_mask;
        for (uint c = tid; c < cols; c += tgsize) {
            const ulong key = topk_key(ldv(x, base + c), c);
            if ((key & mask) >= prefix) {
                const uint slot = atomic_fetch_add_explicit(&count, 1u, memory_order_relaxed);
                if (slot < KMAX) {
                    cand[slot] = key;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = k + tid; i < p2; i += tgsize) {
            cand[i] = 0ul;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint size = 2u; size <= p2; size <<= 1) {
            for (uint stride = size >> 1; stride > 0u; stride >>= 1) {
                for (uint i = tid; i < p2 / 2u; i += tgsize) {
                    const uint lo = 2u * i - (i & (stride - 1u));
                    const uint hi = lo + stride;
                    const bool desc = (lo & size) == 0u;
                    const ulong a = cand[lo];
                    const ulong b = cand[hi];
                    if ((a < b) == desc) {
                        cand[lo] = b;
                        cand[hi] = a;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        for (uint j = tid; j < k; j += tgsize) {
            out[r * k + j] = int(~uint(cand[j] & 0xfffffffful));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
