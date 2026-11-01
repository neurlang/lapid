// Matrix-vector product with a ggml K-quant B [n, k] (k % 256 == 0) for few rows (m = MR <= 4):
// out[bt, i, j] = epilogue(sum_k A[bt, i, k] * B[j, k]).
//
// Threadgroup = 4 simdgroups; each simdgroup owns NR = 4 output columns (A values are reused
// across them). A simdgroup step covers 4 super-blocks: lanes 8b..8b+7 share block b, lane
// `it` = lane % 8 owns 32 of its values (`kq_lane_*` / `kq_aoff_*` in the K-quant helpers),
// so the scales are decoded once per 32 values. Partial sums are combined with simd_sum.
//
// P: [0] m, [1] n, [2] k, [3] a.offset, [4] a.batch_stride, [5] a.row_stride,
//    [6] b.offset, [7] b.batch_stride, [8] b.row_stride, [9] unused (0),
//    [10] out.offset, [11] out.batch_stride, [12] out.row_stride (unit column stride),
//    [13 + 4e ..] extra e view over [batch, m, n]: offset, stride0, stride1, stride2.
//    MATMUL_ID (MR = 1, threadgroup z = pair): [13] R, [14] E, [15] expert stride (bytes).
#define MR {{MR}}
#define NR 4

kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
{{BINDINGS}}    uint3 tgp [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint n = P[1];
    const uint k = P[2];
    const uint bt = tgp.z;
    const uint col0 = (tgp.x * 4 + sg) * NR;
    if (col0 >= n) {
        return;
    }
// #if !MATMUL_ID
    const uint a_base = P[3] + bt * P[4];
    const uint a_rs = P[5];
    const uint o_base = P[10] + bt * P[11];
// #endif
// #if MATMUL_ID
    const int expert = ids[bt];
    if (expert < 0 || uint(expert) >= P[14]) {
        if (tgp.x == 0 && sg == 0 && lane == 0) {
            status[0] = 1u;
        }
        return;
    }
    device const uchar* b = w + uint(expert) * P[15];
    const uint a_base = (bt / P[13]) * k;
    const uint a_rs = 0;
    const uint o_base = bt * n;
// #endif
    const uint nb = k >> 8;
    const uint it = lane & 7u;
    const uint ncols = min(uint(NR), n - col0);

    float acc[MR][NR];
    for (uint i = 0; i < MR; i++) {
        for (uint r = 0; r < NR; r++) {
            acc[i][r] = 0.0f;
        }
    }
    for (uint ib = lane >> 3; ib < nb; ib += 4) {
        for (uint i = 0; i < MR; i++) {
            float4 av[8];
            const uint ar = a_base + i * a_rs + ib * 256u;
            for (uint j = 0; j < 8; j++) {
                av[j] = ld4(a, ar + kq_aoff_{{KQ}}(it, j));
            }
            float4 sums;
            kq_prep_{{KQ}}(av, sums);
            for (uint r = 0; r < NR; r++) {
                if (r < ncols) {
                    device const uchar* blk = b + ((col0 + r) * nb + ib) * {{KQ_BYTES}}u;
                    acc[i][r] += kq_lane_{{KQ}}(blk, it, av, sums);
                }
            }
        }
    }

    for (uint i = 0; i < MR; i++) {
        for (uint r = 0; r < NR; r++) {
            const float x0 = simd_sum(acc[i][r]);
            const uint gc = col0 + r;
            if (lane == 0 && gc < n) {
                const uint gr = i;
{{EXTRA_LOADS}}                stv(out, o_base + gr * P[12] + gc, {{EPILOGUE}});
            }
        }
    }
}
