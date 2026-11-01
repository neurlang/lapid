// Matrix-vector product for few rows (m = MR <= 4, the decoder) with transposed B [n, k]:
// out[bt, i, j] = epilogue(sum_k A[bt, i, k] * B[j, k]).
//
// Threadgroup = 4 simdgroups; each simdgroup owns NR = 2 output columns. Lanes stride over
// k in chunks of 8 (a simdgroup covers 256 consecutive k per step, so B rows are read
// fully coalesced), partial dot products are combined with simd_sum. Quantized B: a lane's
// values always lie in one 32-group, so the int dot product is scaled once; int4 uses its
// own loop (two lanes per group, both nibbles of every loaded byte).
// MR is a compile-time constant (one pipeline per m) so the accumulators stay in registers.
//
// MATMUL_ID (MoE, MR = 1): threadgroup z = pair p = t * R + r; B is expert ids[p] of the
// stack `w` (P[15] elements of w per expert), A row t = p / R, output row p.
//
// P: [0] m, [1] n, [2] k, [3] a.offset, [4] a.batch_stride, [5] a.row_stride,
//    [6] b.offset, [7] b.batch_stride, [8] b.row_stride, [9] B scales offset (halves),
//    [10] out.offset, [11] out.batch_stride, [12] out.row_stride (unit column stride),
//    [13 + 4e ..] extra e view over [batch, m, n]: offset, stride0, stride1, stride2.
//    MATMUL_ID: [13] R, [14] E, [15] expert stride (elements of w).
#define MR {{MR}}
#define NR 2

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
    const uint b_base = P[6] + bt * P[7];
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
    device const {{BTY}}* b = w + uint(expert) * P[15];
    const uint a_base = (bt / P[13]) * k;
    const uint a_rs = 0;
    const uint b_base = 0;
    const uint o_base = bt * n;
// #endif
    const uint b_rs = P[8];

    float acc[MR][NR];
    for (uint i = 0; i < MR; i++) {
        for (uint r = 0; r < NR; r++) {
            acc[i][r] = 0.0f;
        }
    }

// #if B_INT4
    // Two lanes per 32-group: lane half h reads bytes 8h..8h+8 of the group's 16, whose low
    // nibbles are elements 8h.. and high nibbles elements 16+8h.. (every loaded bit is used).
    for (uint g = lane >> 1; g * 32 < k; g += 16) {
        const uint e0 = g * 32 + (lane & 1u) * 8u;
        float4 alo[MR][2];
        float4 ahi[MR][2];
        for (uint i = 0; i < MR; i++) {
            const uint ar = a_base + i * a_rs + e0;
            alo[i][0] = ld4(a, ar);
            alo[i][1] = ld4(a, ar + 4);
            ahi[i][0] = ld4(a, ar + 16);
            ahi[i][1] = ld4(a, ar + 20);
        }
        for (uint r = 0; r < NR; r++) {
            const uint gc = col0 + r;
            if (gc >= n) {
                break;
            }
            const uint2 qw = *(device const uint2*)(b + ((gc * k + e0 - (lane & 1u) * 8u) >> 1) + (lane & 1u) * 8u);
            const uint4 b0 = uint4(as_type<uchar4>(qw.x));
            const uint4 b1 = uint4(as_type<uchar4>(qw.y));
            const float4 l0 = float4(b0 & 15u) - 8.0f;
            const float4 l1 = float4(b1 & 15u) - 8.0f;
            const float4 h0 = float4(b0 >> 4) - 8.0f;
            const float4 h1 = float4(b1 >> 4) - 8.0f;
            const float d = float(((device const half*)b)[P[9] + gc * (k >> 5) + g]);
            for (uint i = 0; i < MR; i++) {
                const float s = dot(l0, alo[i][0]) + dot(l1, alo[i][1]) + dot(h0, ahi[i][0]) + dot(h1, ahi[i][1]);
                acc[i][r] = fma(s, d, acc[i][r]);
            }
        }
    }
// #endif
// #if B_INT8
    // 16 int8 per lane (16-byte aligned, one 32-group: k % 32 == 0), a simdgroup covers 512
    // consecutive bytes of a row per step.
    for (uint k0 = lane * 16; k0 < k; k0 += 512) {
        float4 av[MR][4];
        for (uint i = 0; i < MR; i++) {
            const uint ar = a_base + i * a_rs + k0;
            for (uint t = 0; t < 4; t++) {
                av[i][t] = ld4(a, ar + 4 * t);
            }
        }
        for (uint r = 0; r < NR; r++) {
            const uint gc = col0 + r;
            if (gc >= n) {
                break;
            }
            const uint4 qw = *(device const uint4*)(b + gc * k + k0);
            const float d = float(((device const half*)b)[P[9] + gc * (k >> 5) + (k0 >> 5)]);
            for (uint i = 0; i < MR; i++) {
                float s = 0.0f;
                for (uint t = 0; t < 4; t++) {
                    s += dot(float4(as_type<char4>(qw[t])), av[i][t]);
                }
                acc[i][r] = fma(s, d, acc[i][r]);
            }
        }
    }
// #endif
// #if B_FLOAT
    for (uint k0 = lane * 8; k0 < k; k0 += 256) {
        const bool full = k0 + 8 <= k;
        float av[MR][8];
        for (uint i = 0; i < MR; i++) {
            const uint ar = a_base + i * a_rs + k0;
            if (full) {
// #if A_F32
                const float4 lo = float4(*(device const packed_float4*)(a + ar));
                const float4 hi = float4(*(device const packed_float4*)(a + ar + 4));
                av[i][0] = lo.x; av[i][1] = lo.y; av[i][2] = lo.z; av[i][3] = lo.w;
                av[i][4] = hi.x; av[i][5] = hi.y; av[i][6] = hi.z; av[i][7] = hi.w;
// #endif
// #if A_F16
                const float4 lo = float4(*(device const packed_half4*)(a + ar));
                const float4 hi = float4(*(device const packed_half4*)(a + ar + 4));
                av[i][0] = lo.x; av[i][1] = lo.y; av[i][2] = lo.z; av[i][3] = lo.w;
                av[i][4] = hi.x; av[i][5] = hi.y; av[i][6] = hi.z; av[i][7] = hi.w;
// #endif
// #if A_ANY
                for (uint t = 0; t < 8; t++) {
                    av[i][t] = ldv(a, ar + t);
                }
// #endif
            } else {
                for (uint t = 0; t < 8; t++) {
                    av[i][t] = k0 + t < k ? ldv(a, ar + t) : 0.0f;
                }
            }
        }
        for (uint r = 0; r < NR; r++) {
            const uint gc = col0 + r;
            if (gc >= n) {
                break;
            }
// #if B_FLOAT
            const uint br = b_base + gc * b_rs + k0;
            float4 w0;
            float4 w1;
            if (full) {
// #if B_F32
                w0 = float4(*(device const packed_float4*)(b + br));
                w1 = float4(*(device const packed_float4*)(b + br + 4));
// #endif
// #if B_F16
                w0 = float4(*(device const packed_half4*)(b + br));
                w1 = float4(*(device const packed_half4*)(b + br + 4));
// #endif
// #if B_ANY
                w0 = float4(ldv(b, br), ldv(b, br + 1), ldv(b, br + 2), ldv(b, br + 3));
                w1 = float4(ldv(b, br + 4), ldv(b, br + 5), ldv(b, br + 6), ldv(b, br + 7));
// #endif
            } else {
                float w[8];
                for (uint t = 0; t < 8; t++) {
                    w[t] = k0 + t < k ? ldv(b, br + t) : 0.0f;
                }
                w0 = float4(w[0], w[1], w[2], w[3]);
                w1 = float4(w[4], w[5], w[6], w[7]);
            }
            for (uint i = 0; i < MR; i++) {
                acc[i][r] += dot(w0, float4(av[i][0], av[i][1], av[i][2], av[i][3]))
                    + dot(w1, float4(av[i][4], av[i][5], av[i][6], av[i][7]));
            }
// #endif
        }
    }
// #endif

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
