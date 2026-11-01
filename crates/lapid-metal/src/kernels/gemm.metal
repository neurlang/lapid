// Tiled GEMM on simdgroup matrices: out[bt, i, j] = epilogue(sum_k A[bt, i, k] * B[bt, k, j]).
//
// Threadgroup = 4 simdgroups (128 threads) computing a BM x BN = 64 x 64 output tile; each
// simdgroup owns a 32 x 32 quadrant (4 x 4 accumulators of 8 x 8, f32). K advances in
// BK = 32 steps (= one quantization group): the A tile is staged in threadgroup memory as
// f32, the B tile as {{TB}} ([BN][BK], dequantized on the fly for int8 / int4 weights),
// then multiplied with simdgroup_multiply_accumulate. After the K loop the accumulators are
// staged through the same threadgroup memory so the fused epilogue runs per element with
// coalesced stores.
//
// MATMUL_ID (MoE): threadgroup z = expert e; its rows are the (t, r) pairs routed to it by
// the route stage (`route` = scratch: counts[E], then E lists of `cap` pair indices). Row i
// of the tile is pair p = list[row0 + i]: A row t = p / R, output row p, B = expert e.
// K-quant B (k % 256 == 0): a 32-step lies in one super-block; each thread dequantizes 16
// values of one row.
//
// P: [0] m, [1] n, [2] k, [3] a.offset, [4] a.batch_stride, [5] a.row_stride,
//    [6] b.offset, [7] b.batch_stride, [8] b.row_stride, [9] B scales offset (halves),
//    [10] out.offset, [11] out.batch_stride, [12] out.row_stride (unit column stride),
//    [13 + 4e ..] extra e view over [batch, m, n]: offset, stride0, stride1, stride2.
//    MATMUL_ID: [13] R, [14] E, [15] expert stride (elements of w), [16] cap.
#define BM 64
#define BN 64
#define BK 32

kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
{{BINDINGS}}    uint3 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    // 16 KB: A tile (BM*BK f32) + B tile (BN*BK {{TB}}); reused for the C tile (BM*BN f32).
    threadgroup float smem[BM * BN];
    threadgroup float* As = smem;
    threadgroup {{TB}}* Bs = (threadgroup {{TB}}*)(smem + BM * BK);

    const uint n = P[1];
    const uint k = P[2];
    const uint bt = tgp.z;
    const uint row0 = tgp.y * BM;
    const uint col0 = tgp.x * BN;
// #if !MATMUL_ID
    const uint m = P[0];
    const uint a_base = P[3] + bt * P[4];
    const uint a_rs = P[5];
    const uint b_base = P[6] + bt * P[7];
// #endif
// #if MATMUL_ID
    const uint m = route[bt];
    if (row0 >= m) {
        return;
    }
    device const uint* pairs = route + P[14] + bt * P[16];
    device const {{BTY}}* b = w + bt * P[15];
    const uint b_base = 0;
// #endif
    const uint b_rs = P[8];

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    const uint sr = (sg >> 1) * 32;
    const uint sc = (sg & 1) * 32;

    for (uint k0 = 0; k0 < k; k0 += BK) {
        // A tile [BM][BK]: thread -> row tid / 2, 16 consecutive k (vector loads in bounds).
        {
            const uint r = tid >> 1;
            const uint h = (tid & 1u) * 16u;
            const uint gr = row0 + r;
            threadgroup float* dst = As + r * BK + h;
// #if !MATMUL_ID
            const uint a_row = a_base + gr * a_rs;
// #endif
// #if MATMUL_ID
            const uint a_row = gr < m ? (pairs[gr] / P[13]) * k : 0u;
// #endif
            if (gr < m && k0 + BK <= k) {
                const uint src = a_row + k0 + h;
                for (uint t = 0; t < 4; t++) {
                    *(threadgroup float4*)(dst + 4 * t) = ld4(a, src + 4 * t);
                }
            } else {
                for (uint t = 0; t < 16; t++) {
                    const uint gk = k0 + h + t;
                    dst[t] = (gr < m && gk < k) ? ldv(a, a_row + gk) : 0.0f;
                }
            }
        }
        // B tile [BN][BK].
        {
// #if B_INT8
            // Row gc, 16 of the group's 32 int8 values (16-byte aligned), one f16 scale.
            const uint c = tid >> 1;
            const uint h = (tid & 1u) * 16u;
            const uint gc = col0 + c;
            threadgroup {{TB}}* dst = Bs + c * BK + h;
            if (gc < n) {
                const float d = float(((device const half*)b)[P[9] + gc * (k >> 5) + (k0 >> 5)]);
                device const char4* q = (device const char4*)(b + gc * k + k0 + h);
                for (uint t = 0; t < 4; t++) {
                    *(threadgroup {{TB}}4*)(dst + 4 * t) = {{TB}}4(float4(q[t]) * d);
                }
            } else {
                for (uint t = 0; t < 16; t++) {
                    dst[t] = {{TB}}(0.0f);
                }
            }
// #endif
// #if B_INT4
            // Row gc: the group's 16 bytes; low nibbles = elements 0..15, high = 16..31.
            const uint c = tid >> 1;
            const uint h = (tid & 1u) * 16u;
            const uint gc = col0 + c;
            threadgroup {{TB}}* dst = Bs + c * BK + h;
            if (gc < n) {
                const float d = float(((device const half*)b)[P[9] + gc * (k >> 5) + (k0 >> 5)]);
                device const uchar4* q = (device const uchar4*)(b + ((gc * k + k0) >> 1));
                const uint sh = h >> 2;
                for (uint t = 0; t < 4; t++) {
                    *(threadgroup {{TB}}4*)(dst + 4 * t) = {{TB}}4((float4((uint4(q[t]) >> sh) & 15u) - 8.0f) * d);
                }
            } else {
                for (uint t = 0; t < 16; t++) {
                    dst[t] = {{TB}}(0.0f);
                }
            }
// #endif
// #if B_KQ
            // Row gc: 16 values of super-block k0 / 256 (two 8-value halves).
            const uint c = tid >> 1;
            const uint h = (tid & 1u) * 16u;
            const uint gc = col0 + c;
            threadgroup {{TB}}* dst = Bs + c * BK + h;
            if (gc < n) {
                device const uchar* blk = b + (gc * (k >> 8) + (k0 >> 8)) * {{KQ_BYTES}}u;
                const uint j0 = (k0 & 255u) + h;
                float4 v0, v1, v2, v3;
                kq_deq8_{{KQ}}(blk, j0, v0, v1);
                kq_deq8_{{KQ}}(blk, j0 + 8u, v2, v3);
                *(threadgroup {{TB}}4*)(dst) = {{TB}}4(v0);
                *(threadgroup {{TB}}4*)(dst + 4) = {{TB}}4(v1);
                *(threadgroup {{TB}}4*)(dst + 8) = {{TB}}4(v2);
                *(threadgroup {{TB}}4*)(dst + 12) = {{TB}}4(v3);
            } else {
                for (uint t = 0; t < 16; t++) {
                    dst[t] = {{TB}}(0.0f);
                }
            }
// #endif
// #if B_TRANSPOSED
            // B[n][k] (k contiguous): thread -> row tid / 2, 16 consecutive k.
            const uint c = tid >> 1;
            const uint h = (tid & 1u) * 16u;
            const uint gc = col0 + c;
            if (gc < n && k0 + BK <= k) {
                const uint src = b_base + gc * b_rs + k0 + h;
                for (uint t = 0; t < 4; t++) {
                    *(threadgroup {{TB}}4*)(Bs + c * BK + h + 4 * t) = {{TB}}4(ld4(b, src + 4 * t));
                }
            } else {
                for (uint t = 0; t < 16; t++) {
                    const uint gk = k0 + h + t;
                    Bs[c * BK + h + t] = {{TB}}((gc < n && gk < k) ? ldv(b, b_base + gc * b_rs + gk) : 0.0f);
                }
            }
// #endif
// #if B_PLAIN
            // B[k][n] (n contiguous): each simdgroup reads 32 consecutive columns of one k.
            const uint c = tid & 63u;
            const uint gc = col0 + c;
            for (uint j = 0; j < 16; j++) {
                const uint kk = (tid >> 6) + 2 * j;
                const uint gk = k0 + kk;
                Bs[c * BK + kk] = {{TB}}((gc < n && gk < k) ? ldv(b, b_base + gk * b_rs + gc) : 0.0f);
            }
// #endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 ma[4];
            simdgroup_{{TB}}8x8 mb[4];
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(ma[i], As + (sr + 8 * i) * BK + kk, BK);
            }
            for (uint j = 0; j < 4; j++) {
                simdgroup_load(mb[j], Bs + (sc + 8 * j) * BK + kk, BK, ulong2(0, 0), true);
            }
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Stage C [BM][BN] and run the epilogue with one column per thread (coalesced stores).
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            simdgroup_store(acc[i][j], smem + (sr + 8 * i) * BN + sc + 8 * j, BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint c = tid & 63u;
    const uint gc = col0 + c;
    if (gc < n) {
        for (uint j = 0; j < 32; j++) {
            const uint r = (tid >> 6) + 2 * j;
            const uint gr = row0 + r;
            if (gr < m) {
                const float x0 = smem[r * BN + c];
// #if !MATMUL_ID
{{EXTRA_LOADS}}                stv(out, P[10] + bt * P[11] + gr * P[12] + gc, {{EPILOGUE}});
// #endif
// #if MATMUL_ID
                stv(out, pairs[gr] * n + gc, x0);
// #endif
            }
        }
    }
}
