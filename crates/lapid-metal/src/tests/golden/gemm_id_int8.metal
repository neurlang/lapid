// lapid Metal prelude: typed loads / stores, quantized reads, math helpers and threadgroup
// reductions shared by every kernel. All arithmetic is f32; f16 / bf16 / i32 are storage.
//
// Buffer element types: f32 = float, f16 = half, bf16 = ushort (raw bits), i32 = int,
// int8 / int4 = uchar (lapid-quant SoA bytes). Element indices are uint; signed strides are
// stored as two's complement, so `offset + coord * stride` wraps to the right index.
#include <metal_stdlib>
using namespace metal;

static inline float ldv(device const float* p, uint e) { return p[e]; }
static inline float ldv(device const half* p, uint e) { return float(p[e]); }
static inline float ldv(device const ushort* p, uint e) { return as_type<float>(uint(p[e]) << 16); }
static inline float ldv(device const int* p, uint e) { return float(p[e]); }

// Four consecutive elements (any alignment of the element type: packed vector loads).
static inline float4 ld4(device const float* p, uint e) { return float4(*(device const packed_float4*)(p + e)); }
static inline float4 ld4(device const half* p, uint e) { return float4(*(device const packed_half4*)(p + e)); }
static inline float4 ld4(device const ushort* p, uint e) {
    return float4(ldv(p, e), ldv(p, e + 1), ldv(p, e + 2), ldv(p, e + 3));
}
static inline float4 ld4(device const int* p, uint e) { return float4(*(device const packed_int4*)(p + e)); }

static inline void stv(device float* p, uint e, float v) { p[e] = v; }
static inline void stv(device half* p, uint e, float v) { p[e] = half(v); }
static inline void stv(device ushort* p, uint e, float v) {
    // Round to nearest even; NaN stays a quiet NaN.
    const uint bits = as_type<uint>(v);
    p[e] = isnan(v) ? ushort(0x7fc0) : ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
// f32 -> i32 / u32: truncation toward zero, saturating, NaN -> 0 (Rust `as`).
static inline uint f2i_(float v) {
    const int r = isnan(v) ? 0 : (v >= 2147483648.0f ? 2147483647 : (v <= -2147483648.0f ? int(0x80000000u) : int(v)));
    return as_type<uint>(r);
}
static inline uint f2u_(float v) {
    return (isnan(v) || v <= 0.0f) ? 0u : (v >= 4294967296.0f ? 0xffffffffu : uint(v));
}
static inline void stv(device int* p, uint e, float v) { p[e] = as_type<int>(f2i_(v)); }

// Raw 32-bit words (integer expressions): f32 / i32 storage bit patterns; 16-bit storage
// (not produced by the compiler) reads the word of the f32 value.
static inline uint ldw(device const float* p, uint e) { return as_type<uint>(p[e]); }
static inline uint ldw(device const int* p, uint e) { return as_type<uint>(p[e]); }
static inline uint ldw(device const half* p, uint e) { return as_type<uint>(float(p[e])); }
static inline uint ldw(device const ushort* p, uint e) { return as_type<uint>(ldv(p, e)); }
// Store a raw word (i32 output with an integer expression root).
static inline void stw(device int* p, uint e, uint w) { p[e] = as_type<int>(w); }

// Integer division / remainder that never trap: x / 0 and INT_MIN / -1 give 0 (as the CPU VM).
static inline uint divs_(uint a, uint b) {
    const int x = as_type<int>(a);
    const int y = as_type<int>(b);
    const bool bad = y == 0 || (x == int(0x80000000u) && y == -1);
    return bad ? 0u : as_type<uint>(x / (bad ? 1 : y));
}
static inline uint rems_(uint a, uint b) {
    const int x = as_type<int>(a);
    const int y = as_type<int>(b);
    const bool bad = y == 0 || (x == int(0x80000000u) && y == -1);
    return bad ? 0u : as_type<uint>(x % (bad ? 1 : y));
}
static inline uint divu_(uint a, uint b) { return b == 0u ? 0u : a / max(b, 1u); }
static inline uint remu_(uint a, uint b) { return b == 0u ? 0u : a % max(b, 1u); }
// Wrapping |x| of a signed word.
static inline uint iabs_(uint a) { return as_type<int>(a) < 0 ? 0u - a : a; }

// lapid-quant SoA matrix [rows, cols] (cols % 32 == 0): element e = r * cols + c, group e / 32,
// `so` = offset of the f16 scales plane in halves.
static inline float ld_int8(device const uchar* q, uint so, uint e) {
    const float d = float(((device const half*)q)[so + (e >> 5)]);
    return float(as_type<char>(q[e])) * d;
}
static inline float ld_int4(device const uchar* q, uint so, uint e) {
    const uint g = e >> 5;
    const uint j = e & 31u;
    const uint b = q[g * 16u + (j & 15u)];
    const uint nib = j < 16u ? (b & 15u) : (b >> 4);
    return (float(nib) - 8.0f) * float(((device const half*)q)[so + g]);
}

// erf: 6-term Taylor series for |x| < 0.5 (abs error < 2e-8), Numerical Recipes erfc
// Chebyshev fit (relative error < 1.2e-7) elsewhere.
static inline float erf_(float x) {
    const float a = fabs(x);
    if (a < 0.5f) {
        const float x2 = x * x;
        return x * (1.1283791671f + x2 * (-0.3761263890f + x2 * (0.1128379167f + x2 * (-0.0268661706f
            + x2 * (0.0052239776f + x2 * -0.0008548327f)))));
    }
    const float t = 1.0f / (1.0f + 0.5f * a);
    const float r = t * precise::exp(-a * a - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f
        + t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f
        + t * (-0.82215223f + t * 0.17087277f)))))))));
    const float y = 1.0f - r;
    return x < 0.0f ? -y : y;
}

static inline float tanh_(float x) {
    return precise::tanh(clamp(x, -15.0f, 15.0f));
}

// pow with IEEE-like handling of negative bases (integer exponents only).
static inline float pow_(float a, float b) {
    if (b == 0.0f) {
        return 1.0f;
    }
    const float r = precise::pow(fabs(a), b);
    if (a < 0.0f) {
        if (floor(b) != b) {
            return as_type<float>(0x7fc00000u);
        }
        const float h = b * 0.5f;
        if (floor(h) != h) {
            return -r;
        }
    }
    return r;
}

// Threadgroup-wide sum / max. `red` holds >= 32 floats; every thread gets the result.
// Must be called in uniform control flow by all threads of the threadgroup.
static inline float tg_sum(float v, threadgroup float* red, uint lane, uint sg, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) {
        red[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = lane < nsg ? red[lane] : 0.0f;
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return v;
}

static inline float tg_max(float v, threadgroup float* red, uint lane, uint sg, uint nsg) {
    v = simd_max(v);
    if (lane == 0) {
        red[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = lane < nsg ? red[lane] : -INFINITY;
    v = simd_max(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return v;
}

// Order-preserving map of an f32 to a uint (larger value -> larger key; -0 == +0).
static inline uint ord_key(float v) {
    // -0 -> +0 on the bits (a float compare would be folded away under relaxed math).
    uint b = as_type<uint>(v);
    b = (b & 0x7fffffffu) == 0u ? 0u : b;
    return (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u);
}

// Top-k sort key: value order in the high word, lower index wins ties (~index).
static inline ulong topk_key(float v, uint c) {
    return (ulong(ord_key(v)) << 32) | ulong(~c);
}

// Simdgroup argmax of (key, index) pairs: larger key, then lower index. Every lane gets the result.
static inline void argmax_simd(thread uint& bk, thread uint& bi) {
    for (ushort off = 16; off > 0; off >>= 1) {
        const uint ok = simd_shuffle_xor(bk, off);
        const uint oi = simd_shuffle_xor(bi, off);
        if (ok > bk || (ok == bk && oi < bi)) {
            bk = ok;
            bi = oi;
        }
    }
}

// Tiled GEMM on simdgroup matrices: out[bt, i, j] = epilogue(sum_k A[bt, i, k] * B[bt, k, j]).
//
// Threadgroup = 4 simdgroups (128 threads) computing a BM x BN = 64 x 64 output tile; each
// simdgroup owns a 32 x 32 quadrant (4 x 4 accumulators of 8 x 8, f32). K advances in
// BK = 32 steps (= one quantization group): the A tile is staged in threadgroup memory as
// f32, the B tile as half ([BN][BK], dequantized on the fly for int8 / int4 weights),
// then multiplied with simdgroup_multiply_accumulate. After the K loop the accumulators are
// staged through the same threadgroup memory so the fused epilogue runs per element with
// coalesced stores.
//
// MATMUL_ID (MoE): threadgroup z = expert e, its row tiles strided by the grid's y extent;
// its rows are the (t, r) pairs routed to it by the route stage (`route` = scratch: counts[E], then E lists of `cap` pair indices). Row i
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
    device const float* a [[buffer(1)]],
    device const uchar* w [[buffer(2)]],
    device float* out [[buffer(3)]],
    device const uint* route [[buffer(4)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint3 ntg [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    // 16 KB: A tile (BM*BK f32) + B tile (BN*BK half); reused for the C tile (BM*BN f32).
    threadgroup float smem[BM * BN];
    threadgroup float* As = smem;
    threadgroup half* Bs = (threadgroup half*)(smem + BM * BK);

    const uint n = P[1];
    const uint k = P[2];
    const uint bt = tgp.z;
    const uint col0 = tgp.x * BN;
    const uint m = route[bt];
    device const uint* pairs = route + P[14] + bt * P[16];
    device const uchar* b = w + bt * P[15];
    const uint b_base = 0;
    const uint b_rs = P[8];
    // Row tiles of this expert, strided by the grid's y extent (sized for the average count).
    for (uint row0 = tgp.y * BM; row0 < m; row0 += ntg.y * BM) {

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    const uint sr = (sg >> 1) * 32;
    const uint sc = (sg & 1) * 32;
    // 8-row blocks of this simdgroup's quadrant that hold routed rows (experts often get far
    // fewer than BM rows: the empty blocks skip their multiplies; uniform per simdgroup).
    const uint mrows = min(m - row0, uint(BM));
    const uint nbi = mrows > sr ? min(4u, (mrows - sr + 7u) / 8u) : 0u;

    for (uint k0 = 0; k0 < k; k0 += BK) {
        // A tile [BM][BK]: thread -> row tid / 2, 16 consecutive k (vector loads in bounds).
        {
            const uint r = tid >> 1;
            const uint h = (tid & 1u) * 16u;
            const uint gr = row0 + r;
            threadgroup float* dst = As + r * BK + h;
            const uint a_row = gr < m ? (pairs[gr] / P[13]) * k : 0u;
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
            // Row gc, 16 of the group's 32 int8 values (16-byte aligned), one f16 scale.
            const uint c = tid >> 1;
            const uint h = (tid & 1u) * 16u;
            const uint gc = col0 + c;
            threadgroup half* dst = Bs + c * BK + h;
            if (gc < n) {
                const float d = float(((device const half*)b)[P[9] + gc * (k >> 5) + (k0 >> 5)]);
                device const char4* q = (device const char4*)(b + gc * k + k0 + h);
                for (uint t = 0; t < 4; t++) {
                    *(threadgroup half4*)(dst + 4 * t) = half4(float4(q[t]) * d);
                }
            } else {
                for (uint t = 0; t < 16; t++) {
                    dst[t] = half(0.0f);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK && nbi > 0; kk += 8) {
            simdgroup_float8x8 ma[4];
            simdgroup_half8x8 mb[4];
            for (uint i = 0; i < 4; i++) {
                if (i < nbi) {
                    simdgroup_load(ma[i], As + (sr + 8 * i) * BK + kk, BK);
                }
            }
            for (uint j = 0; j < 4; j++) {
                simdgroup_load(mb[j], Bs + (sc + 8 * j) * BK + kk, BK, ulong2(0, 0), true);
            }
            for (uint i = 0; i < 4; i++) {
                if (i < nbi) {
                    for (uint j = 0; j < 4; j++) {
                        simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
                    }
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
                stv(out, pairs[gr] * n + gc, x0);
            }
        }
    }
    // smem (C tile) is reused by the next row tile's A / B tiles.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
