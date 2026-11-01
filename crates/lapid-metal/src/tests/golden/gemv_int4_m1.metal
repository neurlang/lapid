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
#define MR 1
#define NR 2

kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device const uchar* b [[buffer(2)]],
    device const float* e0 [[buffer(3)]],
    device float* out [[buffer(4)]],
    uint3 tgp [[threadgroup_position_in_grid]],
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
    const uint a_base = P[3] + bt * P[4];
    const uint a_rs = P[5];
    const uint b_base = P[6] + bt * P[7];
    const uint o_base = P[10] + bt * P[11];
    const uint b_rs = P[8];

    float acc[MR][NR];
    for (uint i = 0; i < MR; i++) {
        for (uint r = 0; r < NR; r++) {
            acc[i][r] = 0.0f;
        }
    }

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

    for (uint i = 0; i < MR; i++) {
        for (uint r = 0; r < NR; r++) {
            const float x0 = simd_sum(acc[i][r]);
            const uint gc = col0 + r;
            if (lane == 0 && gc < n) {
                const uint gr = i;
                const float x1 = ldv(e0, P[13] + bt * P[14] + gr * P[15] + gc * P[16]);
                stv(out, o_base + gr * P[12] + gc, (((x0 * 0.5f) * (1.0f + erf_((x0 * 0.70710677f)))) + x1));
            }
        }
    }
}
