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

// Attention, vector variant (few queries, e.g. decoding): one threadgroup (128 threads) per
// (batch * head, query, key split of KS = 128 keys). Keys are scored by groups of LK lanes
// (K rows read coalesced, partial dots combined with shuffles) against the pre-scaled query
// in threadgroup memory; max / sum over the split use simd + threadgroup reductions; P V is
// split over (key group, head-dim column) so V rows are read coalesced.
// DIRECT (a single split): the normalized output is written directly. Otherwise each split
// writes (m, l, o[D]) to scratch and `attn_merge` combines the splits.
//
// D (head dim), LK (lanes per key: 8 when D % 32 == 0, else 1) and G (key groups of the
// P V reduction) are compile-time constants.
// P: [0] sq, [1] sk, [2] scale (f32 bits), [3] heads, then rank-4 views over [B, H, S, D]
//    (offset, s0, s1, s2, s3): [4] q, [9] k, [14] v, [19] mask (over [B, H, Sq, Sk]);
//    [24] splits.
#define D 64
#define G 2
#define LK 8
#define DL (D / LK)
#define KPS (32 / LK)
#define KS 128

kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* k [[buffer(2)]],
    device const float* v [[buffer(3)]],
    device const float* mask [[buffer(4)]],
    device float* scratch [[buffer(5)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float qs[D];
    threadgroup float ps[KS];
    threadgroup float red[32];
    threadgroup float part[G * D];

    const uint sq = P[0];
    const uint sk = P[1];
    const float scale = as_type<float>(P[2]);
    const uint heads = P[3];
    const uint split = tgp.x;
    const uint qi = tgp.y;
    const uint bh = tgp.z;
    const uint b = bh / heads;
    const uint h = bh - b * heads;
    const uint qb = P[4] + b * P[5] + h * P[6] + qi * P[7];
    const uint kb = P[9] + b * P[10] + h * P[11];
    const uint vb = P[14] + b * P[15] + h * P[16];
    const uint mb = P[19] + b * P[20] + h * P[21] + qi * P[22];

    for (uint c = tid; c < D; c += KS) {
        qs[c] = ldv(q, qb + c * P[8]) * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scores: LK lanes per key (DL = D / LK dims each, vector loads when the head dim is
    // contiguous), KPS keys per simdgroup step; then one probability per thread.
    const uint k0 = split * KS;
    const uint sub = lane % LK;
    const bool kvec = P[13] == 1u && (DL % 4) == 0;
    for (uint j = sg * KPS + lane / LK; j < KS; j += 4 * KPS) {
        const uint kj = k0 + j;
        bool vis = kj < sk;
        float dot = 0.0f;
        if (vis) {
            const uint kr = kb + kj * P[12] + sub * DL * P[13];
            const uint qo = sub * DL;
            if (kvec) {
                for (uint t = 0; t < DL; t += 4) {
                    dot += metal::dot(ld4(k, kr + t), float4(qs[qo + t], qs[qo + t + 1], qs[qo + t + 2], qs[qo + t + 3]));
                }
            } else {
                for (uint t = 0; t < DL; t++) {
                    dot = fma(qs[qo + t], ldv(k, kr + t * P[13]), dot);
                }
            }
        }
        for (uint o = 1; o < LK; o <<= 1) {
            dot += simd_shuffle_xor(dot, o);
        }
        if (sub == 0) {
            if (vis) {
                dot += ldv(mask, mb + kj * P[23]);
            }
            ps[j] = vis ? dot : -INFINITY;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float s = ps[tid];
    const float m = tg_max(s, red, lane, sg, 4);
    const float p = m == -INFINITY ? 0.0f : exp(s - m);
    ps[tid] = p;
    const float l = tg_sum(p, red, lane, sg, 4);

    const uint nk = min(uint(KS), sk - min(sk, k0));
    for (uint e = tid; e < G * D; e += KS) {
        const uint g = e / D;
        const uint c = e - g * D;
        float o = 0.0f;
        for (uint j = g; j < nk; j += G) {
            o = fma(ps[j], ldv(v, vb + (k0 + j) * P[17] + c * P[18]), o);
        }
        part[e] = o;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint obase = ((bh * sq + qi) * P[24] + split) * (D + 2);
    if (tid == 0) {
        scratch[obase] = m;
        scratch[obase + 1] = l;
    }
    for (uint c = tid; c < D; c += KS) {
        float o = 0.0f;
        for (uint g = 0; g < G; g++) {
            o += part[g * D + c];
        }
        scratch[obase + 2 + c] = o;
    }
}
