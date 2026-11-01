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

// ggml K-quant super-blocks (AoS, exactly as stored in GGUF; 256 elements per block):
//   Q4_K (144 B): half d, half dmin, uchar scales[12] (8 x 6-bit scale + 6-bit min), uchar qs[128]
//   Q5_K (176 B): half d, half dmin, uchar scales[12], uchar qh[32], uchar qs[128]
//   Q6_K (210 B): uchar ql[128], uchar qh[64], char scales[16], half d
// Q4/Q5: element j of sub-block sb = j / 32 is d * sc[sb] * q - dmin * m[sb]; q is the low
// (even sb) or high (odd sb) nibble of qs[(sb / 2) * 32 + j % 32] (+ 16 * bit sb of qh[j % 32]).
// Q6: half n = j / 128, quadrant qd = (j % 128) / 32, l = j % 32:
//   q = (ql[64n + l + 32 (qd & 1)] >> 4 (qd >> 1)) & 15 | ((qh[32n + l] >> 2 qd) & 3) << 4,
//   x = d * scales[8n + l / 16 + 2 qd] * (q - 32).
// Q6_K blocks are only 2-byte aligned: byte / packed loads only. Q4_K / Q5_K blocks are
// 16-byte aligned (144 = 9 * 16, 176 = 11 * 16) relative to the matrix start.

// 6-bit (scale, min) of sub-block j from the 12 packed bytes at s.
static inline uint2 kq_scale_min(device const uchar* s, uint j) {
    if (j < 4u) {
        return uint2(s[j] & 63u, s[j + 4u] & 63u);
    }
    return uint2((s[j + 4u] & 15u) | ((s[j - 4u] >> 6) << 4), (s[j + 4u] >> 4) | ((s[j] >> 6) << 4));
}

static inline uint4 kq_u4(uint w) {
    return uint4(as_type<uchar4>(w));
}

// Raw 4-bit values of elements j0..j0+8 (j0 % 8 == 0) of a Q4_K / Q5_K block (qs at `qs`),
// plus the 5th bit for Q5_K (qh at `qh`, null for Q4_K).
static inline void kq_q45(device const uchar* qs, device const uchar* qh, uint j0, thread uint4& lo, thread uint4& hi) {
    const uint sb = j0 >> 5;
    const uint2 w = *(device const uint2*)(qs + (sb >> 1) * 32u + (j0 & 31u));
    const uint sh = (sb & 1u) * 4u;
    lo = (kq_u4(w.x) >> sh) & 15u;
    hi = (kq_u4(w.y) >> sh) & 15u;
    if (qh) {
        const uint2 h = *(device const uint2*)(qh + (j0 & 31u));
        lo |= ((kq_u4(h.x) >> sb) & 1u) << 4;
        hi |= ((kq_u4(h.y) >> sb) & 1u) << 4;
    }
}

// Q6_K values (q - 32) of elements j0..j0+8 and their (shared) scale d * scales[..].
static inline float kq_q6(device const uchar* blk, uint j0, thread float4& lo, thread float4& hi) {
    const uint n = j0 >> 7;
    const uint qd = (j0 >> 5) & 3u;
    const uint l = j0 & 31u;
    device const uchar* ql = blk + 64u * n + l + (qd & 1u) * 32u;
    device const uchar* qh = blk + 128u + 32u * n + l;
    const uint sl = (qd >> 1) * 4u;
    const uint sh = 2u * qd;
    const uint4 l0 = uint4(*(device const packed_uchar4*)ql);
    const uint4 l1 = uint4(*(device const packed_uchar4*)(ql + 4));
    const uint4 h0 = uint4(*(device const packed_uchar4*)qh);
    const uint4 h1 = uint4(*(device const packed_uchar4*)(qh + 4));
    lo = float4(((l0 >> sl) & 15u) | (((h0 >> sh) & 3u) << 4)) - 32.0f;
    hi = float4(((l1 >> sl) & 15u) | (((h1 >> sh) & 3u) << 4)) - 32.0f;
    const float d = float(*(device const half*)(blk + 208));
    return d * float(as_type<char>(blk[192u + 8u * n + (l >> 4) + 2u * qd]));
}

// GEMV lane split (`gemv_kq.metal`): 8 lanes per block, lane `it` (0..7) owns 32 values
// read as 8 float4 runs a[0..8]; kq_aoff_* is the element offset of run j inside the block.
// Q4/Q5: 64-group g = it / 2, half hh = it % 2: runs 0..3 = low nibbles of qs[32g + 16hh ..
// + 16] (sub-block 2g), runs 4..7 = their high nibbles (sub-block 2g + 1).
// Q6: half n = it / 4, l0 = 8 (it % 4): run j = quadrant j / 2, elements l0 + 4 (j % 2) ..
static inline uint kq_aoff_q45(uint it, uint j) {
    return (it >> 1) * 64u + (j >> 2) * 32u + (it & 1u) * 16u + (j & 3u) * 4u;
}
static inline uint kq_aoff_q6(uint it, uint j) {
    return (it >> 2) * 128u + (j >> 1) * 32u + (it & 3u) * 8u + (j & 1u) * 4u;
}
static inline uint kq_aoff_q4k(uint it, uint j) { return kq_aoff_q45(it, j); }
static inline uint kq_aoff_q5k(uint it, uint j) { return kq_aoff_q45(it, j); }
static inline uint kq_aoff_q6k(uint it, uint j) { return kq_aoff_q6(it, j); }

// A prepared for the lane functions: every float4 run is scaled by 2^-8j per component j, so
// a byte j of a 32-bit word w is weighted by (w & (0xFF << 8j)) * 2^-8j = byte value without
// extracting it; `sums` holds the run sums the zero points need (Q4/Q5: x = sub-block 2g,
// y = 2g + 1; Q6: one per quadrant).
constant float4 KQ_BYTE_SCALE = float4(1.0f, 1.0f / 256.0f, 1.0f / 65536.0f, 1.0f / 16777216.0f);
constant uint4 KQ_BYTE_MASK = uint4(0xFFu, 0xFF00u, 0xFF0000u, 0xFF000000u);

static inline float kq_bytes_dot(uint w, float4 a) {
    return dot(float4(uint4(w) & KQ_BYTE_MASK), a);
}

static inline void kq_prep_q45(thread float4* a, thread float4& sums) {
    float4 lo = a[0] + a[1] + a[2] + a[3];
    float4 hi = a[4] + a[5] + a[6] + a[7];
    sums = float4(dot(lo, float4(1.0f)), dot(hi, float4(1.0f)), 0.0f, 0.0f);
    for (uint j = 0; j < 8; j++) {
        a[j] *= KQ_BYTE_SCALE;
    }
}
static inline void kq_prep_q4k(thread float4* a, thread float4& sums) { kq_prep_q45(a, sums); }
static inline void kq_prep_q5k(thread float4* a, thread float4& sums) { kq_prep_q45(a, sums); }
static inline void kq_prep_q6k(thread float4* a, thread float4& sums) {
    for (uint q = 0; q < 4; q++) {
        sums[q] = dot(a[2 * q] + a[2 * q + 1], float4(1.0f));
    }
    for (uint j = 0; j < 8; j++) {
        a[j] *= KQ_BYTE_SCALE;
    }
}

static inline float kq_lane_q45(device const uchar* blk, uint it, thread const float4* a, float4 sums, bool q5) {
    const uint g = it >> 1;
    const uint hh = it & 1u;
    const float2 dm = float2(*(device const half2*)blk);
    const uint2 s0 = kq_scale_min(blk + 4, 2u * g);
    const uint2 s1 = kq_scale_min(blk + 4, 2u * g + 1u);
    const uint4 w = *(device const uint4*)(blk + (q5 ? 48u : 16u) + g * 32u + hh * 16u);
    uint4 hb = uint4(0u);
    if (q5) {
        hb = *(device const uint4*)(blk + 16u + hh * 16u);
    }
    float lo = 0.0f;
    float hi = 0.0f;
    for (uint t = 0; t < 4; t++) {
        uint wl = w[t] & 0x0F0F0F0Fu;
        uint wh = (w[t] >> 4) & 0x0F0F0F0Fu;
        if (q5) {
            wl |= ((hb[t] >> (2u * g)) & 0x01010101u) << 4;
            wh |= ((hb[t] >> (2u * g + 1u)) & 0x01010101u) << 4;
        }
        lo += kq_bytes_dot(wl, a[t]);
        hi += kq_bytes_dot(wh, a[4 + t]);
    }
    return dm.x * (float(s0.x) * lo + float(s1.x) * hi) - dm.y * (float(s0.y) * sums.x + float(s1.y) * sums.y);
}
static inline float kq_lane_q4k(device const uchar* blk, uint it, thread const float4* a, float4 sums) {
    return kq_lane_q45(blk, it, a, sums, false);
}
static inline float kq_lane_q5k(device const uchar* blk, uint it, thread const float4* a, float4 sums) {
    return kq_lane_q45(blk, it, a, sums, true);
}
static inline uint kq_word2(device const uchar* p) {
    // 2-byte aligned 32-bit load (Q6_K blocks are 210 bytes).
    return as_type<uint>(ushort2(*(device const packed_ushort2*)p));
}
static inline float kq_lane_q6k(device const uchar* blk, uint it, thread const float4* a, float4 sums) {
    const uint n = it >> 2;
    const uint l0 = (it & 3u) * 8u;
    device const uchar* ql = blk + 64u * n + l0;
    device const uchar* qh = blk + 128u + 32u * n + l0;
    float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint t = 0; t < 2; t++) {
        const uint l0w = kq_word2(ql + 4u * t);
        const uint l1w = kq_word2(ql + 32u + 4u * t);
        const uint h = kq_word2(qh + 4u * t);
        s[0] += kq_bytes_dot((l0w & 0x0F0F0F0Fu) | ((h & 0x03030303u) << 4), a[t]);
        s[1] += kq_bytes_dot((l1w & 0x0F0F0F0Fu) | (((h >> 2) & 0x03030303u) << 4), a[2 + t]);
        s[2] += kq_bytes_dot(((l0w >> 4) & 0x0F0F0F0Fu) | (((h >> 4) & 0x03030303u) << 4), a[4 + t]);
        s[3] += kq_bytes_dot(((l1w >> 4) & 0x0F0F0F0Fu) | (((h >> 6) & 0x03030303u) << 4), a[6 + t]);
    }
    device const uchar* sc = blk + 192u + 8u * n + (l0 >> 4);
    const float d = float(*(device const half*)(blk + 208));
    return d * (float(as_type<char>(sc[0])) * (s[0] - 32.0f * sums[0]) + float(as_type<char>(sc[2])) * (s[1] - 32.0f * sums[1])
        + float(as_type<char>(sc[4])) * (s[2] - 32.0f * sums[2]) + float(as_type<char>(sc[6])) * (s[3] - 32.0f * sums[3]));
}

// Dequantize elements j0..j0+8 of a block.
static inline void kq_deq8_q4k(device const uchar* blk, uint j0, thread float4& v0, thread float4& v1) {
    const float2 dm = float2(*(device const half2*)blk);
    const uint2 sm = kq_scale_min(blk + 4, j0 >> 5);
    uint4 lo, hi;
    kq_q45(blk + 16, nullptr, j0, lo, hi);
    const float d1 = dm.x * float(sm.x);
    const float m1 = dm.y * float(sm.y);
    v0 = float4(lo) * d1 - m1;
    v1 = float4(hi) * d1 - m1;
}
static inline void kq_deq8_q5k(device const uchar* blk, uint j0, thread float4& v0, thread float4& v1) {
    const float2 dm = float2(*(device const half2*)blk);
    const uint2 sm = kq_scale_min(blk + 4, j0 >> 5);
    uint4 lo, hi;
    kq_q45(blk + 48, blk + 16, j0, lo, hi);
    const float d1 = dm.x * float(sm.x);
    const float m1 = dm.y * float(sm.y);
    v0 = float4(lo) * d1 - m1;
    v1 = float4(hi) * d1 - m1;
}
static inline void kq_deq8_q6k(device const uchar* blk, uint j0, thread float4& v0, thread float4& v1) {
    float4 lo, hi;
    const float d = kq_q6(blk, j0, lo, hi);
    v0 = lo * d;
    v1 = hi * d;
}

// Element e of a [rows, cols] matrix (cols % 256 == 0): block e / 256, element e % 256.
static inline float ld_q4k(device const uchar* q, uint e) {
    device const uchar* blk = q + (e >> 8) * 144u;
    const uint j = e & 255u;
    const uint sb = j >> 5;
    const float2 dm = float2(*(device const half2*)blk);
    const uint2 sm = kq_scale_min(blk + 4, sb);
    const uint v = (blk[16u + (sb >> 1) * 32u + (j & 31u)] >> ((sb & 1u) * 4u)) & 15u;
    return dm.x * float(sm.x) * float(v) - dm.y * float(sm.y);
}
static inline float ld_q5k(device const uchar* q, uint e) {
    device const uchar* blk = q + (e >> 8) * 176u;
    const uint j = e & 255u;
    const uint sb = j >> 5;
    const float2 dm = float2(*(device const half2*)blk);
    const uint2 sm = kq_scale_min(blk + 4, sb);
    const uint v = ((blk[48u + (sb >> 1) * 32u + (j & 31u)] >> ((sb & 1u) * 4u)) & 15u) | (((blk[16u + (j & 31u)] >> sb) & 1u) << 4);
    return dm.x * float(sm.x) * float(v) - dm.y * float(sm.y);
}
static inline float ld_q6k(device const uchar* q, uint e) {
    device const uchar* blk = q + (e >> 8) * 210u;
    const uint j = e & 255u;
    const uint n = j >> 7;
    const uint qd = (j >> 5) & 3u;
    const uint l = j & 31u;
    const uint lo = (blk[64u * n + l + (qd & 1u) * 32u] >> ((qd >> 1) * 4u)) & 15u;
    const uint hi = (blk[128u + 32u * n + l] >> (2u * qd)) & 3u;
    const float sc = float(as_type<char>(blk[192u + 8u * n + (l >> 4) + 2u * qd]));
    return float(*(device const half*)(blk + 208)) * sc * (float(lo | (hi << 4)) - 32.0f);
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
    device const uchar* b [[buffer(2)]],
    device float* out [[buffer(3)]],
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
    const uint m = P[0];
    const uint a_base = P[3] + bt * P[4];
    const uint a_rs = P[5];
    const uint b_base = P[6] + bt * P[7];
    const uint b_rs = P[8];
    {
    const uint row0 = tgp.y * BM;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    const uint sr = (sg >> 1) * 32;
    const uint sc = (sg & 1) * 32;
    // Compile-time 4: full tiles (the dynamic skip costs ~10 % on dense GEMMs).
    const uint nbi = 4u;

    for (uint k0 = 0; k0 < k; k0 += BK) {
        // A tile [BM][BK]: thread -> row tid / 2, 16 consecutive k (vector loads in bounds).
        {
            const uint r = tid >> 1;
            const uint h = (tid & 1u) * 16u;
            const uint gr = row0 + r;
            threadgroup float* dst = As + r * BK + h;
            const uint a_row = a_base + gr * a_rs;
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
            // Row gc: 16 values of super-block k0 / 256 (two 8-value halves).
            const uint c = tid >> 1;
            const uint h = (tid & 1u) * 16u;
            const uint gc = col0 + c;
            threadgroup half* dst = Bs + c * BK + h;
            if (gc < n) {
                device const uchar* blk = b + (gc * (k >> 8) + (k0 >> 8)) * 210u;
                const uint j0 = (k0 & 255u) + h;
                float4 v0, v1, v2, v3;
                kq_deq8_q6k(blk, j0, v0, v1);
                kq_deq8_q6k(blk, j0 + 8u, v2, v3);
                *(threadgroup half4*)(dst) = half4(v0);
                *(threadgroup half4*)(dst + 4) = half4(v1);
                *(threadgroup half4*)(dst + 8) = half4(v2);
                *(threadgroup half4*)(dst + 12) = half4(v3);
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
                stv(out, P[10] + bt * P[11] + gr * P[12] + gc, x0);
            }
        }
    }
    }
}
