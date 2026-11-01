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

// Elementwise: out[out_view(idx)] = expr(inputs[i][in_view_i(idx)]) over `shape`.
// P: [0] numel, [1..=rank] shape, then per input and the output: offset, strides[rank].
// Grid-stride loop over the flat index; coordinates are decomposed only when some view is
// strided (contiguous views index with offset + i, broadcast scalars with offset).
kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
    device const int* in0 [[buffer(1)]],
    device const float* in1 [[buffer(2)]],
    device int* out [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint gsize [[threads_per_grid]])
{
    const uint n = P[0];
    for (uint i = gid; i < n; i += gsize) {
        uint rem = i;
        const uint c1 = rem % P[2];
        rem = rem / P[2];
        const uint c0 = rem;
        const uint w0 = ldw(in0, P[3] + c0 * P[4] + c1 * P[5]);
        const float x1 = ldv(in1, P[6] + i);
        const uint r = select((as_type<uint>((float(as_type<int>(w0)) + (float(w0) + as_type<float>(0x3f800000u)))) - f2u_(x1)), (((((((((((((((((((((f2i_(x1) ^ (w0 + w0)) ^ (w0 - 0x4u)) ^ (w0 * w0)) ^ divs_(w0, 0xau)) ^ divu_(w0, w0)) ^ rems_(w0, 0x10u)) ^ remu_(w0, w0)) ^ as_type<uint>(min(as_type<int>(w0), as_type<int>(0x16u)))) ^ min(w0, w0)) ^ as_type<uint>(max(as_type<int>(w0), as_type<int>(0x1cu)))) ^ max(w0, w0)) ^ (w0 & 0x22u)) ^ (w0 | w0)) ^ (w0 ^ 0x28u)) ^ (w0 << (w0 & 31u))) ^ as_type<uint>(as_type<int>(w0) >> (0x2eu & 31u))) ^ (w0 >> (w0 & 31u))) ^ uint(as_type<int>(w0) < as_type<int>(0x34u))) ^ uint(w0 < w0)) ^ uint(w0 == 0x3au)) + (0u - iabs_((~w0)))), (w0 & 0x1u) != 0u);
        stw(out, P[9] + i, r);
    }
}
