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

// Flash attention, tile variant (many queries, e.g. the encoder): out = softmax(q k^T * scale
// + mask, causal) v without materializing the Sq x Sk matrix.
//
// Threadgroup = NSG simdgroups x (8 * RB) queries = BQ queries of one (batch, head). Keys are
// processed in tiles of BKV staged in threadgroup memory (f32). Per simdgroup and tile:
//   S[8RB x BKV] = Q K^T on 8x8 simdgroup matrices (Q pre-scaled, kept in registers; each
//                  K fragment is reused for the RB row blocks),
//   online softmax over S by lanes (LPR lanes per query row, LC columns per lane),
//   O = diag(alpha) O + P V with O[8RB x D] kept in registers.
// Compile-time constants: head dim D (multiple of 8, <= 128) so Q and O fragments live in
// registers; RB row blocks per simdgroup, NSG simdgroups and BKV = 32 for D <= 64 (else 16)
// come from the register / threadgroup memory budget (see codegen/attention.rs).
//
// P: [0] sq, [1] sk, [2] scale (f32 bits), [3] heads, then rank-4 views over [B, H, S, D]
//    (offset, s0, s1, s2, s3): [4] q, [9] k, [14] v, [19] mask (over [B, H, Sq, Sk]).
#define D 64
#define RB 1
#define BKV 32
#define ROWS (8 * RB)
#define NSG 8
#define NT (32 * NSG)
#define BQ (NSG * ROWS)
#define DC (D / 8)
#define KC (BKV / 8)
#define LPR (32 / ROWS)
#define LC (BKV / LPR)

kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* k [[buffer(2)]],
    device const float* v [[buffer(3)]],
    device float* out [[buffer(4)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    // K tile then V tile ([BKV][D] each); also stages Q ([BQ][D]) before and O after the loop.
    threadgroup float kv[2 * BKV * D];
    threadgroup float sbuf[NSG * ROWS * BKV];
    threadgroup float dbuf[NSG * RB * 64];

    const uint sq = P[0];
    const uint sk = P[1];
    const float scale = as_type<float>(P[2]);
    const uint b = tgp.z;
    const uint h = tgp.y;
    const uint q0 = tgp.x * BQ;
    const uint qb = P[4] + b * P[5] + h * P[6];
    const uint kb = P[9] + b * P[10] + h * P[11];
    const uint vb = P[14] + b * P[15] + h * P[16];
    // Head-dim contiguous views load 4 elements at a time.
    const bool qvec = P[8] == 1u && (D % 4) == 0;
    const bool kvvec = P[13] == 1u && P[18] == 1u && (D % 4) == 0;

    if (qvec) {
        for (uint e = tid * 4; e < BQ * D; e += NT * 4) {
            const uint r = e / D;
            const uint c = e - r * D;
            const uint qi = q0 + r;
            *(threadgroup float4*)(kv + e) = qi < sq ? ld4(q, qb + qi * P[7] + c) * scale : float4(0.0f);
        }
    } else {
        for (uint e = tid; e < BQ * D; e += NT) {
            const uint r = e / D;
            const uint c = e - r * D;
            const uint qi = q0 + r;
            kv[e] = qi < sq ? ldv(q, qb + qi * P[7] + c * P[8]) * scale : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 qm[RB][DC];
    simdgroup_float8x8 om[RB][DC];
    for (uint i = 0; i < RB; i++) {
        for (uint j = 0; j < DC; j++) {
            simdgroup_load(qm[i][j], kv + (sg * ROWS + 8 * i) * D + j * 8, D);
            om[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* sb = sbuf + sg * ROWS * BKV;
    threadgroup float* db = dbuf + sg * RB * 64;
    const uint rr = lane / LPR;
    const uint cq = (lane % LPR) * LC;
    const uint my_q = q0 + sg * ROWS + rr;
    float m_run = -INFINITY;
    float l_run = 0.0f;

    uint kend = sk;
    {
        // Keys after the last visible key of the tile's last query are skipped.
        const int qlast = int(min(q0 + BQ, sq)) - 1;
        kend = uint(clamp(qlast + int(sk) - int(sq) + 1, 0, int(sk)));
    }

    for (uint k0 = 0; k0 < kend; k0 += BKV) {
        if (kvvec && k0 + BKV <= sk) {
            for (uint e = tid * 4; e < BKV * D; e += NT * 4) {
                const uint r = e / D;
                const uint c = e - r * D;
                const uint kj = k0 + r;
                *(threadgroup float4*)(kv + e) = ld4(k, kb + kj * P[12] + c);
                *(threadgroup float4*)(kv + BKV * D + e) = ld4(v, vb + kj * P[17] + c);
            }
        } else {
            for (uint e = tid; e < BKV * D; e += NT) {
                const uint r = e / D;
                const uint c = e - r * D;
                const uint kj = k0 + r;
                float kx = 0.0f;
                float vx = 0.0f;
                if (kj < sk) {
                    kx = ldv(k, kb + kj * P[12] + c * P[13]);
                    vx = ldv(v, vb + kj * P[17] + c * P[18]);
                }
                kv[e] = kx;
                kv[BKV * D + e] = vx;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q K^T
        for (uint c = 0; c < KC; c++) {
            simdgroup_float8x8 s[RB];
            for (uint i = 0; i < RB; i++) {
                s[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
            for (uint j = 0; j < DC; j++) {
                simdgroup_float8x8 kt;
                simdgroup_load(kt, kv + c * 8 * D + j * 8, D, ulong2(0, 0), true);
                for (uint i = 0; i < RB; i++) {
                    simdgroup_multiply_accumulate(s[i], qm[i][j], kt, s[i]);
                }
            }
            for (uint i = 0; i < RB; i++) {
                simdgroup_store(s[i], sb + 8 * i * BKV + c * 8, BKV);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax: LPR lanes per query row.
        float sv[LC];
        float mx = -INFINITY;
        for (uint t = 0; t < LC; t++) {
            const uint kj = k0 + cq + t;
            float x = sb[rr * BKV + cq + t];
            bool vis = kj < sk;
            vis = vis && int(kj) <= int(my_q) + int(sk) - int(sq);
            x = vis ? x : -INFINITY;
            sv[t] = x;
            mx = max(mx, x);
        }
        for (uint o = 1; o < LPR; o <<= 1) {
            mx = max(mx, simd_shuffle_xor(mx, o));
        }
        const float m_new = max(m_run, mx);
        const bool dead = m_new == -INFINITY;
        const float alpha = dead ? 1.0f : exp(m_run - m_new);
        float ls = 0.0f;
        for (uint t = 0; t < LC; t++) {
            const float p = dead ? 0.0f : exp(sv[t] - m_new);
            sb[rr * BKV + cq + t] = p;
            ls += p;
        }
        for (uint o = 1; o < LPR; o <<= 1) {
            ls += simd_shuffle_xor(ls, o);
        }
        l_run = l_run * alpha + ls;
        m_run = m_new;
        if ((lane % LPR) == 0) {
            // Row rr lives in row block rr / 8, diagonal position rr % 8.
            threadgroup float* dg = db + (rr / 8) * 64 + (rr % 8) * 8;
            for (uint c = 0; c < 8; c++) {
                dg[c] = c == (rr % 8) ? alpha : 0.0f;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(alpha) O + P V
        for (uint i = 0; i < RB; i++) {
            simdgroup_float8x8 dm;
            simdgroup_load(dm, db + i * 64, 8);
            for (uint j = 0; j < DC; j++) {
                simdgroup_multiply(om[i][j], dm, om[i][j]);
            }
        }
        for (uint c = 0; c < KC; c++) {
            simdgroup_float8x8 pm[RB];
            for (uint i = 0; i < RB; i++) {
                simdgroup_load(pm[i], sb + 8 * i * BKV + c * 8, BKV);
            }
            for (uint j = 0; j < DC; j++) {
                simdgroup_float8x8 vm;
                simdgroup_load(vm, kv + BKV * D + c * 8 * D + j * 8, D);
                for (uint i = 0; i < RB; i++) {
                    simdgroup_multiply_accumulate(om[i][j], pm[i], vm, om[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // O / l, staged per simdgroup through kv (free after the loop's final barrier).
    threadgroup float* ob = kv + sg * ROWS * D;
    for (uint i = 0; i < RB; i++) {
        for (uint j = 0; j < DC; j++) {
            simdgroup_store(om[i][j], ob + 8 * i * D + j * 8, D);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = l_run > 0.0f ? 1.0f / l_run : 0.0f;
    if (my_q < sq) {
        const uint obase = ((b * P[3] + h) * sq + my_q) * D;
        for (uint c = lane % LPR; c < D; c += LPR) {
            stv(out, obase + c, ob[rr * D + c] * inv);
        }
    }
}
