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
