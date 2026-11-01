// Attention, vector variant (few queries, e.g. decoding): one threadgroup (128 threads) per
// (batch * head, query, key split of KS = 128 keys). Keys are scored by groups of LK lanes
// (K rows read coalesced, partial dots combined with shuffles) against the pre-scaled query
// in threadgroup memory; max / sum over the split use simd + threadgroup reductions; P V is
// split over (key group, head-dim column) so V rows are read coalesced.
// DIRECT (a single split): the normalized output is written directly. Otherwise each split
// writes (m, l, o[D]) to scratch and `attn_merge` combines the splits.
//
// D (head dim), LK (lanes per key: 8 when D % 32 == 0, else 1) and G (key groups of the
// P V reduction) are compile-time constants; WIDE (D / KS, only when D % KS == 0) selects
// the multi-column P V loop for large head dims (256, 512).
// P: [0] sq, [1] sk, [2] scale (f32 bits), [3] heads, then rank-4 views over [B, H, S, D]
//    (offset, s0, s1, s2, s3): [4] q, [9] k, [14] v, [19] mask (over [B, H, Sq, Sk]);
//    [24] splits.
#define D {{D}}
#define G {{G}}
#define LK {{LK}}
#define WIDE {{WIDE}}
#define DL (D / LK)
#define KPS (32 / LK)
#define KS 128

kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
{{BINDINGS}}    uint3 tgp [[threadgroup_position_in_grid]],
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
// #if MASK
    const uint mb = P[19] + b * P[20] + h * P[21] + qi * P[22];
// #endif

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
// #if CAUSAL
        vis = vis && int(kj) <= int(qi) + int(sk) - int(sq);
// #endif
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
// #if MASK
            if (vis) {
                dot += ldv(mask, mb + kj * P[23]);
            }
// #endif
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
// #if WIDE
    // D a multiple of KS (G = 1): WIDE = D / KS independent columns per thread, keys outer
    // (independent accumulator chains instead of one long dependent chain per column).
    {
        float o[WIDE];
        for (uint i = 0; i < WIDE; i++) {
            o[i] = 0.0f;
        }
        for (uint j = 0; j < nk; j++) {
            const float pj = ps[j];
            const uint vr = vb + (k0 + j) * P[17];
            for (uint i = 0; i < WIDE; i++) {
                o[i] = fma(pj, ldv(v, vr + (tid + i * KS) * P[18]), o[i]);
            }
        }
        for (uint i = 0; i < WIDE; i++) {
            part[tid + i * KS] = o[i];
        }
    }
// #endif
// #if !WIDE
    for (uint e = tid; e < G * D; e += KS) {
        const uint g = e / D;
        const uint c = e - g * D;
        float o = 0.0f;
        for (uint j = g; j < nk; j += G) {
            o = fma(ps[j], ldv(v, vb + (k0 + j) * P[17] + c * P[18]), o);
        }
        part[e] = o;
    }
// #endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
// #if DIRECT
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    const uint obase = (bh * sq + qi) * D;
// #endif
// #if !DIRECT
    const uint obase = ((bh * sq + qi) * P[24] + split) * (D + 2);
    if (tid == 0) {
        scratch[obase] = m;
        scratch[obase + 1] = l;
    }
// #endif
    for (uint c = tid; c < D; c += KS) {
        float o = 0.0f;
        for (uint g = 0; g < G; g++) {
            o += part[g * D + c];
        }
// #if DIRECT
        stv(out, obase + c, o * inv);
// #endif
// #if !DIRECT
        scratch[obase + 2 + c] = o;
// #endif
    }
}
