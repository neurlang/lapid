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
#define D {{D}}
#define RB {{RB}}
#define BKV {{BKV}}
#define ROWS (8 * RB)
#define NSG {{NSG}}
#define NT (32 * NSG)
#define BQ (NSG * ROWS)
#define DC (D / 8)
#define KC (BKV / 8)
#define LPR (32 / ROWS)
#define LC (BKV / LPR)

kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
{{BINDINGS}}    uint3 tgp [[threadgroup_position_in_grid]],
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
// #if MASK
    const uint mb = P[19] + b * P[20] + h * P[21];
// #endif

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
// #if CAUSAL
    {
        // Keys after the last visible key of the tile's last query are skipped.
        const int qlast = int(min(q0 + BQ, sq)) - 1;
        kend = uint(clamp(qlast + int(sk) - int(sq) + 1, 0, int(sk)));
    }
// #endif

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
// #if CAUSAL
            vis = vis && int(kj) <= int(my_q) + int(sk) - int(sq);
// #endif
// #if MASK
            if (vis && my_q < sq) {
                x += ldv(mask, mb + my_q * P[22] + kj * P[23]);
            }
// #endif
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
