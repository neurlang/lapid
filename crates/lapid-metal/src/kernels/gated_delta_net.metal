// Gated delta rule (ggml GATED_DELTA_NET semantics), sequential over t per (b, h):
//   S = exp(g_t) * S;  S += k_t (x) (beta_t * (v_t - S^T k_t));  o_t = S^T (q_t * scale)
// q, k [B, T, H, dk], v [B, T, H, dv], g, beta [B, T, H], state [B, H, dk, dv] (f32,
// contiguous); out [B, H, T + dk, dv]: rows < T are o_t, then the final state.
//
// One simdgroup per state column j (threadgroup = 4 columns, grid (dv / 4, H, B)): lane l
// keeps S[l + 32 c][j] for c < DKC (= ceil(dk / 32), compile-time) in registers; the two
// dot products over dk are simd_sums.
// P: [0] T, [1] H, [2] dk, [3] dv, [4] scale (f32 bits).
#define DKC {{DKC}}

kernel void lapid_kernel(
    constant uint* P [[buffer(0)]],
{{BINDINGS}}    uint3 tgp [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint T = P[0];
    const uint H = P[1];
    const uint dk = P[2];
    const uint dv = P[3];
    const float scale = as_type<float>(P[4]);
    const uint j = tgp.x * 4 + sg;
    if (j >= dv) {
        return;
    }
    const uint bh = tgp.z * H + tgp.y;
    float s[DKC];
    for (uint c = 0; c < DKC; c++) {
        const uint i = lane + 32 * c;
        s[c] = i < dk ? state[(bh * dk + i) * dv + j] : 0.0f;
    }
    const uint obase = bh * (T + dk);
    for (uint t = 0; t < T; t++) {
        const uint row = (tgp.z * T + t) * H + tgp.y;
        const float decay = precise::exp(g[row]);
        const float bv = beta[row];
        float kv[DKC];
        float sk = 0.0f;
        for (uint c = 0; c < DKC; c++) {
            const uint i = lane + 32 * c;
            kv[c] = i < dk ? k[row * dk + i] : 0.0f;
            s[c] *= decay;
            sk = fma(s[c], kv[c], sk);
        }
        sk = simd_sum(sk);
        const float d = (v[row * dv + j] - sk) * bv;
        float y = 0.0f;
        for (uint c = 0; c < DKC; c++) {
            const uint i = lane + 32 * c;
            s[c] = fma(kv[c], d, s[c]);
            y = fma(s[c], i < dk ? q[row * dk + i] : 0.0f, y);
        }
        y = simd_sum(y);
        if (lane == 0) {
            stv(out, (obase + t) * dv + j, y * scale);
        }
    }
    for (uint c = 0; c < DKC; c++) {
        const uint i = lane + 32 * c;
        if (i < dk) {
            stv(out, (obase + T + i) * dv + j, s[c]);
        }
    }
}
