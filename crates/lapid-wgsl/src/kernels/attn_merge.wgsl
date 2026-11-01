// Combine the per-chunk partials of `attn_split`: one thread per output element.
// out[bhq, d] = sum_s e^(m_s - M) o_s[d] / sum_s e^(m_s - M) l_s,  M = max_s m_s.
// Params: [rows = B * H * sq, d, nsplit]
{{BINDINGS}}
@group(0) @binding({{SCRATCH_BINDING}}) var<storage, read_write> S: array<f32>;

// lowest finite f32 ("no score yet"; real -inf mask values compare below it)
const NEG_INF: f32 = -3.4028235e38;

fn kernel(wg_lin: u32, wg_total: u32) {
    let rows = P[0];
    let D = P[1];
    let NS = P[2];
    let total = rows * D;
    let stride = D + 2u;
    for (var e = wg_lin * 256u + g_lid; e < total; e += wg_total * 256u) {
        let r = e / D;
        let d = e % D;
        let base = r * NS * stride;
        var M = NEG_INF;
        for (var s = 0u; s < NS; s += 1u) {
            M = max(M, S[base + s * stride]);
        }
        var l: f32 = 0.0;
        var o: f32 = 0.0;
        if (M > NEG_INF) {
            for (var s = 0u; s < NS; s += 1u) {
                let sb = base + s * stride;
                let w = exp(S[sb] - M);
                l = l + w * S[sb + 1u];
                o = o + w * S[sb + 2u + d];
            }
        }
        st(e, select(0.0, o / l, l > 0.0));
    }
}
