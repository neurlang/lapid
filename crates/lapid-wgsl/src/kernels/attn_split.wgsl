// Attention for few queries (decoder), split over keys ("flash decoding").
// One workgroup per (batch, head, query, key chunk of CH keys):
//   scores: one subgroup per key (SG32), else TPK threads per key reduced in shared memory; chunk max / exp / sum (workgroup reductions); then the weighted V sum split over (key group, dim lane).
// DIRECT (one chunk covers all keys): normalize and write the output.
// Otherwise write the partial (m, l, o[D]) to scratch S; `attn_merge` combines the chunks.
// Head dim D is baked in (constant trip counts let the compiler unroll the dot loops).
// D > 256: each thread covers DC = ceil(D / 256) dims in the V phase (one key group).
// Params: [B, H, sq, sk, d, scale_bits, nsplit, q(off, s0..s3), k(..), v(..), mask(..)?]
{{BINDINGS}}
// #if !DIRECT
@group(0) @binding({{SCRATCH_BINDING}}) var<storage, read_write> S: array<f32>;
// #endif

const CH: u32 = {{CH}}u;
const D: u32 = {{D}}u;
// V-phase lane width (dims per pass) and passes per thread
const DW: u32 = min(D, 256u);
const DC: u32 = (D + DW - 1u) / DW;
// key groups in the V phase
const G: u32 = 256u / DW;
// threads per key in the score phase
const TPK: u32 = 256u / CH;
// lowest finite f32 ("no score yet"; real -inf mask values compare below it)
const NEG_INF: f32 = -3.4028235e38;

var<workgroup> qsh: array<f32, DC * 256u>;
var<workgroup> dots: array<f32, 256>;
var<workgroup> pp: array<f32, CH>;
var<workgroup> part: array<f32, DC * 256u>;

fn kernel(wg_lin: u32, wg_total: u32) {
    let H = P[1];
    let SQ = P[2];
    let SK = P[3];
    let scale = bitcast<f32>(P[5]);
    let NS = P[6];
    let items = P[0] * H * SQ * NS;
    let kl = g_lid / TPK;
    let quarter = g_lid % TPK;
    let dl = g_lid % DW;
    let grp = g_lid / DW;
    let lane_on = grp < G;
    for (var item = wg_lin; item < items; item += wg_total) {
        let split = item % NS;
        let bhq = item / NS;
        let qi = bhq % SQ;
        let bh = bhq / SQ;
        let h = bh % H;
        let b = bh / H;
        let qbase = P[7u] + b * P[8u] + h * P[9u] + qi * P[10u];
        let kbase = P[12u] + b * P[13u] + h * P[14u];
        let vbase = P[17u] + b * P[18u] + h * P[19u];
        let ks = P[15u];
        let kd = P[16u];
        let vs = P[20u];
        let vd = P[21u];
// #if MASK
        let mrow = P[22u] + b * P[23u] + h * P[24u] + qi * P[25u];
        let mstep = P[26u];
// #endif
// #if CAUSAL
        let kend = u32(clamp(i32(qi) + 1 + i32(SK) - i32(SQ), 0, i32(SK)));
// #endif
// #if !CAUSAL
        let kend = SK;
// #endif
        let k0 = split * CH;
        let kn = select(0u, min(CH, kend - k0), kend > k0);
        workgroupBarrier();
        for (var c = 0u; c < DC; c += 1u) {
            let qd = g_lid + c * 256u;
            if (qd < D) {
                qsh[qd] = ld_q(qbase + qd * P[11u]) * scale;
            }
        }
        workgroupBarrier();
// #if SG32
        // one 32-lane subgroup per key (8 subgroups): lane covers dims lane, lane + 32, ...
        // (coalesced 128-byte rows); all loads issued before the subgroup sums
        var xs: array<f32, CH / 8u>;
        for (var u = 0u; u < CH / 8u; u += 1u) {
            let kj = g_sgid + u * 8u;
            var x: f32 = 0.0;
            if (kj < kn) {
                let kr = kbase + (k0 + kj) * ks;
                for (var c = 0u; c < (D + 31u) / 32u; c += 1u) {
                    let dd = g_sgi + c * 32u;
                    if (dd < D) {
                        x = x + qsh[dd] * ld_k(kr + dd * kd);
                    }
                }
            }
            xs[u] = x;
        }
        for (var u = 0u; u < CH / 8u; u += 1u) {
            let tot = subgroupAdd(xs[u]);
            if (g_sgi == 0u) {
                dots[g_sgid + u * 8u] = tot;
            }
        }
        workgroupBarrier();
        // thread j < kn owns key k0 + j
        var sc = NEG_INF;
        if (g_lid < kn) {
            sc = dots[g_lid];
// #endif
// #if !SG32
        var x: f32 = 0.0;
        if (kl < kn) {
            let kr = kbase + (k0 + kl) * ks;
            for (var dd = quarter; dd < D; dd += TPK) {
                x = x + qsh[dd] * ld_k(kr + dd * kd);
            }
        }
        dots[g_lid] = x;
        workgroupBarrier();
        // thread j < kn owns key k0 + j
        var sc = NEG_INF;
        if (g_lid < kn) {
            sc = 0.0;
            for (var t = 0u; t < TPK; t += 1u) {
                sc = sc + dots[g_lid * TPK + t];
            }
// #endif
// #if MASK
            sc = sc + ld_msk(mrow + (k0 + g_lid) * mstep);
// #endif
        }
        let m = wg_max(sc);
        var p: f32 = 0.0;
        if (g_lid < kn && m > NEG_INF) {
            p = exp(sc - m);
        }
        if (g_lid < CH) {
            pp[g_lid] = p;
        }
        // (wg_sum's barriers also publish pp)
        let l = wg_sum(p);
        for (var c = 0u; c < DC; c += 1u) {
            let dd = dl + c * DW;
            var acc: f32 = 0.0;
            if (lane_on && dd < D) {
                let vcol = vbase + k0 * vs + dd * vd;
                // constant trip count (unrolled: all loads in flight at once); rounded up so
                // every key is covered when G does not divide CH (e.g. D = 80 -> G = 3)
                for (var u = 0u; u < (CH + G - 1u) / G; u += 1u) {
                    let t = grp + u * G;
                    if (t < kn) {
                        acc = acc + pp[t] * ld_v(vcol + t * vs);
                    }
                }
            }
            part[c * 256u + g_lid] = select(0.0, acc, lane_on);
        }
        workgroupBarrier();
        if (g_lid < DW) {
            for (var c = 0u; c < DC; c += 1u) {
                let dd = g_lid + c * DW;
                if (dd < D) {
                    var tot: f32 = 0.0;
                    for (var g = 0u; g < G; g += 1u) {
                        tot = tot + part[c * 256u + g * DW + g_lid];
                    }
// #if DIRECT
                    st(bhq * D + dd, select(0.0, tot / l, l > 0.0));
// #endif
// #if !DIRECT
                    let sb = item * (D + 2u);
                    S[sb + 2u + dd] = tot;
// #endif
                }
            }
// #if !DIRECT
            if (g_lid == 0u) {
                let sb0 = item * (D + 2u);
                S[sb0] = m;
                S[sb0 + 1u] = l;
            }
// #endif
        }
    }
}
