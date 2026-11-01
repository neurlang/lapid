// Attention for many queries with wide heads (D = 256 / 512) on cooperative matrices
// (16x16x16 f16, f32 accumulate). One workgroup (8 subgroups) per (batch, head, block of
// 16 queries); keys in tiles of 16, K and V staged through the same shared buffer.
// Two passes over the keys (no rescaling of the output accumulators):
//   pass 1: S = Q K^T per tile, each subgroup a D/8 slice of the head dim (partials summed
//           in shared memory) -> per-row running max / sum;
//   pass 2: S again, P = exp(S - m) / l (f16), O += P V with each subgroup owning D/8
//           output columns (NT 16x16 accumulators).
// Q is pre-scaled; masks / causal limits are applied in the scalar phase.
// Params: [B, H, sq, sk, d, scale_bits, q_blocks, q(off, s0..s3), k(..), v(..), mask(..)?]
{{BINDINGS}}
const BQ: u32 = 16u;
const BKV: u32 = 16u;
const D: u32 = {{D}}u;
const LDQ: u32 = D + 8u;   // f16 pitch of Qs / KVs rows
const DS: u32 = D / 8u;    // head-dim slice per subgroup
const NEG_INF: f32 = -3.4028235e38;

var<workgroup> Qs: array<f16, BQ * LDQ>;
var<workgroup> KVs: array<f16, BKV * LDQ>;    // [key][d]: K, then V of the tile
var<workgroup> Sp: array<f32, 2048>;           // 8 partial 16x16 score tiles; output staging
var<workgroup> Ss: array<f32, 256>;            // scores of the tile (masked)
var<workgroup> Ps: array<f16, 256>;            // probabilities (16 x 16)
var<workgroup> rowm: array<f32, 16>;
var<workgroup> rowl: array<f32, 16>;

// Load key (is_v = false) or value tile k0 into KVs (keys >= kend read as zero).
fn load_tile(base: u32, k0: u32, kend: u32, is_v: bool) {
    let st_ = select(P[15u], P[20u], is_v);
    let sd = select(P[16u], P[21u], is_v);
    for (var e = g_lid; e < BKV * D; e += 256u) {
        let key = e / D;
        let dd = e % D;
        var x: f32 = 0.0;
        if (k0 + key < kend) {
            if (is_v) {
                x = ld_v(base + (k0 + key) * st_ + dd * sd);
            } else {
                x = ld_k(base + (k0 + key) * st_ + dd * sd);
            }
        }
        KVs[key * LDQ + dd] = f16(x);
    }
}

// Ss = masked Q K^T of the tile in KVs (all threads; barriers inside).
fn scores(q0: u32, k0: u32, kend: u32, mrow0: u32, SQ: u32, SK: u32) {
    var acc = coop_mat16x16<f32, C>();
    let d0 = g_sgid * DS;
    for (var dk = 0u; dk < DS; dk += 16u) {
        let a = coopLoadT<coop_mat16x16<f16, A>>(&Qs[d0 + dk], LDQ);
        let b = coopLoad<coop_mat16x16<f16, B>>(&KVs[d0 + dk], LDQ);
        acc = coopMultiplyAdd(a, b, acc);
    }
    coopStoreT(acc, &Sp[g_sgid * 256u], 16u);
    workgroupBarrier();
    let row = g_lid / 16u;
    let col = g_lid % 16u;
    var s: f32 = 0.0;
    for (var x = 0u; x < 8u; x += 1u) {
        s = s + Sp[x * 256u + g_lid];
    }
    let key = k0 + col;
    if (key >= kend) {
        s = NEG_INF;
    }
// #if CAUSAL
    if (i32(key) > i32(q0 + row) + i32(SK) - i32(SQ)) {
        s = NEG_INF;
    }
// #endif
// #if MASK
    if (s > NEG_INF && q0 + row < SQ) {
        s = s + ld_msk(mrow0 + (q0 + row) * P[25u] + key * P[26u]);
    }
// #endif
    Ss[g_lid] = s;
    workgroupBarrier();
}

fn kernel(wg_lin: u32, wg_total: u32) {
    let H = P[1];
    let SQ = P[2];
    let SK = P[3];
    let scale = bitcast<f32>(P[5]);
    let nqb = P[6];
    let items = P[0] * H * nqb;
    for (var item = wg_lin; item < items; item += wg_total) {
        let qb = item % nqb;
        let bh = item / nqb;
        let h = bh % H;
        let b = bh / H;
        let q0 = qb * BQ;
        let qbase = P[7u] + b * P[8u] + h * P[9u];
        let kbase = P[12u] + b * P[13u] + h * P[14u];
        let vbase = P[17u] + b * P[18u] + h * P[19u];
// #if MASK
        let mrow0 = P[22u] + b * P[23u] + h * P[24u];
// #endif
// #if !MASK
        let mrow0 = 0u;
// #endif
// #if CAUSAL
        let last_q = min(q0 + BQ - 1u, SQ - 1u);
        let kend = u32(clamp(i32(last_q) + 1 + i32(SK) - i32(SQ), 0, i32(SK)));
// #endif
// #if !CAUSAL
        let kend = SK;
// #endif
        workgroupBarrier();
        for (var e = g_lid; e < BQ * D; e += 256u) {
            let r = e / D;
            let dd = e % D;
            var v: f32 = 0.0;
            if (q0 + r < SQ) {
                v = ld_q(qbase + (q0 + r) * P[10u] + dd * P[11u]) * scale;
            }
            Qs[r * LDQ + dd] = f16(v);
        }
        // pass 1: row max / sum (thread r < 16 owns row r)
        var pm = NEG_INF;
        var pl: f32 = 0.0;
        for (var k0 = 0u; k0 < kend; k0 += BKV) {
            workgroupBarrier();
            load_tile(kbase, k0, kend, false);
            workgroupBarrier();
            scores(q0, k0, kend, mrow0, SQ, SK);
            if (g_lid < 16u) {
                var tm = NEG_INF;
                for (var c = 0u; c < 16u; c += 1u) {
                    tm = max(tm, Ss[g_lid * 16u + c]);
                }
                let nm = max(pm, tm);
                if (nm > NEG_INF) {
                    var ts: f32 = 0.0;
                    for (var c = 0u; c < 16u; c += 1u) {
                        ts = ts + exp(Ss[g_lid * 16u + c] - nm);
                    }
                    pl = pl * exp(pm - nm) + ts;
                    pm = nm;
                }
            }
        }
        if (g_lid < 16u) {
            rowm[g_lid] = pm;
            rowl[g_lid] = select(0.0, 1.0 / pl, pl > 0.0);
        }
        // pass 2: O = P V
        var o0 = coop_mat16x16<f32, C>();
// #if NT2
        var o1 = coop_mat16x16<f32, C>();
// #endif
// #if NT4
        var o1 = coop_mat16x16<f32, C>();
        var o2 = coop_mat16x16<f32, C>();
        var o3 = coop_mat16x16<f32, C>();
// #endif
        let d0 = g_sgid * DS;
        for (var k0 = 0u; k0 < kend; k0 += BKV) {
            workgroupBarrier();
            load_tile(kbase, k0, kend, false);
            workgroupBarrier();
            scores(q0, k0, kend, mrow0, SQ, SK);
            let r = g_lid / 16u;
            let m = rowm[r];
            var p: f32 = 0.0;
            if (m > NEG_INF) {
                p = exp(Ss[g_lid] - m) * rowl[r];
            }
            Ps[g_lid] = f16(p);
            load_tile(vbase, k0, kend, true);
            workgroupBarrier();
            let a = coopLoadT<coop_mat16x16<f16, A>>(&Ps[0], 16u);
            o0 = coopMultiplyAdd(a, coopLoadT<coop_mat16x16<f16, B>>(&KVs[d0], LDQ), o0);
// #if NT2
            o1 = coopMultiplyAdd(a, coopLoadT<coop_mat16x16<f16, B>>(&KVs[d0 + 16u], LDQ), o1);
// #endif
// #if NT4
            o1 = coopMultiplyAdd(a, coopLoadT<coop_mat16x16<f16, B>>(&KVs[d0 + 16u], LDQ), o1);
            o2 = coopMultiplyAdd(a, coopLoadT<coop_mat16x16<f16, B>>(&KVs[d0 + 32u], LDQ), o2);
            o3 = coopMultiplyAdd(a, coopLoadT<coop_mat16x16<f16, B>>(&KVs[d0 + 48u], LDQ), o3);
// #endif
        }
        // output: stage each 16x16 tile in this subgroup's 256-float slice of Sp
        let ob = ((b * H + h) * SQ) * D;
        for (var t = 0u; t < DS / 16u; t += 1u) {
            workgroupBarrier();
            if (t == 0u) {
                coopStoreT(o0, &Sp[g_sgid * 256u], 16u);
            }
// #if NT2
            if (t == 1u) {
                coopStoreT(o1, &Sp[g_sgid * 256u], 16u);
            }
// #endif
// #if NT4
            if (t == 1u) {
                coopStoreT(o1, &Sp[g_sgid * 256u], 16u);
            } else if (t == 2u) {
                coopStoreT(o2, &Sp[g_sgid * 256u], 16u);
            } else if (t == 3u) {
                coopStoreT(o3, &Sp[g_sgid * 256u], 16u);
            }
// #endif
            workgroupBarrier();
            for (var e = g_sgi; e < 256u; e += g_sgs) {
                let qi = q0 + e / 16u;
                let dd = d0 + t * 16u + e % 16u;
                if (qi < SQ) {
                    st(ob + qi * D + dd, Sp[g_sgid * 256u + e]);
                }
            }
        }
    }
}
