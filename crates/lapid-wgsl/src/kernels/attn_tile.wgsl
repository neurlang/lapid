// Flash attention without matrix cores (many queries, head dim D baked in, multiple of 4,
// <= 128): 64 queries per workgroup, one thread per query holding q and the output
// accumulator in vec4 registers; K/V tiles of BK keys staged through shared memory;
// per-key online softmax.
// Params: [B, H, sq, sk, d, scale_bits, q_blocks, q(off, s0..s3), k(..), v(..), mask(..)?]
{{BINDINGS}}
const D4: u32 = {{D4}}u;
const BK: u32 = {{BK}}u;

var<workgroup> Ks: array<vec4<f32>, {{KV_SIZE}}>;
var<workgroup> Vs: array<vec4<f32>, {{KV_SIZE}}>;

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
        let qstart = qb * 64u;
        let qi = qstart + g_lid;
        let valid = qi < SQ;
        let qbase = P[7u] + b * P[8u] + h * P[9u];
        let kbase = P[12u] + b * P[13u] + h * P[14u];
        let vbase = P[17u] + b * P[18u] + h * P[19u];
        let qrow = qbase + qi * P[10u];
        let qs = P[11u];
        var q: array<vec4<f32>, D4>;
        var o: array<vec4<f32>, D4>;
        for (var i = 0u; i < D4; i += 1u) {
            q[i] = vec4<f32>(0.0);
            o[i] = vec4<f32>(0.0);
            if (valid) {
                let e = qrow + i * 4u * qs;
                q[i] = vec4<f32>(ld_q(e), ld_q(e + qs), ld_q(e + 2u * qs), ld_q(e + 3u * qs)) * scale;
            }
        }
// #if MASK
        let mrow = P[22u] + b * P[23u] + h * P[24u] + qi * P[25u];
        let mstep = P[26u];
// #endif
// #if CAUSAL
        let last_q = min(qstart + 63u, SQ - 1u);
        let kend = u32(clamp(i32(last_q) + 1 + i32(SK) - i32(SQ), 0, i32(SK)));
// #endif
// #if !CAUSAL
        let kend = SK;
// #endif
        let ks = P[15u];
        let kd = P[16u];
        let vs = P[20u];
        let vd = P[21u];
        var m: f32 = -3.0e38;
        var l: f32 = 0.0;
        for (var k0 = 0u; k0 < kend; k0 += BK) {
            workgroupBarrier();
            for (var t = g_lid; t < BK * D4; t += 64u) {
                let kj = t / D4;
                let dd = (t % D4) * 4u;
                let key = k0 + kj;
                var kv = vec4<f32>(0.0);
                var vv = vec4<f32>(0.0);
                if (key < kend) {
                    let kr = kbase + key * ks + dd * kd;
                    let vr = vbase + key * vs + dd * vd;
                    kv = vec4<f32>(ld_k(kr), ld_k(kr + kd), ld_k(kr + 2u * kd), ld_k(kr + 3u * kd));
                    vv = vec4<f32>(ld_v(vr), ld_v(vr + vd), ld_v(vr + 2u * vd), ld_v(vr + 3u * vd));
                }
                Ks[t] = kv;
                Vs[t] = vv;
            }
            workgroupBarrier();
            if (valid) {
                for (var kj = 0u; kj < BK; kj += 1u) {
                    let key = k0 + kj;
                    let kb = kj * D4;
                    var x: f32 = 0.0;
                    for (var i = 0u; i < D4; i += 1u) {
                        x = x + dot(q[i], Ks[kb + i]);
                    }
// #if MASK
                    if (key < kend) {
                        x = x + ld_msk(mrow + key * mstep);
                    }
// #endif
// #if CAUSAL
                    let ok = key < kend && i32(key) <= i32(qi) + i32(SK) - i32(SQ);
// #endif
// #if !CAUSAL
                    let ok = key < kend;
// #endif
                    x = select(bitcast<f32>(0xff800000u), x, ok);
                    if (x > m) {
                        let c = exp(m - x);
                        l = l * c;
                        for (var i = 0u; i < D4; i += 1u) {
                            o[i] = o[i] * c;
                        }
                        m = x;
                    }
                    let p = exp(x - m);
                    l = l + p;
                    for (var i = 0u; i < D4; i += 1u) {
                        o[i] = fma(vec4<f32>(p), Vs[kb + i], o[i]);
                    }
                }
            }
        }
        if (valid) {
            let inv = select(0.0, 1.0 / l, l > 0.0);
            let ob = ((b * H + h) * SQ + qi) * D4 * 4u;
            for (var i = 0u; i < D4; i += 1u) {
                st(ob + i * 4u, o[i].x * inv);
                st(ob + i * 4u + 1u, o[i].y * inv);
                st(ob + i * 4u + 2u, o[i].z * inv);
                st(ob + i * 4u + 3u, o[i].w * inv);
            }
        }
    }
}
