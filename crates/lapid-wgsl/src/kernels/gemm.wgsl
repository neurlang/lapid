// Tiled GEMM without matrix cores: 64x64 output tile per 256-thread workgroup, 4x4 per
// thread, k-step 32 (one quant group, dequantized into shared memory), f32 throughout.
// out[bt, i, j] = epilogue(sum_k A[bt, i, k] * B[bt, k, j]).
// Params: [batch, m, n, k, a.off, a.bs, a.rs, b.off, b.bs, b.rs, scales_words, tiles_n,
//          tiles_m, out(off, bs, rs), extra_e(off, s_batch, s_m, s_n)...]
{{BINDINGS}}
var<workgroup> As: array<f32, 2048>;
var<workgroup> Bs: array<f32, 2048>;

fn kernel(wg_lin: u32, wg_total: u32) {
    let batch = P[0];
    let M = P[1];
    let N = P[2];
    let K = P[3];
    let tiles_n = P[11];
    let tiles_m = P[12];
    let items = batch * tiles_m * tiles_n;
    let tx = g_lid % 16u;
    let ty = g_lid / 16u;
    let lr = g_lid / 4u;
    let lk = (g_lid % 4u) * 8u;
    for (var item = wg_lin; item < items; item += wg_total) {
        let tn = item % tiles_n;
        let rest = item / tiles_n;
        let tm = rest % tiles_m;
        let bt = rest / tiles_m;
        let row0 = tm * 64u;
        let col0 = tn * 64u;
        let a_base = P[4] + bt * P[5];
        let b_base = P[7] + bt * P[8];
        var acc0 = vec4<f32>(0.0);
        var acc1 = vec4<f32>(0.0);
        var acc2 = vec4<f32>(0.0);
        var acc3 = vec4<f32>(0.0);
        for (var k0 = 0u; k0 < K; k0 += 32u) {
            workgroupBarrier();
            {
                let gi = row0 + lr;
                let arow = a_base + gi * P[6];
                for (var e = 0u; e < 8u; e += 1u) {
                    let kk = k0 + lk + e;
                    var v: f32 = 0.0;
                    if (gi < M && kk < K) {
                        v = ld_a(arow + kk);
                    }
                    As[(lk + e) * 64u + lr] = v;
                }
            }
            // B tile into Bs[k * 64 + j]
{{B_TILE}}            workgroupBarrier();
            for (var kk = 0u; kk < 32u; kk += 1u) {
                let ao = kk * 64u + ty * 4u;
                let bo = kk * 64u + tx * 4u;
                let av = vec4<f32>(As[ao], As[ao + 1u], As[ao + 2u], As[ao + 3u]);
                let bv = vec4<f32>(Bs[bo], Bs[bo + 1u], Bs[bo + 2u], Bs[bo + 3u]);
                acc0 = fma(vec4<f32>(av.x), bv, acc0);
                acc1 = fma(vec4<f32>(av.y), bv, acc1);
                acc2 = fma(vec4<f32>(av.z), bv, acc2);
                acc3 = fma(vec4<f32>(av.w), bv, acc3);
            }
        }
        for (var r = 0u; r < 4u; r += 1u) {
            let i = row0 + ty * 4u + r;
            var accr = acc0;
            if (r == 1u) {
                accr = acc1;
            } else if (r == 2u) {
                accr = acc2;
            } else if (r == 3u) {
                accr = acc3;
            }
            if (i < M) {
                for (var c = 0u; c < 4u; c += 1u) {
                    let j = col0 + tx * 4u + c;
                    if (j < N) {
                        let acc = accr[c];
{{EPILOGUE}}                    }
                }
            }
        }
    }
}
