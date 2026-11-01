// Tiled GEMM on cooperative matrices (16x16x16, f16 inputs, f32 accumulators).
// out[bt, i, j] = epilogue(sum_k A[bt, i, k] * B[bt, k, j]).
// Workgroup tile BM x BN, 8 subgroups of 32 lanes, each owning a 32x32 block
// (2x2 accumulators). A and B tiles are converted to f16 (B dequantized) into shared
// memory per BK-step. The next step's global loads are issued into registers before the
// current step's matrix multiplies (register prefetch), so load latency overlaps math.
// The epilogue stages each accumulator through shared memory.
// Params: [batch, m, n, k, a.off, a.bs, a.rs, b.off, b.bs, b.rs, scales_words, tiles_n,
//          tiles_m, out(off, bs, rs), extra_e(off, s_batch, s_m, s_n)...]
{{BINDINGS}}
const BM: u32 = {{BM}}u;
const BN: u32 = {{BN}}u;
const BK: u32 = {{BK}}u;
// shared row pitch in halves (BK + 8: keeps rows 16-byte aligned, staggers banks)
const LDS: u32 = BK + 8u;
// A elements per thread per step
const A_PER: u32 = BM * BK / 256u;

var<private> ra: array<f32, A_PER>;
{{B_REGS}}
// Load the A / B tiles of k-step `k0` into registers.
fn load_tiles(row0: u32, col0: u32, a_base: u32, b_base: u32, k0: u32, M: u32, N: u32, K: u32) {
    for (var t = 0u; t < A_PER; t += 1u) {
        let e = g_lid + t * 256u;
        let gi = row0 + e / BK;
        let gk = k0 + e % BK;
        var v: f32 = 0.0;
        if (gi < M && gk < K) {
            v = ld_a(a_base + gi * P[6] + gk);
        }
        ra[t] = v;
    }
{{B_LOAD}}}

// Store the register tiles to shared memory as f16 (A row-major, B column-major).
fn store_tiles() {
    for (var t = 0u; t < A_PER; t += 1u) {
        let e = g_lid + t * 256u;
        As[(e / BK) * LDS + e % BK] = f16(ra[t]);
    }
{{B_STORE}}}

var<workgroup> As: array<f16, {{AS_SIZE}}>;
var<workgroup> Bs: array<f16, {{BS_SIZE}}>;
var<workgroup> Cs: array<f32, 2048>;

fn kernel(wg_lin: u32, wg_total: u32) {
    let batch = P[0];
    let M = P[1];
    let N = P[2];
    let K = P[3];
    let tiles_n = P[11];
    let tiles_m = P[12];
    let items = batch * tiles_m * tiles_n;
    let wm = g_sgid / (BN / 32u);
    let wn = g_sgid % (BN / 32u);
    for (var item = wg_lin; item < items; item += wg_total) {
        let tn = item % tiles_n;
        let rest = item / tiles_n;
        let tm = rest % tiles_m;
        let bt = rest / tiles_m;
        let row0 = tm * BM;
        let col0 = tn * BN;
        let a_base = P[4] + bt * P[5];
        let b_base = P[7] + bt * P[8];
        var c00 = coop_mat16x16<f32, C>();
        var c01 = coop_mat16x16<f32, C>();
        var c10 = coop_mat16x16<f32, C>();
        var c11 = coop_mat16x16<f32, C>();
        load_tiles(row0, col0, a_base, b_base, 0u, M, N, K);
        for (var k0 = 0u; k0 < K; k0 += BK) {
            workgroupBarrier();
            store_tiles();
            workgroupBarrier();
            if (k0 + BK < K) {
                load_tiles(row0, col0, a_base, b_base, k0 + BK, M, N, K);
            }
            for (var ks = 0u; ks < BK; ks += 16u) {
                let a0 = coopLoadT<coop_mat16x16<f16, A>>(&As[(wm * 32u) * LDS + ks], LDS);
                let a1 = coopLoadT<coop_mat16x16<f16, A>>(&As[(wm * 32u + 16u) * LDS + ks], LDS);
                let b0 = coopLoad<coop_mat16x16<f16, B>>(&Bs[(wn * 32u) * LDS + ks], LDS);
                let b1 = coopLoad<coop_mat16x16<f16, B>>(&Bs[(wn * 32u + 16u) * LDS + ks], LDS);
                c00 = coopMultiplyAdd(a0, b0, c00);
                c01 = coopMultiplyAdd(a0, b1, c01);
                c10 = coopMultiplyAdd(a1, b0, c10);
                c11 = coopMultiplyAdd(a1, b1, c11);
            }
        }
        // epilogue: stage each 16x16 accumulator in this subgroup's slice of Cs
        let cs = g_sgid * 256u;
        for (var t = 0u; t < 4u; t += 1u) {
            workgroupBarrier();
            if (t == 0u) {
                coopStoreT(c00, &Cs[cs], 16u);
            } else if (t == 1u) {
                coopStoreT(c01, &Cs[cs], 16u);
            } else if (t == 2u) {
                coopStoreT(c10, &Cs[cs], 16u);
            } else {
                coopStoreT(c11, &Cs[cs], 16u);
            }
            workgroupBarrier();
            let ib = row0 + wm * 32u + (t / 2u) * 16u;
            let jb = col0 + wn * 32u + (t % 2u) * 16u;
            for (var e = g_sgi; e < 256u; e += g_sgs) {
                let i = ib + e / 16u;
                let j = jb + e % 16u;
                if (i < M && j < N) {
                    let acc = Cs[cs + e];
{{EPILOGUE}}                }
            }
        }
    }
}
