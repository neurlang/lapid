// lapid prelude: extensions, the params binding, builtin mirrors and math helpers.
enable f16;
enable wgpu_cooperative_matrix;

// Launch params (sizes, strides, offsets) as u32 words; layout documented per kernel.
@group(0) @binding(0) var<storage, read> P: array<u32>;

var<private> g_lid: u32;
var<private> g_sgi: u32;
var<private> g_sgid: u32;
var<private> g_nsg: u32;
var<private> g_sgs: u32;

// erf: Taylor series near 0, Abramowitz & Stegun 7.1.26 elsewhere.
fn erf_(x: f32) -> f32 {
    let a = abs(x);
    if (a < 0.5) {
        let x2 = x * x;
        return x * (1.1283791671 + x2 * (-0.3761263890 + x2 * (0.1128379167 + x2 * (-0.0268661706 + x2 * 0.0052239776))));
    }
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * exp(-a * a);
    return select(y, -y, x < 0.0);
}

fn tanh_(x: f32) -> f32 {
    return tanh(clamp(x, -15.0, 15.0));
}

// pow with negative bases for integer exponents.
fn pow_(a: f32, b: f32) -> f32 {
    if (b == 0.0) {
        return 1.0;
    }
    let r = exp2(b * log2(abs(a)));
    if (a < 0.0) {
        if (floor(b) != b) {
            return bitcast<f32>(0x7fc00000u);
        }
        let h = b * 0.5;
        if (floor(h) != h) {
            return -r;
        }
    }
    return r;
}

fn idiv_up(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}
// Tiled GEMM on cooperative matrices (16x16x16, f16 inputs, f32 accumulators).
// out[bt, i, j] = epilogue(sum_k A[bt, i, k] * B[bt, k, j]).
// Workgroup tile BM x BN, 8 subgroups of 32 lanes, each owning a 32x32 block
// (2x2 accumulators). A and B tiles are converted to f16 (B dequantized) into shared
// memory per BK-step. The next step's global loads are issued into registers before the
// current step's matrix multiplies (register prefetch), so load latency overlaps math.
// The epilogue stages each accumulator through shared memory.
// Params: [batch, m, n, k, a.off, a.bs, a.rs, b.off, b.bs, b.rs, scales_words, tiles_n,
//          tiles_m, out(off, bs, rs), extra_e(off, s_batch, s_m, s_n)...]
@group(0) @binding(1) var<storage, read> a: array<f32>;
fn ld_a(e: u32) -> f32 {
    return a[e];
}
@group(0) @binding(2) var<storage, read> b: array<u32>;
fn b_q(e: u32) -> f32 {
    let r = e & 31u;
    let byte = (e >> 5u) * 16u + (r & 15u);
    let w = b[byte >> 2u];
    let sh = (byte & 3u) * 8u + (r >> 4u) * 4u;
    return f32((w >> sh) & 15u) - 8.0;
}
fn b_scale(so: u32, gi: u32) -> f32 {
    return unpack2x16float(b[so + (gi >> 1u)])[gi & 1u];
}
fn b_deq(so: u32, e: u32) -> f32 {
    return b_q(e) * b_scale(so, e >> 5u);
}
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
fn st(e: u32, v: f32) {
    out[e] = v;
}

const BM: u32 = 64u;
const BN: u32 = 128u;
const BK: u32 = 32u;
// shared row pitch in halves (BK + 8: keeps rows 16-byte aligned, staggers banks)
const LDS: u32 = BK + 8u;
// A elements per thread per step
const A_PER: u32 = BM * BK / 256u;

var<private> ra: array<f32, A_PER>;
const B_PER: u32 = 2u;
var<private> rbw: array<u32, B_PER>;
var<private> rbs: array<f32, B_PER>;

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
    for (var t = 0u; t < B_PER; t += 1u) {
        let e = g_lid + t * 256u;
        let gj = col0 + e / (BK / 8u);
        let w = e % (BK / 8u);
        let gk = k0 + (w / 4u) * 32u;
        var dsc: f32 = 0.0;
        var word: u32 = 0x88888888u;
        if (gj < N && gk < K) {
            dsc = b_scale(P[10], (gj * K + gk) / 32u);
            word = b[(gj * K + k0) / 8u + w];
        }
        rbw[t] = word;
        rbs[t] = dsc;
    }
}

// Store the register tiles to shared memory as f16 (A row-major, B column-major).
fn store_tiles() {
    for (var t = 0u; t < A_PER; t += 1u) {
        let e = g_lid + t * 256u;
        As[(e / BK) * LDS + e % BK] = f16(ra[t]);
    }
    for (var t = 0u; t < B_PER; t += 1u) {
        let e = g_lid + t * 256u;
        let w = e % (BK / 8u);
        let o = (e / (BK / 8u)) * LDS + (w / 4u) * 32u + (w % 4u) * 4u;
        let word = rbw[t];
        let dsc = rbs[t];
        for (var i = 0u; i < 4u; i += 1u) {
            Bs[o + i] = f16((f32((word >> (i * 8u)) & 15u) - 8.0) * dsc);
            Bs[o + 16u + i] = f16((f32((word >> (i * 8u + 4u)) & 15u) - 8.0) * dsc);
        }
    }
}

var<workgroup> As: array<f16, 2560>;
var<workgroup> Bs: array<f16, 5120>;
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
                    st(P[13u] + bt * P[14u] + i * P[15u] + j, acc);
                }
            }
        }
    }
}

// Entry point: mirrors builtins into private globals and runs `kernel` with the
// linearized workgroup id / count (kernels grid-stride over their work items).
@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_index) lidx: u32,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(subgroup_invocation_id) sgi: u32,
    @builtin(subgroup_id) sgid: u32,
    @builtin(num_subgroups) nsg: u32,
    @builtin(subgroup_size) sgs: u32,
) {
    g_lid = lidx;
    g_sgi = sgi;
    g_sgid = sgid;
    g_nsg = nsg;
    g_sgs = sgs;
    let wg_lin = wgid.x + nwg.x * (wgid.y + nwg.y * wgid.z);
    let wg_total = nwg.x * nwg.y * nwg.z;
    kernel(wg_lin, wg_total);
}
