// Bucket the (token, slot) pairs of a MatMulId by expert for the grouped GEMM (one
// workgroup). Scratch S (u32): [0, E] row offsets, [E + 1, 2E + 1] tile offsets
// (BM-row tiles per expert), [2E + 2, 2E + 2 + T*R) pair list ordered by expert.
// An expert id outside [0, E) raises the error flag and counts as expert 0.
// Params: [T * R, E, BM]
@group(0) @binding(3) var<storage, read> ids: array<i32>;
@group(0) @binding(5) var<storage, read_write> S: array<u32>;
@group(0) @binding(6) var<storage, read_write> status: array<u32>;

const EMAX: u32 = {{EMAX}}u;
var<workgroup> cnt: array<atomic<u32>, EMAX>;
var<workgroup> base: array<u32, EMAX>;

fn expert(p: u32, E: u32) -> u32 {
    let raw = ids[p];
    if (raw < 0 || u32(raw) >= E) {
        status[0] = 1u;
        return 0u;
    }
    return u32(raw);
}

fn kernel(wg_lin: u32, wg_total: u32) {
    let TR = P[0];
    let E = P[1];
    let BM = P[2];
    for (var e = g_lid; e < E; e += 256u) {
        atomicStore(&cnt[e], 0u);
    }
    workgroupBarrier();
    for (var p = g_lid; p < TR; p += 256u) {
        atomicAdd(&cnt[expert(p, E)], 1u);
    }
    workgroupBarrier();
    if (g_lid == 0u) {
        var r = 0u;
        var tl = 0u;
        for (var e = 0u; e < E; e += 1u) {
            let c = atomicLoad(&cnt[e]);
            S[e] = r;
            S[E + 1u + e] = tl;
            base[e] = r;
            r += c;
            tl += (c + BM - 1u) / BM;
            atomicStore(&cnt[e], 0u);
        }
        S[E] = r;
        S[2u * E + 1u] = tl;
    }
    workgroupBarrier();
    for (var p = g_lid; p < TR; p += 256u) {
        let x = expert(p, E);
        let pos = atomicAdd(&cnt[x], 1u);
        S[2u * E + 2u + base[x] + pos] = p;
    }
}
