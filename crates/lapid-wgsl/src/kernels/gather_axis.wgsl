// take_along_axis over the canonical [outer, n, inner] space: exact element copy of x at the
// index read from `idx`. Out-of-range indices raise the error flag and write 0 (x is not read).
// HALF: 16-bit elements, one thread per output word (two elements, no atomics).
// Params: [outer, n, inner, axis_len, total, x(off, s0, s1, s2), idx(off, s0, s1, s2)]
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read> idx: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
@group(0) @binding(4) var<storage, read_write> status: array<u32>;

// Source element of output element `e`, or 0xffffffff when its index is out of range.
fn src_of(e: u32) -> u32 {
    let inner = P[2];
    let i = e % inner;
    let rest = e / inner;
    let j = rest % P[1];
    let o = rest / P[1];
    let raw = idx[P[9] + o * P[10] + j * P[11] + i * P[12]];
    if (raw < 0 || u32(raw) >= P[3]) {
        status[0] = 1u;
        return 0xffffffffu;
    }
    return P[5] + o * P[6] + u32(raw) * P[7] + i * P[8];
}

// #if HALF
fn half_at(s: u32) -> u32 {
    if (s == 0xffffffffu) {
        return 0u;
    }
    return (x[s >> 1u] >> ((s & 1u) * 16u)) & 0xffffu;
}
// #endif

fn kernel(wg_lin: u32, wg_total: u32) {
    let total = P[4];
    let gstride = wg_total * 256u;
// #if !HALF
    for (var e = wg_lin * 256u + g_lid; e < total; e += gstride) {
        let s = src_of(e);
        var v = 0u;
        if (s != 0xffffffffu) {
            v = x[s];
        }
        out[e] = v;
    }
// #endif
// #if HALF
    let words = (total + 1u) / 2u;
    for (var w = wg_lin * 256u + g_lid; w < words; w += gstride) {
        var v = half_at(src_of(2u * w));
        if (2u * w + 1u < total) {
            v = v | (half_at(src_of(2u * w + 1u)) << 16u);
        }
        out[w] = v;
    }
// #endif
}
