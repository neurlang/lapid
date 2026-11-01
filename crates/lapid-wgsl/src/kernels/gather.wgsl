// Gather rows of a (possibly quantized) [rows, cols] table by i32 indices. An index outside
// [0, all_rows) raises the backend's error flag (`status[0] = 1`) and reads row 0 instead.
// A split launch binds rows [base, base + rows) only and skips indices outside them.
// Params: [rows, cols, count, scales_words, total = count * cols, base, all_rows]
{{BINDINGS}}
@group(0) @binding(2) var<storage, read> idx: array<i32>;
@group(0) @binding(4) var<storage, read_write> status: array<u32>;

fn kernel(wg_lin: u32, wg_total: u32) {
    let rows = P[0];
    let cols = P[1];
    let so = P[3];
    let total = P[4];
    let base = P[5];
    let all_rows = P[6];
    let gstride = wg_total * 256u;
    for (var e = wg_lin * 256u + g_lid; e < total; e += gstride) {
        let i = e / cols;
        let c = e - i * cols;
        let raw = idx[i];
        var r = 0u;
        if (raw < 0 || u32(raw) >= all_rows) {
            status[0] = 1u;
            if (base != 0u) {
                continue;
            }
        } else {
            if (u32(raw) < base || u32(raw) - base >= rows) {
                continue;
            }
            r = u32(raw) - base;
        }
        let src = r * cols + c;
        st(e, {{LOAD}});
    }
}
