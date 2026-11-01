// Gather rows of a (possibly quantized) [rows, cols] table by i32 indices. An index outside
// [0, rows) raises the backend's error flag (`status[0] = 1`) and reads row 0 instead.
// Params: [rows, cols, count, scales_words, total = count * cols]
{{BINDINGS}}
@group(0) @binding(2) var<storage, read> idx: array<i32>;
@group(0) @binding(4) var<storage, read_write> status: array<u32>;

fn kernel(wg_lin: u32, wg_total: u32) {
    let rows = P[0];
    let cols = P[1];
    let so = P[3];
    let total = P[4];
    let gstride = wg_total * 256u;
    for (var e = wg_lin * 256u + g_lid; e < total; e += gstride) {
        let i = e / cols;
        let c = e - i * cols;
        let raw = idx[i];
        var r = 0u;
        if (raw < 0 || u32(raw) >= rows) {
            status[0] = 1u;
        } else {
            r = u32(raw);
        }
        let src = r * cols + c;
        st(e, {{LOAD}});
    }
}
