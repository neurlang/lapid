// Row softmax, one workgroup per row. Params: [rows, cols]
{{BINDINGS}}
fn kernel(wg_lin: u32, wg_total: u32) {
    let rows = P[0];
    let cols = P[1];
    for (var r = wg_lin; r < rows; r += wg_total) {
        let base = r * cols;
        var m: f32 = bitcast<f32>(0xff800000u);
        for (var j = g_lid; j < cols; j += 256u) {
            m = max(m, ld_x(base + j));
        }
        m = wg_max(m);
        var sum: f32 = 0.0;
        for (var j = g_lid; j < cols; j += 256u) {
            sum += exp(ld_x(base + j) - m);
        }
        sum = wg_sum(sum);
        let inv = 1.0 / sum;
        for (var j = g_lid; j < cols; j += 256u) {
            st(base + j, exp(ld_x(base + j) - m) * inv);
        }
    }
}
