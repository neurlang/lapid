// RMS norm over rows, one workgroup per row: x * rsqrt(mean(x^2) + eps) * w. Rows up to
// 4 * 256 columns are read once into registers; longer rows are re-read.
// Params: [rows, cols, eps_bits]
{{BINDINGS}}
fn kernel(wg_lin: u32, wg_total: u32) {
    let rows = P[0];
    let cols = P[1];
    let eps = bitcast<f32>(P[2]);
    let fcols = f32(cols);
    let cached = cols <= 1024u;
    for (var r = wg_lin; r < rows; r += wg_total) {
        let base = r * cols;
        var xv: array<f32, 4>;
        var sq: f32 = 0.0;
        if (cached) {
            for (var u = 0u; u < 4u; u += 1u) {
                let j = g_lid + u * 256u;
                xv[u] = 0.0;
                if (j < cols) {
                    xv[u] = ld_x(base + j);
                }
                sq += xv[u] * xv[u];
            }
        } else {
            for (var j = g_lid; j < cols; j += 256u) {
                let v = ld_x(base + j);
                sq += v * v;
            }
        }
        let rstd = inverseSqrt(wg_sum(sq) / fcols + eps);
        if (cached) {
            for (var u = 0u; u < 4u; u += 1u) {
                let j = g_lid + u * 256u;
                if (j < cols) {
                    st(base + j, xv[u] * rstd * ld_w(j));
                }
            }
        } else {
            for (var j = g_lid; j < cols; j += 256u) {
                st(base + j, ld_x(base + j) * rstd * ld_w(j));
            }
        }
    }
}
