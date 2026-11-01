// Reduce over the last axis with a fused prologue expression.
// ROW: one workgroup per output (threads stride the reduced axis, then a workgroup
// reduction); otherwise one thread per output.
// Params: [outer, inner, shape[0..R-1], in_i(off, strides[R])...]
{{BINDINGS}}
fn kernel(wg_lin: u32, wg_total: u32) {
    let outer = P[0];
    let inner = P[1];
// #if ROW
    for (var o = wg_lin; o < outer; o += wg_total) {
// #endif
// #if !ROW
    for (var o = wg_lin * 256u + g_lid; o < outer; o += wg_total * 256u) {
// #endif
{{COORDS}}{{BASES}}        var acc: f32 = {{IDENTITY}};
// #if ROW
        for (var j = g_lid; j < inner; j += 256u) {
// #endif
// #if !ROW
        for (var j = 0u; j < inner; j += 1u) {
// #endif
{{LOADS}}            let v = {{VALUE}};
            acc = {{COMBINE}};
        }
// #if ROW
        let total = {{WG_REDUCE}}(acc);
        if (g_lid == 0u) {
            st(o, total);
        }
// #endif
// #if !ROW
        st(o, acc);
// #endif
    }
}
