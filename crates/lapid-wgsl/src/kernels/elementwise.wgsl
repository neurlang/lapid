// Elementwise: out[out_view(idx)] = expr(in_i[in_view_i(idx)]) for every idx of a rank-R
// space, grid-stride over the linear index. Views are Flat (offset + lin), Scalar (offset)
// or Strided (offset + sum(c_d * stride_d)); the coordinates are only decoded when needed.
// Params: [numel, shape[R], out(off, strides[R]), in_i(off, strides[R])...]
{{BINDINGS}}
fn kernel(wg_lin: u32, wg_total: u32) {
    let numel = P[0];
    let gstride = wg_total * 256u;
    for (var lin = wg_lin * 256u + g_lid; lin < numel; lin += gstride) {
{{COORDS}}{{LOADS}}        {{STORE}}({{OUT_INDEX}}, {{VALUE}});
    }
}
