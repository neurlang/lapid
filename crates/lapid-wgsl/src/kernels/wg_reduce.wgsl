
// Workgroup-wide sum / max over 256 threads; every thread gets the result.
// Must be called in workgroup-uniform control flow.
var<workgroup> red: array<f32, 256>;

// #if SUBGROUP
fn wg_sum(v: f32) -> f32 {
    let s = subgroupAdd(v);
    if (g_sgi == 0u) {
        red[g_sgid] = s;
    }
    workgroupBarrier();
    var r: f32 = 0.0;
    for (var i = 0u; i < g_nsg; i += 1u) {
        r = r + red[i];
    }
    workgroupBarrier();
    return r;
}

fn wg_max(v: f32) -> f32 {
    let s = subgroupMax(v);
    if (g_sgi == 0u) {
        red[g_sgid] = s;
    }
    workgroupBarrier();
    var r: f32 = bitcast<f32>(0xff800000u);
    for (var i = 0u; i < g_nsg; i += 1u) {
        r = max(r, red[i]);
    }
    workgroupBarrier();
    return r;
}
// #endif
// #if !SUBGROUP
fn wg_sum(v: f32) -> f32 {
    red[g_lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (g_lid < s) {
            red[g_lid] = red[g_lid] + red[g_lid + s];
        }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

fn wg_max(v: f32) -> f32 {
    red[g_lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (g_lid < s) {
            red[g_lid] = max(red[g_lid], red[g_lid + s]);
        }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}
// #endif
