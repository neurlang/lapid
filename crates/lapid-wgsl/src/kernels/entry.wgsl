
// Entry point: mirrors builtins into private globals and runs `kernel` with the
// linearized workgroup id / count (kernels grid-stride over their work items).
@compute @workgroup_size({{WG}})
fn main(
    @builtin(local_invocation_index) lidx: u32,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
// #if SUBGROUP
    @builtin(subgroup_invocation_id) sgi: u32,
// #if !WEBGPU
    @builtin(subgroup_id) sgid: u32,
    @builtin(num_subgroups) nsg: u32,
// #endif
    @builtin(subgroup_size) sgs: u32,
// #endif
) {
    g_lid = lidx;
// #if SUBGROUP
    g_sgi = sgi;
// #if WEBGPU
    // WebGPU has no subgroup_id builtin: lanes are packed in local-index order
    g_sgid = lidx / sgs;
    g_nsg = ({{WG}}u + sgs - 1u) / sgs;
// #endif
// #if !WEBGPU
    g_sgid = sgid;
    g_nsg = nsg;
// #endif
    g_sgs = sgs;
// #endif
    let wg_lin = wgid.x + nwg.x * (wgid.y + nwg.y * wgid.z);
    let wg_total = nwg.x * nwg.y * nwg.z;
    kernel(wg_lin, wg_total);
}
