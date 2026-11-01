// lapid prelude: extensions, the params binding, builtin mirrors and math helpers.

// Launch params (sizes, strides, offsets) as u32 words; layout documented per kernel.
@group(0) @binding(0) var<storage, read> P: array<u32>;

var<private> g_lid: u32;
var<private> g_sgi: u32;
var<private> g_sgid: u32;
var<private> g_nsg: u32;
var<private> g_sgs: u32;

// erf: Taylor series near 0, Abramowitz & Stegun 7.1.26 elsewhere.
fn erf_(x: f32) -> f32 {
    let a = abs(x);
    if (a < 0.5) {
        let x2 = x * x;
        return x * (1.1283791671 + x2 * (-0.3761263890 + x2 * (0.1128379167 + x2 * (-0.0268661706 + x2 * 0.0052239776))));
    }
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * exp(-a * a);
    return select(y, -y, x < 0.0);
}

fn tanh_(x: f32) -> f32 {
    return tanh(clamp(x, -15.0, 15.0));
}

// pow with negative bases for integer exponents.
fn pow_(a: f32, b: f32) -> f32 {
    if (b == 0.0) {
        return 1.0;
    }
    let r = exp2(b * log2(abs(a)));
    if (a < 0.0) {
        if (floor(b) != b) {
            return bitcast<f32>(0x7fc00000u);
        }
        let h = b * 0.5;
        if (floor(h) != h) {
            return -r;
        }
    }
    return r;
}

fn idiv_up(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}
// Tiled GEMM without matrix cores: 64x64 output tile per 256-thread workgroup, 4x4 per
// thread, k-step 32 (one quant group, dequantized into shared memory), f32 throughout.
// out[bt, i, j] = epilogue(sum_k A[bt, i, k] * B[bt, k, j]).
// Params: [batch, m, n, k, a.off, a.bs, a.rs, b.off, b.bs, b.rs, scales_words, tiles_n,
//          tiles_m, out(off, bs, rs), extra_e(off, s_batch, s_m, s_n)...]
@group(0) @binding(1) var<storage, read> a: array<f32>;
fn ld_a(e: u32) -> f32 {
    return a[e];
}
@group(0) @binding(2) var<storage, read> b: array<u32>;
fn b_b8(o: u32) -> u32 {
    return (b[o >> 2u] >> ((o & 3u) * 8u)) & 255u;
}
fn b_h(o: u32) -> f32 {
    return unpack2x16float(b[o >> 2u])[(o >> 1u) & 1u];
}
// 32-bit word at a 2-byte aligned offset
fn b_w(o: u32) -> u32 {
    let w0 = b[o >> 2u];
    if ((o & 2u) == 0u) {
        return w0;
    }
    return (w0 >> 16u) | (b[(o >> 2u) + 1u] << 16u);
}
fn b_kdeq(bb: u32, i: u32) -> f32 {
    let h = i >> 7u;
    let wi = i & 127u;
    let quad = wi >> 5u;
    let l = wi & 31u;
    let lo = (b_b8(bb + h * 64u + l + (quad & 1u) * 32u) >> ((quad >> 1u) * 4u)) & 15u;
    let hb = (b_b8(bb + 128u + h * 32u + l) >> (quad * 2u)) & 3u;
    let q = i32(lo | (hb << 4u)) - 32;
    let sc = extractBits(bitcast<i32>(b_b8(bb + 192u + h * 8u + (l >> 4u) + 2u * quad)), 0u, 8u);
    return b_h(bb + 208u) * f32(sc) * f32(q);
}
fn b_kdeq4(bb: u32, i: u32) -> vec4<f32> {
    let h = i >> 7u;
    let quad = (i >> 5u) & 3u;
    let l = i & 31u;
    let ql = b_w(bb + h * 64u + l + (quad & 1u) * 32u);
    let qh = b_w(bb + 128u + h * 32u + l);
    let lo = (vec4<u32>(ql, ql >> 8u, ql >> 16u, ql >> 24u) >> vec4<u32>((quad >> 1u) * 4u)) & vec4<u32>(15u);
    let hb = (vec4<u32>(qh, qh >> 8u, qh >> 16u, qh >> 24u) >> vec4<u32>(quad * 2u)) & vec4<u32>(3u);
    let q = vec4<f32>(lo | (hb << vec4<u32>(4u))) - vec4<f32>(32.0);
    let sc = extractBits(bitcast<i32>(b_b8(bb + 192u + h * 8u + (l >> 4u) + 2u * quad)), 0u, 8u);
    return q * (b_h(bb + 208u) * f32(sc));
}
fn b_kdeq16(bb: u32, i: u32) -> array<vec4<f32>, 4> {
    let h = i >> 7u;
    let quad = (i >> 5u) & 3u;
    let l = i & 31u;
    let sc = extractBits(bitcast<i32>(b_b8(bb + 192u + h * 8u + (l >> 4u) + 2u * quad)), 0u, 8u);
    let f = b_h(bb + 208u) * f32(sc);
    var r: array<vec4<f32>, 4>;
    for (var u = 0u; u < 4u; u += 1u) {
        let ql = b_w(bb + h * 64u + l + (quad & 1u) * 32u + 4u * u);
        let qh = b_w(bb + 128u + h * 32u + l + 4u * u);
        let lo = (vec4<u32>(ql, ql >> 8u, ql >> 16u, ql >> 24u) >> vec4<u32>((quad >> 1u) * 4u)) & vec4<u32>(15u);
        let hb = (vec4<u32>(qh, qh >> 8u, qh >> 16u, qh >> 24u) >> vec4<u32>(quad * 2u)) & vec4<u32>(3u);
        r[u] = (vec4<f32>(lo | (hb << vec4<u32>(4u))) - vec4<f32>(32.0)) * f;
    }
    return r;
}
fn b_deq(so: u32, e: u32) -> f32 {
    return b_kdeq((e >> 8u) * 210u, e & 255u);
}
fn ld_b(e: u32) -> f32 {
    return b_deq(0u, e);
}
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
fn st(e: u32, v: f32) {
    out[e] = v;
}

var<workgroup> As: array<f32, 2048>;
var<workgroup> Bs: array<f32, 2048>;

fn kernel(wg_lin: u32, wg_total: u32) {
    let batch = P[0];
    let M = P[1];
    let N = P[2];
    let K = P[3];
    let tiles_n = P[11];
    let tiles_m = P[12];
    let items = batch * tiles_m * tiles_n;
    let tx = g_lid % 16u;
    let ty = g_lid / 16u;
    let lr = g_lid / 4u;
    let lk = (g_lid % 4u) * 8u;
    for (var item = wg_lin; item < items; item += wg_total) {
        let tn = item % tiles_n;
        let rest = item / tiles_n;
        let tm = rest % tiles_m;
        let bt = rest / tiles_m;
        let row0 = tm * 64u;
        let col0 = tn * 64u;
        let a_base = P[4] + bt * P[5];
        let b_base = P[7] + bt * P[8];
        var acc0 = vec4<f32>(0.0);
        var acc1 = vec4<f32>(0.0);
        var acc2 = vec4<f32>(0.0);
        var acc3 = vec4<f32>(0.0);
        for (var k0 = 0u; k0 < K; k0 += 32u) {
            workgroupBarrier();
            {
                let gi = row0 + lr;
                let arow = a_base + gi * P[6];
                for (var e = 0u; e < 8u; e += 1u) {
                    let kk = k0 + lk + e;
                    var v: f32 = 0.0;
                    if (gi < M && kk < K) {
                        v = ld_a(arow + kk);
                    }
                    As[(lk + e) * 64u + lr] = v;
                }
            }
            // B tile into Bs[k * 64 + j]
            {
                // 8 consecutive k of one row (K % 256 == 0: the tile never crosses K)
                let gj = col0 + lr;
                var v0 = vec4<f32>(0.0);
                var v1 = vec4<f32>(0.0);
                if (gj < N && k0 + lk < K) {
                    let kk = k0 + lk;
                    let bb = (gj * (K / 256u) + kk / 256u) * 210u;
                    v0 = b_kdeq4(bb, kk % 256u);
                    v1 = b_kdeq4(bb, kk % 256u + 4u);
                }
                for (var e = 0u; e < 4u; e += 1u) {
                    Bs[(lk + e) * 64u + lr] = v0[e];
                    Bs[(lk + 4u + e) * 64u + lr] = v1[e];
                }
            }
            workgroupBarrier();
            for (var kk = 0u; kk < 32u; kk += 1u) {
                let ao = kk * 64u + ty * 4u;
                let bo = kk * 64u + tx * 4u;
                let av = vec4<f32>(As[ao], As[ao + 1u], As[ao + 2u], As[ao + 3u]);
                let bv = vec4<f32>(Bs[bo], Bs[bo + 1u], Bs[bo + 2u], Bs[bo + 3u]);
                acc0 = fma(vec4<f32>(av.x), bv, acc0);
                acc1 = fma(vec4<f32>(av.y), bv, acc1);
                acc2 = fma(vec4<f32>(av.z), bv, acc2);
                acc3 = fma(vec4<f32>(av.w), bv, acc3);
            }
        }
        for (var r = 0u; r < 4u; r += 1u) {
            let i = row0 + ty * 4u + r;
            var accr = acc0;
            if (r == 1u) {
                accr = acc1;
            } else if (r == 2u) {
                accr = acc2;
            } else if (r == 3u) {
                accr = acc3;
            }
            if (i < M) {
                for (var c = 0u; c < 4u; c += 1u) {
                    let j = col0 + tx * 4u + c;
                    if (j < N) {
                        let acc = accr[c];
                        st(P[13u] + bt * P[14u] + i * P[15u] + j, acc);
                    }
                }
            }
        }
    }
}

// Entry point: mirrors builtins into private globals and runs `kernel` with the
// linearized workgroup id / count (kernels grid-stride over their work items).
@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_index) lidx: u32,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    g_lid = lidx;
    let wg_lin = wgid.x + nwg.x * (wgid.y + nwg.y * wgid.z);
    let wg_total = nwg.x * nwg.y * nwg.z;
    kernel(wg_lin, wg_total);
}
