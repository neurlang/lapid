//! ggml K-quant super-blocks (Q4_K / Q5_K / Q6_K), stored exactly as in GGUF (AoS, 256
//! elements per block, row-major blocks). WGSL helpers over an `array<u32>` binding:
//!
//! - `{n}_kdeq(bb, i)`: element `i` (0..256) of the block starting at byte `bb`;
//! - `{n}_deq(so, e)` / `ld_{n}(e)`: row-major element `e` (`so` unused; cols % 256 == 0,
//!   so block = e / 256);
//! - `{n}_kdot(bb, s, a0)`: lane `s` (0..8) share of the dot product of one block with
//!   `A[a0 .. a0 + 256]` (32 elements per lane, word loads; needs `lda4`).
//!
//! Block layouts (bytes): Q4_K = d f16, dmin f16, scales[12], qs[128] (144);
//! Q5_K = d, dmin, scales[12], qh[32], qs[128] (176); Q6_K = ql[128], qh[64], scales i8[16],
//! d f16 (210 — blocks are only 2-byte aligned).

use lapid_tensor::DType;

pub fn is_k(d: DType) -> bool {
    matches!(d, DType::Q4K | DType::Q5K | DType::Q6K)
}

pub fn block_bytes(d: DType) -> usize {
    d.quant_format().map(|f| f.block_bytes()).unwrap_or(0)
}

/// Byte / half / word readers shared by all K formats.
fn readers(n: &str) -> String {
    format!(
        "fn {n}_b8(o: u32) -> u32 {{\n    return ({n}[o >> 2u] >> ((o & 3u) * 8u)) & 255u;\n}}\n\
         fn {n}_h(o: u32) -> f32 {{\n    return unpack2x16float({n}[o >> 2u])[(o >> 1u) & 1u];\n}}\n\
         // 32-bit word at a 2-byte aligned offset\n\
         fn {n}_w(o: u32) -> u32 {{\n    let w0 = {n}[o >> 2u];\n    if ((o & 2u) == 0u) {{\n        return w0;\n    }}\n    return (w0 >> 16u) | ({n}[(o >> 2u) + 1u] << 16u);\n}}\n"
    )
}

/// (scale, min) of sub-block `j` (0..8) of a Q4_K / Q5_K block (ggml get_scale_min_k4).
fn scale_min(n: &str) -> String {
    format!(
        r#"fn {n}_smk(bb: u32, j: u32) -> vec2<f32> {{
    let q = bb + 4u;
    if (j < 4u) {{
        return vec2<f32>(f32({n}_b8(q + j) & 63u), f32({n}_b8(q + j + 4u) & 63u));
    }}
    let sc = ({n}_b8(q + j + 4u) & 15u) | (({n}_b8(q + j - 4u) >> 6u) << 4u);
    let m = ({n}_b8(q + j + 4u) >> 4u) | (({n}_b8(q + j) >> 6u) << 4u);
    return vec2<f32>(f32(sc), f32(m));
}}
"#
    )
}

fn kdeq(n: &str, d: DType) -> String {
    match d {
        DType::Q4K => format!(
            r#"fn {n}_kdeq(bb: u32, i: u32) -> f32 {{
    let c = i >> 6u;
    let wi = i & 63u;
    let hi = wi >> 5u;
    let q = ({n}_b8(bb + 16u + c * 32u + (wi & 31u)) >> (hi * 4u)) & 15u;
    let sm = {n}_smk(bb, 2u * c + hi);
    let dm = unpack2x16float({n}[bb >> 2u]);
    return dm.x * sm.x * f32(q) - dm.y * sm.y;
}}
"#
        ),
        DType::Q5K => format!(
            r#"fn {n}_kdeq(bb: u32, i: u32) -> f32 {{
    let c = i >> 6u;
    let wi = i & 63u;
    let hi = wi >> 5u;
    let l = wi & 31u;
    let q = (({n}_b8(bb + 48u + c * 32u + l) >> (hi * 4u)) & 15u) | ((({n}_b8(bb + 16u + l) >> (2u * c + hi)) & 1u) << 4u);
    let sm = {n}_smk(bb, 2u * c + hi);
    let dm = unpack2x16float({n}[bb >> 2u]);
    return dm.x * sm.x * f32(q) - dm.y * sm.y;
}}
"#
        ),
        _ => format!(
            r#"fn {n}_kdeq(bb: u32, i: u32) -> f32 {{
    let h = i >> 7u;
    let wi = i & 127u;
    let quad = wi >> 5u;
    let l = wi & 31u;
    let lo = ({n}_b8(bb + h * 64u + l + (quad & 1u) * 32u) >> ((quad >> 1u) * 4u)) & 15u;
    let hb = ({n}_b8(bb + 128u + h * 32u + l) >> (quad * 2u)) & 3u;
    let q = i32(lo | (hb << 4u)) - 32;
    let sc = extractBits(bitcast<i32>({n}_b8(bb + 192u + h * 8u + (l >> 4u) + 2u * quad)), 0u, 8u);
    return {n}_h(bb + 208u) * f32(sc) * f32(q);
}}
"#
        ),
    }
}

fn kdot(n: &str, d: DType) -> String {
    match d {
        DType::Q4K | DType::Q5K => {
            let (qs, qh) = if d == DType::Q4K {
                ("4u", String::new())
            } else {
                (
                    "12u",
                    format!(
                        "        let qh = {n}[wb + 4u + (l0 >> 2u) + t];\n        \
                         let qhv = vec4<u32>(qh, qh >> 8u, qh >> 16u, qh >> 24u) >> vec4<u32>(2u * c);\n        \
                         lo = lo + vec4<f32>((qhv & vec4<u32>(1u)) * 16u);\n        \
                         hi = hi + vec4<f32>(((qhv >> vec4<u32>(1u)) & vec4<u32>(1u)) * 16u);\n"
                    ),
                )
            };
            format!(
                r#"fn {n}_kdot(bb: u32, s: u32, a0: u32) -> f32 {{
    // lane s (0..8) of a block: qs bytes 16s..16s+15 -> elements c*64 + l0 + [0, 16)
    // (low nibbles, sub-block 2c) and c*64 + 32 + l0 + [0, 16) (high nibbles, 2c + 1)
    let c = s >> 1u;
    let l0 = (s & 1u) * 16u;
    let wb = bb >> 2u;
    let sw = vec3<u32>({n}[wb + 1u], {n}[wb + 2u], {n}[wb + 3u]);
    let s1 = kq_smr(sw, 2u * c);
    let s2 = kq_smr(sw, 2u * c + 1u);
    let dm = unpack2x16float({n}[wb]);
    let one = vec4<f32>(1.0);
    var dlo: f32 = 0.0;
    var dhi: f32 = 0.0;
    var alo_s: f32 = 0.0;
    var ahi_s: f32 = 0.0;
    for (var t = 0u; t < 4u; t += 1u) {{
        let qw = {n}[wb + {qs} + 4u * s + t];
        let qv = vec4<u32>(qw, qw >> 8u, qw >> 16u, qw >> 24u);
        var lo = vec4<f32>(qv & vec4<u32>(15u));
        var hi = vec4<f32>((qv >> vec4<u32>(4u)) & vec4<u32>(15u));
{qh}        let alo = lda4(a0 + c * 64u + l0 + 4u * t);
        let ahi = lda4(a0 + c * 64u + 32u + l0 + 4u * t);
        dlo = dlo + dot(lo, alo);
        dhi = dhi + dot(hi, ahi);
        alo_s = alo_s + dot(alo, one);
        ahi_s = ahi_s + dot(ahi, one);
    }}
    return dm.x * (s1.x * dlo + s2.x * dhi) - dm.y * (s1.y * alo_s + s2.y * ahi_s);
}}
"#
            )
        }
        _ => format!(
            r#"fn {n}_kdot(bb: u32, s: u32, a0: u32) -> f32 {{
    // lane s (0..8): half h, ql bytes h*64 + p*32 + l0 + [0, 16) -> quads p (low nibbles)
    // and p + 2 (high nibbles)
    let h = s >> 2u;
    let p = (s >> 1u) & 1u;
    let l0 = (s & 1u) * 16u;
    let so = bb + 192u + h * 8u + (s & 1u) + 2u * p;
    let sc_lo = f32(extractBits(bitcast<i32>({n}_b8(so)), 0u, 8u));
    let sc_hi = f32(extractBits(bitcast<i32>({n}_b8(so + 4u)), 0u, 8u));
    var dlo: f32 = 0.0;
    var dhi: f32 = 0.0;
    for (var t = 0u; t < 4u; t += 1u) {{
        let ql = {n}_w(bb + h * 64u + p * 32u + l0 + 4u * t);
        let qh = {n}_w(bb + 128u + h * 32u + l0 + 4u * t);
        let qlv = vec4<u32>(ql, ql >> 8u, ql >> 16u, ql >> 24u);
        let qhv = vec4<u32>(qh, qh >> 8u, qh >> 16u, qh >> 24u);
        let qlo = vec4<f32>((qlv & vec4<u32>(15u)) | (((qhv >> vec4<u32>(2u * p)) & vec4<u32>(3u)) << vec4<u32>(4u))) - vec4<f32>(32.0);
        let qhi = vec4<f32>(((qlv >> vec4<u32>(4u)) & vec4<u32>(15u)) | (((qhv >> vec4<u32>(2u * p + 4u)) & vec4<u32>(3u)) << vec4<u32>(4u))) - vec4<f32>(32.0);
        dlo = dlo + dot(qlo, lda4(a0 + h * 128u + p * 32u + l0 + 4u * t));
        dhi = dhi + dot(qhi, lda4(a0 + h * 128u + (p + 2u) * 32u + l0 + 4u * t));
    }}
    return {n}_h(bb + 208u) * (sc_lo * dlo + sc_hi * dhi);
}}
"#
        ),
    }
}

/// `{n}_kdeq4(bb, i)`: elements i..i+4 (i % 4 == 0) of a block, one word per plane.
fn kdeq4(n: &str, d: DType) -> String {
    match d {
        DType::Q4K | DType::Q5K => {
            let qh = if d == DType::Q5K {
                format!(
                    "    let qh = {n}[(bb >> 2u) + 4u + (l >> 2u)];\n    \
                     q = q | (((vec4<u32>(qh, qh >> 8u, qh >> 16u, qh >> 24u) >> vec4<u32>(2u * c + hi)) & vec4<u32>(1u)) << vec4<u32>(4u));\n"
                )
            } else {
                String::new()
            };
            let qs = if d == DType::Q4K { 16 } else { 48 };
            format!(
                r#"fn {n}_kdeq4(bb: u32, i: u32) -> vec4<f32> {{
    let c = i >> 6u;
    let hi = (i >> 5u) & 1u;
    let l = i & 31u;
    let qw = {n}[(bb + {qs}u + c * 32u + l) >> 2u];
    var q = (vec4<u32>(qw, qw >> 8u, qw >> 16u, qw >> 24u) >> vec4<u32>(hi * 4u)) & vec4<u32>(15u);
{qh}    let sm = {n}_smk(bb, 2u * c + hi);
    let dm = unpack2x16float({n}[bb >> 2u]);
    return vec4<f32>(q) * (dm.x * sm.x) - vec4<f32>(dm.y * sm.y);
}}
"#
            )
        }
        _ => format!(
            r#"fn {n}_kdeq4(bb: u32, i: u32) -> vec4<f32> {{
    let h = i >> 7u;
    let quad = (i >> 5u) & 3u;
    let l = i & 31u;
    let ql = {n}_w(bb + h * 64u + l + (quad & 1u) * 32u);
    let qh = {n}_w(bb + 128u + h * 32u + l);
    let lo = (vec4<u32>(ql, ql >> 8u, ql >> 16u, ql >> 24u) >> vec4<u32>((quad >> 1u) * 4u)) & vec4<u32>(15u);
    let hb = (vec4<u32>(qh, qh >> 8u, qh >> 16u, qh >> 24u) >> vec4<u32>(quad * 2u)) & vec4<u32>(3u);
    let q = vec4<f32>(lo | (hb << vec4<u32>(4u))) - vec4<f32>(32.0);
    let sc = extractBits(bitcast<i32>({n}_b8(bb + 192u + h * 8u + (l >> 4u) + 2u * quad)), 0u, 8u);
    return q * ({n}_h(bb + 208u) * f32(sc));
}}
"#
        ),
    }
}

/// `{n}_kdeq16(bb, i)`: elements i..i+16 (i % 16 == 0) of a block: one sub-block, so the
/// scale (and min) are decoded once.
fn kdeq16(n: &str, d: DType) -> String {
    match d {
        DType::Q4K | DType::Q5K => {
            let qh = if d == DType::Q5K {
                format!(
                    "        let qh = {n}[(bb >> 2u) + 4u + (l >> 2u) + u];\n        \
                     q = q | (((vec4<u32>(qh, qh >> 8u, qh >> 16u, qh >> 24u) >> vec4<u32>(2u * c + hi)) & vec4<u32>(1u)) << vec4<u32>(4u));\n"
                )
            } else {
                String::new()
            };
            let qs = if d == DType::Q4K { 16 } else { 48 };
            format!(
                r#"fn {n}_kdeq16(bb: u32, i: u32) -> array<vec4<f32>, 4> {{
    let c = i >> 6u;
    let hi = (i >> 5u) & 1u;
    let l = i & 31u;
    let sm = {n}_smk(bb, 2u * c + hi);
    let dm = unpack2x16float({n}[bb >> 2u]);
    let sc = dm.x * sm.x;
    let mn = vec4<f32>(dm.y * sm.y);
    let w0 = (bb + {qs}u + c * 32u + l) >> 2u;
    var r: array<vec4<f32>, 4>;
    for (var u = 0u; u < 4u; u += 1u) {{
        let qw = {n}[w0 + u];
        var q = (vec4<u32>(qw, qw >> 8u, qw >> 16u, qw >> 24u) >> vec4<u32>(hi * 4u)) & vec4<u32>(15u);
{qh}        r[u] = vec4<f32>(q) * sc - mn;
    }}
    return r;
}}
"#
            )
        }
        _ => format!(
            r#"fn {n}_kdeq16(bb: u32, i: u32) -> array<vec4<f32>, 4> {{
    let h = i >> 7u;
    let quad = (i >> 5u) & 3u;
    let l = i & 31u;
    let sc = extractBits(bitcast<i32>({n}_b8(bb + 192u + h * 8u + (l >> 4u) + 2u * quad)), 0u, 8u);
    let f = {n}_h(bb + 208u) * f32(sc);
    var r: array<vec4<f32>, 4>;
    for (var u = 0u; u < 4u; u += 1u) {{
        let ql = {n}_w(bb + h * 64u + l + (quad & 1u) * 32u + 4u * u);
        let qh = {n}_w(bb + 128u + h * 32u + l + 4u * u);
        let lo = (vec4<u32>(ql, ql >> 8u, ql >> 16u, ql >> 24u) >> vec4<u32>((quad >> 1u) * 4u)) & vec4<u32>(15u);
        let hb = (vec4<u32>(qh, qh >> 8u, qh >> 16u, qh >> 24u) >> vec4<u32>(quad * 2u)) & vec4<u32>(3u);
        r[u] = (vec4<f32>(lo | (hb << vec4<u32>(4u))) - vec4<f32>(32.0)) * f;
    }}
    return r;
}}
"#
        ),
    }
}

/// Element helpers (`{n}_kdeq`, `{n}_kdeq4`, `{n}_deq`, `ld_{n}`) over binding `{n}`
/// (declared by the caller).
pub fn fns(n: &str, d: DType) -> String {
    let mut s = readers(n);
    if d != DType::Q6K {
        s.push_str(&scale_min(n));
    }
    s.push_str(&kdeq(n, d));
    s.push_str(&kdeq4(n, d));
    s.push_str(&kdeq16(n, d));
    let bb = block_bytes(d);
    s.push_str(&format!(
        "fn {n}_deq(so: u32, e: u32) -> f32 {{\n    return {n}_kdeq((e >> 8u) * {bb}u, e & 255u);\n}}\n\
         fn ld_{n}(e: u32) -> f32 {{\n    return {n}_deq(0u, e);\n}}\n"
    ));
    s
}

/// (scale, min) of sub-block `j` from the 12 scale bytes held in registers.
const SMR: &str = r#"fn kq_sb(sw: vec3<u32>, i: u32) -> u32 {
    return (sw[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}
fn kq_smr(sw: vec3<u32>, j: u32) -> vec2<f32> {
    if (j < 4u) {
        return vec2<f32>(f32(kq_sb(sw, j) & 63u), f32(kq_sb(sw, j + 4u) & 63u));
    }
    let hi = kq_sb(sw, j + 4u);
    return vec2<f32>(f32((hi & 15u) | ((kq_sb(sw, j - 4u) >> 6u) << 4u)), f32((hi >> 4u) | ((kq_sb(sw, j) >> 6u) << 4u)));
}
"#;

/// `fns` plus the lane dot product `{n}_kdot` (the caller provides `lda4`).
pub fn fns_dot(n: &str, d: DType) -> String {
    let mut s = fns(n, d);
    if d != DType::Q6K {
        s.push_str(SMR);
    }
    s.push_str(&kdot(n, d));
    s
}
