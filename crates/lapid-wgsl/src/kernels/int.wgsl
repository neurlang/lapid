// Integer-word helpers for typed integer expressions (included when a kernel calls one).
// Words are u32; signedness is a property of the operation. Division / remainder by zero
// (and i32::MIN / -1) give 0 and never trap; f32 -> int conversions saturate, NaN -> 0.
fn gi_div_s(a: u32, b: u32) -> u32 {
    let x = bitcast<i32>(a);
    let y = bitcast<i32>(b);
    let bad = y == 0 || (x == bitcast<i32>(0x80000000u) && y == -1);
    return select(bitcast<u32>(x / select(y, 1, bad)), 0u, bad);
}

fn gi_rem_s(a: u32, b: u32) -> u32 {
    let x = bitcast<i32>(a);
    let y = bitcast<i32>(b);
    let bad = y == 0 || (x == bitcast<i32>(0x80000000u) && y == -1);
    return select(bitcast<u32>(x % select(y, 1, bad)), 0u, bad);
}

fn gi_div_u(a: u32, b: u32) -> u32 {
    return select(a / max(b, 1u), 0u, b == 0u);
}

fn gi_rem_u(a: u32, b: u32) -> u32 {
    return select(a % max(b, 1u), 0u, b == 0u);
}

fn gi_f2i(x: f32) -> u32 {
    // NaN test on the bits (drivers may fold `x != x`)
    if ((bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u) {
        return 0u;
    }
    if (x >= 2147483648.0) {
        return 0x7fffffffu;
    }
    if (x <= -2147483648.0) {
        return 0x80000000u;
    }
    return bitcast<u32>(i32(x));
}

fn gi_f2u(x: f32) -> u32 {
    if (!(x > 0.0)) {
        return 0u;
    }
    if (x >= 4294967296.0) {
        return 0xffffffffu;
    }
    return u32(x);
}

fn gi_abs(a: u32) -> u32 {
    return select(a, 0u - a, bitcast<i32>(a) < 0);
}
