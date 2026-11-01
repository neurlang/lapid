//! `Expr` → WGSL expression. F32-typed nodes render as `f32`, integer nodes as `u32`
//! words (signedness per operation via `bitcast<i32>`); integer helpers (`gi_*`) live in
//! `kernels/int.wgsl` and are included automatically when used.

use lapid_compiler::{Conv, Expr, ExprType, IntOp, IntUnaryOp};
use lapid_tensor::{BinaryOp, DType, Error, Result, UnaryOp};

use super::common::flit;

/// Render `e`; `load(i, bits)` gives the WGSL expression for `Expr::Load(i)` (`bits ==
/// false`, an f32) or `Expr::LoadBits(i)` (`bits == true`, a u32 word) — normally
/// `let`-bound variables so repeated loads cost nothing.
pub fn render(e: &Expr, load: &dyn Fn(usize, bool) -> String) -> String {
    match e {
        Expr::Load(i) => load(*i, false),
        Expr::LoadBits(i) => load(*i, true),
        Expr::Const(bits) => flit(f32::from_bits(*bits)),
        Expr::ConstBits(v) => format!("{v:#x}u"),
        Expr::Unary(op, a) => {
            let a = render(a, load);
            match op {
                UnaryOp::Neg => format!("(-{a})"),
                UnaryOp::Exp => format!("exp({a})"),
                UnaryOp::Log => format!("log({a})"),
                UnaryOp::Sqrt => format!("sqrt({a})"),
                UnaryOp::Rsqrt => format!("inverseSqrt({a})"),
                UnaryOp::Recip => format!("(1.0 / {a})"),
                UnaryOp::Sin => format!("sin({a})"),
                UnaryOp::Cos => format!("cos({a})"),
                UnaryOp::Tanh => format!("tanh_({a})"),
                UnaryOp::Erf => format!("erf_({a})"),
                UnaryOp::Abs => format!("abs({a})"),
            }
        }
        Expr::Binary(op, a, b) => {
            if *op == BinaryOp::Pow
                && let Expr::Const(bits) = **b {
                    let c = f32::from_bits(bits);
                    let ra = render(a, load);
                    if c == 1.0 {
                        return ra;
                    } else if c == 2.0 {
                        return format!("({ra} * {ra})");
                    } else if c == 0.5 {
                        return format!("sqrt({ra})");
                    } else if c == -0.5 {
                        return format!("inverseSqrt({ra})");
                    } else if c == -1.0 {
                        return format!("(1.0 / {ra})");
                    }
                }
            let ra = render(a, load);
            let rb = render(b, load);
            match op {
                BinaryOp::Add => format!("({ra} + {rb})"),
                BinaryOp::Sub => format!("({ra} - {rb})"),
                BinaryOp::Mul => format!("({ra} * {rb})"),
                BinaryOp::Div => format!("({ra} / {rb})"),
                BinaryOp::Max => format!("max({ra}, {rb})"),
                BinaryOp::Min => format!("min({ra}, {rb})"),
                BinaryOp::Pow => format!("pow_({ra}, {rb})"),
                BinaryOp::CmpLt => format!("select(0.0, 1.0, {ra} < {rb})"),
                BinaryOp::CmpEq => format!("select(0.0, 1.0, {ra} == {rb})"),
                // integer-only ops never reach f32 expressions (lowered to `IntBinary`)
                BinaryOp::Rem | BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor | BinaryOp::Shl | BinaryOp::Shr => {
                    "bitcast<f32>(0x7fc00000u)".to_string()
                }
            }
        }
        Expr::Where(c, a, b) => {
            let rc = render(c, load);
            let ra = render(a, load);
            let rb = render(b, load);
            let zero = if c.ty() == Some(ExprType::Int) { "0u" } else { "0.0" };
            format!("select({rb}, {ra}, {rc} != {zero})")
        }
        Expr::IntUnary(op, a) => {
            let a = render(a, load);
            match op {
                IntUnaryOp::Neg => format!("(0u - {a})"),
                IntUnaryOp::Abs => format!("gi_abs({a})"),
                IntUnaryOp::Not => format!("(~{a})"),
            }
        }
        Expr::IntBinary(op, a, b) => {
            let a = render(a, load);
            let b = render(b, load);
            let s = |x: &str| format!("bitcast<i32>({x})");
            match op {
                IntOp::Add => format!("({a} + {b})"),
                IntOp::Sub => format!("({a} - {b})"),
                IntOp::Mul => format!("({a} * {b})"),
                IntOp::DivS => format!("gi_div_s({a}, {b})"),
                IntOp::DivU => format!("gi_div_u({a}, {b})"),
                IntOp::RemS => format!("gi_rem_s({a}, {b})"),
                IntOp::RemU => format!("gi_rem_u({a}, {b})"),
                IntOp::MinS => format!("bitcast<u32>(min({}, {}))", s(&a), s(&b)),
                IntOp::MaxS => format!("bitcast<u32>(max({}, {}))", s(&a), s(&b)),
                IntOp::MinU => format!("min({a}, {b})"),
                IntOp::MaxU => format!("max({a}, {b})"),
                IntOp::And => format!("({a} & {b})"),
                IntOp::Or => format!("({a} | {b})"),
                IntOp::Xor => format!("({a} ^ {b})"),
                IntOp::Shl => format!("({a} << ({b} & 31u))"),
                IntOp::ShrU => format!("({a} >> ({b} & 31u))"),
                IntOp::ShrS => format!("bitcast<u32>({} >> ({b} & 31u))", s(&a)),
                IntOp::CmpLtS => format!("select(0u, 1u, {} < {})", s(&a), s(&b)),
                IntOp::CmpLtU => format!("select(0u, 1u, {a} < {b})"),
                IntOp::CmpEq => format!("select(0u, 1u, {a} == {b})"),
            }
        }
        Expr::Convert(c, a) => {
            let a = render(a, load);
            match c {
                Conv::F32ToI32 => format!("gi_f2i({a})"),
                Conv::F32ToU32 => format!("gi_f2u({a})"),
                Conv::I32ToF32 => format!("f32(bitcast<i32>({a}))"),
                Conv::U32ToF32 => format!("f32({a})"),
                Conv::F32Bits => format!("bitcast<u32>({a})"),
                Conv::BitsF32 => format!("bitcast<f32>({a})"),
            }
        }
    }
}

/// Validate an expression for a kernel: well-typed, root of type `root` (or either type
/// when `root` is `None`), loads within `dtypes`, `LoadBits` only on 32-bit plain inputs.
pub fn check(e: &Expr, dtypes: &[DType], root: Option<ExprType>, what: &str) -> Result<()> {
    let bad = |m: String| Error::new("bad_launch", format!("{what}: {m}"));
    let Some(t) = e.ty() else {
        return Err(bad("ill-typed expression".into()));
    };
    if let Some(r) = root
        && t != r {
            return Err(bad(format!("expression root is {t:?}, expected {r:?}")));
        }
    if e.num_loads() > dtypes.len() {
        return Err(bad("expression loads more inputs than bound".into()));
    }
    for (i, d) in dtypes.iter().enumerate() {
        if uses_bits(e, i) && !matches!(d, DType::F32 | DType::I32) {
            return Err(bad(format!("LoadBits of a {} input", d.name())));
        }
    }
    Ok(())
}

/// Whether `e` reads `Load(i)` (as f32).
pub fn uses_load(e: &Expr, i: usize) -> bool {
    walk(e, &|x| matches!(x, Expr::Load(j) if *j == i))
}

/// Whether `e` reads `LoadBits(i)` (raw word).
pub fn uses_bits(e: &Expr, i: usize) -> bool {
    walk(e, &|x| matches!(x, Expr::LoadBits(j) if *j == i))
}

fn walk(e: &Expr, f: &dyn Fn(&Expr) -> bool) -> bool {
    if f(e) {
        return true;
    }
    match e {
        Expr::Load(_) | Expr::LoadBits(_) | Expr::Const(_) | Expr::ConstBits(_) => false,
        Expr::Unary(_, a) | Expr::IntUnary(_, a) | Expr::Convert(_, a) => walk(a, f),
        Expr::Binary(_, a, b) | Expr::IntBinary(_, a, b) => walk(a, f) || walk(b, f),
        Expr::Where(c, a, b) => walk(c, f) || walk(a, f) || walk(b, f),
    }
}

/// Whether `e` reads input `i` at all.
pub fn uses(e: &Expr, i: usize) -> bool {
    uses_load(e, i) || uses_bits(e, i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_gelu() {
        // x * 0.5 * (1 + erf(x * 0.70710678))
        let x = || Box::new(Expr::Load(0));
        let e = Expr::Binary(
            BinaryOp::Mul,
            Box::new(Expr::Binary(BinaryOp::Mul, x(), Box::new(Expr::konst(0.5)))),
            Box::new(Expr::Binary(
                BinaryOp::Add,
                Box::new(Expr::konst(1.0)),
                Box::new(Expr::Unary(
                    UnaryOp::Erf,
                    Box::new(Expr::Binary(BinaryOp::Mul, x(), Box::new(Expr::konst(std::f32::consts::FRAC_1_SQRT_2)))),
                )),
            )),
        );
        let s = render(&e, &|i, _| format!("x{i}"));
        assert_eq!(s, "((x0 * 0.5f) * (1.0f + erf_((x0 * 0.70710677f))))");
        assert!(uses(&e, 0));
        assert!(!uses(&e, 1));
    }
}
