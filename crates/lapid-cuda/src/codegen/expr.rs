//! `Expr` → PTX instructions: f32 math plus typed 32-bit integer words (`b32` registers,
//! wrapping, shifts `& 31`, division by zero selects 0 instead of trapping). Transcendentals are accurate to a few ulp:
//! `exp` with Cody-Waite range reduction around `ex2.approx`, `erf` with a Taylor series
//! near 0 and Abramowitz & Stegun 7.1.26 elsewhere (|err| < 1.5e-7), `tanh` via exp,
//! `sin`/`cos` with 2π range reduction, IEEE division / sqrt / reciprocal.

use lapid_compiler::{Conv, Expr, ExprType, IntOp, IntUnaryOp};
use lapid_tensor::{BinaryOp, UnaryOp};

use super::ptx::{Gen, fimm};

const LOG2E: f32 = std::f32::consts::LOG2_E;
const LN2: f32 = std::f32::consts::LN_2;

/// Register holding a rendered value and its type (`f32` or a 32-bit word in a `b32`).
pub(crate) struct Val {
    pub reg: String,
    pub ty: ExprType,
}

/// Render `e` into `g` (f32 root); `load(i, bits)` names the register holding
/// `Expr::Load(i)` (f32, `bits == false`) or `Expr::LoadBits(i)` (b32 word, `bits == true`).
/// An integer root is reinterpreted as f32 bits (roots are validated by the planners).
pub(crate) fn render(g: &mut Gen, e: &Expr, load: &dyn Fn(usize, bool) -> String) -> String {
    let v = render_typed(g, e, load);
    match v.ty {
        ExprType::F32 => v.reg,
        ExprType::Int => {
            let r = g.reg("f32");
            g.op(format!("mov.b32 {r}, {}", v.reg));
            r
        }
    }
}

/// Render `e` and report the root's type.
pub(crate) fn render_typed(g: &mut Gen, e: &Expr, load: &dyn Fn(usize, bool) -> String) -> Val {
    let f = |reg: String| Val { reg, ty: ExprType::F32 };
    let w = |reg: String| Val { reg, ty: ExprType::Int };
    match e {
        Expr::Load(i) => f(load(*i, false)),
        Expr::Const(bits) => f(g.konst(f32::from_bits(*bits))),
        Expr::Unary(op, a) => {
            let a = render(g, a, load);
            f(unary(g, *op, &a))
        }
        Expr::Binary(op, a, b) => {
            if *op == BinaryOp::Pow
                && let Expr::Const(bits) = **b
            {
                let c = f32::from_bits(bits);
                let special = [1.0f32, 2.0, 3.0, 0.5, -0.5, -1.0].contains(&c);
                if special {
                    let ra = render(g, a, load);
                    return f(pow_const(g, &ra, c));
                }
            }
            let ra = render(g, a, load);
            let rb = render(g, b, load);
            f(binary(g, *op, &ra, &rb))
        }
        Expr::Where(c, a, b) => {
            let rc = render_typed(g, c, load);
            let ra = render_typed(g, a, load);
            let rb = render_typed(g, b, load);
            let p = g.reg("pred");
            match rc.ty {
                // NaN condition counts as true (`c != 0`)
                ExprType::F32 => g.op(format!("setp.neu.f32 {p}, {}, 0f00000000", rc.reg)),
                ExprType::Int => g.op(format!("setp.ne.b32 {p}, {}, 0", rc.reg)),
            }
            if ra.ty == ExprType::Int && rb.ty == ExprType::Int {
                let r = g.reg("b32");
                g.op(format!("selp.b32 {r}, {}, {}, {p}", ra.reg, rb.reg));
                w(r)
            } else {
                let (a, b) = (as_f32(g, ra), as_f32(g, rb));
                let r = g.reg("f32");
                g.op(format!("selp.f32 {r}, {a}, {b}, {p}"));
                f(r)
            }
        }
        Expr::LoadBits(i) => w(load(*i, true)),
        Expr::ConstBits(v) => {
            let r = g.reg("b32");
            g.op(format!("mov.b32 {r}, {v}"));
            w(r)
        }
        Expr::IntUnary(op, a) => {
            let a = render_typed(g, a, load);
            let a = as_int(g, a);
            w(int_unary(g, *op, &a))
        }
        Expr::IntBinary(op, a, b) => {
            let a = render_typed(g, a, load);
            let a = as_int(g, a);
            let b = render_typed(g, b, load);
            let b = as_int(g, b);
            w(int_binary(g, *op, &a, &b))
        }
        Expr::Convert(c, a) => {
            let a = render_typed(g, a, load);
            convert(g, *c, a)
        }
    }
}

fn as_f32(g: &mut Gen, v: Val) -> String {
    match v.ty {
        ExprType::F32 => v.reg,
        ExprType::Int => {
            let r = g.reg("f32");
            g.op(format!("mov.b32 {r}, {}", v.reg));
            r
        }
    }
}

fn as_int(g: &mut Gen, v: Val) -> String {
    match v.ty {
        ExprType::Int => v.reg,
        ExprType::F32 => {
            let r = g.reg("b32");
            g.op(format!("mov.b32 {r}, {}", v.reg));
            r
        }
    }
}

fn convert(g: &mut Gen, c: Conv, a: Val) -> Val {
    match c {
        Conv::F32ToI32 | Conv::F32ToU32 => {
            let a = as_f32(g, a);
            let r = g.reg("b32");
            // truncate toward zero, saturating, NaN → 0
            let t = if c == Conv::F32ToI32 { "s32" } else { "u32" };
            g.op(format!("cvt.rzi.sat.{t}.f32 {r}, {a}"));
            Val { reg: r, ty: ExprType::Int }
        }
        Conv::F32Bits => Val { reg: as_int(g, a), ty: ExprType::Int },
        Conv::I32ToF32 | Conv::U32ToF32 => {
            let a = as_int(g, a);
            let r = g.reg("f32");
            let t = if c == Conv::I32ToF32 { "s32" } else { "u32" };
            g.op(format!("cvt.rn.f32.{t} {r}, {a}"));
            Val { reg: r, ty: ExprType::F32 }
        }
        Conv::BitsF32 => Val { reg: as_f32(g, a), ty: ExprType::F32 },
    }
}

fn int_unary(g: &mut Gen, op: IntUnaryOp, a: &str) -> String {
    let r = g.reg("b32");
    match op {
        IntUnaryOp::Neg => g.op(format!("neg.s32 {r}, {a}")),
        IntUnaryOp::Not => g.op(format!("not.b32 {r}, {a}")),
        IntUnaryOp::Abs => {
            // wrapping: (a ^ s) - s with s = a >> 31 (abs(i32::MIN) = i32::MIN)
            let s = g.reg("b32");
            g.op(format!("shr.s32 {s}, {a}, 31"));
            g.op(format!("xor.b32 {r}, {a}, {s}"));
            g.op(format!("sub.s32 {r}, {r}, {s}"));
        }
    }
    r
}

fn int_binary(g: &mut Gen, op: IntOp, a: &str, b: &str) -> String {
    let r = g.reg("b32");
    let simple = |i: &str| format!("{i} {r}, {a}, {b}");
    match op {
        IntOp::Add => g.op(simple("add.s32")),
        IntOp::Sub => g.op(simple("sub.s32")),
        IntOp::Mul => g.op(simple("mul.lo.s32")),
        IntOp::MinS => g.op(simple("min.s32")),
        IntOp::MinU => g.op(simple("min.u32")),
        IntOp::MaxS => g.op(simple("max.s32")),
        IntOp::MaxU => g.op(simple("max.u32")),
        IntOp::And => g.op(simple("and.b32")),
        IntOp::Or => g.op(simple("or.b32")),
        IntOp::Xor => g.op(simple("xor.b32")),
        IntOp::Shl | IntOp::ShrS | IntOp::ShrU => {
            let s = g.reg("b32");
            g.op(format!("and.b32 {s}, {b}, 31"));
            let i = match op {
                IntOp::Shl => "shl.b32",
                IntOp::ShrS => "shr.s32",
                _ => "shr.u32",
            };
            g.op(format!("{i} {r}, {a}, {s}"));
        }
        IntOp::CmpLtS | IntOp::CmpLtU | IntOp::CmpEq => {
            let p = g.reg("pred");
            let c = match op {
                IntOp::CmpLtS => "lt.s32",
                IntOp::CmpLtU => "lt.u32",
                _ => "eq.b32",
            };
            g.op(format!("setp.{c} {p}, {a}, {b}"));
            g.op(format!("selp.b32 {r}, 1, 0, {p}"));
        }
        IntOp::DivS | IntOp::DivU | IntOp::RemS | IntOp::RemU => {
            // never divide by zero (or i32::MIN / -1): divide by 1, then select 0
            let bad = g.reg("pred");
            let d = g.reg("b32");
            g.op(format!("setp.eq.b32 {bad}, {b}, 0"));
            let signed = matches!(op, IntOp::DivS | IntOp::RemS);
            if signed {
                let p = g.reg("pred");
                g.op(format!("setp.eq.b32 {p}, {a}, 0x80000000"));
                g.op(format!("setp.eq.and.b32 {p}, {b}, 0xFFFFFFFF, {p}"));
                g.op(format!("or.pred {bad}, {bad}, {p}"));
            }
            g.op(format!("selp.b32 {d}, 1, {b}, {bad}"));
            let i = match op {
                IntOp::DivS => "div.s32",
                IntOp::DivU => "div.u32",
                IntOp::RemS => "rem.s32",
                _ => "rem.u32",
            };
            g.op(format!("{i} {r}, {a}, {d}"));
            g.op(format!("selp.b32 {r}, 0, {r}, {bad}"));
        }
    }
    r
}

/// Whether `e` reads input `i` through `Load` (`bits == false`) or `LoadBits` (`true`).
pub(crate) fn uses_as(e: &Expr, i: usize, bits: bool) -> bool {
    match e {
        Expr::Load(j) => !bits && *j == i,
        Expr::LoadBits(j) => bits && *j == i,
        Expr::Const(_) | Expr::ConstBits(_) => false,
        Expr::Unary(_, a) | Expr::IntUnary(_, a) | Expr::Convert(_, a) => uses_as(a, i, bits),
        Expr::Binary(_, a, b) | Expr::IntBinary(_, a, b) => uses_as(a, i, bits) || uses_as(b, i, bits),
        Expr::Where(c, a, b) => uses_as(c, i, bits) || uses_as(a, i, bits) || uses_as(b, i, bits),
    }
}

/// Whether `e` reads input `i` at all.
pub(crate) fn uses(e: &Expr, i: usize) -> bool {
    uses_as(e, i, false) || uses_as(e, i, true)
}

/// Check the typing of `e` (`root`: the type the kernel stores).
pub(crate) fn check_root(e: &Expr, allow_int_root: bool, what: &str) -> lapid_tensor::Result<ExprType> {
    match e.ty() {
        Some(ExprType::F32) => Ok(ExprType::F32),
        Some(ExprType::Int) if allow_int_root => Ok(ExprType::Int),
        Some(ExprType::Int) => Err(super::bad(format!("{what}: integer expression root"))),
        None => Err(super::bad(format!("{what}: ill-typed expression"))),
    }
}

fn pow_const(g: &mut Gen, a: &str, c: f32) -> String {
    if c == 1.0 {
        return a.to_string();
    }
    let r = g.reg("f32");
    if c == 2.0 {
        g.op(format!("mul.f32 {r}, {a}, {a}"));
    } else if c == 3.0 {
        g.op(format!("mul.f32 {r}, {a}, {a}"));
        g.op(format!("mul.f32 {r}, {r}, {a}"));
    } else if c == 0.5 {
        g.op(format!("sqrt.rn.f32 {r}, {a}"));
    } else if c == -0.5 {
        g.op(format!("sqrt.rn.f32 {r}, {a}"));
        g.op(format!("rcp.rn.f32 {r}, {r}"));
    } else {
        g.op(format!("rcp.rn.f32 {r}, {a}"));
    }
    r
}

fn unary(g: &mut Gen, op: UnaryOp, a: &str) -> String {
    let r = g.reg("f32");
    match op {
        UnaryOp::Neg => g.op(format!("neg.f32 {r}, {a}")),
        UnaryOp::Abs => g.op(format!("abs.f32 {r}, {a}")),
        UnaryOp::Sqrt => g.op(format!("sqrt.rn.f32 {r}, {a}")),
        UnaryOp::Rsqrt => {
            g.op(format!("sqrt.rn.f32 {r}, {a}"));
            g.op(format!("rcp.rn.f32 {r}, {r}"));
        }
        UnaryOp::Recip => g.op(format!("rcp.rn.f32 {r}, {a}")),
        UnaryOp::Exp => return exp(g, a),
        UnaryOp::Log => {
            g.op(format!("lg2.approx.f32 {r}, {a}"));
            g.op(format!("mul.f32 {r}, {r}, {}", fimm(LN2)));
        }
        UnaryOp::Sin | UnaryOp::Cos => {
            // r = a - k * 2pi (two-constant split), then the hardware approximation
            let k = g.reg("f32");
            g.op(format!(
                "mul.f32 {k}, {a}, {}",
                fimm(1.0 / (2.0 * std::f32::consts::PI))
            ));
            g.op(format!("cvt.rni.f32.f32 {k}, {k}"));
            g.op(format!("neg.f32 {k}, {k}"));
            g.op(format!("fma.rn.f32 {r}, {k}, {}, {a}", fimm(6.283_185_5)));
            g.op(format!(
                "fma.rn.f32 {r}, {k}, {}, {r}",
                fimm(-1.748_455_6e-7)
            ));
            let f = if op == UnaryOp::Sin { "sin" } else { "cos" };
            g.op(format!("{f}.approx.f32 {r}, {r}"));
        }
        UnaryOp::Tanh => return tanh(g, a),
        UnaryOp::Erf => return erf(g, a),
    }
    r
}

/// exp(x): k = rint(x log2 e), r = x - k ln2 (hi/lo split), exp = 2^k * 2^(r log2 e).
/// Input clamped to [-104, 89] (results 0 / +inf beyond).
pub(crate) fn exp(g: &mut Gen, a: &str) -> String {
    let x = g.reg("f32");
    let k = g.reg("f32");
    let r = g.reg("f32");
    let y = g.reg("f32");
    g.op(format!("max.f32 {x}, {a}, {}", fimm(-104.0)));
    g.op(format!("min.f32 {x}, {x}, {}", fimm(89.0)));
    g.op(format!("mul.f32 {k}, {x}, {}", fimm(LOG2E)));
    g.op(format!("cvt.rni.f32.f32 {k}, {k}"));
    g.op(format!("neg.f32 {r}, {k}"));
    g.op(format!("fma.rn.f32 {r}, {r}, {}, {x}", fimm(0.693_145_75)));
    g.op(format!(
        "fma.rn.f32 {r}, {k}, {}, {r}",
        fimm(-1.428_606_8e-6)
    ));
    g.op(format!("mul.f32 {r}, {r}, {}", fimm(LOG2E)));
    g.op(format!("ex2.approx.f32 {y}, {r}"));
    // 2^k in two halves so k up to 128 / down to -150 stays representable
    let h = g.reg("f32");
    g.op(format!("mul.f32 {h}, {k}, 0f3F000000"));
    g.op(format!("cvt.rmi.f32.f32 {h}, {h}"));
    g.op(format!("sub.f32 {k}, {k}, {h}"));
    g.op(format!("ex2.approx.f32 {h}, {h}"));
    g.op(format!("ex2.approx.f32 {k}, {k}"));
    g.op(format!("mul.f32 {y}, {y}, {h}"));
    g.op(format!("mul.f32 {y}, {y}, {k}"));
    y
}

fn tanh(g: &mut Gen, a: &str) -> String {
    // |x| < 0.3: odd Taylor series to x^9; else 1 - 2 / (e^{2|x|} + 1), sign restored
    let ax = g.reg("f32");
    let x2 = g.reg("f32");
    let s = g.reg("f32");
    g.op(format!("abs.f32 {ax}, {a}"));
    g.op(format!("mul.f32 {x2}, {a}, {a}"));
    g.op(format!("mov.f32 {s}, {}", fimm(62.0 / 2835.0)));
    for c in [-17.0f32 / 315.0, 2.0 / 15.0, -1.0 / 3.0, 1.0] {
        g.op(format!("fma.rn.f32 {s}, {s}, {x2}, {}", fimm(c)));
    }
    g.op(format!("mul.f32 {s}, {s}, {a}"));
    let t2 = g.reg("f32");
    g.op(format!("add.f32 {t2}, {ax}, {ax}"));
    let e = exp(g, &t2);
    let big = g.reg("f32");
    g.op(format!("add.f32 {big}, {e}, 0f3F800000"));
    g.op(format!("rcp.rn.f32 {big}, {big}"));
    g.op(format!("fma.rn.f32 {big}, {big}, 0fC0000000, 0f3F800000"));
    // copy the sign of x
    let ub = g.reg("b32");
    let ux = g.reg("b32");
    g.op(format!("mov.b32 {ub}, {big}"));
    g.op(format!("mov.b32 {ux}, {a}"));
    g.op(format!("and.b32 {ux}, {ux}, -2147483648"));
    g.op(format!("or.b32 {ub}, {ub}, {ux}"));
    g.op(format!("mov.b32 {big}, {ub}"));
    let p = g.reg("pred");
    let r = g.reg("f32");
    g.op(format!("setp.lt.f32 {p}, {ax}, {}", fimm(0.3)));
    g.op(format!("selp.f32 {r}, {s}, {big}, {p}"));
    r
}

fn erf(g: &mut Gen, a: &str) -> String {
    let ax = g.reg("f32");
    g.op(format!("abs.f32 {ax}, {a}"));
    // small: x * (c0 + x2 (c1 + x2 (c2 + x2 (c3 + x2 c4))))
    let x2 = g.reg("f32");
    let s = g.reg("f32");
    g.op(format!("mul.f32 {x2}, {a}, {a}"));
    g.op(format!("mov.f32 {s}, {}", fimm(0.005_223_977_6)));
    for c in [
        -0.026_866_17f32,
        0.112_837_92,
        -0.376_126_38,
        std::f32::consts::FRAC_2_SQRT_PI,
    ] {
        g.op(format!("fma.rn.f32 {s}, {s}, {x2}, {}", fimm(c)));
    }
    g.op(format!("mul.f32 {s}, {s}, {a}"));
    // large: 1 - poly(t) t e^{-x^2}, t = 1 / (1 + p |x|)
    let t = g.reg("f32");
    g.op(format!(
        "fma.rn.f32 {t}, {ax}, {}, 0f3F800000",
        fimm(0.327_591_1)
    ));
    g.op(format!("rcp.rn.f32 {t}, {t}"));
    let y = g.reg("f32");
    g.op(format!("mov.f32 {y}, {}", fimm(1.061_405_4)));
    for c in [-1.453_152_1_f32, 1.421_413_8, -0.284_496_72, 0.254_829_6] {
        g.op(format!("fma.rn.f32 {y}, {y}, {t}, {}", fimm(c)));
    }
    g.op(format!("mul.f32 {y}, {y}, {t}"));
    let nx2 = g.reg("f32");
    g.op(format!("neg.f32 {nx2}, {x2}"));
    let e = exp(g, &nx2);
    g.op(format!("mul.f32 {y}, {y}, {e}"));
    g.op(format!("sub.f32 {y}, 0f3F800000, {y}"));
    let neg = g.reg("pred");
    let ny = g.reg("f32");
    g.op(format!("setp.lt.f32 {neg}, {a}, 0f00000000"));
    g.op(format!("neg.f32 {ny}, {y}"));
    g.op(format!("selp.f32 {y}, {ny}, {y}, {neg}"));
    let small = g.reg("pred");
    let r = g.reg("f32");
    g.op(format!("setp.lt.f32 {small}, {ax}, 0f3F000000"));
    g.op(format!("selp.f32 {r}, {s}, {y}, {small}"));
    r
}

fn binary(g: &mut Gen, op: BinaryOp, a: &str, b: &str) -> String {
    let r = g.reg("f32");
    match op {
        BinaryOp::Add => g.op(format!("add.f32 {r}, {a}, {b}")),
        BinaryOp::Sub => g.op(format!("sub.f32 {r}, {a}, {b}")),
        BinaryOp::Mul => g.op(format!("mul.f32 {r}, {a}, {b}")),
        BinaryOp::Div => g.op(format!("div.rn.f32 {r}, {a}, {b}")),
        BinaryOp::Max => g.op(format!("max.f32 {r}, {a}, {b}")),
        BinaryOp::Min => g.op(format!("min.f32 {r}, {a}, {b}")),
        BinaryOp::CmpLt | BinaryOp::CmpEq => {
            let p = g.reg("pred");
            let c = if op == BinaryOp::CmpLt { "lt" } else { "eq" };
            g.op(format!("setp.{c}.f32 {p}, {a}, {b}"));
            g.op(format!("selp.f32 {r}, 0f3F800000, 0f00000000, {p}"));
        }
        BinaryOp::Pow => return pow(g, a, b),
        // Integer-only ops are lowered to `IntBinary` by the compiler.
        BinaryOp::Rem | BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor | BinaryOp::Shl | BinaryOp::Shr => {
            g.op(format!("mov.f32 {r}, 0f7FC00000"))
        }
    }
    r
}

/// pow(a, b) = 2^(b log2|a|); negative bases only for integer exponents (sign from the
/// parity), NaN otherwise; pow(a, 0) = 1.
fn pow(g: &mut Gen, a: &str, b: &str) -> String {
    let r = g.reg("f32");
    g.op(format!("abs.f32 {r}, {a}"));
    g.op(format!("lg2.approx.f32 {r}, {r}"));
    g.op(format!("mul.f32 {r}, {r}, {b}"));
    g.op(format!("ex2.approx.f32 {r}, {r}"));
    let fb = g.reg("f32");
    let isint = g.reg("pred");
    let odd = g.reg("pred");
    let neg = g.reg("pred");
    let h = g.reg("f32");
    let fh = g.reg("f32");
    g.op(format!("cvt.rmi.f32.f32 {fb}, {b}"));
    g.op(format!("setp.eq.f32 {isint}, {fb}, {b}"));
    g.op(format!("mul.f32 {h}, {b}, 0f3F000000"));
    g.op(format!("cvt.rmi.f32.f32 {fh}, {h}"));
    g.op(format!("setp.neu.f32 {odd}, {fh}, {h}"));
    g.op(format!("and.pred {odd}, {odd}, {isint}"));
    g.op(format!("setp.lt.f32 {neg}, {a}, 0f00000000"));
    let nr = g.reg("f32");
    g.op(format!("neg.f32 {nr}, {r}"));
    let flip = g.reg("pred");
    g.op(format!("and.pred {flip}, {neg}, {odd}"));
    g.op(format!("selp.f32 {r}, {nr}, {r}, {flip}"));
    let bad = g.reg("pred");
    g.op(format!("not.pred {bad}, {isint}"));
    g.op(format!("and.pred {bad}, {bad}, {neg}"));
    g.op(format!("selp.f32 {r}, 0f7FC00000, {r}, {bad}"));
    let zero = g.reg("pred");
    g.op(format!("setp.eq.f32 {zero}, {b}, 0f00000000"));
    g.op(format!("selp.f32 {r}, 0f3F800000, {r}, {zero}"));
    r
}
