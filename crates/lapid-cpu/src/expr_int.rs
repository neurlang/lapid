//! Integer-word part of the expression VM. Words live in the f32 register lanes as bit
//! patterns (`f32::from_bits`), so registers stay one type; integer instructions reinterpret.
//! Also: expression normalization for raw (bit-exact) i32 inputs.

use lapid_compiler::{Conv, Expr, ExprType, IntOp, IntUnaryOp};
use lapid_tensor::DType;

pub(crate) fn conv_result(c: Conv) -> ExprType {
    match c {
        Conv::F32ToI32 | Conv::F32ToU32 | Conv::F32Bits => ExprType::Int,
        Conv::I32ToF32 | Conv::U32ToF32 | Conv::BitsF32 => ExprType::F32,
    }
}

/// `c` applied to the lane bits `x`; returns lane bits.
#[inline(always)]
pub(crate) fn conv_scalar(c: Conv, x: u32) -> u32 {
    match c {
        Conv::F32ToI32 => (f32::from_bits(x) as i32) as u32,
        Conv::F32ToU32 => f32::from_bits(x) as u32,
        Conv::I32ToF32 => (x as i32 as f32).to_bits(),
        Conv::U32ToF32 => (x as f32).to_bits(),
        Conv::F32Bits | Conv::BitsF32 => x,
    }
}

#[inline(always)]
pub(crate) fn int_bin_scalar(op: IntOp, a: u32, b: u32) -> u32 {
    let (sa, sb) = (a as i32, b as i32);
    match op {
        IntOp::Add => a.wrapping_add(b),
        IntOp::Sub => a.wrapping_sub(b),
        IntOp::Mul => a.wrapping_mul(b),
        IntOp::DivS => sa.checked_div(sb).unwrap_or(0) as u32,
        IntOp::DivU => a.checked_div(b).unwrap_or(0),
        IntOp::RemS => sa.checked_rem(sb).unwrap_or(0) as u32,
        IntOp::RemU => a.checked_rem(b).unwrap_or(0),
        IntOp::MinS => sa.min(sb) as u32,
        IntOp::MinU => a.min(b),
        IntOp::MaxS => sa.max(sb) as u32,
        IntOp::MaxU => a.max(b),
        IntOp::And => a & b,
        IntOp::Or => a | b,
        IntOp::Xor => a ^ b,
        IntOp::Shl => a << (b & 31),
        IntOp::ShrS => (sa >> (b & 31)) as u32,
        IntOp::ShrU => a >> (b & 31),
        IntOp::CmpLtS => u32::from(sa < sb),
        IntOp::CmpLtU => u32::from(a < b),
        IntOp::CmpEq => u32::from(a == b),
    }
}

#[inline(always)]
pub(crate) fn int_un_scalar(op: IntUnaryOp, a: u32) -> u32 {
    match op {
        IntUnaryOp::Neg => (a as i32).wrapping_neg() as u32,
        IntUnaryOp::Abs => (a as i32).wrapping_abs() as u32,
        IntUnaryOp::Not => !a,
    }
}

/// An operand: a register of lanes or a constant word.
#[derive(Clone, Copy)]
pub(crate) enum Words<'a> {
    Reg(&'a [f32]),
    Const(u32),
}

impl Words<'_> {
    #[inline(always)]
    fn at(&self, t: usize) -> u32 {
        match self {
            Words::Reg(r) => r[t].to_bits(),
            Words::Const(c) => *c,
        }
    }
}

pub(crate) fn binary(op: IntOp, d: &mut [f32], a: Words, b: Words) {
    macro_rules! go {
        ($f:expr) => {
            match (a, b) {
                (Words::Reg(x), Words::Reg(y)) => {
                    for ((o, x), y) in d.iter_mut().zip(x).zip(y) {
                        *o = f32::from_bits($f(x.to_bits(), y.to_bits()));
                    }
                }
                (Words::Reg(x), Words::Const(c)) => {
                    for (o, x) in d.iter_mut().zip(x) {
                        *o = f32::from_bits($f(x.to_bits(), c));
                    }
                }
                _ => {
                    for (t, o) in d.iter_mut().enumerate() {
                        *o = f32::from_bits($f(a.at(t), b.at(t)));
                    }
                }
            }
        };
    }
    match op {
        IntOp::Add => go!(|x: u32, y: u32| x.wrapping_add(y)),
        IntOp::Sub => go!(|x: u32, y: u32| x.wrapping_sub(y)),
        IntOp::Mul => go!(|x: u32, y: u32| x.wrapping_mul(y)),
        IntOp::And => go!(|x: u32, y: u32| x & y),
        IntOp::Or => go!(|x: u32, y: u32| x | y),
        IntOp::Xor => go!(|x: u32, y: u32| x ^ y),
        _ => go!(|x: u32, y: u32| int_bin_scalar(op, x, y)),
    }
}

pub(crate) fn unary(op: IntUnaryOp, d: &mut [f32], a: &[f32]) {
    for (o, x) in d.iter_mut().zip(a) {
        *o = f32::from_bits(int_un_scalar(op, x.to_bits()));
    }
}

pub(crate) fn convert(c: Conv, d: &mut [f32], a: &[f32]) {
    match c {
        Conv::I32ToF32 => {
            for (o, x) in d.iter_mut().zip(a) {
                *o = x.to_bits() as i32 as f32;
            }
        }
        Conv::F32ToI32 => {
            for (o, x) in d.iter_mut().zip(a) {
                *o = f32::from_bits(*x as i32 as u32);
            }
        }
        Conv::F32Bits | Conv::BitsF32 => d.copy_from_slice(a),
        _ => {
            for (o, x) in d.iter_mut().zip(a) {
                *o = f32::from_bits(conv_scalar(c, x.to_bits()));
            }
        }
    }
}

pub(crate) fn select(d: &mut [f32], c: Words, a: Words, b: Words) {
    for (t, o) in d.iter_mut().enumerate() {
        *o = f32::from_bits(if c.at(t) != 0 { a.at(t) } else { b.at(t) });
    }
}

/// Inputs to load as raw words: i32 inputs read by `LoadBits` (their `Load`s become
/// `I32ToF32` of the raw word). Empty when the expression has no integer nodes.
pub(crate) fn raw_inputs(e: &Expr, in_dtypes: &[DType]) -> Vec<bool> {
    let mut raw = vec![false; in_dtypes.len()];
    if !e.has_int() {
        return raw;
    }
    fn walk(e: &Expr, dts: &[DType], raw: &mut [bool]) {
        match e {
            Expr::LoadBits(i) => {
                if dts.get(*i) == Some(&DType::I32) {
                    raw[*i] = true;
                }
            }
            Expr::Load(_) | Expr::Const(_) | Expr::ConstBits(_) => {}
            Expr::Unary(_, a) | Expr::IntUnary(_, a) | Expr::Convert(_, a) => walk(a, dts, raw),
            Expr::Binary(_, a, b) | Expr::IntBinary(_, a, b) => {
                walk(a, dts, raw);
                walk(b, dts, raw);
            }
            Expr::Where(c, a, b) => {
                walk(c, dts, raw);
                walk(a, dts, raw);
                walk(b, dts, raw);
            }
        }
    }
    walk(e, in_dtypes, &mut raw);
    raw
}

/// `LoadBits` of f32 inputs is the lane itself: rewrite to `F32Bits(Load)` so the VM only
/// sees raw loads for i32 inputs.
pub(crate) fn normalize(e: &Expr, in_dtypes: &[DType]) -> Expr {
    let b = |x: &Expr| Box::new(normalize(x, in_dtypes));
    match e {
        Expr::LoadBits(i) if in_dtypes.get(*i) == Some(&DType::F32) => {
            Expr::Convert(Conv::F32Bits, Box::new(Expr::Load(*i)))
        }
        Expr::Load(_) | Expr::Const(_) | Expr::ConstBits(_) | Expr::LoadBits(_) => e.clone(),
        Expr::Unary(op, a) => Expr::Unary(*op, b(a)),
        Expr::IntUnary(op, a) => Expr::IntUnary(*op, b(a)),
        Expr::Convert(c, a) => Expr::Convert(*c, b(a)),
        Expr::Binary(op, x, y) => Expr::Binary(*op, b(x), b(y)),
        Expr::IntBinary(op, x, y) => Expr::IntBinary(*op, b(x), b(y)),
        Expr::Where(c, x, y) => Expr::Where(b(c), b(x), b(y)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{Program, Regs};

    fn w(i: usize) -> Box<Expr> {
        Box::new(Expr::LoadBits(i))
    }
    fn ib(op: IntOp, a: Box<Expr>, b: Box<Expr>) -> Box<Expr> {
        Box::new(Expr::IntBinary(op, a, b))
    }

    #[test]
    fn integer_ops_are_exact() {
        let ops = [
            IntOp::Add, IntOp::Sub, IntOp::Mul, IntOp::DivS, IntOp::DivU, IntOp::RemS, IntOp::RemU, IntOp::MinS,
            IntOp::MinU, IntOp::MaxS, IntOp::MaxU, IntOp::And, IntOp::Or, IntOp::Xor, IntOp::Shl, IntOp::ShrS,
            IntOp::ShrU, IntOp::CmpLtS, IntOp::CmpLtU, IntOp::CmpEq,
        ];
        let vals: Vec<i32> = vec![0, 1, -1, 7, -7, 1 << 24 | 1, i32::MAX, i32::MIN, 123_456_789, -98_765, 33, 3];
        for op in ops {
            let e = ib(op, w(0), w(1));
            let p = Program::compile_raw(&e, &[true, true]);
            let mut regs = Regs::default();
            p.prepare(&mut regs);
            let n = vals.len() * vals.len();
            let (mut xs, mut ys) = (Vec::new(), Vec::new());
            for &a in &vals {
                for &b in &vals {
                    xs.push(f32::from_bits(a as u32));
                    ys.push(f32::from_bits(b as u32));
                }
            }
            p.input(&mut regs, 0).unwrap()[..n].copy_from_slice(&xs);
            p.input(&mut regs, 1).unwrap()[..n].copy_from_slice(&ys);
            let out = p.run(&mut regs, n).to_vec();
            for t in 0..n {
                let want = int_bin_scalar(op, xs[t].to_bits(), ys[t].to_bits());
                assert_eq!(out[t].to_bits(), want, "{op:?} {} {}", xs[t].to_bits() as i32, ys[t].to_bits() as i32);
            }
        }
    }

    #[test]
    fn conversions_and_select() {
        // where(x < 3, -(x * 1000003), f32 -> i32 of (x as f32) * 0.5)
        let x = w(0);
        let e = Expr::Where(
            ib(IntOp::CmpLtS, x.clone(), Box::new(Expr::konst_int(3))),
            Box::new(Expr::IntUnary(IntUnaryOp::Neg, ib(IntOp::Mul, x.clone(), Box::new(Expr::konst_int(1_000_003))))),
            Box::new(Expr::Convert(
                Conv::F32ToI32,
                Box::new(Expr::Binary(
                    lapid_tensor::BinaryOp::Mul,
                    Box::new(Expr::Convert(Conv::I32ToF32, x)),
                    Box::new(Expr::konst(0.5)),
                )),
            )),
        );
        assert_eq!(e.ty(), Some(ExprType::Int));
        let p = Program::compile_raw(&e, &[true]);
        let mut regs = Regs::default();
        p.prepare(&mut regs);
        let vals: Vec<i32> = vec![-5, 0, 2, 3, 9, 1 << 25, -(1 << 26) - 3];
        let xs: Vec<f32> = vals.iter().map(|&v| f32::from_bits(v as u32)).collect();
        p.input(&mut regs, 0).unwrap()[..xs.len()].copy_from_slice(&xs);
        let out = p.run(&mut regs, xs.len()).to_vec();
        for (i, &v) in vals.iter().enumerate() {
            let want = if v < 3 { v.wrapping_mul(1_000_003).wrapping_neg() } else { ((v as f32) * 0.5) as i32 };
            assert_eq!(out[i].to_bits() as i32, want, "x = {v}");
        }
        // Numeric (non-raw) input: LoadBits converts the f32 value.
        let p = Program::compile(&Expr::IntBinary(IntOp::Add, w(0), Box::new(Expr::konst_int(1))));
        let mut regs = Regs::default();
        p.prepare(&mut regs);
        p.input(&mut regs, 0).unwrap()[..2].copy_from_slice(&[5.0, -3.0]);
        let out: Vec<i32> = p.run(&mut regs, 2).iter().map(|v| v.to_bits() as i32).collect();
        assert_eq!(out, [6, -2]);
    }
}

#[cfg(test)]
mod epilogue_tests {
    use super::*;
    use crate::expr::Regs;

    /// `LoadBits` of an f32 input (e.g. a matmul accumulator) is its bit pattern.
    #[test]
    fn f32_bits_of_numeric_input() {
        let e = Expr::Convert(
            Conv::BitsF32,
            Box::new(Expr::IntBinary(IntOp::And, Box::new(Expr::LoadBits(0)), Box::new(Expr::konst_int(!0xff)))),
        );
        let p = crate::expr::Program::compile_raw(&normalize(&e, &[DType::F32]), &[]);
        let mut regs = Regs::default();
        p.prepare(&mut regs);
        let xs = [1.234_567_f32, -3.5e7];
        p.input(&mut regs, 0).unwrap()[..2].copy_from_slice(&xs);
        let out = p.run(&mut regs, 2).to_vec();
        for (o, x) in out.iter().zip(xs) {
            assert_eq!(o.to_bits(), x.to_bits() & !0xff);
        }
    }
}
