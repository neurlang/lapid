//! Lazy elementwise expressions — the lowering's value representation.
//!
//! Every lowered node gets an `LExpr` over the node's shape. Leaves are strided views of
//! buffers (or of a *pending* matmul whose launch is deferred so its consumer can become
//! the epilogue); interior nodes are f32 math. A materialized node is a single `Leaf`,
//! a scalar broadcast is `Const`. Movement ops rewrite leaf views and never emit kernels.
//! Values of i32 nodes are typed integer expressions (`IntLeaf`, `ConstInt`, `IntUnary`,
//! `IntBinary`, `Convert` to int); see `ty` / `as_int` / `as_f32`.

use lapid_tensor::{BinaryOp, DType, Error, Result, UnaryOp};

use crate::ir::{BufId, Conv, Expr, ExprType, IntOp, IntUnaryOp, View};

/// Where a leaf reads from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Src {
    /// A buffer of the schedule.
    Buf(BufId),
    /// The (not yet emitted) output of `Lowerer::pendings[i]`, contiguous `[batch, m, n]`.
    Pending(usize),
}

/// A strided read of `src`, whose elements have type `dtype` (the buffer's storage dtype).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Leaf {
    pub src: Src,
    pub dtype: DType,
    pub view: View,
}

#[derive(Clone, Debug)]
pub(crate) enum LExpr {
    Leaf(Leaf),
    Const(f32),
    Unary(UnaryOp, Box<LExpr>),
    Binary(BinaryOp, Box<LExpr>, Box<LExpr>),
    Where(Box<LExpr>, Box<LExpr>, Box<LExpr>),
    /// Raw 32-bit word of an I32 leaf.
    IntLeaf(Leaf),
    ConstInt(i32),
    IntUnary(IntUnaryOp, Box<LExpr>),
    IntBinary(IntOp, Box<LExpr>, Box<LExpr>),
    Convert(Conv, Box<LExpr>),
}

impl LExpr {
    /// Value type (`Where` takes the type of its branches).
    pub fn ty(&self) -> ExprType {
        match self {
            LExpr::Leaf(_) | LExpr::Const(_) | LExpr::Unary(..) | LExpr::Binary(..) => ExprType::F32,
            LExpr::IntLeaf(_) | LExpr::ConstInt(_) | LExpr::IntUnary(..) | LExpr::IntBinary(..) => ExprType::Int,
            LExpr::Convert(c, _) => match c {
                Conv::F32ToI32 | Conv::F32ToU32 | Conv::F32Bits => ExprType::Int,
                _ => ExprType::F32,
            },
            LExpr::Where(_, a, _) => a.ty(),
        }
    }

    pub fn buf(buf: BufId, dtype: DType, view: View) -> LExpr {
        LExpr::Leaf(Leaf { src: Src::Buf(buf), dtype, view })
    }

    /// `Some((buf, dtype, view))` when the value is a plain view of a materialized buffer.
    pub fn as_buf(&self) -> Option<(BufId, DType, &View)> {
        match self {
            LExpr::Leaf(Leaf { src: Src::Buf(b), dtype, view }) => Some((*b, *dtype, view)),
            _ => None,
        }
    }

    /// Number of math nodes (leaves and constants are free).
    pub fn op_count(&self) -> usize {
        match self {
            LExpr::Leaf(_) | LExpr::Const(_) | LExpr::IntLeaf(_) | LExpr::ConstInt(_) => 0,
            LExpr::Unary(_, a) | LExpr::IntUnary(_, a) | LExpr::Convert(_, a) => 1 + a.op_count(),
            LExpr::Binary(_, a, b) | LExpr::IntBinary(_, a, b) => 1 + a.op_count() + b.op_count(),
            LExpr::Where(c, a, b) => 1 + c.op_count() + a.op_count() + b.op_count(),
        }
    }

    pub fn visit_leaves<'a>(&'a self, f: &mut dyn FnMut(&'a Leaf)) {
        match self {
            LExpr::Leaf(l) | LExpr::IntLeaf(l) => f(l),
            LExpr::Const(_) | LExpr::ConstInt(_) => {}
            LExpr::Unary(_, a) | LExpr::IntUnary(_, a) | LExpr::Convert(_, a) => a.visit_leaves(f),
            LExpr::Binary(_, a, b) | LExpr::IntBinary(_, a, b) => {
                a.visit_leaves(f);
                b.visit_leaves(f);
            }
            LExpr::Where(c, a, b) => {
                c.visit_leaves(f);
                a.visit_leaves(f);
                b.visit_leaves(f);
            }
        }
    }

    /// Number of distinct leaves (= kernel inputs after dedupe).
    pub fn distinct_leaves(&self) -> usize {
        if let LExpr::Leaf(_) = self {
            return 1;
        }
        let mut seen: Vec<&Leaf> = Vec::new();
        self.visit_leaves(&mut |l| {
            if !seen.contains(&l) {
                seen.push(l);
            }
        });
        seen.len()
    }

    /// Distinct pending-matmul ids referenced by the expression.
    pub fn pending_ids(&self) -> Vec<usize> {
        let mut ids = Vec::new();
        self.visit_leaves(&mut |l| {
            if let Src::Pending(p) = l.src
                && !ids.contains(&p) {
                    ids.push(p);
                }
        });
        ids
    }

    pub fn has_pending(&self) -> bool {
        match self {
            LExpr::Leaf(l) | LExpr::IntLeaf(l) => matches!(l.src, Src::Pending(_)),
            LExpr::Const(_) | LExpr::ConstInt(_) => false,
            LExpr::Unary(_, a) | LExpr::IntUnary(_, a) | LExpr::Convert(_, a) => a.has_pending(),
            LExpr::Binary(_, a, b) | LExpr::IntBinary(_, a, b) => a.has_pending() || b.has_pending(),
            LExpr::Where(c, a, b) => c.has_pending() || a.has_pending() || b.has_pending(),
        }
    }

    /// Rewrite every leaf; stops at the first error.
    pub fn map_leaves(&self, f: &mut dyn FnMut(&Leaf) -> Result<Leaf>) -> Result<LExpr> {
        Ok(match self {
            LExpr::Leaf(l) => LExpr::Leaf(f(l)?),
            LExpr::IntLeaf(l) => LExpr::IntLeaf(f(l)?),
            LExpr::Const(c) => LExpr::Const(*c),
            LExpr::ConstInt(c) => LExpr::ConstInt(*c),
            LExpr::Unary(op, a) => LExpr::Unary(*op, Box::new(a.map_leaves(f)?)),
            LExpr::IntUnary(op, a) => LExpr::IntUnary(*op, Box::new(a.map_leaves(f)?)),
            LExpr::Convert(c, a) => LExpr::Convert(*c, Box::new(a.map_leaves(f)?)),
            LExpr::IntBinary(op, a, b) => {
                let a = a.map_leaves(f)?;
                let b = b.map_leaves(f)?;
                LExpr::IntBinary(*op, Box::new(a), Box::new(b))
            }
            LExpr::Binary(op, a, b) => {
                let a = a.map_leaves(f)?;
                let b = b.map_leaves(f)?;
                LExpr::Binary(*op, Box::new(a), Box::new(b))
            }
            LExpr::Where(c, a, b) => {
                let c = c.map_leaves(f)?;
                let a = a.map_leaves(f)?;
                let b = b.map_leaves(f)?;
                LExpr::Where(Box::new(c), Box::new(a), Box::new(b))
            }
        })
    }

    /// Rewrite every leaf view; `None` if any view cannot be expressed.
    pub fn map_views(&self, f: &mut dyn FnMut(&View) -> Option<View>) -> Option<LExpr> {
        self.map_leaves(&mut |l| match f(&l.view) {
            Some(view) => Ok(Leaf { src: l.src, dtype: l.dtype, view }),
            None => Err(Error::new("internal", "view not expressible")),
        })
        .ok()
    }
}

// ------------------------------------------------------------------ builders (with folding)

fn is_const(e: &LExpr, v: f32) -> bool {
    matches!(e, LExpr::Const(c) if *c == v)
}

pub(crate) fn unary(op: UnaryOp, a: LExpr) -> LExpr {
    if let LExpr::Const(x) = &a
        && let Some(v) = fold_unary(op, *x) {
            return LExpr::Const(v);
        }
    LExpr::Unary(op, Box::new(a))
}

pub(crate) fn binary(op: BinaryOp, a: LExpr, b: LExpr) -> LExpr {
    if let (LExpr::Const(x), LExpr::Const(y)) = (&a, &b) {
        return LExpr::Const(fold_binary(op, *x, *y));
    }
    // Identities that are exact in f32.
    match op {
        BinaryOp::Add if is_const(&b, 0.0) => return a,
        BinaryOp::Add if is_const(&a, 0.0) => return b,
        BinaryOp::Sub if is_const(&b, 0.0) => return a,
        BinaryOp::Mul if is_const(&b, 1.0) => return a,
        BinaryOp::Mul if is_const(&a, 1.0) => return b,
        BinaryOp::Div if is_const(&b, 1.0) => return a,
        _ => {}
    }
    LExpr::Binary(op, Box::new(a), Box::new(b))
}

pub(crate) fn where_(c: LExpr, a: LExpr, b: LExpr) -> LExpr {
    if let LExpr::Const(x) = &c {
        return if *x != 0.0 || x.is_nan() { a } else { b };
    }
    if let LExpr::ConstInt(x) = &c {
        return if *x != 0 { a } else { b };
    }
    LExpr::Where(Box::new(c), Box::new(a), Box::new(b))
}

/// Host-side constant folding. `Erf` is left to the backend (no std implementation).
pub(crate) fn fold_unary(op: UnaryOp, x: f32) -> Option<f32> {
    Some(match op {
        UnaryOp::Neg => -x,
        UnaryOp::Exp => x.exp(),
        UnaryOp::Log => x.ln(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Rsqrt => 1.0 / x.sqrt(),
        UnaryOp::Recip => 1.0 / x,
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Abs => x.abs(),
        UnaryOp::Erf => return None,
    })
}

pub(crate) fn fold_binary(op: BinaryOp, a: f32, b: f32) -> f32 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Max => a.max(b),
        BinaryOp::Min => a.min(b),
        BinaryOp::Pow => a.powf(b),
        BinaryOp::CmpLt => f32::from(u8::from(a < b)),
        BinaryOp::CmpEq => f32::from(u8::from(a == b)),
        // Integer-only ops never reach the f32 path (lowered through `int_binary`).
        BinaryOp::Rem | BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor | BinaryOp::Shl | BinaryOp::Shr => {
            f32::NAN
        }
    }
}

// ------------------------------------------------------------------ integer values

/// The value as an integer word: I32 leaves are read raw, f32 values are converted
/// (truncating, saturating — the store rule of f32 → i32).
pub(crate) fn as_int(e: LExpr) -> LExpr {
    match e {
        LExpr::Leaf(l) if l.dtype == DType::I32 => LExpr::IntLeaf(l),
        LExpr::Const(c) => LExpr::ConstInt(c as i32),
        LExpr::Convert(Conv::I32ToF32, a) => *a,
        e if e.ty() == ExprType::Int => e,
        e => LExpr::Convert(Conv::F32ToI32, Box::new(e)),
    }
}

/// The value as f32 (integer words converted numerically).
pub(crate) fn as_f32(e: LExpr) -> LExpr {
    match e {
        LExpr::IntLeaf(l) => LExpr::Leaf(l),
        LExpr::ConstInt(c) => LExpr::Const(c as f32),
        e if e.ty() == ExprType::Int => LExpr::Convert(Conv::I32ToF32, Box::new(e)),
        e => e,
    }
}

/// The signed i32 op implementing `op` on i32 tensors (`None`: no integer form, e.g. Pow).
pub(crate) fn int_op(op: BinaryOp) -> Option<IntOp> {
    Some(match op {
        BinaryOp::Add => IntOp::Add,
        BinaryOp::Sub => IntOp::Sub,
        BinaryOp::Mul => IntOp::Mul,
        BinaryOp::Div => IntOp::DivS,
        BinaryOp::Max => IntOp::MaxS,
        BinaryOp::Min => IntOp::MinS,
        BinaryOp::CmpLt => IntOp::CmpLtS,
        BinaryOp::CmpEq => IntOp::CmpEq,
        BinaryOp::Rem => IntOp::RemS,
        BinaryOp::BitAnd => IntOp::And,
        BinaryOp::BitOr => IntOp::Or,
        BinaryOp::BitXor => IntOp::Xor,
        BinaryOp::Shl => IntOp::Shl,
        BinaryOp::Shr => IntOp::ShrS,
        BinaryOp::Pow => return None,
    })
}

pub(crate) fn int_binary(op: IntOp, a: LExpr, b: LExpr) -> LExpr {
    let (a, b) = (as_int(a), as_int(b));
    if let (LExpr::ConstInt(x), LExpr::ConstInt(y)) = (&a, &b)
        && let Some(v) = fold_int(op, *x, *y) {
            return LExpr::ConstInt(v);
        }
    match op {
        IntOp::Add | IntOp::Sub | IntOp::Or | IntOp::Xor | IntOp::Shl | IntOp::ShrS | IntOp::ShrU
            if matches!(b, LExpr::ConstInt(0)) =>
        {
            return a;
        }
        IntOp::Mul | IntOp::DivS | IntOp::DivU if matches!(b, LExpr::ConstInt(1)) => return a,
        _ => {}
    }
    LExpr::IntBinary(op, Box::new(a), Box::new(b))
}

pub(crate) fn int_unary(op: IntUnaryOp, a: LExpr) -> LExpr {
    let a = as_int(a);
    if let LExpr::ConstInt(x) = a {
        return LExpr::ConstInt(match op {
            IntUnaryOp::Neg => x.wrapping_neg(),
            IntUnaryOp::Abs => x.wrapping_abs(),
            IntUnaryOp::Not => !x,
        });
    }
    LExpr::IntUnary(op, Box::new(a))
}

/// Host folding of integer ops (`None` where the result is unspecified: division by zero).
pub(crate) fn fold_int(op: IntOp, a: i32, b: i32) -> Option<i32> {
    let (ua, ub) = (a as u32, b as u32);
    Some(match op {
        IntOp::Add => a.wrapping_add(b),
        IntOp::Sub => a.wrapping_sub(b),
        IntOp::Mul => a.wrapping_mul(b),
        IntOp::DivS => a.checked_div(b)?,
        IntOp::DivU => ua.checked_div(ub)? as i32,
        IntOp::RemS => a.checked_rem(b)?,
        IntOp::RemU => ua.checked_rem(ub)? as i32,
        IntOp::MinS => a.min(b),
        IntOp::MinU => ua.min(ub) as i32,
        IntOp::MaxS => a.max(b),
        IntOp::MaxU => ua.max(ub) as i32,
        IntOp::And => a & b,
        IntOp::Or => a | b,
        IntOp::Xor => a ^ b,
        IntOp::Shl => a.wrapping_shl(ub & 31),
        IntOp::ShrS => a.wrapping_shr(ub & 31),
        IntOp::ShrU => ua.wrapping_shr(ub & 31) as i32,
        IntOp::CmpLtS => i32::from(a < b),
        IntOp::CmpLtU => i32::from(ua < ub),
        IntOp::CmpEq => i32::from(a == b),
    })
}

// ------------------------------------------------------------------ LExpr -> kernel Expr

/// Collects the distinct buffer loads of an expression while converting it to `Expr`.
/// Load indices start at `base`; leaves of pending matmul `acc` become `Load(0)` (the
/// matmul accumulator in an epilogue).
pub(crate) struct Loads {
    pub bufs: Vec<BufId>,
    pub dtypes: Vec<DType>,
    pub views: Vec<View>,
    base: usize,
    acc: Option<usize>,
}

impl Loads {
    pub fn new(base: usize, acc: Option<usize>) -> Loads {
        Loads { bufs: Vec::new(), dtypes: Vec::new(), views: Vec::new(), base, acc }
    }

    pub fn expr(&mut self, e: &LExpr) -> Result<Expr> {
        Ok(match e {
            LExpr::Leaf(l) => self.load(l)?,
            LExpr::IntLeaf(l) => match self.load(l)? {
                Expr::Load(i) => Expr::LoadBits(i),
                _ => return Err(Error::new("internal", "integer read of a matmul accumulator")),
            },
            LExpr::Const(c) => Expr::konst(*c),
            LExpr::ConstInt(c) => Expr::konst_int(*c),
            LExpr::Unary(op, a) => Expr::Unary(*op, Box::new(self.expr(a)?)),
            LExpr::IntUnary(op, a) => Expr::IntUnary(*op, Box::new(self.expr(a)?)),
            LExpr::Convert(c, a) => Expr::Convert(*c, Box::new(self.expr(a)?)),
            LExpr::IntBinary(op, a, b) => {
                let a = self.expr(a)?;
                let b = self.expr(b)?;
                Expr::IntBinary(*op, Box::new(a), Box::new(b))
            }
            LExpr::Binary(op, a, b) => {
                let a = self.expr(a)?;
                let b = self.expr(b)?;
                Expr::Binary(*op, Box::new(a), Box::new(b))
            }
            LExpr::Where(c, a, b) => {
                let c = self.expr(c)?;
                let a = self.expr(a)?;
                let b = self.expr(b)?;
                Expr::Where(Box::new(c), Box::new(a), Box::new(b))
            }
        })
    }

    fn load(&mut self, l: &Leaf) -> Result<Expr> {
        match l.src {
            Src::Pending(p) if Some(p) == self.acc => Ok(Expr::Load(0)),
            Src::Pending(p) => Err(Error::new(
                "internal",
                format!("pending matmul #{p} reached a kernel expression unflushed"),
            )),
            Src::Buf(b) => {
                if l.dtype.is_quant() {
                    return Err(Error::new(
                        "unsupported",
                        format!(
                            "quantized ({}) tensors can only feed MatMul (as b) or Gather (as table)",
                            l.dtype.name()
                        ),
                    ));
                }
                for i in 0..self.bufs.len() {
                    if self.bufs[i] == b && self.dtypes[i] == l.dtype && self.views[i] == l.view {
                        return Ok(Expr::Load(self.base + i));
                    }
                }
                self.bufs.push(b);
                self.dtypes.push(l.dtype);
                self.views.push(l.view.clone());
                Ok(Expr::Load(self.base + self.bufs.len() - 1))
            }
        }
    }
}
