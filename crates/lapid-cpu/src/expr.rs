//! Expression compiler: `Expr` tree -> flat register bytecode evaluated over f32 chunks.
//!
//! Compilation does common-subexpression elimination, constant folding, constant operands
//! (no register fills for `x * 0.5`), strength reduction of `pow` with constant exponents and
//! register reuse by liveness. Each instruction runs a tight, auto-vectorized loop over up to
//! `CHUNK` elements, so dispatch overhead is amortized over the chunk.

use std::collections::HashMap;

use lapid_compiler::{Conv, Expr, ExprType, IntOp, IntUnaryOp};
use lapid_tensor::{BinaryOp, UnaryOp};

use crate::expr_int::{self, conv_scalar, int_bin_scalar, int_un_scalar};
use crate::math;

/// Elements per evaluation chunk.
pub(crate) const CHUNK: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Opd {
    R(u16),
    C(f32),
}

#[derive(Clone, Debug)]
enum Inst {
    Fill(u16, f32),
    Un(UnaryOp, u16, u16),
    Bin(BinaryOp, u16, Opd, Opd),
    /// dst = a ^ c for a constant exponent c.
    PowC(u16, u16, f32),
    Sel(u16, Opd, Opd, Opd),
    /// Select with an integer-word condition.
    SelI(u16, Opd, Opd, Opd),
    IBin(IntOp, u16, Opd, Opd),
    IUn(IntUnaryOp, u16, u16),
    Conv(Conv, u16, u16),
}

/// Compiled expression.
#[derive(Debug)]
pub(crate) struct Program {
    code: Vec<Inst>,
    n_regs: usize,
    /// Register holding expression input `i`, `None` if the input is unused.
    inputs: Vec<Option<u16>>,
    result: u16,
    /// `Some(i)` when the whole expression is `Load(i)`.
    identity: Option<usize>,
    uses_erf: bool,
}

/// Per-thread register file for a program.
#[derive(Default)]
pub(crate) struct Regs {
    buf: Vec<f32>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Node {
    Load(usize),
    Const(u32),
    Un(UnaryOp, usize),
    Bin(BinaryOp, usize, usize),
    Sel(usize, usize, usize),
    SelI(usize, usize, usize),
    IBin(IntOp, usize, usize),
    IUn(IntUnaryOp, usize),
    Conv(Conv, usize),
}

impl Node {
    /// Operand node ids.
    fn operands(&self) -> ([usize; 3], usize) {
        match *self {
            Node::Load(_) | Node::Const(_) => ([0; 3], 0),
            Node::Un(_, a) | Node::IUn(_, a) | Node::Conv(_, a) => ([a, 0, 0], 1),
            Node::Bin(_, a, b) | Node::IBin(_, a, b) => ([a, b, 0], 2),
            Node::Sel(c, a, b) | Node::SelI(c, a, b) => ([c, a, b], 3),
        }
    }
}

struct Builder {
    nodes: Vec<Node>,
    map: HashMap<Node, usize>,
    /// Value type per node (integer words live in f32 lanes as bit patterns).
    tys: Vec<ExprType>,
    /// Inputs loaded as raw words (`LoadBits`) instead of numeric f32.
    raw: Vec<bool>,
}

pub(crate) fn unary_scalar(op: UnaryOp, a: f32) -> f32 {
    match op {
        UnaryOp::Neg => -a,
        UnaryOp::Exp => math::exp(a),
        UnaryOp::Log => a.ln(),
        UnaryOp::Sqrt => a.sqrt(),
        UnaryOp::Rsqrt => 1.0 / a.sqrt(),
        UnaryOp::Recip => 1.0 / a,
        UnaryOp::Sin => a.sin(),
        UnaryOp::Cos => a.cos(),
        UnaryOp::Tanh => a.tanh(),
        UnaryOp::Erf => math::erf(a),
        UnaryOp::Abs => a.abs(),
    }
}

#[inline(always)]
pub(crate) fn binary_scalar(op: BinaryOp, a: f32, b: f32) -> f32 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Max => a.max(b),
        BinaryOp::Min => a.min(b),
        BinaryOp::Pow => a.powf(b),
        BinaryOp::CmpLt => {
            if a < b {
                1.0
            } else {
                0.0
            }
        }
        BinaryOp::CmpEq => {
            if a == b {
                1.0
            } else {
                0.0
            }
        }
        // Integer-only ops: the compiler lowers them to `IntBinary`.
        BinaryOp::Rem | BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor | BinaryOp::Shl | BinaryOp::Shr => f32::NAN,
    }
}

impl Builder {
    fn intern(&mut self, n: Node, ty: ExprType) -> usize {
        if let Some(&id) = self.map.get(&n) {
            return id;
        }
        let id = self.nodes.len();
        self.nodes.push(n);
        self.tys.push(ty);
        self.map.insert(n, id);
        id
    }

    fn konst(&self, id: usize) -> Option<f32> {
        match self.nodes[id] {
            Node::Const(b) => Some(f32::from_bits(b)),
            _ => None,
        }
    }

    fn kbits(&self, id: usize) -> Option<u32> {
        match self.nodes[id] {
            Node::Const(b) => Some(b),
            _ => None,
        }
    }

    fn is_raw(&self, i: usize) -> bool {
        self.raw.get(i).copied().unwrap_or(false)
    }

    fn conv(&mut self, c: Conv, a: usize) -> usize {
        let ty = expr_int::conv_result(c);
        if let Some(b) = self.kbits(a) {
            return self.intern(Node::Const(conv_scalar(c, b)), ty);
        }
        self.intern(Node::Conv(c, a), ty)
    }

    fn build(&mut self, e: &Expr) -> usize {
        use ExprType::{F32, Int};
        match e {
            Expr::Load(i) if self.is_raw(*i) => {
                let w = self.intern(Node::Load(*i), Int);
                self.conv(Conv::I32ToF32, w)
            }
            Expr::Load(i) => self.intern(Node::Load(*i), F32),
            Expr::LoadBits(i) if self.is_raw(*i) => self.intern(Node::Load(*i), Int),
            // Numerically loaded i32 input (matmul epilogues, reduce prologues).
            Expr::LoadBits(i) => {
                let x = self.intern(Node::Load(*i), F32);
                self.conv(Conv::F32ToI32, x)
            }
            Expr::Const(b) => self.intern(Node::Const(*b), F32),
            Expr::ConstBits(b) => self.intern(Node::Const(*b), Int),
            Expr::Unary(op, a) => {
                let a = self.build(a);
                if let Some(c) = self.konst(a) {
                    return self.intern(Node::Const(unary_scalar(*op, c).to_bits()), F32);
                }
                self.intern(Node::Un(*op, a), F32)
            }
            Expr::Binary(op, a, b) => {
                let a = self.build(a);
                let b = self.build(b);
                if let (Some(x), Some(y)) = (self.konst(a), self.konst(b)) {
                    return self.intern(Node::Const(binary_scalar(*op, x, y).to_bits()), F32);
                }
                self.intern(Node::Bin(*op, a, b), F32)
            }
            Expr::IntUnary(op, a) => {
                let a = self.build(a);
                if let Some(x) = self.kbits(a) {
                    return self.intern(Node::Const(int_un_scalar(*op, x)), Int);
                }
                self.intern(Node::IUn(*op, a), Int)
            }
            Expr::IntBinary(op, a, b) => {
                let a = self.build(a);
                let b = self.build(b);
                if let (Some(x), Some(y)) = (self.kbits(a), self.kbits(b)) {
                    return self.intern(Node::Const(int_bin_scalar(*op, x, y)), Int);
                }
                self.intern(Node::IBin(*op, a, b), Int)
            }
            Expr::Convert(c, a) => {
                let a = self.build(a);
                self.conv(*c, a)
            }
            Expr::Where(c, a, b) => {
                let c = self.build(c);
                let a = self.build(a);
                let b = self.build(b);
                let int_cond = self.tys[c] == Int;
                if let Some(cv) = self.kbits(c) {
                    let t = if int_cond { cv != 0 } else { f32::from_bits(cv) != 0.0 || f32::from_bits(cv).is_nan() };
                    return if t { a } else { b };
                }
                if a == b {
                    return a;
                }
                let ty = self.tys[a];
                if int_cond { self.intern(Node::SelI(c, a, b), ty) } else { self.intern(Node::Sel(c, a, b), ty) }
            }
        }
    }
}

impl Program {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn compile(expr: &Expr) -> Program {
        Program::compile_raw(expr, &[])
    }

    /// `raw[i]`: input `i` is filled with raw 32-bit words (i32 storage read bit-exactly)
    /// instead of numeric f32 values.
    pub(crate) fn compile_raw(expr: &Expr, raw: &[bool]) -> Program {
        let mut b = Builder { nodes: Vec::new(), map: HashMap::new(), tys: Vec::new(), raw: raw.to_vec() };
        let root = b.build(expr);
        let nodes = b.nodes;
        let n_inputs = expr.num_loads();

        let mut uses = vec![0usize; nodes.len()];
        uses[root] += 1;
        for n in &nodes {
            let (ops, k) = n.operands();
            for &o in &ops[..k] {
                uses[o] += 1;
            }
        }
        // Liveness must only count uses from live nodes; dead nodes (folded away) are
        // never emitted, so subtract their contributions.
        let mut live = vec![false; nodes.len()];
        live[root] = true;
        for id in (0..nodes.len()).rev() {
            if !live[id] {
                continue;
            }
            let (ops, k) = nodes[id].operands();
            for &o in &ops[..k] {
                live[o] = true;
            }
        }
        for (id, n) in nodes.iter().enumerate() {
            if live[id] {
                continue;
            }
            let (ops, k) = n.operands();
            for &o in &ops[..k] {
                uses[o] -= 1;
            }
        }

        let mut n_regs = 0usize;
        fn new_reg(n_regs: &mut usize) -> u16 {
            let r = *n_regs as u16;
            *n_regs += 1;
            r
        }
        // Inputs get dedicated registers (filled by the caller before `run`).
        let mut inputs = vec![None; n_inputs];
        let mut reg_of: Vec<Option<u16>> = vec![None; nodes.len()];
        for (id, n) in nodes.iter().enumerate() {
            if let Node::Load(i) = *n
                && live[id] {
                    let r = new_reg(&mut n_regs);
                    inputs[i] = Some(r);
                    reg_of[id] = Some(r);
                }
        }
        let identity = match nodes[root] {
            Node::Load(i) => Some(i),
            _ => None,
        };

        let mut free: Vec<u16> = Vec::new();
        let mut code = Vec::new();
        let mut uses_erf = false;
        let opd = |id: usize, reg_of: &[Option<u16>]| -> Opd {
            match nodes[id] {
                Node::Const(b) => Opd::C(f32::from_bits(b)),
                _ => Opd::R(reg_of[id].unwrap_or(0)),
            }
        };
        let release = |id: usize, uses: &mut [usize], reg_of: &[Option<u16>], free: &mut Vec<u16>| {
            uses[id] -= 1;
            if uses[id] == 0 && !matches!(nodes[id], Node::Load(_) | Node::Const(_))
                && let Some(r) = reg_of[id] {
                    free.push(r);
                }
        };

        for id in 0..nodes.len() {
            if !live[id] {
                continue;
            }
            let node = nodes[id];
            if matches!(node, Node::Load(_) | Node::Const(_)) {
                continue;
            }
            let dst = free.pop().unwrap_or_else(|| new_reg(&mut n_regs));
            reg_of[id] = Some(dst);
            match node {
                Node::Un(op, a) => {
                    uses_erf |= op == UnaryOp::Erf;
                    let ra = reg_of[a].unwrap_or(0);
                    code.push(Inst::Un(op, dst, ra));
                    release(a, &mut uses, &reg_of, &mut free);
                }
                Node::Bin(op, a, c) => {
                    let (oa, oc) = (opd(a, &reg_of), opd(c, &reg_of));
                    match (op, oa, oc) {
                        (BinaryOp::Pow, Opd::R(ra), Opd::C(e)) => code.push(Inst::PowC(dst, ra, e)),
                        _ => code.push(Inst::Bin(op, dst, oa, oc)),
                    }
                    release(a, &mut uses, &reg_of, &mut free);
                    release(c, &mut uses, &reg_of, &mut free);
                }
                Node::Sel(c, x, y) | Node::SelI(c, x, y) => {
                    let (oc, ox, oy) = (opd(c, &reg_of), opd(x, &reg_of), opd(y, &reg_of));
                    code.push(if matches!(node, Node::SelI(..)) {
                        Inst::SelI(dst, oc, ox, oy)
                    } else {
                        Inst::Sel(dst, oc, ox, oy)
                    });
                    release(c, &mut uses, &reg_of, &mut free);
                    release(x, &mut uses, &reg_of, &mut free);
                    release(y, &mut uses, &reg_of, &mut free);
                }
                Node::IBin(op, a, c) => {
                    code.push(Inst::IBin(op, dst, opd(a, &reg_of), opd(c, &reg_of)));
                    release(a, &mut uses, &reg_of, &mut free);
                    release(c, &mut uses, &reg_of, &mut free);
                }
                Node::IUn(op, a) => {
                    code.push(Inst::IUn(op, dst, reg_of[a].unwrap_or(0)));
                    release(a, &mut uses, &reg_of, &mut free);
                }
                Node::Conv(c, a) => {
                    code.push(Inst::Conv(c, dst, reg_of[a].unwrap_or(0)));
                    release(a, &mut uses, &reg_of, &mut free);
                }
                Node::Load(_) | Node::Const(_) => {}
            }
        }

        let result = match nodes[root] {
            Node::Const(b) => {
                let r = new_reg(&mut n_regs);
                code.push(Inst::Fill(r, f32::from_bits(b)));
                r
            }
            _ => reg_of[root].unwrap_or(0),
        };
        Program { code, n_regs: n_regs.max(1), inputs, result, identity, uses_erf }
    }

    /// Number of expression inputs (`Load` indices) the program was compiled for.
    pub(crate) fn n_inputs(&self) -> usize {
        self.inputs.len()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn uses_input(&self, i: usize) -> bool {
        self.inputs.get(i).is_some_and(|r| r.is_some())
    }

    pub(crate) fn identity(&self) -> Option<usize> {
        self.identity
    }

    pub(crate) fn prepare(&self, regs: &mut Regs) {
        let need = self.n_regs * CHUNK;
        if regs.buf.len() < need {
            regs.buf.resize(need, 0.0);
        }
    }

    /// Register slot of input `i` (length `CHUNK`), or None if the input is unused.
    /// `prepare` must have been called.
    pub(crate) fn input<'r>(&self, regs: &'r mut Regs, i: usize) -> Option<&'r mut [f32]> {
        let r = (*self.inputs.get(i)?)? as usize;
        Some(&mut regs.buf[r * CHUNK..(r + 1) * CHUNK])
    }

    /// Evaluate over the first `len` (<= CHUNK) elements; returns the result slice.
    pub(crate) fn run<'r>(&self, regs: &'r mut Regs, len: usize) -> &'r [f32] {
        assert!(len <= CHUNK && regs.buf.len() >= self.n_regs * CHUNK, "cpu: bad program evaluation");
        let erf = if self.uses_erf { Some(math::erf_table()) } else { None };
        let base = regs.buf.as_mut_ptr();
        // SAFETY: the compiler never assigns an instruction's destination register to any of
        // its operands (dst is allocated before operands are released), so the mutable dst
        // slice never aliases the shared operand slices. All registers are in bounds.
        let reg = |r: u16| unsafe { std::slice::from_raw_parts(base.add(r as usize * CHUNK), len) };
        let reg_mut = |r: u16| unsafe { std::slice::from_raw_parts_mut(base.add(r as usize * CHUNK), len) };
        for inst in &self.code {
            match *inst {
                Inst::Fill(d, c) => reg_mut(d).fill(c),
                Inst::Un(op, d, a) => unary(op, reg_mut(d), reg(a), erf),
                Inst::PowC(d, a, e) => pow_const(reg_mut(d), reg(a), e),
                Inst::Bin(op, d, a, b) => match (a, b) {
                    (Opd::R(a), Opd::R(b)) => binary_rr(op, reg_mut(d), reg(a), reg(b)),
                    (Opd::R(a), Opd::C(c)) => binary_rc(op, reg_mut(d), reg(a), c),
                    (Opd::C(c), Opd::R(b)) => binary_cr(op, reg_mut(d), c, reg(b)),
                    (Opd::C(x), Opd::C(y)) => reg_mut(d).fill(binary_scalar(op, x, y)),
                },
                Inst::Sel(d, c, a, b) => {
                    let split = |o: Opd| match o {
                        Opd::R(r) => (Some(reg(r)), 0.0),
                        Opd::C(v) => (None, v),
                    };
                    let ((cs, cv), (as_, av), (bs, bv)) = (split(c), split(a), split(b));
                    let dst = reg_mut(d);
                    for (t, o) in dst.iter_mut().enumerate() {
                        let cond = cs.map_or(cv, |s| s[t]);
                        *o = if cond != 0.0 || cond.is_nan() { as_.map_or(av, |s| s[t]) } else { bs.map_or(bv, |s| s[t]) };
                    }
                }
                Inst::SelI(d, c, a, b) => {
                    let w = |o: Opd| match o {
                        Opd::R(r) => expr_int::Words::Reg(reg(r)),
                        Opd::C(v) => expr_int::Words::Const(v.to_bits()),
                    };
                    expr_int::select(reg_mut(d), w(c), w(a), w(b));
                }
                Inst::IBin(op, d, a, b) => {
                    let w = |o: Opd| match o {
                        Opd::R(r) => expr_int::Words::Reg(reg(r)),
                        Opd::C(v) => expr_int::Words::Const(v.to_bits()),
                    };
                    expr_int::binary(op, reg_mut(d), w(a), w(b));
                }
                Inst::IUn(op, d, a) => expr_int::unary(op, reg_mut(d), reg(a)),
                Inst::Conv(c, d, a) => expr_int::convert(c, reg_mut(d), reg(a)),
            }
        }
        // SAFETY: result register in bounds; no other borrow is alive.
        unsafe { std::slice::from_raw_parts(base.add(self.result as usize * CHUNK), len) }
    }
}

fn unary(op: UnaryOp, d: &mut [f32], a: &[f32], erf: Option<&math::ErfTable>) {
    macro_rules! map {
        (|$x:ident| $body:expr) => {
            for (o, &$x) in d.iter_mut().zip(a) {
                *o = $body;
            }
        };
    }
    match op {
        UnaryOp::Neg => map!(|x| -x),
        UnaryOp::Exp => map!(|x| math::exp(x)),
        UnaryOp::Log => map!(|x| x.ln()),
        UnaryOp::Sqrt => map!(|x| x.sqrt()),
        UnaryOp::Rsqrt => map!(|x| 1.0 / x.sqrt()),
        UnaryOp::Recip => map!(|x| 1.0 / x),
        UnaryOp::Sin => map!(|x| x.sin()),
        UnaryOp::Cos => map!(|x| x.cos()),
        UnaryOp::Tanh => map!(|x| x.tanh()),
        UnaryOp::Abs => map!(|x| x.abs()),
        UnaryOp::Erf => {
            let t: &math::ErfTable = match erf {
                Some(t) => t,
                None => math::erf_table(),
            };
            map!(|x| t.erf(x))
        }
    }
}

fn pow_const(d: &mut [f32], a: &[f32], e: f32) {
    macro_rules! map {
        (|$x:ident| $body:expr) => {
            for (o, &$x) in d.iter_mut().zip(a) {
                *o = $body;
            }
        };
    }
    if e == 1.0 {
        d.copy_from_slice(a)
    } else if e == 2.0 {
        map!(|x| x * x)
    } else if e == 3.0 {
        map!(|x| x * x * x)
    } else if e == 0.5 {
        map!(|x| x.sqrt())
    } else if e == -1.0 {
        map!(|x| 1.0 / x)
    } else if e == -0.5 {
        map!(|x| 1.0 / x.sqrt())
    } else if e == 0.0 {
        d.fill(1.0)
    } else {
        map!(|x| x.powf(e))
    }
}

macro_rules! bin_match {
    ($op:expr, $f:ident, $($arg:expr),*) => {
        match $op {
            BinaryOp::Add => $f($($arg,)* |x: f32, y: f32| x + y),
            BinaryOp::Sub => $f($($arg,)* |x: f32, y: f32| x - y),
            BinaryOp::Mul => $f($($arg,)* |x: f32, y: f32| x * y),
            BinaryOp::Div => $f($($arg,)* |x: f32, y: f32| x / y),
            BinaryOp::Max => $f($($arg,)* |x: f32, y: f32| x.max(y)),
            BinaryOp::Min => $f($($arg,)* |x: f32, y: f32| x.min(y)),
            BinaryOp::Pow => $f($($arg,)* |x: f32, y: f32| x.powf(y)),
            BinaryOp::CmpLt => $f($($arg,)* |x: f32, y: f32| if x < y { 1.0 } else { 0.0 }),
            BinaryOp::CmpEq => $f($($arg,)* |x: f32, y: f32| if x == y { 1.0 } else { 0.0 }),
            BinaryOp::Rem | BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor | BinaryOp::Shl | BinaryOp::Shr => {
                $f($($arg,)* |_: f32, _: f32| f32::NAN)
            }
        }
    };
}

#[inline(always)]
fn zip_rr(d: &mut [f32], a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> f32) {
    for ((o, &x), &y) in d.iter_mut().zip(a).zip(b) {
        *o = f(x, y);
    }
}

#[inline(always)]
fn zip_rc(d: &mut [f32], a: &[f32], c: f32, f: impl Fn(f32, f32) -> f32) {
    for (o, &x) in d.iter_mut().zip(a) {
        *o = f(x, c);
    }
}

#[inline(always)]
fn zip_cr(d: &mut [f32], c: f32, b: &[f32], f: impl Fn(f32, f32) -> f32) {
    for (o, &y) in d.iter_mut().zip(b) {
        *o = f(c, y);
    }
}

fn binary_rr(op: BinaryOp, d: &mut [f32], a: &[f32], b: &[f32]) {
    bin_match!(op, zip_rr, d, a, b)
}

fn binary_rc(op: BinaryOp, d: &mut [f32], a: &[f32], c: f32) {
    bin_match!(op, zip_rc, d, a, c)
}

fn binary_cr(op: BinaryOp, d: &mut [f32], c: f32, b: &[f32]) {
    bin_match!(op, zip_cr, d, c, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(i: usize) -> Box<Expr> {
        Box::new(Expr::Load(i))
    }
    fn k(v: f32) -> Box<Expr> {
        Box::new(Expr::konst(v))
    }

    fn eval_ref(e: &Expr, ins: &[f32]) -> f32 {
        match e {
            Expr::Load(i) => ins[*i],
            Expr::Const(b) => f32::from_bits(*b),
            Expr::Unary(op, a) => unary_scalar(*op, eval_ref(a, ins)),
            Expr::Binary(op, a, b) => binary_scalar(*op, eval_ref(a, ins), eval_ref(b, ins)),
            Expr::Where(c, a, b) => {
                if eval_ref(c, ins) != 0.0 {
                    eval_ref(a, ins)
                } else {
                    eval_ref(b, ins)
                }
            }
            _ => unreachable!("integer nodes are tested in expr_int"),
        }
    }

    fn check(e: &Expr, n_in: usize) {
        let p = Program::compile(e);
        let mut regs = Regs::default();
        p.prepare(&mut regs);
        let len = 300;
        let inputs: Vec<Vec<f32>> =
            (0..n_in).map(|i| (0..len).map(|t| ((t * 7 + i * 13) % 23) as f32 * 0.25 - 2.0).collect()).collect();
        for (i, v) in inputs.iter().enumerate() {
            if let Some(r) = p.input(&mut regs, i) {
                r[..len].copy_from_slice(v);
            }
        }
        let out = p.run(&mut regs, len).to_vec();
        for t in 0..len {
            let ins: Vec<f32> = inputs.iter().map(|v| v[t]).collect();
            let want = eval_ref(e, &ins);
            let got = out[t];
            assert!(
                (got - want).abs() <= 1e-5 * want.abs().max(1.0) || (got.is_nan() && want.is_nan()),
                "t={t}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn gelu_and_friends() {
        // x * 0.5 * (1 + erf(x / sqrt(2)))
        let x = load(0);
        let lapid = Expr::Binary(
            BinaryOp::Mul,
            Box::new(Expr::Binary(BinaryOp::Mul, x.clone(), k(0.5))),
            Box::new(Expr::Binary(
                BinaryOp::Add,
                k(1.0),
                Box::new(Expr::Unary(UnaryOp::Erf, Box::new(Expr::Binary(BinaryOp::Div, x, k(2f32.sqrt()))))),
            )),
        );
        check(&lapid, 1);
        let sel = Expr::Where(
            Box::new(Expr::Binary(BinaryOp::CmpLt, load(0), load(1))),
            Box::new(Expr::Binary(BinaryOp::Pow, load(2), k(2.0))),
            Box::new(Expr::Binary(BinaryOp::Sub, k(3.0), load(0))),
        );
        check(&sel, 3);
        let folded = Expr::Binary(BinaryOp::Add, k(1.0), k(2.0));
        let p = Program::compile(&folded);
        let mut regs = Regs::default();
        p.prepare(&mut regs);
        assert!(p.run(&mut regs, 5).iter().all(|&v| v == 3.0));
        let id = Program::compile(&Expr::Load(2));
        assert_eq!(id.identity(), Some(2));
        assert!(!id.uses_input(0));
        // CSE: (a+b)*(a+b) uses a single add.
        let s = Box::new(Expr::Binary(BinaryOp::Add, load(0), load(1)));
        let sq = Expr::Binary(BinaryOp::Mul, s.clone(), s);
        assert_eq!(Program::compile(&sq).code.len(), 2);
        check(&sq, 2);
    }
}
