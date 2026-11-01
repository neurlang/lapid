//! Template syntax tree.

use super::value::{Arith, Value};

#[derive(Clone, Debug)]
pub(crate) struct Arg {
    pub name: Option<String>,
    pub value: Expr,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    In,
    NotIn,
}

#[derive(Clone, Debug)]
pub(crate) enum Expr {
    Const(Value),
    Name(String),
    List(Vec<Expr>),
    Dict(Vec<(Expr, Expr)>),
    Attr(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    Slice(Box<Expr>, [Option<Box<Expr>>; 3]),
    Call(Box<Expr>, Vec<Arg>),
    Filter(Box<Expr>, String, Vec<Arg>),
    Test { value: Box<Expr>, name: String, args: Vec<Arg>, negated: bool },
    Not(Box<Expr>),
    Neg(Box<Expr>),
    Arith(Arith, Box<Expr>, Box<Expr>),
    Concat(Box<Expr>, Box<Expr>),
    /// Chained comparison `a < b <= c`.
    Compare(Box<Expr>, Vec<(CmpOp, Expr)>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Cond { cond: Box<Expr>, then: Box<Expr>, otherwise: Option<Box<Expr>> },
}

#[derive(Clone, Debug)]
pub(crate) enum Target {
    Name(String),
    Attr(String, String),
    Tuple(Vec<String>),
}

#[derive(Clone, Debug)]
pub(crate) enum Stmt {
    Text(String),
    Output(Expr),
    If(Vec<(Expr, Vec<Stmt>)>, Vec<Stmt>),
    For { target: Target, iter: Expr, filter: Option<Expr>, body: Vec<Stmt>, otherwise: Vec<Stmt> },
    Set(Target, Expr),
    SetBlock(String, Vec<Stmt>),
    /// Bind macro `id` (index into the macro table) to `name`.
    Macro(String, usize),
    /// `{% generation %}`: rendered as-is in its own scope.
    Scoped(Vec<Stmt>),
    Break,
    Continue,
}

#[derive(Clone, Debug)]
pub(crate) struct MacroDef {
    pub params: Vec<(String, Option<Expr>)>,
    pub body: Vec<Stmt>,
}
