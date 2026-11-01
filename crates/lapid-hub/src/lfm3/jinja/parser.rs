//! Recursive-descent parser following Jinja's grammar and operator precedence
//! (filters and `is` tests bind tighter than arithmetic).

use lapid::{Error, Result};

use super::ast::{Arg, CmpOp, Expr, MacroDef, Stmt, Target};
use super::lexer::{Spanned, Tok};
use super::value::{Arith, Value};

pub(crate) struct Parsed {
    pub body: Vec<Stmt>,
    pub macros: Vec<MacroDef>,
}

pub(crate) fn parse(tokens: Vec<Spanned>) -> Result<Parsed> {
    let mut p = Parser { toks: tokens, pos: 0, macros: Vec::new(), depth: 0 };
    let (body, end) = p.body(&[])?;
    if let Some(tag) = end {
        return Err(p.err(format!("unexpected '{tag}'")));
    }
    Ok(Parsed { body, macros: p.macros })
}

struct Parser {
    toks: Vec<Spanned>,
    pos: usize,
    macros: Vec<MacroDef>,
    /// Expression / block nesting depth (guards the native stack).
    depth: usize,
}

const MAX_NESTING: usize = 64;

fn boxed(e: Expr) -> Box<Expr> {
    Box::new(e)
}

impl Parser {
    fn err(&self, msg: impl std::fmt::Display) -> Error {
        let line = self.toks.get(self.pos).or(self.toks.last()).map_or(0, |t| t.1);
        Error::new("template", format!("line {line}: {msg}"))
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|t| &t.0)
    }

    fn peek_at(&self, n: usize) -> Option<&Tok> {
        self.toks.get(self.pos + n).map(|t| &t.0)
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).map(|t| t.0.clone());
        self.pos += 1;
        t
    }

    fn is_op(&self, op: &str) -> bool {
        matches!(self.peek(), Some(Tok::Op(o)) if *o == op)
    }

    fn is_name(&self, name: &str) -> bool {
        matches!(self.peek(), Some(Tok::Name(n)) if n == name)
    }

    fn eat_op(&mut self, op: &str) -> bool {
        let hit = self.is_op(op);
        if hit {
            self.pos += 1;
        }
        hit
    }

    fn eat_name(&mut self, name: &str) -> bool {
        let hit = self.is_name(name);
        if hit {
            self.pos += 1;
        }
        hit
    }

    fn expect_op(&mut self, op: &str) -> Result<()> {
        if self.eat_op(op) { Ok(()) } else { Err(self.err(format!("expected '{op}', found {:?}", self.peek()))) }
    }

    fn expect_tok(&mut self, tok: Tok) -> Result<()> {
        if self.peek() == Some(&tok) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.err(format!("expected {tok:?}, found {:?}", self.peek())))
        }
    }

    fn ident(&mut self) -> Result<String> {
        match self.next() {
            Some(Tok::Name(n)) => Ok(n),
            other => {
                self.pos -= 1;
                Err(self.err(format!("expected a name, found {other:?}")))
            }
        }
    }

    // ---- statements -------------------------------------------------------------------

    /// Parse statements until a block tag named in `ends` (consumed up to its name).
    fn body(&mut self, ends: &[&str]) -> Result<(Vec<Stmt>, Option<String>)> {
        let mut out = Vec::new();
        while let Some(tok) = self.next() {
            match tok {
                Tok::Data(s) => out.push(Stmt::Text(s)),
                Tok::VarBegin => {
                    let e = self.expr()?;
                    self.expect_tok(Tok::VarEnd)?;
                    out.push(Stmt::Output(e));
                }
                Tok::BlockBegin => {
                    let kw = self.ident()?;
                    if ends.contains(&kw.as_str()) {
                        return Ok((out, Some(kw)));
                    }
                    out.push(self.statement(&kw)?);
                }
                other => return Err(self.err(format!("unexpected token {other:?}"))),
            }
        }
        if ends.is_empty() {
            Ok((out, None))
        } else {
            Err(self.err(format!("unexpected end of template, expected one of {ends:?}")))
        }
    }

    fn block_end(&mut self) -> Result<()> {
        self.expect_tok(Tok::BlockEnd)
    }

    fn statement(&mut self, kw: &str) -> Result<Stmt> {
        if self.depth >= MAX_NESTING {
            return Err(self.err("blocks nested too deeply"));
        }
        self.depth += 1;
        let r = self.statement_inner(kw);
        self.depth -= 1;
        r
    }

    fn statement_inner(&mut self, kw: &str) -> Result<Stmt> {
        match kw {
            "if" => {
                let mut branches = Vec::new();
                let mut cond = self.expr()?;
                loop {
                    self.block_end()?;
                    let (body, end) = self.body(&["elif", "else", "endif"])?;
                    branches.push((cond, body));
                    match end.as_deref() {
                        Some("elif") => cond = self.expr()?,
                        Some("else") => {
                            self.block_end()?;
                            let (otherwise, _) = self.body(&["endif"])?;
                            self.block_end()?;
                            return Ok(Stmt::If(branches, otherwise));
                        }
                        _ => {
                            self.block_end()?;
                            return Ok(Stmt::If(branches, Vec::new()));
                        }
                    }
                }
            }
            "for" => {
                let first = self.ident()?;
                let target = if self.is_op(",") {
                    let mut names = vec![first];
                    while self.eat_op(",") {
                        names.push(self.ident()?);
                    }
                    Target::Tuple(names)
                } else {
                    Target::Name(first)
                };
                if !self.eat_name("in") {
                    return Err(self.err("expected 'in' in for loop"));
                }
                let iter = self.or_expr()?;
                let filter = if self.eat_name("if") { Some(self.expr()?) } else { None };
                self.block_end()?;
                let (body, end) = self.body(&["else", "endfor"])?;
                let mut otherwise = Vec::new();
                if end.as_deref() == Some("else") {
                    self.block_end()?;
                    otherwise = self.body(&["endfor"])?.0;
                }
                self.block_end()?;
                Ok(Stmt::For { target, iter, filter, body, otherwise })
            }
            "set" => {
                let name = self.ident()?;
                let target = if self.eat_op(".") {
                    Target::Attr(name, self.ident()?)
                } else if self.is_op(",") {
                    let mut names = vec![name];
                    while self.eat_op(",") {
                        names.push(self.ident()?);
                    }
                    Target::Tuple(names)
                } else {
                    Target::Name(name)
                };
                if self.eat_op("=") {
                    let value = self.expr()?;
                    self.block_end()?;
                    return Ok(Stmt::Set(target, value));
                }
                let Target::Name(name) = target else {
                    return Err(self.err("block set needs a plain variable name"));
                };
                self.block_end()?;
                let (body, _) = self.body(&["endset"])?;
                self.block_end()?;
                Ok(Stmt::SetBlock(name, body))
            }
            "macro" => {
                let name = self.ident()?;
                self.expect_op("(")?;
                let mut params = Vec::new();
                while !self.eat_op(")") {
                    if !params.is_empty() {
                        self.expect_op(",")?;
                        if self.eat_op(")") {
                            break;
                        }
                    }
                    let p = self.ident()?;
                    let default = if self.eat_op("=") { Some(self.expr()?) } else { None };
                    params.push((p, default));
                }
                self.block_end()?;
                let (body, _) = self.body(&["endmacro"])?;
                if !matches!(self.peek(), Some(Tok::BlockEnd)) {
                    self.ident()?; // optional `{% endmacro name %}`
                }
                self.block_end()?;
                self.macros.push(MacroDef { params, body });
                Ok(Stmt::Macro(name, self.macros.len() - 1))
            }
            "generation" => {
                self.block_end()?;
                let (body, _) = self.body(&["endgeneration"])?;
                self.block_end()?;
                Ok(Stmt::Scoped(body))
            }
            "break" | "continue" => {
                self.block_end()?;
                Ok(if kw == "break" { Stmt::Break } else { Stmt::Continue })
            }
            other => Err(self.err(format!("unknown tag '{other}'"))),
        }
    }

    // ---- expressions ------------------------------------------------------------------

    pub(crate) fn expr(&mut self) -> Result<Expr> {
        if self.depth >= MAX_NESTING {
            return Err(self.err("expression nested too deeply"));
        }
        self.depth += 1;
        let r = self.expr_inner();
        self.depth -= 1;
        r
    }

    fn expr_inner(&mut self) -> Result<Expr> {
        let mut node = self.or_expr()?;
        while self.eat_name("if") {
            let cond = self.or_expr()?;
            let otherwise = if self.eat_name("else") { Some(boxed(self.expr()?)) } else { None };
            node = Expr::Cond { cond: boxed(cond), then: boxed(node), otherwise };
        }
        Ok(node)
    }

    fn or_expr(&mut self) -> Result<Expr> {
        let mut node = self.and_expr()?;
        while self.eat_name("or") {
            node = Expr::Or(boxed(node), boxed(self.and_expr()?));
        }
        Ok(node)
    }

    fn and_expr(&mut self) -> Result<Expr> {
        let mut node = self.not_expr()?;
        while self.eat_name("and") {
            node = Expr::And(boxed(node), boxed(self.not_expr()?));
        }
        Ok(node)
    }

    fn not_expr(&mut self) -> Result<Expr> {
        if self.eat_name("not") {
            return Ok(Expr::Not(boxed(self.not_expr()?)));
        }
        self.compare()
    }

    fn compare(&mut self) -> Result<Expr> {
        let first = self.math1()?;
        let mut ops = Vec::new();
        loop {
            let op = match self.peek() {
                Some(Tok::Op("==")) => CmpOp::Eq,
                Some(Tok::Op("!=")) => CmpOp::Ne,
                Some(Tok::Op("<")) => CmpOp::Lt,
                Some(Tok::Op("<=")) => CmpOp::Le,
                Some(Tok::Op(">")) => CmpOp::Gt,
                Some(Tok::Op(">=")) => CmpOp::Ge,
                Some(Tok::Name(n)) if n == "in" => CmpOp::In,
                Some(Tok::Name(n))
                    if n == "not" && matches!(self.peek_at(1), Some(Tok::Name(m)) if m == "in") =>
                {
                    self.pos += 1;
                    CmpOp::NotIn
                }
                _ => break,
            };
            self.pos += 1;
            ops.push((op, self.math1()?));
        }
        Ok(if ops.is_empty() { first } else { Expr::Compare(boxed(first), ops) })
    }

    fn math1(&mut self) -> Result<Expr> {
        let mut node = self.concat()?;
        loop {
            let op = if self.eat_op("+") {
                Arith::Add
            } else if self.eat_op("-") {
                Arith::Sub
            } else {
                return Ok(node);
            };
            node = Expr::Arith(op, boxed(node), boxed(self.concat()?));
        }
    }

    fn concat(&mut self) -> Result<Expr> {
        let mut node = self.math2()?;
        while self.eat_op("~") {
            node = Expr::Concat(boxed(node), boxed(self.math2()?));
        }
        Ok(node)
    }

    fn math2(&mut self) -> Result<Expr> {
        let mut node = self.pow()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Op("*")) => Arith::Mul,
                Some(Tok::Op("/")) => Arith::Div,
                Some(Tok::Op("//")) => Arith::FloorDiv,
                Some(Tok::Op("%")) => Arith::Mod,
                _ => return Ok(node),
            };
            self.pos += 1;
            node = Expr::Arith(op, boxed(node), boxed(self.pow()?));
        }
    }

    fn pow(&mut self) -> Result<Expr> {
        let mut node = self.unary(true)?;
        while self.eat_op("**") {
            node = Expr::Arith(Arith::Pow, boxed(node), boxed(self.unary(true)?));
        }
        Ok(node)
    }

    fn unary(&mut self, with_filter: bool) -> Result<Expr> {
        let node = if self.eat_op("-") {
            Expr::Neg(boxed(self.nested_unary()?))
        } else if self.eat_op("+") {
            self.nested_unary()?
        } else {
            self.primary()?
        };
        let node = self.postfix(node)?;
        if with_filter { self.filter_expr(node) } else { Ok(node) }
    }

    fn nested_unary(&mut self) -> Result<Expr> {
        if self.depth >= MAX_NESTING {
            return Err(self.err("expression nested too deeply"));
        }
        self.depth += 1;
        let r = self.unary(false);
        self.depth -= 1;
        r
    }

    fn primary(&mut self) -> Result<Expr> {
        let tok = self.next().ok_or_else(|| self.err("unexpected end of expression"))?;
        Ok(match tok {
            Tok::Name(n) => match n.as_str() {
                "true" | "True" => Expr::Const(Value::Bool(true)),
                "false" | "False" => Expr::Const(Value::Bool(false)),
                "none" | "None" => Expr::Const(Value::None),
                _ => Expr::Name(n),
            },
            Tok::Str(mut s) => {
                while let Some(Tok::Str(more)) = self.peek() {
                    s.push_str(more);
                    self.pos += 1;
                }
                Expr::Const(Value::Str(s))
            }
            Tok::Int(i) => Expr::Const(Value::Int(i)),
            Tok::Float(f) => Expr::Const(Value::Float(f)),
            Tok::Op("(") => {
                if self.eat_op(")") {
                    return Ok(Expr::List(Vec::new()));
                }
                let first = self.expr()?;
                if !self.is_op(",") {
                    self.expect_op(")")?;
                    return Ok(first);
                }
                let mut items = vec![first];
                while self.eat_op(",") && !self.is_op(")") {
                    items.push(self.expr()?);
                }
                self.expect_op(")")?;
                Expr::List(items)
            }
            Tok::Op("[") => {
                let mut items = Vec::new();
                while !self.eat_op("]") {
                    if !items.is_empty() {
                        self.expect_op(",")?;
                        if self.eat_op("]") {
                            break;
                        }
                    }
                    items.push(self.expr()?);
                }
                Expr::List(items)
            }
            Tok::Op("{") => {
                let mut pairs = Vec::new();
                while !self.eat_op("}") {
                    if !pairs.is_empty() {
                        self.expect_op(",")?;
                        if self.eat_op("}") {
                            break;
                        }
                    }
                    let k = self.expr()?;
                    self.expect_op(":")?;
                    pairs.push((k, self.expr()?));
                }
                Expr::Dict(pairs)
            }
            other => {
                self.pos -= 1;
                return Err(self.err(format!("unexpected {other:?}")));
            }
        })
    }

    fn postfix(&mut self, mut node: Expr) -> Result<Expr> {
        loop {
            if self.eat_op(".") {
                node = match self.next() {
                    Some(Tok::Name(n)) => Expr::Attr(boxed(node), n),
                    Some(Tok::Int(i)) => Expr::Index(boxed(node), boxed(Expr::Const(Value::Int(i)))),
                    _ => return Err(self.err("expected attribute name after '.'")),
                };
            } else if self.eat_op("[") {
                node = self.subscript(node)?;
            } else if self.is_op("(") {
                node = Expr::Call(boxed(node), self.call_args()?);
            } else {
                return Ok(node);
            }
        }
    }

    fn subscript(&mut self, node: Expr) -> Result<Expr> {
        let start = if self.is_op(":") { None } else { Some(boxed(self.expr()?)) };
        if !self.eat_op(":") {
            self.expect_op("]")?;
            let idx = start.ok_or_else(|| self.err("empty subscript"))?;
            return Ok(Expr::Index(boxed(node), idx));
        }
        let stop = if self.is_op(":") || self.is_op("]") { None } else { Some(boxed(self.expr()?)) };
        let step = if self.eat_op(":") && !self.is_op("]") { Some(boxed(self.expr()?)) } else { None };
        self.expect_op("]")?;
        Ok(Expr::Slice(boxed(node), [start, stop, step]))
    }

    fn call_args(&mut self) -> Result<Vec<Arg>> {
        self.expect_op("(")?;
        let mut args = Vec::new();
        while !self.eat_op(")") {
            if !args.is_empty() {
                self.expect_op(",")?;
                if self.eat_op(")") {
                    break;
                }
            }
            let kw = matches!(self.peek(), Some(Tok::Name(_))) && matches!(self.peek_at(1), Some(Tok::Op("=")));
            let name = if kw {
                let n = self.ident()?;
                self.pos += 1;
                Some(n)
            } else {
                None
            };
            args.push(Arg { name, value: self.expr()? });
        }
        Ok(args)
    }

    fn dotted_name(&mut self) -> Result<String> {
        let mut name = self.ident()?;
        while self.eat_op(".") {
            name.push('.');
            name.push_str(&self.ident()?);
        }
        Ok(name)
    }

    fn filter_expr(&mut self, mut node: Expr) -> Result<Expr> {
        loop {
            if self.eat_op("|") {
                let name = self.dotted_name()?;
                let args = if self.is_op("(") { self.call_args()? } else { Vec::new() };
                node = Expr::Filter(boxed(node), name, args);
            } else if self.eat_name("is") {
                let negated = self.eat_name("not");
                let name = self.dotted_name()?;
                let args = if self.is_op("(") {
                    self.call_args()?
                } else if self.bare_test_arg() {
                    let arg = self.primary()?;
                    vec![Arg { name: None, value: self.postfix(arg)? }]
                } else {
                    Vec::new()
                };
                node = Expr::Test { value: boxed(node), name, args, negated };
            } else if self.is_op("(") {
                node = Expr::Call(boxed(node), self.call_args()?);
            } else {
                return Ok(node);
            }
        }
    }

    /// Jinja accepts one argument without parentheses: `x is divisibleby 3`.
    fn bare_test_arg(&self) -> bool {
        match self.peek() {
            Some(Tok::Name(n)) => !matches!(n.as_str(), "else" | "or" | "and"),
            Some(Tok::Str(_) | Tok::Int(_) | Tok::Float(_)) => true,
            Some(Tok::Op(o)) => matches!(*o, "(" | "[" | "{"),
            _ => false,
        }
    }
}
