//! Tree-walking renderer: statements, scoping, expressions, macro and builtin calls.

use std::collections::HashMap;

use lapid::{Error, Result};

use super::ast::{Arg, CmpOp, Expr, MacroDef, Stmt, Target};
use super::builtins;
use super::value::{Value, arith};

const MAX_DEPTH: usize = 48;
const BUILTINS: [&str; 5] = ["namespace", "range", "raise_exception", "dict", "strftime_now"];

fn terr(msg: impl Into<String>) -> Error {
    Error::new("template", msg)
}

/// Keyword arguments in call order.
type Kwargs = Vec<(String, Value)>;

enum Flow {
    Normal,
    Break,
    Continue,
}

pub(crate) struct Renderer<'t> {
    macros: &'t [MacroDef],
    /// Scope stack; frame 0 holds the globals and top-level assignments.
    frames: Vec<HashMap<String, Value>>,
    namespaces: Vec<Vec<(String, Value)>>,
    depth: usize,
}

impl<'t> Renderer<'t> {
    pub(crate) fn new(macros: &'t [MacroDef], globals: &[(&str, Value)]) -> Self {
        let root = globals.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
        Renderer { macros, frames: vec![root], namespaces: Vec::new(), depth: 0 }
    }

    pub(crate) fn render(&mut self, body: &[Stmt]) -> Result<String> {
        let mut out = String::new();
        self.exec(body, &mut out)?;
        Ok(out)
    }

    fn lookup(&self, name: &str) -> Value {
        for frame in self.frames.iter().rev() {
            if let Some(v) = frame.get(name) {
                return v.clone();
            }
        }
        match BUILTINS.iter().find(|b| **b == name) {
            Some(b) => Value::Builtin(b),
            None => Value::Undefined,
        }
    }

    fn assign(&mut self, name: &str, v: Value) {
        if let Some(frame) = self.frames.last_mut() {
            frame.insert(name.to_string(), v);
        }
    }

    fn bind(&mut self, target: &Target, v: Value) -> Result<()> {
        match target {
            Target::Name(n) => self.assign(n, v),
            Target::Tuple(names) => {
                let items = builtins::iterate(&v)?;
                if items.len() != names.len() {
                    return Err(terr(format!(
                        "cannot unpack {} values into {} names",
                        items.len(),
                        names.len()
                    )));
                }
                for (n, item) in names.iter().zip(items) {
                    self.assign(n, item);
                }
            }
            Target::Attr(obj, attr) => match self.lookup(obj) {
                Value::Namespace(id) => {
                    let ns = self.namespaces.get_mut(id).ok_or_else(|| terr("dangling namespace"))?;
                    match ns.iter_mut().find(|(k, _)| k == attr) {
                        Some(slot) => slot.1 = v,
                        None => ns.push((attr.clone(), v)),
                    }
                }
                other => {
                    return Err(terr(format!(
                        "cannot assign attribute '{attr}' on a {} (only namespace objects)",
                        other.type_name()
                    )));
                }
            },
        }
        Ok(())
    }

    fn scoped<R>(&mut self, f: impl FnOnce(&mut Self) -> Result<R>) -> Result<R> {
        self.frames.push(HashMap::new());
        let r = f(self);
        self.frames.pop();
        r
    }

    fn exec(&mut self, body: &[Stmt], out: &mut String) -> Result<Flow> {
        for stmt in body {
            match stmt {
                Stmt::Text(s) => out.push_str(s),
                Stmt::Output(e) => out.push_str(&self.eval(e)?.to_str()),
                Stmt::If(branches, otherwise) => {
                    let mut chosen = otherwise;
                    for (cond, branch) in branches {
                        if self.eval(cond)?.truthy() {
                            chosen = branch;
                            break;
                        }
                    }
                    match self.exec(chosen, out)? {
                        Flow::Normal => {}
                        flow => return Ok(flow),
                    }
                }
                Stmt::For { target, iter, filter, body, otherwise } => {
                    self.exec_for(target, iter, filter.as_ref(), body, otherwise, out)?;
                }
                Stmt::Set(target, e) => {
                    let v = self.eval(e)?;
                    self.bind(target, v)?;
                }
                Stmt::SetBlock(name, body) => {
                    let mut captured = String::new();
                    self.exec(body, &mut captured)?;
                    self.assign(name, Value::Str(captured));
                }
                Stmt::Macro(name, id) => self.assign(name, Value::Macro(*id)),
                Stmt::Scoped(body) => match self.scoped(|r| r.exec(body, out))? {
                    Flow::Normal => {}
                    flow => return Ok(flow),
                },
                Stmt::Break => return Ok(Flow::Break),
                Stmt::Continue => return Ok(Flow::Continue),
            }
        }
        Ok(Flow::Normal)
    }

    fn exec_for(
        &mut self,
        target: &Target,
        iter: &Expr,
        filter: Option<&Expr>,
        body: &[Stmt],
        otherwise: &[Stmt],
        out: &mut String,
    ) -> Result<()> {
        let mut items = builtins::iterate(&self.eval(iter)?)?;
        if let Some(cond) = filter {
            let mut kept = Vec::with_capacity(items.len());
            for item in items {
                let keep = self.scoped(|r| {
                    r.bind(target, item.clone())?;
                    Ok(r.eval(cond)?.truthy())
                })?;
                if keep {
                    kept.push(item);
                }
            }
            items = kept;
        }
        if items.is_empty() {
            return self.scoped(|r| r.exec(otherwise, out)).map(|_| ());
        }
        let n = items.len();
        for i in 0..n {
            let mut info = vec![
                ("index".to_string(), Value::Int(i as i64 + 1)),
                ("index0".to_string(), Value::Int(i as i64)),
                ("revindex".to_string(), Value::Int((n - i) as i64)),
                ("revindex0".to_string(), Value::Int((n - i - 1) as i64)),
                ("first".to_string(), Value::Bool(i == 0)),
                ("last".to_string(), Value::Bool(i + 1 == n)),
                ("length".to_string(), Value::Int(n as i64)),
                ("depth".to_string(), Value::Int(1)),
                ("depth0".to_string(), Value::Int(0)),
            ];
            if i > 0 {
                info.push(("previtem".to_string(), items[i - 1].clone()));
            }
            if i + 1 < n {
                info.push(("nextitem".to_string(), items[i + 1].clone()));
            }
            let item = items[i].clone();
            let flow = self.scoped(|r| {
                r.bind(target, item)?;
                r.assign("loop", Value::Map(info));
                r.exec(body, out)
            })?;
            if let Flow::Break = flow {
                break;
            }
        }
        Ok(())
    }

    fn args(&mut self, args: &[Arg]) -> Result<(Vec<Value>, Kwargs)> {
        let mut pos = Vec::new();
        let mut kw = Vec::new();
        for a in args {
            let v = self.eval(&a.value)?;
            match &a.name {
                Some(n) => kw.push((n.clone(), v)),
                None => pos.push(v),
            }
        }
        Ok((pos, kw))
    }

    fn get_attr(&self, obj: &Value, name: &str) -> Value {
        match obj {
            Value::Namespace(id) => self
                .namespaces
                .get(*id)
                .and_then(|ns| ns.iter().find(|(k, _)| k == name))
                .map_or(Value::Undefined, |(_, v)| v.clone()),
            other => builtins::get_item(other, &Value::Str(name.to_string())),
        }
    }

    pub(crate) fn eval(&mut self, e: &Expr) -> Result<Value> {
        Ok(match e {
            Expr::Const(v) => v.clone(),
            Expr::Name(n) => self.lookup(n),
            Expr::List(items) => {
                Value::List(items.iter().map(|i| self.eval(i)).collect::<Result<_>>()?)
            }
            Expr::Dict(pairs) => {
                let mut map: Vec<(String, Value)> = Vec::with_capacity(pairs.len());
                for (k, v) in pairs {
                    let k = self.eval(k)?.to_str();
                    let v = self.eval(v)?;
                    match map.iter_mut().find(|(k2, _)| *k2 == k) {
                        Some(slot) => slot.1 = v,
                        None => map.push((k, v)),
                    }
                }
                Value::Map(map)
            }
            Expr::Attr(obj, name) => {
                let obj = self.eval(obj)?;
                self.get_attr(&obj, name)
            }
            Expr::Index(obj, idx) => {
                let obj = self.eval(obj)?;
                let idx = self.eval(idx)?;
                match (&obj, &idx) {
                    (Value::Namespace(_), Value::Str(s)) => self.get_attr(&obj, s),
                    _ => builtins::get_item(&obj, &idx),
                }
            }
            Expr::Slice(obj, parts) => {
                let obj = self.eval(obj)?;
                let mut bounds = [None, None, None];
                for (slot, part) in bounds.iter_mut().zip(parts) {
                    if let Some(p) = part {
                        *slot = match self.eval(p)? {
                            Value::Int(i) => Some(i),
                            Value::None => None,
                            other => {
                                return Err(terr(format!("slice index must be int, not {}", other.type_name())));
                            }
                        };
                    }
                }
                builtins::slice(&obj, bounds)?
            }
            Expr::Call(callee, args) => self.call(callee, args)?,
            Expr::Filter(value, name, args) => {
                let v = self.eval(value)?;
                let (pos, kw) = self.args(args)?;
                builtins::filter(name, v, &pos, &kw)?
            }
            Expr::Test { value, name, args, negated } => {
                let v = self.eval(value)?;
                let (pos, _) = self.args(args)?;
                Value::Bool(builtins::test(name, &v, &pos)? != *negated)
            }
            Expr::Not(x) => Value::Bool(!self.eval(x)?.truthy()),
            Expr::Neg(x) => match self.eval(x)? {
                Value::Int(i) => Value::Int(i.checked_neg().ok_or_else(|| terr("integer overflow"))?),
                Value::Float(f) => Value::Float(-f),
                Value::Bool(b) => Value::Int(-(b as i64)),
                other => return Err(terr(format!("bad operand type for unary -: '{}'", other.type_name()))),
            },
            Expr::Arith(op, a, b) => {
                let a = self.eval(a)?;
                let b = self.eval(b)?;
                arith(*op, &a, &b)?
            }
            Expr::Concat(a, b) => {
                let a = self.eval(a)?.to_str();
                Value::Str(a + &self.eval(b)?.to_str())
            }
            Expr::Compare(first, ops) => {
                let mut left = self.eval(first)?;
                for (op, rhs) in ops {
                    let right = self.eval(rhs)?;
                    let ok = match op {
                        CmpOp::Eq => left.py_eq(&right),
                        CmpOp::Ne => !left.py_eq(&right),
                        CmpOp::Lt => left.py_cmp(&right)?.is_lt(),
                        CmpOp::Le => left.py_cmp(&right)?.is_le(),
                        CmpOp::Gt => left.py_cmp(&right)?.is_gt(),
                        CmpOp::Ge => left.py_cmp(&right)?.is_ge(),
                        CmpOp::In => builtins::contains(&right, &left)?,
                        CmpOp::NotIn => !builtins::contains(&right, &left)?,
                    };
                    if !ok {
                        return Ok(Value::Bool(false));
                    }
                    left = right;
                }
                Value::Bool(true)
            }
            Expr::And(a, b) => {
                let a = self.eval(a)?;
                if a.truthy() { self.eval(b)? } else { a }
            }
            Expr::Or(a, b) => {
                let a = self.eval(a)?;
                if a.truthy() { a } else { self.eval(b)? }
            }
            Expr::Cond { cond, then, otherwise } => {
                if self.eval(cond)?.truthy() {
                    self.eval(then)?
                } else if let Some(o) = otherwise {
                    self.eval(o)?
                } else {
                    Value::Undefined
                }
            }
        })
    }

    fn call(&mut self, callee: &Expr, args: &[Arg]) -> Result<Value> {
        if let Expr::Attr(obj, method) = callee {
            let obj = self.eval(obj)?;
            let (pos, _) = self.args(args)?;
            if let Some(v) = builtins::method(&obj, method, &pos)? {
                return Ok(v);
            }
            let f = self.get_attr(&obj, method);
            return self.call_value(&f, pos, Vec::new(), method);
        }
        let f = self.eval(callee)?;
        let (pos, kw) = self.args(args)?;
        let name = match callee {
            Expr::Name(n) => n.as_str(),
            _ => "expression",
        };
        self.call_value(&f, pos, kw, name)
    }

    fn call_value(
        &mut self,
        f: &Value,
        pos: Vec<Value>,
        kw: Vec<(String, Value)>,
        name: &str,
    ) -> Result<Value> {
        match f {
            Value::Macro(id) => self.call_macro(*id, pos, kw, name),
            Value::Builtin("namespace") | Value::Builtin("dict") => {
                let mut attrs: Vec<(String, Value)> = Vec::new();
                for p in pos {
                    match p {
                        Value::Map(m) => attrs.extend(m),
                        other => return Err(terr(format!("{name}() argument must be a mapping, not {}", other.type_name()))),
                    }
                }
                for (k, v) in kw {
                    match attrs.iter_mut().find(|(k2, _)| *k2 == k) {
                        Some(slot) => slot.1 = v,
                        None => attrs.push((k, v)),
                    }
                }
                if matches!(f, Value::Builtin("dict")) {
                    return Ok(Value::Map(attrs));
                }
                self.namespaces.push(attrs);
                Ok(Value::Namespace(self.namespaces.len() - 1))
            }
            Value::Builtin("range") => builtins::range(&pos),
            Value::Builtin("strftime_now") => {
                let fmt = pos.first().map(Value::to_str).unwrap_or_default();
                Ok(Value::Str(super::time::strftime_now(&fmt)))
            }
            Value::Builtin("raise_exception") => {
                Err(terr(pos.first().map(Value::to_str).unwrap_or_default()))
            }
            Value::Undefined => Err(terr(format!("'{name}' is undefined"))),
            other => Err(terr(format!("'{name}' ({}) is not callable", other.type_name()))),
        }
    }

    fn call_macro(
        &mut self,
        id: usize,
        pos: Vec<Value>,
        kw: Vec<(String, Value)>,
        name: &str,
    ) -> Result<Value> {
        let macros = self.macros;
        let def = macros.get(id).ok_or_else(|| terr("unknown macro"))?;
        if pos.len() > def.params.len() {
            return Err(terr(format!("macro '{name}' takes {} arguments, got {}", def.params.len(), pos.len())));
        }
        if let Some((k, _)) = kw.iter().find(|(k, _)| !def.params.iter().any(|(p, _)| p == k)) {
            return Err(terr(format!("macro '{name}' has no argument '{k}'")));
        }
        if self.depth >= MAX_DEPTH {
            return Err(terr("maximum macro recursion depth exceeded"));
        }
        self.depth += 1;
        let saved = self.frames.split_off(1);
        self.frames.push(HashMap::new());
        let mut pos = pos.into_iter();
        let result = (|| -> Result<String> {
            for (param, default) in &def.params {
                let v = match (pos.next(), kw.iter().find(|(k, _)| k == param)) {
                    (_, Some((_, v))) => v.clone(),
                    (Some(v), None) => v,
                    (None, None) => match default {
                        Some(d) => self.eval(d)?,
                        None => Value::Undefined,
                    },
                };
                self.assign(param, v);
            }
            let mut out = String::new();
            self.exec(&def.body, &mut out)?;
            Ok(out)
        })();
        self.frames.truncate(1);
        self.frames.extend(saved);
        self.depth -= 1;
        result.map(Value::Str)
    }
}
