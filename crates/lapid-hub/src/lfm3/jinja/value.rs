//! Runtime values and their Python-flavoured semantics: truthiness, `str()` / `repr()`,
//! equality, ordering and arithmetic.

use std::cmp::Ordering;

use lapid::{Error, Result};

/// A template value. Maps keep insertion order (like Python dicts).
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Undefined,
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    Map(Vec<(String, Value)>),
    /// Internal: handle to a `namespace()` object owned by the running render.
    Namespace(usize),
    /// Internal: a template macro (index into the template's macro table).
    Macro(usize),
    /// Internal: a builtin global function (`namespace`, `range`, `raise_exception`).
    Builtin(&'static str),
}

impl Value {
    pub fn str(s: impl Into<String>) -> Value {
        Value::Str(s.into())
    }

    pub fn map(pairs: Vec<(String, Value)>) -> Value {
        Value::Map(pairs)
    }

    /// Order-preserving JSON parse. Error code "json".
    pub fn from_json_str(s: &str) -> Result<Value> {
        super::json::parse(s)
    }

    /// Convert a `serde_json` value (object keys come out in serde_json's order).
    pub fn from_json(v: &serde_json::Value) -> Value {
        match v {
            serde_json::Value::Null => Value::None,
            serde_json::Value::Bool(b) => Value::Bool(*b),
            serde_json::Value::Number(n) => match n.as_i64() {
                Some(i) => Value::Int(i),
                None => Value::Float(n.as_f64().unwrap_or(f64::NAN)),
            },
            serde_json::Value::String(s) => Value::Str(s.clone()),
            serde_json::Value::Array(a) => Value::List(a.iter().map(Value::from_json).collect()),
            serde_json::Value::Object(o) => {
                Value::Map(o.iter().map(|(k, v)| (k.clone(), Value::from_json(v))).collect())
            }
        }
    }

    /// Python truthiness (Undefined is false).
    pub(crate) fn truthy(&self) -> bool {
        match self {
            Value::Undefined | Value::None => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::List(l) => !l.is_empty(),
            Value::Map(m) => !m.is_empty(),
            Value::Namespace(_) | Value::Macro(_) | Value::Builtin(_) => true,
        }
    }

    pub(crate) fn type_name(&self) -> &'static str {
        match self {
            Value::Undefined => "undefined",
            Value::None => "none",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "str",
            Value::List(_) => "list",
            Value::Map(_) => "dict",
            Value::Namespace(_) => "namespace",
            Value::Macro(_) => "macro",
            Value::Builtin(_) => "function",
        }
    }

    /// Python `str(value)` as Jinja prints it (Undefined prints as "").
    pub(crate) fn to_str(&self) -> String {
        match self {
            Value::Undefined => String::new(),
            Value::Str(s) => s.clone(),
            other => other.repr(),
        }
    }

    /// Python `repr(value)`.
    pub(crate) fn repr(&self) -> String {
        match self {
            Value::Undefined => "Undefined".into(),
            Value::None => "None".into(),
            Value::Bool(true) => "True".into(),
            Value::Bool(false) => "False".into(),
            Value::Int(i) => i.to_string(),
            Value::Float(f) => float_repr(*f),
            Value::Str(s) => str_repr(s),
            Value::List(l) => {
                let items: Vec<String> = l.iter().map(Value::repr).collect();
                format!("[{}]", items.join(", "))
            }
            Value::Map(m) => {
                let items: Vec<String> =
                    m.iter().map(|(k, v)| format!("{}: {}", str_repr(k), v.repr())).collect();
                format!("{{{}}}", items.join(", "))
            }
            Value::Namespace(_) => "<Namespace>".into(),
            Value::Macro(_) => "<Macro>".into(),
            Value::Builtin(name) => format!("<built-in function {name}>"),
        }
    }

    /// Numeric view (bools count as ints, like Python).
    fn num(&self) -> Option<Num> {
        match self {
            Value::Bool(b) => Some(Num::I(*b as i64)),
            Value::Int(i) => Some(Num::I(*i)),
            Value::Float(f) => Some(Num::F(*f)),
            _ => None,
        }
    }

    /// Python `==`.
    pub(crate) fn py_eq(&self, other: &Value) -> bool {
        if let (Some(a), Some(b)) = (self.num(), other.num()) {
            return match (a, b) {
                (Num::I(x), Num::I(y)) => x == y,
                (x, y) => x.f() == y.f(),
            };
        }
        match (self, other) {
            (Value::List(a), Value::List(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.py_eq(y))
            }
            (Value::Map(a), Value::Map(b)) => {
                a.len() == b.len()
                    && a.iter().all(|(k, v)| b.iter().any(|(k2, v2)| k == k2 && v.py_eq(v2)))
            }
            _ => self == other,
        }
    }

    /// Python ordering (`<` and friends); errors on unorderable types.
    pub(crate) fn py_cmp(&self, other: &Value) -> Result<Ordering> {
        if let (Some(a), Some(b)) = (self.num(), other.num()) {
            let ord = match (a, b) {
                (Num::I(x), Num::I(y)) => Some(x.cmp(&y)),
                (x, y) => x.f().partial_cmp(&y.f()),
            };
            return Ok(ord.unwrap_or(Ordering::Equal));
        }
        match (self, other) {
            (Value::Str(a), Value::Str(b)) => Ok(a.cmp(b)),
            (Value::List(a), Value::List(b)) => {
                for (x, y) in a.iter().zip(b) {
                    if !x.py_eq(y) {
                        return x.py_cmp(y);
                    }
                }
                Ok(a.len().cmp(&b.len()))
            }
            _ => Err(type_error("compare", self, other)),
        }
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Value {
        Value::Bool(b)
    }
}
impl From<&str> for Value {
    fn from(s: &str) -> Value {
        Value::Str(s.to_string())
    }
}
impl From<String> for Value {
    fn from(s: String) -> Value {
        Value::Str(s)
    }
}
impl From<i64> for Value {
    fn from(i: i64) -> Value {
        Value::Int(i)
    }
}

#[derive(Clone, Copy)]
enum Num {
    I(i64),
    F(f64),
}

impl Num {
    fn f(self) -> f64 {
        match self {
            Num::I(i) => i as f64,
            Num::F(f) => f,
        }
    }
}

pub(crate) fn type_error(op: &str, a: &Value, b: &Value) -> Error {
    Error::new(
        "template",
        format!("unsupported operand types for {op}: '{}' and '{}'", a.type_name(), b.type_name()),
    )
}

fn overflow() -> Error {
    Error::new("template", "integer overflow")
}

/// Binary arithmetic operators.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Arith {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
}

/// Python arithmetic on two values.
pub(crate) fn arith(op: Arith, a: &Value, b: &Value) -> Result<Value> {
    let sym = match op {
        Arith::Add => "+",
        Arith::Sub => "-",
        Arith::Mul => "*",
        Arith::Div => "/",
        Arith::FloorDiv => "//",
        Arith::Mod => "%",
        Arith::Pow => "**",
    };
    if let (Some(x), Some(y)) = (a.num(), b.num()) {
        return num_arith(op, x, y);
    }
    match (op, a, b) {
        (Arith::Add, Value::Str(x), Value::Str(y)) => Ok(Value::Str(format!("{x}{y}"))),
        (Arith::Add, Value::List(x), Value::List(y)) => {
            Ok(Value::List(x.iter().chain(y).cloned().collect()))
        }
        (Arith::Mul, Value::Str(s), n) | (Arith::Mul, n, Value::Str(s)) if n.num().is_some() => {
            let k = match n.num() {
                Some(Num::I(k)) => k,
                _ => return Err(type_error(sym, a, b)),
            };
            Ok(Value::Str(s.repeat(k.max(0) as usize)))
        }
        (Arith::Mul, Value::List(l), n) | (Arith::Mul, n, Value::List(l)) if n.num().is_some() => {
            let k = match n.num() {
                Some(Num::I(k)) => k.max(0) as usize,
                _ => return Err(type_error(sym, a, b)),
            };
            let mut out = Vec::with_capacity(l.len() * k);
            for _ in 0..k {
                out.extend(l.iter().cloned());
            }
            Ok(Value::List(out))
        }
        _ => Err(type_error(sym, a, b)),
    }
}

fn num_arith(op: Arith, x: Num, y: Num) -> Result<Value> {
    let zero_div = || Error::new("template", "division by zero");
    if let (Num::I(a), Num::I(b)) = (x, y) {
        let r = match op {
            Arith::Add => a.checked_add(b),
            Arith::Sub => a.checked_sub(b),
            Arith::Mul => a.checked_mul(b),
            Arith::Div => {
                if b == 0 {
                    return Err(zero_div());
                }
                return Ok(Value::Float(a as f64 / b as f64));
            }
            Arith::FloorDiv | Arith::Mod => {
                if b == 0 {
                    return Err(zero_div());
                }
                let (q, r) = (a.div_euclid(b), a.rem_euclid(b));
                // Python floors toward -inf; euclid differs when b < 0 and r != 0.
                let (q, r) = if b < 0 && r != 0 { (q - 1, r + b) } else { (q, r) };
                Some(if op == Arith::FloorDiv { q } else { r })
            }
            Arith::Pow => {
                if b < 0 {
                    return Ok(Value::Float((a as f64).powf(b as f64)));
                }
                u32::try_from(b).ok().and_then(|e| a.checked_pow(e))
            }
        };
        return r.map(Value::Int).ok_or_else(overflow);
    }
    let (a, b) = (x.f(), y.f());
    let r = match op {
        Arith::Add => a + b,
        Arith::Sub => a - b,
        Arith::Mul => a * b,
        Arith::Div => {
            if b == 0.0 {
                return Err(zero_div());
            }
            a / b
        }
        Arith::FloorDiv => {
            if b == 0.0 {
                return Err(zero_div());
            }
            (a / b).floor()
        }
        Arith::Mod => {
            if b == 0.0 {
                return Err(zero_div());
            }
            let r = a % b;
            if r != 0.0 && (r < 0.0) != (b < 0.0) { r + b } else { r }
        }
        Arith::Pow => a.powf(b),
    };
    Ok(Value::Float(r))
}

/// Python `repr(float)`: shortest round-trip digits, exponent form outside [1e-4, 1e16).
pub(crate) fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let sign = if f.is_sign_negative() { "-" } else { "" };
    let sci = format!("{:e}", f.abs()); // e.g. "1.5e20", "1e-7", "0e0"
    let (mant, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    if f == 0.0 {
        return format!("{sign}0.0");
    }
    if !(-4..16).contains(&exp) {
        let rest = &digits[1..];
        let m = if rest.is_empty() { digits[..1].to_string() } else { format!("{}.{rest}", &digits[..1]) };
        let es = if exp < 0 { "-" } else { "+" };
        return format!("{sign}{m}e{es}{:02}", exp.abs());
    }
    let n = digits.len() as i32;
    let point = exp + 1; // digits before the decimal point
    let body = if point <= 0 {
        format!("0.{}{digits}", "0".repeat((-point) as usize))
    } else if point >= n {
        format!("{digits}{}.0", "0".repeat((point - n) as usize))
    } else {
        format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
    };
    format!("{sign}{body}")
}

/// Python `repr(str)`.
pub(crate) fn str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python `str.isspace` for a single char.
pub(crate) fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}
