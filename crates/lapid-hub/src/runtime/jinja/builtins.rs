//! Builtin filters, tests and methods (`dict.get`, `str.split`, ...) on plain values.

use lapid::{Error, Result};

use super::json;
use super::value::{Value, is_py_space};

fn terr(msg: impl Into<String>) -> Error {
    Error::new("template", msg)
}

/// Python `iter(v)` materialized: lists, dict keys, string chars; Undefined is empty.
pub(crate) fn iterate(v: &Value) -> Result<Vec<Value>> {
    match v {
        Value::List(l) => Ok(l.clone()),
        Value::Map(m) => Ok(m.iter().map(|(k, _)| Value::Str(k.clone())).collect()),
        Value::Str(s) => Ok(s.chars().map(|c| Value::Str(c.to_string())).collect()),
        Value::Undefined => Ok(Vec::new()),
        other => Err(terr(format!("'{}' object is not iterable", other.type_name()))),
    }
}

fn index(len: usize, i: i64) -> Option<usize> {
    let i = if i < 0 { i + len as i64 } else { i };
    (0..len as i64).contains(&i).then_some(i as usize)
}

/// `obj[key]` / `obj.key`, Undefined when missing.
pub(crate) fn get_item(obj: &Value, key: &Value) -> Value {
    match (obj, key) {
        (Value::Map(m), Value::Str(k)) => {
            m.iter().find(|(k2, _)| k2 == k).map_or(Value::Undefined, |(_, v)| v.clone())
        }
        (Value::List(l), Value::Int(i)) => index(l.len(), *i).map_or(Value::Undefined, |i| l[i].clone()),
        (Value::Str(s), Value::Int(i)) => {
            let chars: Vec<char> = s.chars().collect();
            index(chars.len(), *i).map_or(Value::Undefined, |i| Value::Str(chars[i].to_string()))
        }
        _ => Value::Undefined,
    }
}

/// Python slicing `obj[start:stop:step]` on lists and strings.
pub(crate) fn slice(obj: &Value, [start, stop, step]: [Option<i64>; 3]) -> Result<Value> {
    let step = step.unwrap_or(1);
    if step == 0 {
        return Err(terr("slice step cannot be zero"));
    }
    let pick = |len: usize| -> Vec<usize> {
        let len = len as i64;
        let clamp = |v: i64, lo: i64, hi: i64| v.clamp(lo, hi);
        let norm = |v: i64| if v < 0 { v + len } else { v };
        let mut out = Vec::new();
        if step > 0 {
            let a = start.map_or(0, |v| clamp(norm(v), 0, len));
            let b = stop.map_or(len, |v| clamp(norm(v), 0, len));
            let mut i = a;
            while i < b {
                out.push(i as usize);
                i += step;
            }
        } else {
            let a = start.map_or(len - 1, |v| clamp(norm(v), -1, len - 1));
            let b = stop.map_or(-1, |v| clamp(norm(v), -1, len - 1));
            let mut i = a;
            while i > b {
                out.push(i as usize);
                i += step;
            }
        }
        out
    };
    match obj {
        Value::List(l) => Ok(Value::List(pick(l.len()).into_iter().map(|i| l[i].clone()).collect())),
        Value::Str(s) => {
            let chars: Vec<char> = s.chars().collect();
            Ok(Value::Str(pick(chars.len()).into_iter().map(|i| chars[i]).collect()))
        }
        Value::Undefined => Ok(Value::Undefined),
        other => Err(terr(format!("'{}' object is not subscriptable", other.type_name()))),
    }
}

/// Python `needle in haystack`.
pub(crate) fn contains(haystack: &Value, needle: &Value) -> Result<bool> {
    match (haystack, needle) {
        (Value::Str(h), Value::Str(n)) => Ok(h.contains(n.as_str())),
        (Value::Str(_), other) => {
            Err(terr(format!("'in <string>' requires string as left operand, not {}", other.type_name())))
        }
        (Value::List(l), n) => Ok(l.iter().any(|x| x.py_eq(n))),
        (Value::Map(m), Value::Str(k)) => Ok(m.iter().any(|(k2, _)| k2 == k)),
        (Value::Map(_), _) | (Value::Undefined, _) => Ok(false),
        (other, _) => Err(terr(format!("argument of type '{}' is not iterable", other.type_name()))),
    }
}

fn length(v: &Value) -> Result<usize> {
    match v {
        Value::Str(s) => Ok(s.chars().count()),
        Value::List(l) => Ok(l.len()),
        Value::Map(m) => Ok(m.len()),
        Value::Undefined => Ok(0),
        other => Err(terr(format!("object of type '{}' has no len()", other.type_name()))),
    }
}

fn arg<'a>(pos: &'a [Value], kw: &'a [(String, Value)], i: usize, name: &str) -> Option<&'a Value> {
    pos.get(i).or_else(|| kw.iter().find(|(k, _)| k == name).map(|(_, v)| v))
}

fn py_strip<'a>(s: &'a str, chars: Option<&Value>, left: bool, right: bool) -> &'a str {
    let set: Option<Vec<char>> = match chars {
        Some(Value::Str(c)) => Some(c.chars().collect()),
        _ => None,
    };
    let pred = |c: char| match &set {
        Some(set) => set.contains(&c),
        None => is_py_space(c),
    };
    let s = if left { s.trim_start_matches(pred) } else { s };
    if right { s.trim_end_matches(pred) } else { s }
}

fn parse_int(s: &str) -> Option<i64> {
    let t = s.trim_matches(is_py_space).replace('_', "");
    t.parse::<i64>().ok().or_else(|| t.parse::<f64>().ok().filter(|f| f.is_finite()).map(|f| f as i64))
}

/// Apply filter `name` to `v`.
pub(crate) fn filter(name: &str, v: Value, pos: &[Value], kw: &[(String, Value)]) -> Result<Value> {
    let a0 = arg(pos, kw, 0, "");
    Ok(match name {
        "default" | "d" => {
            let fallback = arg(pos, kw, 0, "default_value").cloned().unwrap_or(Value::str(""));
            let boolean = arg(pos, kw, 1, "boolean").is_some_and(Value::truthy);
            if matches!(v, Value::Undefined) || (boolean && !v.truthy()) { fallback } else { v }
        }
        "tojson" => match arg(pos, kw, 0, "indent") {
            Some(Value::Int(n)) if *n >= 0 => Value::Str(json::dumps_indent(&v, Some(*n as usize))?),
            _ => Value::Str(json::dumps(&v)?),
        },
        "dictsort" => dictsort(v, pos, kw)?,
        "indent" => Value::Str(indent(&v.to_str(), pos, kw)),
        "string" => Value::Str(v.to_str()),
        "safe" => v,
        "trim" => Value::Str(py_strip(&v.to_str(), arg(pos, kw, 0, "chars"), true, true).to_string()),
        "lower" => Value::Str(v.to_str().to_lowercase()),
        "upper" => Value::Str(v.to_str().to_uppercase()),
        "capitalize" => {
            let s = v.to_str();
            let mut chars = s.chars();
            Value::Str(match chars.next() {
                Some(c) => c.to_uppercase().chain(chars.flat_map(char::to_lowercase)).collect(),
                None => String::new(),
            })
        }
        "join" => {
            let sep = arg(pos, kw, 0, "d").map(Value::to_str).unwrap_or_default();
            let parts: Vec<String> = iterate(&v)?.iter().map(Value::to_str).collect();
            Value::Str(parts.join(&sep))
        }
        "list" => Value::List(iterate(&v)?),
        "length" | "count" => Value::Int(length(&v)? as i64),
        "first" => iterate(&v)?.into_iter().next().unwrap_or(Value::Undefined),
        "last" => iterate(&v)?.pop().unwrap_or(Value::Undefined),
        "reverse" => match v {
            Value::Str(s) => Value::Str(s.chars().rev().collect()),
            other => Value::List(iterate(&other)?.into_iter().rev().collect()),
        },
        "int" => {
            let default = a0.cloned().unwrap_or(Value::Int(0));
            match v {
                Value::Int(i) => Value::Int(i),
                Value::Bool(b) => Value::Int(b as i64),
                Value::Float(f) if f.is_finite() => Value::Int(f.trunc() as i64),
                Value::Str(s) => parse_int(&s).map_or(default, Value::Int),
                _ => default,
            }
        }
        "float" => {
            let default = a0.cloned().unwrap_or(Value::Float(0.0));
            match v {
                Value::Int(i) => Value::Float(i as f64),
                Value::Bool(b) => Value::Float(b as i64 as f64),
                Value::Float(f) => Value::Float(f),
                Value::Str(s) => s.trim_matches(is_py_space).parse().map_or(default, Value::Float),
                _ => default,
            }
        }
        "abs" => match v {
            Value::Int(i) => Value::Int(i.checked_abs().ok_or_else(|| terr("integer overflow"))?),
            Value::Float(f) => Value::Float(f.abs()),
            other => return Err(terr(format!("bad operand type for abs(): '{}'", other.type_name()))),
        },
        "items" => match v {
            Value::Map(m) => Value::List(m.into_iter().map(|(k, v)| Value::List(vec![Value::Str(k), v])).collect()),
            Value::Undefined => Value::List(Vec::new()),
            other => return Err(terr(format!("items filter needs a mapping, got {}", other.type_name()))),
        },
        "replace" => {
            let (old, new) = match (pos.first(), pos.get(1)) {
                (Some(o), Some(n)) => (o.to_str(), n.to_str()),
                _ => return Err(terr("replace filter needs two arguments")),
            };
            Value::Str(replace(&v.to_str(), &old, &new, pos.get(2)))
        }
        other => return Err(terr(format!("no filter named '{other}'"))),
    })
}

/// `dictsort(case_sensitive=False, by="key", reverse=False)`: `[key, value]` pairs sorted.
fn dictsort(v: Value, pos: &[Value], kw: &[(String, Value)]) -> Result<Value> {
    let pairs = match v {
        Value::Map(m) => m,
        Value::Undefined => Vec::new(),
        other => return Err(terr(format!("dictsort filter needs a mapping, got {}", other.type_name()))),
    };
    let case = arg(pos, kw, 0, "case_sensitive").is_some_and(Value::truthy);
    let by_value = arg(pos, kw, 1, "by").is_some_and(|b| b.to_str() == "value");
    let reverse = arg(pos, kw, 2, "reverse").is_some_and(Value::truthy);
    let key = |k: &str, v: &Value| -> Value {
        let x = if by_value { v.clone() } else { Value::str(k) };
        match x {
            Value::Str(s) if !case => Value::Str(s.to_lowercase()),
            other => other,
        }
    };
    let mut keyed: Vec<(Value, String, Value)> = pairs.into_iter().map(|(k, v)| (key(&k, &v), k, v)).collect();
    let mut err = None;
    keyed.sort_by(|a, b| {
        a.0.py_cmp(&b.0).unwrap_or_else(|e| {
            err.get_or_insert(e);
            std::cmp::Ordering::Equal
        })
    });
    if let Some(e) = err {
        return Err(e);
    }
    if reverse {
        keyed.reverse();
    }
    Ok(Value::List(keyed.into_iter().map(|(_, k, v)| Value::List(vec![Value::Str(k), v])).collect()))
}

/// `indent(width=4, first=False, blank=False)`.
fn indent(s: &str, pos: &[Value], kw: &[(String, Value)]) -> String {
    let pad = match arg(pos, kw, 0, "width") {
        Some(Value::Int(n)) => " ".repeat((*n).max(0) as usize),
        Some(Value::Str(p)) => p.clone(),
        _ => "    ".to_string(),
    };
    let first = arg(pos, kw, 1, "first").is_some_and(Value::truthy);
    let blank = arg(pos, kw, 2, "blank").is_some_and(Value::truthy);
    let mut out = String::new();
    for (i, line) in s.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
        }
        if (i > 0 || first) && (blank || !line.trim().is_empty()) {
            out.push_str(&pad);
        }
        out.push_str(line);
    }
    out
}

fn replace(s: &str, old: &str, new: &str, count: Option<&Value>) -> String {
    match count {
        Some(Value::Int(n)) if *n >= 0 => s.replacen(old, new, *n as usize),
        _ => s.replace(old, new),
    }
}

/// Apply test `name` (`v is name(args)`).
pub(crate) fn test(name: &str, v: &Value, args: &[Value]) -> Result<bool> {
    let a0 = || args.first().ok_or_else(|| terr(format!("test '{name}' needs an argument")));
    Ok(match name {
        "defined" => !matches!(v, Value::Undefined),
        "undefined" => matches!(v, Value::Undefined),
        "none" => matches!(v, Value::None),
        "string" => matches!(v, Value::Str(_)),
        "boolean" => matches!(v, Value::Bool(_)),
        "true" => matches!(v, Value::Bool(true)),
        "false" => matches!(v, Value::Bool(false)),
        "mapping" => matches!(v, Value::Map(_)),
        "sequence" => matches!(v, Value::Str(_) | Value::List(_) | Value::Map(_)),
        "iterable" => matches!(v, Value::Str(_) | Value::List(_) | Value::Map(_) | Value::Undefined),
        "number" => matches!(v, Value::Int(_) | Value::Float(_) | Value::Bool(_)),
        "integer" => matches!(v, Value::Int(_)),
        "float" => matches!(v, Value::Float(_)),
        "callable" => matches!(v, Value::Macro(_) | Value::Builtin(_)),
        "lower" => matches!(v, Value::Str(s) if s.to_lowercase() == *s),
        "upper" => matches!(v, Value::Str(s) if s.to_uppercase() == *s),
        "even" | "odd" => match v {
            Value::Int(i) => (i % 2 == 0) == (name == "even"),
            other => return Err(terr(format!("test '{name}' needs an integer, got {}", other.type_name()))),
        },
        "divisibleby" => match (v, a0()?) {
            (Value::Int(a), Value::Int(b)) if *b != 0 => a % b == 0,
            _ => return Err(terr("divisibleby needs non-zero integers")),
        },
        "eq" | "equalto" | "==" => v.py_eq(a0()?),
        "ne" | "!=" => !v.py_eq(a0()?),
        "lt" | "lessthan" | "<" => v.py_cmp(a0()?)?.is_lt(),
        "le" | "<=" => v.py_cmp(a0()?)?.is_le(),
        "gt" | "greaterthan" | ">" => v.py_cmp(a0()?)?.is_gt(),
        "ge" | ">=" => v.py_cmp(a0()?)?.is_ge(),
        "in" => contains(a0()?, v)?,
        "sameas" => v == a0()?,
        other => return Err(terr(format!("no test named '{other}'"))),
    })
}

fn str_arg(args: &[Value], i: usize, method: &str) -> Result<String> {
    match args.get(i) {
        Some(Value::Str(s)) => Ok(s.clone()),
        other => Err(terr(format!(
            "str.{method}() argument {} must be str, not {}",
            i + 1,
            other.map_or("nothing", Value::type_name)
        ))),
    }
}

/// Python `str.split` / `str.rsplit`.
fn split(s: &str, sep: Option<&Value>, maxsplit: Option<&Value>, right: bool) -> Result<Value> {
    let max = match maxsplit {
        Some(Value::Int(n)) if *n >= 0 => Some(*n as usize),
        _ => None,
    };
    let parts: Vec<String> = match sep {
        None | Some(Value::None) => {
            let words: Vec<&str> = s.split(is_py_space).filter(|w| !w.is_empty()).collect();
            match max {
                Some(m) if words.len() > m + 1 && !right => {
                    let mut rest = s.trim_start_matches(is_py_space);
                    let mut out = Vec::new();
                    for _ in 0..m {
                        let end = rest.find(is_py_space).unwrap_or(rest.len());
                        out.push(rest[..end].to_string());
                        rest = rest[end..].trim_start_matches(is_py_space);
                    }
                    out.push(rest.to_string());
                    out
                }
                _ => words.into_iter().map(str::to_string).collect(),
            }
        }
        Some(Value::Str(sep)) if sep.is_empty() => return Err(terr("empty separator")),
        Some(Value::Str(sep)) => match (max, right) {
            (Some(m), false) => s.splitn(m + 1, sep.as_str()).map(str::to_string).collect(),
            (Some(m), true) => {
                let mut v: Vec<String> = s.rsplitn(m + 1, sep.as_str()).map(str::to_string).collect();
                v.reverse();
                v
            }
            (None, _) => s.split(sep.as_str()).map(str::to_string).collect(),
        },
        Some(other) => return Err(terr(format!("split separator must be str, not {}", other.type_name()))),
    };
    Ok(Value::List(parts.into_iter().map(Value::Str).collect()))
}

fn affix_test(s: &str, arg: Option<&Value>, prefix: bool) -> Result<bool> {
    let check = |a: &str| if prefix { s.starts_with(a) } else { s.ends_with(a) };
    match arg {
        Some(Value::Str(a)) => Ok(check(a)),
        Some(Value::List(l)) => Ok(l.iter().any(|x| matches!(x, Value::Str(a) if check(a)))),
        _ => Err(terr("startswith/endswith needs a str or tuple argument")),
    }
}

/// Call a Python method on a value; `Ok(None)` if `obj` has no such method.
pub(crate) fn method(obj: &Value, name: &str, args: &[Value]) -> Result<Option<Value>> {
    let v = match (obj, name) {
        (Value::Map(m), "get") => {
            let key = args.first().ok_or_else(|| terr("dict.get() needs a key"))?;
            let found = m.iter().find(|(k, _)| matches!(key, Value::Str(s) if s == k));
            found.map_or_else(|| args.get(1).cloned().unwrap_or(Value::None), |(_, v)| v.clone())
        }
        (Value::Map(m), "items") => {
            Value::List(m.iter().map(|(k, v)| Value::List(vec![Value::Str(k.clone()), v.clone()])).collect())
        }
        (Value::Map(m), "keys") => Value::List(m.iter().map(|(k, _)| Value::Str(k.clone())).collect()),
        (Value::Map(m), "values") => Value::List(m.iter().map(|(_, v)| v.clone()).collect()),
        (Value::Str(s), "split") => split(s, args.first(), args.get(1), false)?,
        (Value::Str(s), "rsplit") => split(s, args.first(), args.get(1), true)?,
        (Value::Str(s), "startswith") => Value::Bool(affix_test(s, args.first(), true)?),
        (Value::Str(s), "endswith") => Value::Bool(affix_test(s, args.first(), false)?),
        (Value::Str(s), "strip") => Value::str(py_strip(s, args.first(), true, true)),
        (Value::Str(s), "lstrip") => Value::str(py_strip(s, args.first(), true, false)),
        (Value::Str(s), "rstrip") => Value::str(py_strip(s, args.first(), false, true)),
        (Value::Str(s), "lower") => Value::Str(s.to_lowercase()),
        (Value::Str(s), "upper") => Value::Str(s.to_uppercase()),
        (Value::Str(s), "replace") => {
            Value::Str(replace(s, &str_arg(args, 0, name)?, &str_arg(args, 1, name)?, args.get(2)))
        }
        (Value::Str(s), "find") => {
            let needle = str_arg(args, 0, name)?;
            Value::Int(s.find(&needle).map_or(-1, |b| s[..b].chars().count() as i64))
        }
        (Value::Str(s), "count") => Value::Int(s.matches(str_arg(args, 0, name)?.as_str()).count() as i64),
        (Value::Str(s), "join") => {
            let parts: Vec<String> = iterate(args.first().unwrap_or(&Value::Undefined))?
                .iter()
                .map(Value::to_str)
                .collect();
            Value::Str(parts.join(s))
        }
        _ => return Ok(None),
    };
    Ok(Some(v))
}

/// `range(stop)` / `range(start, stop[, step])`.
pub(crate) fn range(args: &[Value]) -> Result<Value> {
    let ints: Vec<i64> = args
        .iter()
        .map(|a| match a {
            Value::Int(i) => Ok(*i),
            other => Err(terr(format!("range() needs ints, got {}", other.type_name()))),
        })
        .collect::<Result<_>>()?;
    let (start, stop, step) = match ints.as_slice() {
        [stop] => (0, *stop, 1),
        [start, stop] => (*start, *stop, 1),
        [start, stop, step] if *step != 0 => (*start, *stop, *step),
        _ => return Err(terr("range() takes 1-3 int arguments, step non-zero")),
    };
    let count = if step > 0 { (stop - start + step - 1) / step } else { (start - stop - step - 1) / -step };
    if count > 100_000 {
        return Err(terr("range() too large"));
    }
    let out = (0..count.max(0)).map(|i| Value::Int(start + i * step)).collect();
    Ok(Value::List(out))
}
