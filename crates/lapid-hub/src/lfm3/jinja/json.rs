//! Order-preserving JSON parser and the `tojson` serializer, which matches Python's
//! `json.dumps(x, ensure_ascii=False)` (separators ", " / ": ", Python float repr).

use lapid::{Error, Result};

use super::value::{Value, float_repr};

fn err(msg: impl Into<String>) -> Error {
    Error::new("json", msg)
}

/// Parse JSON text, keeping object key order.
pub(crate) fn parse(s: &str) -> Result<Value> {
    let mut p = Parser { b: s.as_bytes(), s, i: 0 };
    let v = p.value(0)?;
    p.ws();
    if p.i != p.b.len() {
        return Err(err(format!("trailing characters at byte {}", p.i)));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    s: &'a str,
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn expect(&mut self, c: u8) -> Result<()> {
        self.ws();
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(err(format!("expected '{}' at byte {}", c as char, self.i)))
        }
    }

    fn lit(&mut self, word: &str, v: Value) -> Result<Value> {
        if self.s[self.i..].starts_with(word) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(err(format!("invalid literal at byte {}", self.i)))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > 512 {
            return Err(err("nesting too deep"));
        }
        self.ws();
        match self.peek() {
            None => Err(err("unexpected end of input")),
            Some(b'{') => {
                self.i += 1;
                let mut pairs = Vec::new();
                self.ws();
                if self.peek() == Some(b'}') {
                    self.i += 1;
                    return Ok(Value::Map(pairs));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.expect(b':')?;
                    let v = self.value(depth + 1)?;
                    match pairs.iter_mut().find(|(k2, _)| *k2 == k) {
                        Some(slot) => slot.1 = v,
                        None => pairs.push((k, v)),
                    }
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Value::Map(pairs));
                        }
                        _ => return Err(err(format!("expected ',' or '}}' at byte {}", self.i))),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.peek() == Some(b']') {
                    self.i += 1;
                    return Ok(Value::List(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Value::List(items));
                        }
                        _ => return Err(err(format!("expected ',' or ']' at byte {}", self.i))),
                    }
                }
            }
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b't') => self.lit("true", Value::Bool(true)),
            Some(b'f') => self.lit("false", Value::Bool(false)),
            Some(b'n') => self.lit("null", Value::None),
            Some(b'N') => self.lit("NaN", Value::Float(f64::NAN)),
            Some(b'I') => self.lit("Infinity", Value::Float(f64::INFINITY)),
            Some(_) => self.number(),
        }
    }

    fn number(&mut self) -> Result<Value> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
            if self.s[self.i..].starts_with("Infinity") {
                self.i += "Infinity".len();
                return Ok(Value::Float(f64::NEG_INFINITY));
            }
        }
        let mut float = false;
        while let Some(c) = self.peek() {
            match c {
                b'0'..=b'9' => {}
                b'.' | b'e' | b'E' | b'+' | b'-' => float = true,
                _ => break,
            }
            self.i += 1;
        }
        let text = &self.s[start..self.i];
        if text.is_empty() || text == "-" {
            return Err(err(format!("unexpected character at byte {start}")));
        }
        if !float && let Ok(i) = text.parse::<i64>() {
            return Ok(Value::Int(i));
        }
        text.parse::<f64>()
            .map(Value::Float)
            .map_err(|_| err(format!("invalid number '{text}' at byte {start}")))
    }

    fn hex4(&mut self) -> Result<u32> {
        let h = self.s.get(self.i..self.i + 4).ok_or_else(|| err("truncated \\u escape"))?;
        let v = u32::from_str_radix(h, 16).map_err(|_| err(format!("bad \\u escape '{h}'")))?;
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String> {
        if self.peek() != Some(b'"') {
            return Err(err(format!("expected string at byte {}", self.i)));
        }
        self.i += 1;
        let mut out = String::new();
        loop {
            let run = self.i;
            while let Some(c) = self.peek() {
                if c == b'"' || c == b'\\' {
                    break;
                }
                self.i += 1;
            }
            out.push_str(&self.s[run..self.i]);
            match self.peek() {
                None => return Err(err("unterminated string")),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                _ => {}
            }
            self.i += 1; // backslash
            let c = self.peek().ok_or_else(|| err("unterminated escape"))?;
            self.i += 1;
            match c {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                b'b' => out.push('\u{8}'),
                b'f' => out.push('\u{c}'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'u' => {
                    let mut cp = self.hex4()?;
                    if (0xD800..0xDC00).contains(&cp) && self.s[self.i..].starts_with("\\u") {
                        let save = self.i;
                        self.i += 2;
                        let lo = self.hex4()?;
                        if (0xDC00..0xE000).contains(&lo) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else {
                            self.i = save;
                        }
                    }
                    out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                }
                other => return Err(err(format!("invalid escape '\\{}'", other as char))),
            }
        }
    }
}

/// `json.dumps(v, ensure_ascii=False)` with default separators.
pub(crate) fn dumps(v: &Value) -> Result<String> {
    let mut out = String::new();
    dump(v, &mut out)?;
    Ok(out)
}

fn dump(v: &Value, out: &mut String) -> Result<()> {
    match v {
        Value::None => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Int(i) => out.push_str(&i.to_string()),
        Value::Float(f) => {
            if f.is_nan() {
                out.push_str("NaN");
            } else if f.is_infinite() {
                out.push_str(if *f > 0.0 { "Infinity" } else { "-Infinity" });
            } else {
                out.push_str(&float_repr(*f));
            }
        }
        Value::Str(s) => dump_str(s, out),
        Value::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump(item, out)?;
            }
            out.push(']');
        }
        Value::Map(pairs) => {
            out.push('{');
            for (i, (k, item)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump_str(k, out);
                out.push_str(": ");
                dump(item, out)?;
            }
            out.push('}');
        }
        other => {
            return Err(Error::new(
                "template",
                format!("Object of type {} is not JSON serializable", other.type_name()),
            ));
        }
    }
    Ok(())
}

fn dump_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}
