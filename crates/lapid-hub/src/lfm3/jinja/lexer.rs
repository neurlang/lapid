//! Template lexer: splits source into data and tag tokens, applying Jinja whitespace
//! control (`-`/`+` modifiers, `trim_blocks`, `lstrip_blocks`, dropped trailing newline).

use lapid::{Error, Result};

use super::value::is_py_space;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Tok {
    Data(String),
    VarBegin,
    VarEnd,
    BlockBegin,
    BlockEnd,
    Name(String),
    Str(String),
    Int(i64),
    Float(f64),
    Op(&'static str),
}

/// A token with its 1-based source line.
pub(crate) type Spanned = (Tok, usize);

const OPS: [&str; 22] = [
    "//", "**", "==", "!=", "<=", ">=", "+", "-", "*", "/", "%", "~", "<", ">", "=", "(", ")",
    "[", "]", "{", "}", ",",
];
const OPS_TAIL: [&str; 3] = [".", ":", "|"];

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Var,
    Block,
    Comment,
}

pub(crate) fn tokenize(source: &str) -> Result<Vec<Spanned>> {
    let mut src = source.replace("\r\n", "\n").replace('\r', "\n");
    if src.ends_with('\n') {
        src.pop(); // keep_trailing_newline=False
    }
    Lexer { s: &src, i: 0, line: 1, line_pos: 0, out: Vec::new() }.run()
}

struct Lexer<'a> {
    s: &'a str,
    i: usize,
    line: usize,
    line_pos: usize,
    out: Vec<Spanned>,
}

fn err(line: usize, msg: impl std::fmt::Display) -> Error {
    Error::new("template", format!("line {line}: {msg}"))
}

impl<'a> Lexer<'a> {
    fn line_at(&mut self, pos: usize) -> usize {
        if pos > self.line_pos {
            self.line += self.s[self.line_pos..pos].matches('\n').count();
            self.line_pos = pos;
        }
        self.line
    }

    fn push(&mut self, tok: Tok, pos: usize) {
        let line = self.line_at(pos);
        self.out.push((tok, line));
    }

    fn rest(&self) -> &'a str {
        &self.s[self.i..]
    }

    fn run(mut self) -> Result<Vec<Spanned>> {
        let mut line_starting = true;
        let s: &'a str = self.s;
        loop {
            let found = ["{{", "{%", "{#"]
                .iter()
                .filter_map(|p| self.rest().find(p).map(|off| self.i + off))
                .min();
            let Some(open) = found else {
                if self.i < self.s.len() {
                    self.push(Tok::Data(self.rest().to_string()), self.i);
                }
                return Ok(self.out);
            };
            let kind = match &s[open + 1..open + 2] {
                "{" => Kind::Var,
                "%" => Kind::Block,
                _ => Kind::Comment,
            };
            let modifier = s[open + 2..].chars().next().filter(|c| *c == '-' || *c == '+');
            let mut text = &s[self.i..open];
            if modifier == Some('-') {
                text = text.trim_end_matches(is_py_space);
            } else if modifier.is_none() && kind != Kind::Var {
                let l_pos = text.rfind('\n').map_or(0, |p| p + 1);
                if (l_pos > 0 || line_starting)
                    && text[l_pos..].chars().all(|c| c != '\n' && is_py_space(c))
                {
                    text = &text[..l_pos];
                }
            }
            if !text.is_empty() {
                let t = text.to_string();
                self.push(Tok::Data(t), self.i);
            }
            self.i = open + 2 + modifier.map_or(0, |_| 1);
            let tag_start = self.i;
            match kind {
                Kind::Comment => {
                    let Some(off) = self.rest().find("#}") else {
                        return Err(err(self.line_at(open), "unterminated comment"));
                    };
                    let close = self.i + off;
                    let strip = close > tag_start && &s[close - 1..close] == "-";
                    self.i = close + 2;
                    self.after_close(strip, true);
                }
                Kind::Var => {
                    self.push(Tok::VarBegin, open);
                    self.expr_tokens(Kind::Var)?;
                }
                Kind::Block => {
                    self.push(Tok::BlockBegin, open);
                    self.expr_tokens(Kind::Block)?;
                }
            }
            line_starting = s[..self.i].ends_with('\n');
        }
    }

    /// Whitespace handling after a closing delimiter.
    fn after_close(&mut self, strip: bool, trim_newline: bool) {
        if strip {
            let rest = self.rest();
            self.i += rest.len() - rest.trim_start_matches(is_py_space).len();
        } else if trim_newline && self.rest().starts_with('\n') {
            self.i += 1;
        }
    }

    fn expr_tokens(&mut self, kind: Kind) -> Result<()> {
        let mut depth = 0usize;
        loop {
            let rest = self.rest();
            let ws = rest.len() - rest.trim_start_matches(is_py_space).len();
            self.i += ws;
            let rest = self.rest();
            let start = self.i;
            if rest.is_empty() {
                return Err(err(self.line_at(start), "unexpected end of template inside tag"));
            }
            if depth == 0 {
                let closers: &[(&str, bool, bool)] = match kind {
                    // (text, strip whitespace after, trim_blocks newline)
                    Kind::Block => &[("-%}", true, false), ("+%}", false, false), ("%}", false, true)],
                    _ => &[("-}}", true, false), ("}}", false, false)],
                };
                if let Some((text, strip, trim)) = closers.iter().find(|(t, ..)| rest.starts_with(t)) {
                    self.push(if kind == Kind::Block { Tok::BlockEnd } else { Tok::VarEnd }, start);
                    self.i += text.len();
                    self.after_close(*strip, *trim);
                    return Ok(());
                }
            }
            let c = rest.chars().next().unwrap_or(' ');
            if c == '\'' || c == '"' {
                let s = self.string_lit(c)?;
                self.push(Tok::Str(s), start);
            } else if c.is_ascii_digit() {
                let tok = self.number()?;
                self.push(tok, start);
            } else if c.is_alphabetic() || c == '_' {
                let len = rest
                    .find(|ch: char| !(ch.is_alphanumeric() || ch == '_'))
                    .unwrap_or(rest.len());
                let name = rest[..len].to_string();
                self.i += len;
                self.push(Tok::Name(name), start);
            } else if let Some(op) = OPS.iter().chain(OPS_TAIL.iter()).find(|op| rest.starts_with(**op)) {
                match *op {
                    "(" | "[" | "{" => depth += 1,
                    ")" | "]" | "}" => depth = depth.saturating_sub(1),
                    _ => {}
                }
                self.i += op.len();
                self.push(Tok::Op(op), start);
            } else {
                return Err(err(self.line_at(start), format!("unexpected character {c:?}")));
            }
        }
    }

    fn number(&mut self) -> Result<Tok> {
        let b = self.s.as_bytes();
        let start = self.i;
        let digits = |i: &mut usize| {
            while *i < b.len() && (b[*i].is_ascii_digit() || b[*i] == b'_') {
                *i += 1;
            }
        };
        let mut j = self.i;
        digits(&mut j);
        let mut float = false;
        if j + 1 < b.len() && b[j] == b'.' && b[j + 1].is_ascii_digit() {
            float = true;
            j += 1;
            digits(&mut j);
        }
        if j < b.len() && (b[j] == b'e' || b[j] == b'E') {
            let mut k = j + 1;
            if k < b.len() && (b[k] == b'+' || b[k] == b'-') {
                k += 1;
            }
            if k < b.len() && b[k].is_ascii_digit() {
                float = true;
                j = k;
                digits(&mut j);
            }
        }
        let text: String = self.s[start..j].chars().filter(|c| *c != '_').collect();
        self.i = j;
        let line = self.line_at(start);
        if float {
            text.parse().map(Tok::Float).map_err(|_| err(line, format!("bad float {text}")))
        } else {
            text.parse().map(Tok::Int).map_err(|_| err(line, format!("bad integer {text}")))
        }
    }

    /// Python string literal with backslash escapes.
    fn string_lit(&mut self, quote: char) -> Result<String> {
        let start = self.i;
        let s: &'a str = self.s;
        let mut chars = s[start + 1..].char_indices();
        let mut out = String::new();
        while let Some((off, c)) = chars.next() {
            if c == quote {
                self.i = start + 1 + off + 1;
                return Ok(out);
            }
            if c != '\\' {
                out.push(c);
                continue;
            }
            let Some((_, e)) = chars.next() else { break };
            let mut hex = |n: usize| -> Option<char> {
                let h: String = (0..n).filter_map(|_| chars.next().map(|(_, c)| c)).collect();
                u32::from_str_radix(&h, 16).ok().and_then(char::from_u32)
            };
            let decoded = match e {
                'n' => Some('\n'),
                't' => Some('\t'),
                'r' => Some('\r'),
                '0' => Some('\0'),
                'a' => Some('\u{7}'),
                'b' => Some('\u{8}'),
                'f' => Some('\u{c}'),
                'v' => Some('\u{b}'),
                '\\' | '\'' | '"' => Some(e),
                '\n' => None,
                'x' => Some(hex(2).ok_or_else(|| err(self.line, "bad \\x escape"))?),
                'u' => Some(hex(4).ok_or_else(|| err(self.line, "bad \\u escape"))?),
                'U' => Some(hex(8).ok_or_else(|| err(self.line, "bad \\U escape"))?),
                other => {
                    out.push('\\');
                    Some(other)
                }
            };
            if let Some(d) = decoded {
                out.push(d);
            }
        }
        Err(err(self.line_at(start), "unterminated string literal"))
    }
}
