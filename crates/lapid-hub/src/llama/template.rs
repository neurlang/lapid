//! Chat-template glue on top of the shared Jinja interpreter (`crate::lfm3::jinja`):
//!
//! * `date_string`: Llama 3.x templates print "Today Date: {date_string}" and call
//!   `strftime_now("%d %b %Y")` when it is not given. The interpreter has no `strftime_now`, so
//!   the date is always passed as a global (today's UTC date, or a fixed one for verification).
//! * `tojson(indent=4)`: Llama 3.x templates pretty-print each tool with it; the interpreter's
//!   `tojson` ignores `indent`. Such templates get `| tojson(indent=4)` removed at load time and
//!   their `tools` passed as pre-rendered strings (Python `json.dumps(t, indent=4,
//!   ensure_ascii=False)`), which renders byte-identically.

use lapid::Result;

use crate::lfm3::jinja::{self, Template};

const INDENT4: &str = "tojson(indent=4)";

/// A parsed chat template plus the workarounds it needs.
pub(crate) struct ChatTemplate {
    template: Template,
    source: String,
    /// Tools are passed as `json.dumps(indent=4)` strings.
    pretty_tools: bool,
}

impl ChatTemplate {
    pub fn parse(source: &str) -> Result<ChatTemplate> {
        let pretty_tools = source.contains(INDENT4);
        let src = if pretty_tools {
            source.replace(&format!("| {INDENT4}"), "").replace(&format!("|{INDENT4}"), "")
        } else {
            source.to_string()
        };
        Ok(ChatTemplate { template: Template::parse(&src)?, source: source.to_string(), pretty_tools })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// Prepare the `tools` global.
    pub fn tools(&self, tools: jinja::Value) -> jinja::Value {
        match tools {
            jinja::Value::List(items) if self.pretty_tools => {
                jinja::Value::List(items.iter().map(|t| jinja::Value::Str(pretty_json(t, 4))).collect())
            }
            other => other,
        }
    }

    pub fn render(&self, globals: &[(&str, jinja::Value)]) -> Result<String> {
        self.template.render(globals)
    }
}

fn escape(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn float_repr(f: f64) -> String {
    if f.is_nan() {
        "NaN".into()
    } else if f.is_infinite() {
        if f > 0.0 { "Infinity".into() } else { "-Infinity".into() }
    } else if f == f.trunc() && f.abs() < 1e16 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

fn dump(v: &jinja::Value, indent: usize, level: usize, out: &mut String) {
    let pad = |out: &mut String, l: usize| {
        out.push('\n');
        out.push_str(&" ".repeat(indent * l));
    };
    match v {
        jinja::Value::None | jinja::Value::Undefined => out.push_str("null"),
        jinja::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        jinja::Value::Int(i) => out.push_str(&i.to_string()),
        jinja::Value::Float(f) => out.push_str(&float_repr(*f)),
        jinja::Value::Str(s) => escape(s, out),
        jinja::Value::List(items) if items.is_empty() => out.push_str("[]"),
        jinja::Value::List(items) => {
            out.push('[');
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                pad(out, level + 1);
                dump(x, indent, level + 1, out);
            }
            pad(out, level);
            out.push(']');
        }
        jinja::Value::Map(m) if m.is_empty() => out.push_str("{}"),
        jinja::Value::Map(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                pad(out, level + 1);
                escape(k, out);
                out.push_str(": ");
                dump(x, indent, level + 1, out);
            }
            pad(out, level);
            out.push('}');
        }
        _ => out.push_str("null"),
    }
}

/// Python `json.dumps(v, indent=indent, ensure_ascii=False)`.
pub(crate) fn pretty_json(v: &jinja::Value, indent: usize) -> String {
    let mut out = String::new();
    dump(v, indent, 0, &mut out);
    out
}

/// Today's UTC date as `strftime("%d %b %Y")` ("01 Nov 2026").
pub(crate) fn today() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    date_of_days((secs / 86_400) as i64)
}

/// Civil date of `days` since 1970-01-01 (Howard Hinnant's algorithm).
fn date_of_days(days: i64) -> String {
    const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{d:02} {} {y}", MONTHS[(m - 1) as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates() {
        assert_eq!(date_of_days(0), "01 Jan 1970");
        assert_eq!(date_of_days(19_930), "26 Jul 2024");
        assert_eq!(date_of_days(20_758), "01 Nov 2026");
    }

    #[test]
    fn pretty() {
        let v = jinja::Value::from_json_str(r#"{"a": [1, 2.5, "x\ny"], "b": {}, "c": [], "d": {"e": null, "f": true}, "é": "日本"}"#)
            .expect("json");
        let want = "{\n    \"a\": [\n        1,\n        2.5,\n        \"x\\ny\"\n    ],\n    \"b\": {},\n    \"c\": [],\n    \"d\": {\n        \"e\": null,\n        \"f\": true\n    },\n    \"é\": \"日本\"\n}";
        assert_eq!(pretty_json(&v, 4), want);
    }

    #[test]
    fn indent_rewrite() {
        let t = ChatTemplate::parse("{% for t in tools %}{{ t | tojson(indent=4) }}{% endfor %}").expect("parse");
        let tools = t.tools(jinja::Value::from_json_str(r#"[{"a": 1}]"#).expect("json"));
        assert_eq!(t.render(&[("tools", tools)]).expect("render"), "{\n    \"a\": 1\n}");
    }
}
