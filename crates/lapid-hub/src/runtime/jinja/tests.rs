//! Unit tests: whitespace control, scoping, filters, tojson formatting, JSON parsing, errors.

use super::value::float_repr;
use super::{Template, Value};

fn render(src: &str, globals: &[(&str, Value)]) -> String {
    Template::parse(src).expect("parse").render(globals).expect("render")
}

fn json(s: &str) -> Value {
    Value::from_json_str(s).expect("json")
}

#[test]
fn whitespace_control() {
    assert_eq!(render("a\n  {% if true %}\n    x\n  {% endif %}\nb", &[]), "a\n    x\nb");
    assert_eq!(render("a  {{- 1 -}}  b", &[]), "a1b");
    assert_eq!(render("a {%- if true -%} \n b {%- endif %}", &[]), "ab");
    assert_eq!(render("x {{ 1 }}\ny\n", &[]), "x 1\ny");
    assert_eq!(render("a\n  {# note #}\nb", &[]), "a\nb");
    assert_eq!(render("  {%+ if true %}x{% endif %}", &[]), "  x");
}

#[test]
fn scoping() {
    let src = "{% set x = 1 %}{% for i in [5] %}{% set x = i %}{{ x }}{% endfor %}{{ x }}";
    assert_eq!(render(src, &[]), "51");
    assert_eq!(render("{% set x = 1 %}{% if 1 %}{% set x = 2 %}{% endif %}{{ x }}", &[]), "2");
    let ns = "{% set ns = namespace(n=0, s='') %}{% for i in [1, 2] %}{% set ns.n = ns.n + i %}\
              {% set ns.s = ns.s ~ i %}{% endfor %}{{ ns.n }}/{{ ns.s }}";
    assert_eq!(render(ns, &[]), "3/12");
    // Macros see globals and top-level variables, not the caller's loop locals.
    let mac = "{% macro m() %}{{ g }}{{ i }}{% endmacro %}{% set g = 'G' %}\
               {% for i in [1] %}{{ m() }}{% endfor %}";
    assert_eq!(render(mac, &[]), "G");
    let generation = "{% for m in [1] %}{% generation %}{% set y = 2 %}{{ y }}{% endgeneration %}{{ y }}{% endfor %}";
    assert_eq!(render(generation, &[]), "2");
}

#[test]
fn values_print_like_python() {
    let src = "{{ none }}|{{ true }}|{{ u }}|{{ 1.0 }}|{{ [1, 'a'] }}|{{ {'k': none} }}";
    assert_eq!(render(src, &[]), "None|True||1.0|[1, 'a']|{'k': None}");
    assert_eq!(render("{{ d.missing.deeper }}|{{ d['x'] }}", &[("d", json("{}"))]), "|");
}

#[test]
fn filters_and_precedence() {
    let g = [("c", Value::str("hello")), ("t", Value::str("lo")), ("n", Value::Int(5))];
    assert_eq!(render("{{ (c | length) - (t | length) }}", &g), "3");
    assert_eq!(render("{{ 'x' + n | string + 'y' }}", &g), "x5y");
    assert_eq!(render("{{ 'x' + (['a', 'b'] | join('\\n')) }}", &g), "xa\nb");
    assert_eq!(render("{{ u | default(1) }}{{ c | upper }}{{ ' a ' | trim }}", &g), "1HELLOa");
    assert_eq!(render("{{ c[:2] }}{{ c[-1] }}{{ c.split('l')[-1] }}", &g), "heoo");
    assert_eq!(render("{{ 'a' if 0 }}{{ 'b' if 1 else 'c' }}", &g), "b");
}

#[test]
fn tojson_matches_python_json_dumps() {
    let v = json(r#"{"z": 1, "a": [1.5, 1e20, 1e-5, 100.0, -0.0, true, null], "s": "q\"\n\u0001é", "o": {}}"#);
    let out = render("{{ v | tojson }}", &[("v", v)]);
    assert_eq!(out, r#"{"z": 1, "a": [1.5, 1e+20, 1e-05, 100.0, -0.0, true, null], "s": "q\"\n\u0001é", "o": {}}"#);
}

#[test]
fn float_repr_cases() {
    let cases = [
        (1.5, "1.5"),
        (1e20, "1e+20"),
        (1e16, "1e+16"),
        (1e15, "1000000000000000.0"),
        (0.0001, "0.0001"),
        (0.00001, "1e-05"),
        (2.5e-7, "2.5e-07"),
        (0.1, "0.1"),
        (-3.0, "-3.0"),
        (123456.789, "123456.789"),
        (1.7976931348623157e308, "1.7976931348623157e+308"),
    ];
    for (f, want) in cases {
        assert_eq!(float_repr(f), want, "{f}");
    }
}

#[test]
fn json_parse_keeps_order_and_types() {
    let v = json(r#" {"b": 1, "a": [2.0, "x\u00e9\ud83d\ude00"], "c": {"y": null, "x": false}} "#);
    let Value::Map(m) = &v else { panic!("map") };
    let keys: Vec<&str> = m.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["b", "a", "c"]);
    assert_eq!(m[1].1, Value::List(vec![Value::Float(2.0), Value::str("xé😀")]));
    for bad in ["", "{", "[1,]", "{\"a\" 1}", "tru", "\"\\q\"", "1 2"] {
        let e = Value::from_json_str(bad).expect_err(bad);
        assert_eq!(e.code, "json");
    }
}

#[test]
fn from_serde_json() {
    let v = Value::from_json(&serde_json::json!({"a": [1, 2.5, "s", null, true]}));
    assert_eq!(
        v,
        Value::map(vec![(
            "a".into(),
            Value::List(vec![Value::Int(1), Value::Float(2.5), "s".into(), Value::None, true.into()])
        )])
    );
}

#[test]
fn errors_have_template_code() {
    let e = Template::parse("{% if x %}").expect_err("unclosed");
    assert_eq!(e.code, "template");
    let e = Template::parse("{{ 1 + }}").expect_err("bad expr");
    assert_eq!(e.code, "template");
    let e = Template::parse("{% bogus %}").expect_err("unknown tag");
    assert_eq!(e.code, "template");
    let t = Template::parse("{{ raise_exception('nope') }}").expect("parse");
    let e = t.render(&[]).expect_err("raise");
    assert_eq!((e.code, e.message.as_str()), ("template", "nope"));
    let e = Template::parse("{{ 'a' + 1 }}").expect("parse").render(&[]).expect_err("type");
    assert_eq!(e.code, "template");
    let e = Template::parse("{% macro f() %}{{ f() }}{% endmacro %}{{ f() }}")
        .expect("parse")
        .render(&[])
        .expect_err("recursion");
    assert_eq!(e.code, "template");
}

#[test]
fn deep_nesting_is_an_error_not_a_crash() {
    let src = format!("{{{{ {}1{} }}}}", "(".repeat(500), ")".repeat(500));
    assert_eq!(Template::parse(&src).expect_err("deep").code, "template");
    let src = format!("{{{{ {}1 }}}}", "-".repeat(500));
    assert_eq!(Template::parse(&src).expect_err("deep").code, "template");
    let src = "{% if 1 %}".repeat(200);
    assert_eq!(Template::parse(&src).expect_err("deep").code, "template");
}

#[test]
fn dictsort_tojson_indent_strftime() {
    let m = json(r#"{"b": 1, "A": 2, "c": [1, {"x": null}], "e": []}"#);
    assert_eq!(render("{% for k, v in m | dictsort %}{{ k }}{% endfor %}", &[("m", m.clone())]), "Abce");
    assert_eq!(render("{% for k, v in m | dictsort(true) %}{{ k }}{% endfor %}", &[("m", m.clone())]), "Abce");
    assert_eq!(render("{% for k, v in m | dictsort(reverse=true) %}{{ k }}{% endfor %}", &[("m", m.clone())]), "ecbA");
    // Python: json.dumps(m, indent=2, ensure_ascii=False)
    let want = "{\n  \"b\": 1,\n  \"A\": 2,\n  \"c\": [\n    1,\n    {\n      \"x\": null\n    }\n  ],\n  \"e\": []\n}";
    assert_eq!(render("{{ m | tojson(indent=2) }}", &[("m", m)]), want);
    assert_eq!(render("{{ 'a\nb\n\nc' | indent(2) }}", &[]), "a\n  b\n\n  c");
    let d = render("{{ strftime_now('%Y') }}", &[]);
    assert_eq!(d.len(), 4);
    assert!(d.starts_with("20"));
}
