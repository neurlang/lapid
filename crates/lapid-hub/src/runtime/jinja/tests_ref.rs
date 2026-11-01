//! Byte-exact comparison against real jinja2 output. `testdata/cases.json` was produced by
//! rendering with transformers' exact environment (ImmutableSandboxedEnvironment,
//! trim/lstrip blocks, loopcontrols, generation extension, custom tojson, raise_exception).

use super::{Template, Value};

const CASES: &str = include_str!("testdata/cases.json");
const LFM3_TEMPLATE: &str = include_str!("testdata/lfm3_chat_template.jinja");

fn field<'a>(case: &'a Value, key: &str) -> Option<&'a Value> {
    match case {
        Value::Map(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

fn text(case: &Value, key: &str) -> String {
    match field(case, key) {
        Some(Value::Str(s)) => s.clone(),
        other => panic!("case field {key}: {other:?}"),
    }
}

fn run_group(group: &str, template: Option<&Template>) -> usize {
    let all = Value::from_json_str(CASES).expect("cases.json parses");
    let Some(Value::List(cases)) = field(&all, group) else { panic!("missing group {group}") };
    let mut failures = Vec::new();
    for case in cases {
        let name = text(case, "name");
        let owned;
        let tpl = match template {
            Some(t) => t,
            None => {
                owned = Template::parse(&text(case, "template")).expect("template parses");
                &owned
            }
        };
        let Some(Value::Map(globals)) = field(case, "globals") else { panic!("{name}: globals") };
        let globals: Vec<(&str, Value)> = globals.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        let got = tpl.render(&globals);
        match (field(case, "expected"), got) {
            (Some(Value::Str(want)), Ok(got)) if *want == got => {}
            (None, Err(e)) if e.code == "template" && e.message == text(case, "error") => {}
            (want, got) => failures.push(format!("{name}:\n  want {want:?}\n  got  {got:?}")),
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
    cases.len()
}

#[test]
fn lfm3_template_matches_jinja2() {
    let tpl = Template::parse(LFM3_TEMPLATE).expect("LFM3 template parses");
    assert!(run_group("lfm3", Some(&tpl)) >= 15);
}

#[test]
fn mini_templates_match_jinja2() {
    assert!(run_group("mini", None) >= 20);
}

/// The fixture is a copy of the HF snapshot's template; check it when the snapshot exists.
#[test]
fn fixture_matches_hf_snapshot() {
    let Some(home) = std::env::var_os("HOME") else { return };
    let path = std::path::Path::new(&home).join(
        ".cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/\
         ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7/chat_template.jinja",
    );
    if let Ok(src) = std::fs::read_to_string(path) {
        assert_eq!(src, LFM3_TEMPLATE);
    }
}
