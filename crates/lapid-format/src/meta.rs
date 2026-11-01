//! METADATA validation: required keys, accepted `"format"` ids, and the format 1.1 keys
//! `"components"`, `"kind"` and `"extends"` (docs/spec/format.md).

use std::collections::HashSet;

use lapid_tensor::{Result, bail};
use serde_json::Value;

use crate::{Component, Extends, FORMAT_IDS, FileKind};

pub(crate) const REQUIRED_KEYS: [&str; 7] = ["format", "name", "architecture", "dtype", "license", "source", "converter"];
pub(crate) const MAX_COMPONENTS: usize = 64;
const MAX_PREFIX: usize = 256;
const MAX_NAME: usize = 256;

/// Checks the required metadata keys and the format id. Error "missing_metadata" / "format".
pub(crate) fn check_metadata(m: &Value) -> Result<()> {
    let Some(obj) = m.as_object() else {
        bail!("missing_metadata", "metadata must be a JSON object");
    };
    for k in REQUIRED_KEYS {
        if !obj.contains_key(k) {
            bail!("missing_metadata", "required metadata key {k:?} is missing");
        }
    }
    for k in ["format", "name", "architecture", "dtype", "license", "converter"] {
        if !obj[k].is_string() {
            bail!("missing_metadata", "metadata key {k:?} must be a string");
        }
    }
    if !obj["format"].as_str().is_some_and(|f| FORMAT_IDS.contains(&f)) {
        bail!("format", "metadata \"format\" is {}, expected one of {FORMAT_IDS:?}", obj["format"]);
    }
    if !obj["source"].get("repo").is_some_and(|r| r.is_string()) {
        bail!("missing_metadata", "metadata \"source\" must be an object with a string \"repo\"");
    }
    Ok(())
}

/// True for files of format 1.1 or later (the 1.1 keys are only interpreted there; in
/// "lapid/1" files they are free-form, like every other non-required key).
pub(crate) fn is_v1_1(m: &Value) -> bool {
    m.get("format").and_then(Value::as_str).is_some_and(|f| f != "lapid/1")
}

fn str_field<'a>(v: &'a Value, key: &str, what: &str) -> Result<&'a str> {
    match v.get(key) {
        Some(Value::String(s)) => Ok(s),
        Some(_) => bail!("format", "{what}: {key:?} must be a string"),
        None => bail!("missing_metadata", "{what}: {key:?} is missing"),
    }
}

/// `"kind"` and `"extends"`. Absent kind = "model"; unknown kinds are "unsupported".
pub(crate) fn parse_kind(m: &Value) -> Result<(FileKind, Option<Extends>)> {
    if !is_v1_1(m) {
        return Ok((FileKind::Model, None));
    }
    let kind = match m.get("kind") {
        None => FileKind::Model,
        Some(Value::String(s)) if s == "model" => FileKind::Model,
        Some(Value::String(s)) if s == "extra" => FileKind::Extra,
        Some(Value::String(s)) => bail!("unsupported", "unsupported file kind {s:?} (expected \"model\" or \"extra\")"),
        Some(_) => bail!("format", "metadata \"kind\" must be a string"),
    };
    let extends = match (kind, m.get("extends")) {
        (FileKind::Model, None) => None,
        (FileKind::Model, Some(_)) => bail!("format", "\"extends\" is only allowed with \"kind\": \"extra\""),
        (FileKind::Extra, None) => bail!("missing_metadata", "extras file (\"kind\": \"extra\") without \"extends\""),
        (FileKind::Extra, Some(e)) => {
            if !e.is_object() {
                bail!("format", "\"extends\" must be an object");
            }
            let ext = Extends {
                architecture: str_field(e, "architecture", "extends")?.to_string(),
                component: str_field(e, "component", "extends")?.to_string(),
            };
            let arch = m.get("architecture").and_then(Value::as_str).unwrap_or("");
            if arch != ext.architecture {
                bail!("format", "extras file: \"architecture\" {arch:?} != extends.architecture {:?}", ext.architecture);
            }
            Some(ext)
        }
    };
    Ok((kind, extends))
}

/// `"components"` resolved (config references looked up). Without the key: one implicit
/// component for model files, none for extras.
pub(crate) fn parse_components(m: &Value, kind: FileKind) -> Result<Vec<Component>> {
    let declared = if is_v1_1(m) { m.get("components") } else { None };
    let Some(declared) = declared else {
        if kind == FileKind::Extra {
            return Ok(Vec::new());
        }
        let arch = m.get("architecture").and_then(Value::as_str).unwrap_or("").to_string();
        let config = m.get(&arch).filter(|c| c.is_object()).cloned().unwrap_or(Value::Null);
        return Ok(vec![Component { name: arch.clone(), architecture: arch, config, license: None, tensor_prefix: String::new() }]);
    };
    let Some(obj) = declared.as_object() else {
        bail!("format", "metadata \"components\" must be an object");
    };
    if obj.is_empty() || obj.len() > MAX_COMPONENTS {
        bail!("limit_exceeded", "{} components (allowed 1..={MAX_COMPONENTS})", obj.len());
    }
    let mut prefixes = HashSet::new();
    let mut out = Vec::with_capacity(obj.len());
    for (name, c) in obj {
        let what = format!("component {name:?}");
        if name.is_empty() || name.len() > MAX_NAME {
            bail!("format", "invalid component name {name:?}");
        }
        if !c.is_object() {
            bail!("format", "{what} must be an object");
        }
        let architecture = str_field(c, "architecture", &what)?.to_string();
        let config = match c.get("config") {
            Some(Value::String(key)) => match m.get(key) {
                Some(v) if v.is_object() => v.clone(),
                _ => bail!("missing_metadata", "{what}: config key {key:?} is not a metadata object"),
            },
            Some(v @ Value::Object(_)) => v.clone(),
            Some(_) => bail!("format", "{what}: \"config\" must be an object or a metadata key"),
            None => bail!("missing_metadata", "{what}: \"config\" is missing"),
        };
        let license = match c.get("license") {
            None => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(_) => bail!("format", "{what}: \"license\" must be a string"),
        };
        let tensor_prefix = str_field(c, "tensor_prefix", &what)?.to_string();
        if tensor_prefix.len() > MAX_PREFIX || !(tensor_prefix.is_empty() || tensor_prefix.ends_with('.')) {
            bail!("format", "{what}: tensor_prefix {tensor_prefix:?} must be empty or end with '.'");
        }
        if !prefixes.insert(tensor_prefix.clone()) {
            bail!("format", "{what}: tensor_prefix {tensor_prefix:?} used by another component");
        }
        out.push(Component { name: name.clone(), architecture, config, license, tensor_prefix });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn base() -> Value {
        json!({"format": "lapid/1.1", "name": "m", "architecture": "tts", "dtype": "fp16", "license": "mit",
               "source": {"repo": "x/y"}, "converter": "t", "tts": {"n": 1}})
    }

    #[test]
    fn implicit_component() {
        let c = parse_components(&base(), FileKind::Model).expect("ok");
        assert_eq!(c.len(), 1);
        assert_eq!((c[0].name.as_str(), c[0].tensor_prefix.as_str()), ("tts", ""));
        assert_eq!(c[0].config, json!({"n": 1}));
    }

    #[test]
    fn declared_components() {
        let mut m = base();
        m["codec"] = json!({"k": 2});
        m["components"] = json!({
            "lm": {"architecture": "llama", "config": {"d": 8}, "license": "apache-2.0", "tensor_prefix": "lm."},
            "codec": {"architecture": "codec", "config": "codec", "tensor_prefix": "codec."}
        });
        let c = parse_components(&m, FileKind::Model).expect("ok");
        let codec = c.iter().find(|c| c.name == "codec").expect("codec");
        assert_eq!(codec.config, json!({"k": 2}));
        assert_eq!(codec.license, None);
        let lm = c.iter().find(|c| c.name == "lm").expect("lm");
        assert_eq!(lm.license.as_deref(), Some("apache-2.0"));
        // Bad prefix, duplicate prefix, dangling config reference.
        m["components"]["lm"]["tensor_prefix"] = json!("lm");
        assert_eq!(parse_components(&m, FileKind::Model).err().map(|e| e.code), Some("format"));
        m["components"]["lm"]["tensor_prefix"] = json!("codec.");
        assert_eq!(parse_components(&m, FileKind::Model).err().map(|e| e.code), Some("format"));
        m["components"]["lm"]["tensor_prefix"] = json!("lm.");
        m["components"]["lm"]["config"] = json!("nope");
        assert_eq!(parse_components(&m, FileKind::Model).err().map(|e| e.code), Some("missing_metadata"));
        // In a "lapid/1" file the keys are free-form.
        m["format"] = json!("lapid/1");
        assert_eq!(parse_components(&m, FileKind::Model).expect("v1").len(), 1);
    }

    #[test]
    fn kinds() {
        let mut m = base();
        assert_eq!(parse_kind(&m).expect("ok"), (FileKind::Model, None));
        m["kind"] = json!("extra");
        assert_eq!(parse_kind(&m).err().map(|e| e.code), Some("missing_metadata"));
        m["extends"] = json!({"architecture": "tts", "component": "voice"});
        let (k, e) = parse_kind(&m).expect("extra");
        assert_eq!(k, FileKind::Extra);
        assert_eq!(e.map(|e| e.component), Some("voice".to_string()));
        assert!(parse_components(&m, FileKind::Extra).expect("ok").is_empty());
        m["extends"]["architecture"] = json!("other");
        assert_eq!(parse_kind(&m).err().map(|e| e.code), Some("format"));
        m["kind"] = json!("lora");
        assert_eq!(parse_kind(&m).err().map(|e| e.code), Some("unsupported"));
        m["kind"] = json!("model");
        assert_eq!(parse_kind(&m).err().map(|e| e.code), Some("format"), "extends without extra");
    }
}
