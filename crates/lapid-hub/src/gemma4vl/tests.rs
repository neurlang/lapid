//! Tiny-model check (runs when `data/reference/gemma4vl/tiny` exists; regenerate with
//! `uv run tools/reference/gemma4vl_ref.py --tiny --out data/reference/gemma4vl/tiny`).

use std::path::PathBuf;

fn tiny_dir() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/reference/gemma4vl/tiny");
    root.join("model").is_dir().then_some(root)
}

#[test]
fn tiny_model_matches_reference() {
    let Some(dir) = tiny_dir() else {
        eprintln!("skipping: no data/reference/gemma4vl/tiny");
        return;
    };
    let model = dir.join("model");
    let model = model.to_str().expect("utf-8 path");
    let report = super::verify::verify(&dir, model, "fp32", Some("cpu")).expect("verify");
    assert!(report["ok"].as_bool().unwrap_or(false), "{}", serde_json::to_string_pretty(&report).unwrap_or_default());
}
