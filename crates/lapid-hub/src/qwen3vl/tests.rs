//! Tiny random-weight Qwen3-VL end to end (runs when `data/reference/qwen3vl/tiny` exists:
//! `uv run tools/reference/qwen3vl_ref.py --tiny --out data/reference/qwen3vl/tiny`).

use std::path::PathBuf;

fn tiny_dir() -> Option<PathBuf> {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/reference/qwen3vl/tiny");
    d.join("result.json").is_file().then_some(d)
}

#[test]
fn tiny_model_verifies_on_cpu() {
    let Some(dir) = tiny_dir() else {
        eprintln!("skipping: no data/reference/qwen3vl/tiny");
        return;
    };
    let out = std::env::temp_dir().join(format!("lapid-qwen3vl-tiny-{}.lapid", std::process::id()));
    super::convert::convert_hf(dir.join("model").to_str().expect("utf-8 path"), lapid_convert::OutDType::F32, &out).expect("convert");
    let report = super::verify::verify(&dir, out.to_str().expect("utf-8"), "fp32", Some("cpu")).expect("verify");
    std::fs::remove_file(&out).ok();
    assert_eq!(report["ok"], serde_json::Value::Bool(true), "{report:#}");
}
