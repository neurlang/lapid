//! End-to-end check on the tiny random MoE checkpoint (`tools/reference/gemma4moe_ref.py
//! --tiny`), when `data/reference/gemma4moe/tiny` exists: HF conversion at fp32, then the
//! full `verify` (template, tokenizer, every layer, cached decode, greedy tokens, injected
//! embeddings with the bidirectional image-block mask) on the CPU.

use std::path::PathBuf;

use lapid_convert::OutDType;

fn tiny_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/reference/gemma4moe/tiny")
}

#[test]
fn tiny_moe_matches_reference() {
    let dir = tiny_dir();
    let model = dir.join("model");
    if !dir.join("result.json").is_file() || !model.is_dir() {
        eprintln!("skipped: {} not found (run tools/reference/gemma4moe_ref.py --tiny)", dir.display());
        return;
    }
    let out = std::env::temp_dir().join(format!("lapid-gemma4moe-tiny-{}.lapid", std::process::id()));
    let model_str = model.to_str().expect("utf-8 path");
    super::convert::convert_hf(model_str, OutDType::F32, &out).expect("convert");
    let r = super::verify::verify(&dir, out.to_str().expect("utf-8"), "fp32", Some("cpu")).expect("verify");
    std::fs::remove_file(&out).ok();
    assert_eq!(r["ok"], true, "{}", serde_json::to_string_pretty(&r).unwrap_or_default());
    assert_eq!(r["tokens_match"], true);
    assert!(!r["inject"].is_null(), "reference has no inject case");
}
