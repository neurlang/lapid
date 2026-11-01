//! `lapid info` implementation: a JSON summary of a .lapid file.

use std::collections::BTreeMap;
use std::path::Path;

use lapid_tensor::Result;
use serde_json::{Value, json};

use crate::{FileKind, Reader, VERSION};

/// Arrays longer than this are summarized as `"<array of N>"` in `info` output.
const MAX_SHOWN_ARRAY: usize = 32;

/// Replace every array longer than `MAX_SHOWN_ARRAY` (at any depth) by `"<array of N>"`.
fn summarize(v: &mut Value) {
    match v {
        Value::Array(a) if a.len() > MAX_SHOWN_ARRAY => *v = json!(format!("<array of {}>", a.len())),
        Value::Array(a) => a.iter_mut().for_each(summarize),
        Value::Object(o) => o.values_mut().for_each(summarize),
        _ => {}
    }
}

/// Summary JSON: path, file size, format/version, kind (+ `extends` for extras),
/// metadata (arrays longer than 32 items shown as `"<array of N>"`), components (with
/// their tensor counts/bytes), blobs (name, media type, size), tensor count, per-dtype
/// counts/bytes, total parameters and — with `verify` — the result of checking every
/// tensor and blob hash (a mismatch is returned as an `Err` with code "hash_mismatch").
pub fn info(path: &Path, verify: bool) -> Result<Value> {
    let r = Reader::open(path)?;
    let file_size = r.as_bytes().len() as u64;
    let mut metadata = r.metadata.clone();
    summarize(&mut metadata);
    let mut dtypes: BTreeMap<&'static str, (u64, u64, u64)> = BTreeMap::new();
    let mut total_params: u64 = 0;
    let mut total_bytes: u64 = 0;
    for t in &r.tensors {
        let params = t.shape.iter().map(|&d| d as u64).product::<u64>();
        let e = dtypes.entry(t.dtype.name()).or_default();
        e.0 += 1;
        e.1 += t.nbytes;
        e.2 += params;
        total_params += params;
        total_bytes += t.nbytes;
    }
    let dtypes: serde_json::Map<String, Value> = dtypes
        .into_iter()
        .map(|(k, (count, bytes, params))| (k.to_string(), json!({"count": count, "bytes": bytes, "params": params})))
        .collect();
    // Each tensor belongs to the component with the longest matching prefix.
    let comps = r.components();
    let mut per_comp = vec![(0u64, 0u64); comps.len()];
    for t in &r.tensors {
        let best = comps
            .iter()
            .enumerate()
            .filter(|(_, c)| t.name.starts_with(&c.tensor_prefix))
            .max_by_key(|(_, c)| c.tensor_prefix.len());
        if let Some((i, _)) = best {
            per_comp[i].0 += 1;
            per_comp[i].1 += t.nbytes;
        }
    }
    let components: Vec<Value> = comps
        .iter()
        .zip(&per_comp)
        .map(|(c, (n, bytes))| {
            json!({
                "name": c.name, "architecture": c.architecture,
                "license": c.license.clone().or_else(|| r.metadata.get("license").and_then(Value::as_str).map(str::to_string)),
                "tensor_prefix": c.tensor_prefix, "tensor_count": n, "tensor_bytes": bytes,
            })
        })
        .collect();
    let blobs: Vec<Value> =
        r.blobs().iter().map(|b| json!({"name": b.name, "media_type": b.media_type, "nbytes": b.nbytes})).collect();
    let mut out = json!({
        "path": path.display().to_string(),
        "file_size": file_size,
        "format": r.metadata.get("format").cloned().unwrap_or(Value::Null),
        "version": VERSION,
        "kind": match r.kind() { FileKind::Model => "model", FileKind::Extra => "extra" },
        "metadata": metadata,
        "components": components,
        "blobs": blobs,
        "tensor_count": r.tensors.len(),
        "dtypes": dtypes,
        "total_params": total_params,
        "tensor_bytes": total_bytes,
    });
    if let Some(e) = r.extends() {
        out["extends"] = json!({"architecture": e.architecture, "component": e.component});
    }
    if verify {
        r.verify_tensors()?;
        out["verified"] = json!(true);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Writer;
    use lapid_tensor::DType;

    #[test]
    fn info_summarizes() {
        let p = std::env::temp_dir().join(format!("lapid-info-test-{}.lapid", std::process::id()));
        let meta = json!({
            "name": "t", "architecture": "test", "dtype": "fp32", "license": "mit",
            "source": {"repo": "x/y"}, "converter": "test",
            "model": {"tokens": (0..40).collect::<Vec<u32>>(), "short": [1, 2, 3], "nested": [vec![0u8; 33]]}
        });
        let mut w = Writer::create(&p, meta).expect("create");
        w.add("w", DType::F32, &[2, 2], &[0u8; 16]).expect("add");
        w.add("h", DType::F16, &[3], &[0u8; 6]).expect("add");
        w.add_blob("tokenizer/vocab.json", "application/json", b"[]").expect("blob");
        w.finish().expect("finish");
        let v = info(&p, true).expect("info");
        assert_eq!(v["metadata"]["model"]["tokens"], "<array of 40>");
        assert_eq!(v["metadata"]["model"]["short"], json!([1, 2, 3]));
        assert_eq!(v["metadata"]["model"]["nested"][0], "<array of 33>");
        assert_eq!(v["tensor_count"], 2);
        assert_eq!(v["total_params"], 7);
        assert_eq!(v["dtypes"]["f32"]["bytes"], 16);
        assert_eq!(v["dtypes"]["f16"]["count"], 1);
        assert_eq!(v["verified"], true);
        assert_eq!(v["kind"], "model");
        assert_eq!(v["components"][0]["name"], "test");
        assert_eq!(v["components"][0]["tensor_count"], 2);
        assert_eq!(v["blobs"][0], json!({"name": "tokenizer/vocab.json", "media_type": "application/json", "nbytes": 2}));
        std::fs::remove_file(&p).ok();
    }
}
