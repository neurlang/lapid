//! Model-independent pieces of `lapid verify <model>` / `lapid bench <model>`: the per-model
//! [`Descriptor`], the reference directory ([`Reference`]: `result.json` +
//! `activations.safetensors` written by `tools/reference/<model>_ref.py`), reference
//! accessors, tap-name rules and the fake-quant kernel comparison.
//!
//! Reference activation naming (shared by every causal LM reference script):
//! `<case>.<tap>` (e.g. `short.layers.3.op`) for an uncached forward over all rows,
//! `<case>.<tap>.tail` for the last rows of a long prompt, `<case>.logits.last`, and
//! `<case>.gen.logits` for teacher-forced decode steps. A model emits taps without the case
//! prefix; the reference decides which taps are checked.

use std::path::{Path, PathBuf};

use lapid::{Error, Result};
use serde_json::{Value, json};

use crate::compare::Comparison;
use crate::reference::Activations;
use crate::report::{KERNEL_TOLERANCE, diff};

/// What a model tells the harness about itself.
#[derive(Copy, Clone, Debug)]
pub struct Descriptor {
    /// Name routed by `lapid verify <name>` / `lapid bench <name>`.
    pub name: &'static str,
    /// Default model (HF repo id, directory, `.gguf` or `.lapid`).
    pub default_model: &'static str,
    pub default_dtype: &'static str,
    /// Default reference directory (relative to the repo root).
    pub reference_dir: &'static str,
    /// Script that writes the reference (named in errors).
    pub reference_script: &'static str,
    /// Human description of the reference run.
    pub reference_label: &'static str,
    /// Dtypes whose greedy output must be token-exact.
    pub strict_dtypes: &'static [&'static str],
    /// Bench defaults (prompt tokens, generated tokens); `LAPID_BENCH_PROMPT_TOKENS` /
    /// `LAPID_BENCH_GEN_TOKENS` override.
    pub bench_prompt_tokens: usize,
    pub bench_gen_tokens: usize,
}

impl Descriptor {
    pub fn tokens_must_match(&self, dtype: &str) -> bool {
        self.strict_dtypes.contains(&dtype)
    }

    pub fn bench_sizes(&self) -> Result<(usize, usize)> {
        Ok((env_usize("LAPID_BENCH_PROMPT_TOKENS", self.bench_prompt_tokens)?, env_usize("LAPID_BENCH_GEN_TOKENS", self.bench_gen_tokens)?))
    }
}

/// `key` from the environment as a count, else `default`.
pub fn env_usize(key: &str, default: usize) -> Result<usize> {
    match std::env::var(key) {
        Ok(v) => v.trim().parse().map_err(|_| Error::new("invalid_argument", format!("{key}={v:?} is not a count"))),
        Err(_) => Ok(default),
    }
}

/// A reference directory: `result.json` and `activations.safetensors`.
pub struct Reference {
    pub dir: PathBuf,
    pub result: Value,
    pub acts: Activations,
}

impl Reference {
    pub fn open(dir: &Path, script: &str) -> Result<Reference> {
        let raw = std::fs::read_to_string(dir.join("result.json"))
            .map_err(|e| Error::new("io", format!("{}/result.json: {e} (run {script})", dir.display())))?;
        Ok(Reference { dir: dir.to_path_buf(), result: serde_json::from_str(&raw)?, acts: Activations::open(&dir.join("activations.safetensors"))? })
    }

    /// `result.json` `cases.<name>` (Null when absent).
    pub fn case(&self, name: &str) -> Value {
        self.result.pointer(&format!("/cases/{name}")).cloned().unwrap_or(Value::Null)
    }

    /// Reference taps of `case` (`<case>.<tap>` names, without `<case>.gen.*`).
    pub fn taps(&self, case: &str) -> Vec<String> {
        let p = format!("{case}.");
        let gen_prefix = format!("{case}.gen.");
        self.acts.names().iter().filter(|n| n.starts_with(&p) && !n.starts_with(&gen_prefix)).cloned().collect()
    }

    /// Leading dimension of `name` (e.g. the tail length of `<case>.<tap>.tail`).
    pub fn rows(&self, name: &str) -> Option<usize> {
        self.acts.shape(name).ok().and_then(|s| s.first().copied())
    }
}

/// Integer id array `v[key]`.
pub fn ids(v: &Value, key: &str) -> Result<Vec<u32>> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("reference", format!("result.json: missing array {key:?}")))?
        .iter()
        .map(|x| x.as_u64().map(|n| n as u32).ok_or_else(|| Error::new("reference", format!("{key:?}: non-integer id"))))
        .collect()
}

/// String `v[key]`.
pub fn str_of<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key).and_then(Value::as_str).ok_or_else(|| Error::new("reference", format!("result.json: missing string {key:?}")))
}

/// Rows `[n - tail, n)` of a row-major `[n, d]` buffer.
pub fn tail_rows(v: &[f32], d: usize, tail: usize) -> Vec<f32> {
    let n = v.len() / d.max(1);
    v[(n.saturating_sub(tail)) * d..].to_vec()
}

/// Values of `name` in a named list (empty when missing).
pub fn named<'a>(list: &'a [(String, Vec<f32>)], name: &str) -> &'a [f32] {
    list.iter().find(|(k, _)| k == name).map_or(&[], |(_, v)| v.as_slice())
}

/// Kernel correctness of a quantized model: per tensor, compare ours with the closer of two
/// fake-quant references (`f32_acts`: weights dequantized to f32; `int8_acts`: same with
/// int8 activations) at [`KERNEL_TOLERANCE`]. Returns the comparison and which reference
/// each tensor matched.
pub fn kernel_compare(
    names: &[String],
    ours: &[(String, Vec<f32>)],
    f32_acts: &[(String, Vec<f32>)],
    int8_acts: &[(String, Vec<f32>)],
) -> Result<(Comparison, Value)> {
    let mut merged = Comparison::new();
    let mut which = serde_json::Map::new();
    for n in names {
        let (o, w, a) = (named(ours, n), named(f32_acts, n), named(int8_acts, n));
        let rw = diff(o, w)?;
        let ra = diff(o, a)?;
        let use_a = ra.rel < rw.rel || (rw.rel.is_nan() && !ra.rel.is_nan());
        merged.compare(n, o, if use_a { a } else { w }, KERNEL_TOLERANCE);
        which.insert(n.clone(), json!(if use_a { "int8_activations" } else { "f32_activations" }));
    }
    Ok((merged, Value::Object(which)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers() {
        assert_eq!(tail_rows(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 2), vec![3.0, 4.0, 5.0, 6.0]);
        let v = json!({"a": [1, 2], "s": "x"});
        assert_eq!(ids(&v, "a").expect("ids"), vec![1, 2]);
        assert_eq!(str_of(&v, "s").expect("s"), "x");
        assert!(ids(&v, "b").is_err());
        let d = Descriptor {
            name: "m",
            default_model: "x/y",
            default_dtype: "int8",
            reference_dir: "data/reference/m",
            reference_script: "tools/reference/m_ref.py",
            reference_label: "PyTorch",
            strict_dtypes: &["fp32", "fp16"],
            bench_prompt_tokens: 512,
            bench_gen_tokens: 128,
        };
        assert!(d.tokens_must_match("fp16") && !d.tokens_must_match("int8"));
    }
}
