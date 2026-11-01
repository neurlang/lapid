//! The generic converter: Hugging Face safetensors (single or sharded) and GGUF checkpoints
//! become one `.lapid` file, driven by a model's declarative [`ConvertSpec`] (name rules,
//! role rules) plus small [`ConvertHooks`] for what a table cannot say (config parsing,
//! expected tensors, reshapes, row orders, value transforms).
//!
//! Output layout: our tensor names; metadata `name`, `architecture`, `dtype`, `license`,
//! `source`, the model's config object under its architecture key (plus text-asset keys
//! for language models) and a single-component declaration; text assets as two blobs.
//! Precision follows `crate::policy` through the role rules. GGUF tensors keep their bytes
//! whenever the policy allows it (Q8_0 / Q4_0 repacked to int8 / int4, K-quants copied).
//!
//! ```ignore
//! pub static CONVERT: ConvertSpec = ConvertSpec {
//!     architecture: "mymodel", license: "apache-2.0",
//!     hf_model_types: &["mymodel"], gguf_architectures: &["mymodel"],
//!     hf_names: &[Rule::drop("lm_head.weight")],
//!     gguf_names: &[Rule::new("blk.{i}.attn_q.{w}", "model.layers.{i}.self_attn.q_proj.{w}")],
//!     roles: &[RoleRule::new("model.embed_tokens.weight", Role::LinearInt8)],
//!     text_assets: true,
//!     hooks: &MyHooks,
//! };
//! CONVERT.convert("owner/repo", OutDType::Int8, Path::new("out.lapid"))?;
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use lapid_tensor::{Error, Result, bail};
use serde_json::{Value, json};

use crate::OutDType;
use crate::assets::{HfFiles, TextAssets};
use crate::gguf::{GgufFile, GgufValue, is_gguf, write_ggml_tensor};
use crate::mapping::{Renamed, RoleRule, Rule, rename, role_of};
use crate::policy::Role;
use crate::safetensors::SafeTensors;
use crate::transforms::gather_rows;
use crate::writer::{ModelWriter, RoleArc, file_hash, metadata, read_json};

/// Where a tensor comes from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SourceKind {
    Hf,
    Gguf,
}

/// What to write for one tensor (see [`ConvertHooks::transform`]).
pub enum Emit {
    /// Write it as stored (raw passthrough when possible).
    Keep,
    /// Skip it.
    Drop,
    /// Write these f32 tensors (name, shape, values) instead.
    Replace(Vec<(String, Vec<usize>, Vec<f32>)>),
}

/// Read access to the source tensors by our names (for transforms that combine tensors,
/// e.g. BatchNorm folding).
pub trait TensorSource {
    fn kind(&self) -> SourceKind;
    fn names(&self) -> Vec<String>;
    /// Shape after [`ConvertHooks::reshape`].
    fn shape(&self, name: &str) -> Option<Vec<usize>>;
    /// Values (dequantized) in the [`shape`](Self::shape) layout.
    fn f32(&self, name: &str) -> Result<Vec<f32>>;
}

/// Model-specific conversion logic beyond the name / role tables. Only `expected` is
/// required; `cfg` is the model config object as stored in the metadata.
pub trait ConvertHooks: Sync {
    /// Config object from HF `config.json` (default: stored as is). Validate here.
    fn hf_config(&self, config: &Value) -> Result<Value> {
        Ok(config.clone())
    }

    /// HF-style config object from GGUF metadata (required to convert GGUF).
    fn gguf_config(&self, f: &GgufFile) -> Result<Value> {
        bail!("unsupported_model", "no GGUF converter for architecture {:?}", f.architecture())
    }

    /// Every tensor (our names) the output must contain.
    fn expected(&self, cfg: &Value) -> Result<Vec<String>>;

    /// New shape with the same element order (e.g. `[D, K]` -> `[D, 1, K]`).
    fn reshape(&self, _name: &str, _shape: &[usize], _cfg: &Value, _src: SourceKind) -> Option<Vec<usize>> {
        None
    }

    /// Source row of every output row (applied to raw rows: quantized blocks are kept), e.g.
    /// `transforms::llama_qk_unpermute` for GGUF q/k projections.
    fn row_order(&self, _name: &str, _shape: &[usize], _cfg: &Value, _src: SourceKind) -> Option<Vec<usize>> {
        None
    }

    /// Value transform of one tensor (dequantizes it): fold, split, combine.
    fn transform(&self, _name: &str, _src: &dyn TensorSource, _cfg: &Value) -> Result<Emit> {
        Ok(Emit::Keep)
    }

    /// Final edit of the whole metadata object.
    fn metadata(&self, _meta: &mut Value) {}
}

/// A model's declarative conversion description.
pub struct ConvertSpec {
    /// Metadata `architecture` and config object key.
    pub architecture: &'static str,
    /// SPDX-like license id recorded in the output.
    pub license: &'static str,
    /// HF `config.json` `model_type` values this spec converts.
    pub hf_model_types: &'static [&'static str],
    /// GGUF `general.architecture` values this spec converts.
    pub gguf_architectures: &'static [&'static str],
    /// HF name -> our name (unmatched names are kept as is).
    pub hf_names: &'static [Rule],
    /// GGUF name -> our name (unmatched names are an error).
    pub gguf_names: &'static [Rule],
    /// Our name -> role (unmatched: `mapping::default_role`).
    pub roles: &'static [RoleRule],
    /// Store tokenizer, chat template, special tokens and generation defaults.
    pub text_assets: bool,
    pub hooks: &'static dyn ConvertHooks,
}

enum Src {
    Hf(usize, String),
    Gguf(usize),
}

struct Sources<'a> {
    kind: SourceKind,
    shards: &'a [SafeTensors],
    gguf: Option<&'a GgufFile>,
    map: BTreeMap<String, Src>,
    cfg: &'a Value,
    hooks: &'static dyn ConvertHooks,
}

impl Sources<'_> {
    fn raw_shape(&self, src: &Src) -> Result<Vec<usize>> {
        match src {
            Src::Hf(i, n) => self.shards[*i].shape(n),
            Src::Gguf(i) => Ok(self.gguf.map(|g| g.tensors[*i].shape.clone()).unwrap_or_default()),
        }
    }
}

impl TensorSource for Sources<'_> {
    fn kind(&self) -> SourceKind {
        self.kind
    }

    fn names(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }

    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        let s = self.raw_shape(self.map.get(name)?).ok()?;
        Some(self.hooks.reshape(name, &s, self.cfg, self.kind).unwrap_or(s))
    }

    fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let src = self.map.get(name).ok_or_else(|| Error::new("missing_tensor", format!("no tensor {name:?}")))?;
        let data = match (src, self.gguf) {
            (Src::Hf(i, n), _) => self.shards[*i].to_f32(n)?,
            (Src::Gguf(i), Some(g)) => g.to_f32(&g.tensors[*i])?,
            (Src::Gguf(_), None) => bail!("internal", "gguf source without a file"),
        };
        let shape = self.raw_shape(src)?;
        match self.hooks.row_order(name, &shape, self.cfg, self.kind) {
            Some(order) => gather_rows(&data, &order),
            None => Ok(data),
        }
    }
}

impl ConvertSpec {
    /// Dtype-policy role of one of our tensors.
    pub fn role(&self, name: &str, shape: &[usize]) -> Role {
        role_of(self.roles, name, shape)
    }

    /// The role table as a shared classifier (writers, calibrated conversion).
    pub fn role_arc(&'static self) -> RoleArc {
        Arc::new(move |n: &str, s: &[usize]| role_of(self.roles, n, s))
    }

    /// Whether this spec converts an HF config with `model_type`.
    pub fn accepts_model_type(&self, model_type: &str) -> bool {
        self.hf_model_types.contains(&model_type)
    }

    pub fn accepts_gguf(&self, f: &GgufFile) -> bool {
        f.architecture().is_some_and(|a| self.gguf_architectures.contains(&a))
    }

    /// Convert `src` (GGUF file, HF repo id or HF directory) to `out` at `dtype`.
    pub fn convert(&'static self, src: &str, dtype: OutDType, out: &Path) -> Result<()> {
        let p = Path::new(src);
        if p.is_file() && is_gguf(p)? {
            return self.convert_gguf(p, dtype, out);
        }
        self.convert_hf(src, dtype, out)
    }

    pub fn convert_hf(&'static self, src: &str, dtype: OutDType, out: &Path) -> Result<()> {
        let files = HfFiles::locate(src)?;
        let raw = read_json(&files.config)?;
        let mt = raw.get("model_type").and_then(Value::as_str).unwrap_or("");
        if !self.accepts_model_type(mt) {
            bail!("unsupported_model", "{}: config.json model_type {mt:?} is not {:?}", files.repo, self.architecture);
        }
        let cfg = self.hooks.hf_config(&raw)?;
        let assets = if self.text_assets { Some(TextAssets::from_hf(&files)?) } else { None };
        let shards: Vec<SafeTensors> = files.weights.iter().map(|p| SafeTensors::open(p)).collect::<Result<_>>()?;
        let mut map = BTreeMap::new();
        for (i, st) in shards.iter().enumerate() {
            for n in st.names() {
                match rename(self.hf_names, &n) {
                    Renamed::Drop => {}
                    Renamed::To(o) => {
                        map.insert(o, Src::Hf(i, n));
                    }
                    Renamed::NoMatch => {
                        map.insert(n.clone(), Src::Hf(i, n));
                    }
                }
            }
        }
        let mut source = json!({ "repo": files.repo });
        if files.remote {
            source["revision"] = json!("main");
        }
        if files.weights.len() == 1 {
            source["file_hash"] = json!(file_hash(&files.weights[0])?);
        }
        let srcs = Sources { kind: SourceKind::Hf, shards: &shards, gguf: None, map, cfg: &cfg, hooks: self.hooks };
        self.write(&srcs, &files.name(self.architecture), source, assets.as_ref(), dtype, out)
    }

    pub fn convert_gguf(&'static self, path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
        let f = GgufFile::open(path)?;
        if !self.accepts_gguf(&f) {
            bail!("unsupported_model", "{}: GGUF architecture {:?} is not {:?}", path.display(), f.architecture(), self.architecture);
        }
        let cfg = self.hooks.gguf_config(&f)?;
        let assets = if self.text_assets { Some(TextAssets::from_gguf(&f)?) } else { None };
        let mut map = BTreeMap::new();
        for (i, t) in f.tensors.iter().enumerate() {
            match rename(self.gguf_names, &t.name) {
                Renamed::Drop => {}
                Renamed::To(o) => {
                    map.insert(o, Src::Gguf(i));
                }
                Renamed::NoMatch => bail!("unsupported_model", "GGUF: unknown {} tensor {:?}", self.architecture, t.name),
            }
        }
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or(self.architecture).to_string();
        let source = json!({
            "repo": f.get("general.name").and_then(GgufValue::as_str).map_or_else(|| name.clone(), |n| format!("gguf:{n}")),
            "file": path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
            "format": "gguf",
            "file_hash": file_hash(path)?,
        });
        let srcs = Sources { kind: SourceKind::Gguf, shards: &[], gguf: Some(&f), map, cfg: &cfg, hooks: self.hooks };
        self.write(&srcs, &name, source, assets.as_ref(), dtype, out)
    }

    fn write(&'static self, srcs: &Sources<'_>, name: &str, source: Value, assets: Option<&TextAssets>, dtype: OutDType, out: &Path) -> Result<()> {
        let arch = self.architecture;
        let expected = self.hooks.expected(srcs.cfg)?;
        let mut cfg = srcs.cfg.clone();
        if let Some(a) = assets {
            a.annotate(&mut cfg);
        }
        let mut meta = metadata(name, arch, dtype, self.license, source);
        meta[arch] = cfg;
        meta["components"] = json!({ arch: {"architecture": arch, "config": arch, "license": self.license, "tensor_prefix": ""} });
        self.hooks.metadata(&mut meta);

        let expected_set: BTreeSet<&String> = expected.iter().collect();
        let mut order: Vec<String> = expected.iter().filter(|n| srcs.map.contains_key(*n)).cloned().collect();
        order.extend(srcs.map.keys().filter(|k| !expected_set.contains(k)).cloned());
        let mut w = ModelWriter::create_shared(out, meta, dtype, self.role_arc(), order.len())?;
        if let Some(a) = assets {
            a.write_blobs(&mut w)?;
        }
        for n in &order {
            match self.hooks.transform(n, srcs, srcs.cfg)? {
                Emit::Drop => {}
                Emit::Replace(ts) => {
                    for (tn, shape, data) in ts {
                        w.add_f32(&tn, &shape, &data)?;
                    }
                }
                Emit::Keep => self.write_one(&mut w, srcs, n)?,
            }
        }
        w.finish(&expected)
    }

    fn write_one(&self, w: &mut ModelWriter, srcs: &Sources<'_>, name: &str) -> Result<()> {
        let src = srcs.map.get(name).ok_or_else(|| Error::new("internal", format!("no source for {name:?}")))?;
        let raw = srcs.raw_shape(src)?;
        let shape = self.hooks.reshape(name, &raw, srcs.cfg, srcs.kind).unwrap_or_else(|| raw.clone());
        let order = self.hooks.row_order(name, &raw, srcs.cfg, srcs.kind);
        match (src, srcs.gguf) {
            (Src::Gguf(i), Some(g)) => {
                let t = &g.tensors[*i];
                let data = g.tensor_data(t);
                match order {
                    Some(o) => write_ggml_tensor(w, name, t, &gather_rows(data, &o)?, Some(&shape)),
                    None => write_ggml_tensor(w, name, t, data, Some(&shape)),
                }
            }
            _ => w.add_f32(name, &shape, &srcs.f32(name)?),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Hooks;
    impl ConvertHooks for Hooks {
        fn expected(&self, _cfg: &Value) -> Result<Vec<String>> {
            Ok(vec!["a.weight".into()])
        }
    }

    static SPEC: ConvertSpec = ConvertSpec {
        architecture: "toy",
        license: "mit",
        hf_model_types: &["toy"],
        gguf_architectures: &["toy"],
        hf_names: &[Rule::drop("lm_head.weight")],
        gguf_names: &[],
        roles: &[RoleRule::new("embed.weight", Role::LinearInt8)],
        text_assets: false,
        hooks: &Hooks,
    };

    #[test]
    fn spec_basics() {
        assert!(SPEC.accepts_model_type("toy"));
        assert_eq!(SPEC.role("embed.weight", &[4, 4]), Role::LinearInt8);
        assert_eq!((SPEC.role_arc())("x.weight", &[4, 4]), Role::Linear);
        assert!(SPEC.convert_hf("/nonexistent/dir/x", OutDType::Int8, Path::new("/tmp/x.lapid")).is_err());
    }
}
