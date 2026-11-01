//! GGUF (llama.cpp `general.architecture = "gemma4"`, dense or MoE) → `.lapid`, optionally
//! with the vision tower of an `mmproj` GGUF (`crate::gemma4vl::convert::add_mmproj`).
//!
//! Tensor names (ggml `ne` reversed = row-major; no q/k permutation, NeoX-style RoPE) follow
//! `gemma4::convert::gguf`, plus the MoE block:
//!
//! | GGUF | lapid (HF without `language_model`) |
//! |---|---|
//! | `blk.N.ffn_gate_inp.weight` / `.scale` | `router.proj.weight` / `router.scale` |
//! | `blk.N.ffn_down_exps.scale` | `router.per_expert_scale` |
//! | `blk.N.ffn_gate_up_exps` (or `ffn_gate_exps` + `ffn_up_exps`, fused per expert) / `ffn_down_exps` | `experts.gate_up_proj` / `experts.down_proj` |
//! | `blk.N.pre_ffw_norm_2` / `post_ffw_norm_1` / `post_ffw_norm_2` | `pre_feedforward_layernorm_2` / `post_feedforward_layernorm_1` / `_2` |
//!
//! `use_bidirectional_attention` is not stored in GGUF: like llama.cpp, models without
//! per-layer embeddings (26B-A4B, 31B) get the bidirectional image-block mask.

use std::collections::BTreeSet;
use std::path::Path;

use std::path::PathBuf;

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::gguf::{GgufFile, GgufTensor, GgufValue};
use lapid_convert::writer::{ModelWriter, file_hash};
use serde_json::{Map, Value, json};

use super::{Assets, add_blobs, expected_tensors, metadata, role, write_ggml, write_gguf_tensor};
use crate::gemma4moe::config::Gemma4MoeConfig;

const ARCH: &str = "gemma4";

/// Whether an opened GGUF holds a Gemma 4 MoE model (gemma4 with expert tensors).
pub(crate) fn is_gemma4moe_gguf(f: &GgufFile) -> bool {
    f.architecture() == Some(ARCH) && f.tensors.iter().any(|t| t.name.contains("_exps."))
}

/// A `mmproj*.gguf` next to `model` (llama.cpp / Hugging Face GGUF repos ship them side by side).
pub(crate) fn sibling_mmproj(model: &Path) -> Option<PathBuf> {
    let dir = model.parent()?;
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let n = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            n.starts_with("mmproj") && n.ends_with(".gguf") && p.as_path() != model
        })
        .collect();
    found.sort();
    // Prefer a float projector (F16 / BF16) when several exist.
    let pick = found.iter().find(|p| p.to_string_lossy().contains("F16")).cloned();
    pick.or_else(|| found.into_iter().next())
}

fn key<'a>(f: &'a GgufFile, k: &str) -> Result<&'a GgufValue> {
    f.get(k).ok_or_else(|| Error::new("missing_metadata", format!("GGUF: missing {k:?}")))
}

fn uint(f: &GgufFile, k: &str) -> Result<usize> {
    key(f, k)?.as_u64().map(|v| v as usize).ok_or_else(|| Error::new("metadata", format!("GGUF: {k:?} is not an integer")))
}

fn float(f: &GgufFile, k: &str) -> Option<f64> {
    f.get(k).and_then(GgufValue::as_f64)
}

fn bool_of(v: &GgufValue) -> bool {
    match v {
        GgufValue::Bool(b) => *b,
        other => other.as_u64().is_some_and(|x| x != 0),
    }
}

/// Integer key that may be a per-layer array: (values per layer).
fn per_layer(f: &GgufFile, k: &str, n: usize) -> Result<Vec<usize>> {
    let v = key(f, k)?;
    match v.as_array() {
        Some(a) => Ok(a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect()),
        None => Ok(vec![v.as_u64().unwrap_or(0) as usize; n]),
    }
}

/// HF-style text config JSON from the GGUF metadata (then parsed by `Gemma4Config::from_json`).
fn config_json(f: &GgufFile) -> Result<Value> {
    let p = |k: &str| format!("{ARCH}.{k}");
    let n_layers = uint(f, &p("block_count"))?;
    let swa: Vec<bool> = key(f, &p("attention.sliding_window_pattern"))?
        .as_array()
        .map(|a| a.iter().map(bool_of).collect())
        .unwrap_or_default();
    if swa.len() != n_layers {
        bail!("metadata", "GGUF: sliding_window_pattern has {} entries for {n_layers} layers", swa.len());
    }
    let ff = per_layer(f, &p("feed_forward_length"), n_layers)?;
    let ff_base = ff.iter().copied().min().unwrap_or(0);
    let kv = per_layer(f, &p("attention.head_count_kv"), n_layers)?;
    let kv_swa = (0..n_layers).find(|&i| swa[i]).map_or(kv[0], |i| kv[i]);
    let kv_full = (0..n_layers).find(|&i| !swa[i]).map_or(kv_swa, |i| kv[i]);
    let global_hd = uint(f, &p("attention.key_length"))?;
    let partial = match f.tensor("rope_freqs.weight") {
        Some(t) => {
            let v = f.to_f32(t)?;
            (2 * v.iter().filter(|&&x| x < 1e3).count()) as f64 / global_hd as f64
        }
        None => 1.0,
    };
    let shared = f.get(&p("attention.shared_kv_layers")).and_then(GgufValue::as_u64).unwrap_or(0) as usize;
    let first_full = (0..n_layers.saturating_sub(shared)).find(|&i| !swa[i]);
    let k_eq_v = first_full.is_some_and(|i| f.tensor(&format!("blk.{i}.attn_v.weight")).is_none());
    let rows = |name: &str| f.tensor(name).map(|t| t.shape[0]);
    let vocab = rows("token_embd.weight").ok_or_else(|| Error::new("missing_tensor", "GGUF: no token_embd.weight"))?;
    let tok_id = |k: &str| f.get(&format!("tokenizer.ggml.{k}")).and_then(GgufValue::as_u64);
    let types: Vec<&str> = swa.iter().map(|&s| if s { "sliding_attention" } else { "full_attention" }).collect();
    let n_expert = f.get(&p("expert_count")).and_then(GgufValue::as_u64).unwrap_or(0);
    let ple = f.get(&p("embedding_length_per_layer_input")).and_then(GgufValue::as_u64).unwrap_or(0);
    let moe_ff = match f.get(&p("expert_feed_forward_length")) {
        Some(v) => v.as_u64().or_else(|| v.as_array().and_then(|a| a.iter().find_map(GgufValue::as_u64))),
        None => None,
    };
    let mut cfg = json!({
        "model_type": "gemma4_text",
        "vocab_size": vocab,
        "hidden_size": uint(f, &p("embedding_length"))?,
        "intermediate_size": ff_base,
        "use_double_wide_mlp": ff.iter().any(|&x| x > ff_base),
        "num_hidden_layers": n_layers,
        "num_attention_heads": uint(f, &p("attention.head_count"))?,
        "num_key_value_heads": kv_swa,
        "num_global_key_value_heads": kv_full,
        "attention_k_eq_v": k_eq_v,
        "head_dim": f.get(&p("attention.key_length_swa")).and_then(GgufValue::as_u64).map_or(global_hd, |v| v as usize),
        "global_head_dim": global_hd,
        "rms_norm_eps": float(f, &p("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6),
        "layer_types": types,
        "sliding_window": f.get(&p("attention.sliding_window")).and_then(GgufValue::as_u64).unwrap_or(512),
        "rope_parameters": {
            "full_attention": {"rope_type": "proportional", "rope_theta": float(f, &p("rope.freq_base")).unwrap_or(1e6), "partial_rotary_factor": partial},
            "sliding_attention": {"rope_type": "default", "rope_theta": float(f, &p("rope.freq_base_swa")).unwrap_or(1e4)},
        },
        "num_kv_shared_layers": shared,
        "hidden_size_per_layer_input": ple,
        "vocab_size_per_layer_input": rows("per_layer_token_embd.weight").unwrap_or(0),
        "final_logit_softcapping": float(f, &p("final_logit_softcapping")),
        "max_position_embeddings": f.get(&p("context_length")).and_then(GgufValue::as_u64).unwrap_or(131_072),
        "tie_word_embeddings": true,
        "bos_token_id": tok_id("bos_token_id"),
        "eos_token_id": tok_id("eos_token_id"),
        "pad_token_id": tok_id("padding_token_id"),
    });
    if n_expert > 0 {
        cfg["enable_moe_block"] = json!(true);
        cfg["num_experts"] = json!(n_expert);
        cfg["top_k_experts"] = json!(f.get(&p("expert_used_count")).and_then(GgufValue::as_u64).unwrap_or(0));
        cfg["moe_intermediate_size"] = json!(moe_ff.unwrap_or(0));
    }
    if ple == 0 {
        cfg["use_bidirectional_attention"] = json!("vision");
    }
    Ok(cfg)
}

fn strings(f: &GgufFile, k: &str) -> Result<Vec<String>> {
    key(f, k)?
        .as_array()
        .ok_or_else(|| Error::new("metadata", format!("GGUF: {k:?} is not an array")))?
        .iter()
        .map(|v| v.as_str().map(str::to_string).ok_or_else(|| Error::new("metadata", format!("GGUF: {k:?} holds a non-string"))))
        .collect()
}

/// Hugging Face `tokenizer.json` equivalent of the GGUF Gemma 4 tokenizer (same layout as
/// google/gemma-4-*'s own file): BPE with byte fallback over text whose spaces are
/// replaced by `▁`; control (3) / user-defined (4) tokens are special added tokens.
fn tokenizer_json(f: &GgufFile) -> Result<Value> {
    let model = f.get("tokenizer.ggml.model").and_then(GgufValue::as_str).unwrap_or("");
    if model != "gemma4" {
        bail!("unsupported_model", "GGUF tokenizer {model:?} is not supported for gemma4");
    }
    let tokens = strings(f, "tokenizer.ggml.tokens")?;
    let merges: Vec<Value> = strings(f, "tokenizer.ggml.merges")?
        .into_iter()
        .map(|m| match m.split_once(' ') {
            Some((a, b)) => Ok(json!([a, b])),
            None => Err(Error::new("metadata", format!("GGUF: malformed merge {m:?}"))),
        })
        .collect::<Result<_>>()?;
    let types: Vec<i64> = key(f, "tokenizer.ggml.token_type")?
        .as_array()
        .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(1.0) as i64).collect())
        .unwrap_or_default();
    if types.len() != tokens.len() {
        bail!("metadata", "GGUF: {} token types for {} tokens", types.len(), tokens.len());
    }
    let mut vocab = Map::new();
    let mut added = Vec::new();
    for (id, (tok, ty)) in tokens.iter().zip(&types).enumerate() {
        vocab.insert(tok.clone(), json!(id));
        if matches!(ty, 3 | 4) {
            added.push(json!({
                "id": id, "content": tok, "single_word": false, "lstrip": false, "rstrip": false,
                "normalized": false, "special": true,
            }));
        }
    }
    let unk = f
        .get("tokenizer.ggml.unknown_token_id")
        .and_then(GgufValue::as_u64)
        .and_then(|i| tokens.get(i as usize).cloned());
    Ok(json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"},
        "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false},
        "post_processor": {"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A", "type_id": 0}}],
                           "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
                           "special_tokens": {}},
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ]},
        "model": {"type": "BPE", "dropout": null, "unk_token": unk, "continuing_subword_prefix": null,
                  "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true, "ignore_merges": false,
                  "vocab": vocab, "merges": merges},
    }))
}

/// lapid name of a GGUF tensor, or `None` for tensors that are dropped. Split
/// `ffn_gate_exps` / `ffn_up_exps` map to `None` here (fused by the caller).
fn lapid_name(g: &str) -> Result<Option<String>> {
    let global = match g {
        "token_embd.weight" => Some("model.embed_tokens.weight"),
        "per_layer_token_embd.weight" => Some("model.embed_tokens_per_layer.weight"),
        "per_layer_model_proj.weight" => Some("model.per_layer_model_projection.weight"),
        "per_layer_proj_norm.weight" => Some("model.per_layer_projection_norm.weight"),
        "output_norm.weight" => Some("model.norm.weight"),
        "rope_freqs.weight" | "output.weight" => return Ok(None),
        _ => None,
    };
    if let Some(n) = global {
        return Ok(Some(n.to_string()));
    }
    let unknown = || Error::new("unsupported_model", format!("GGUF: unknown gemma4 tensor {g:?}"));
    let rest = g.strip_prefix("blk.").ok_or_else(unknown)?;
    let (layer, rest) = rest.split_once('.').ok_or_else(unknown)?;
    let hf = match rest {
        "ffn_gate_inp.scale" => "router.scale",
        "ffn_down_exps.scale" => "router.per_expert_scale",
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" => return Ok(None),
        _ => {
            let base = rest.strip_suffix(".weight").ok_or_else(unknown)?;
            match base {
                "attn_norm" => "input_layernorm.weight",
                "post_attention_norm" => "post_attention_layernorm.weight",
                "ffn_norm" => "pre_feedforward_layernorm.weight",
                "post_ffw_norm" => "post_feedforward_layernorm.weight",
                "pre_ffw_norm_2" => "pre_feedforward_layernorm_2.weight",
                "post_ffw_norm_1" => "post_feedforward_layernorm_1.weight",
                "post_ffw_norm_2" => "post_feedforward_layernorm_2.weight",
                "post_norm" => "post_per_layer_input_norm.weight",
                "attn_q" => "self_attn.q_proj.weight",
                "attn_k" => "self_attn.k_proj.weight",
                "attn_v" => "self_attn.v_proj.weight",
                "attn_output" => "self_attn.o_proj.weight",
                "attn_q_norm" => "self_attn.q_norm.weight",
                "attn_k_norm" => "self_attn.k_norm.weight",
                "ffn_gate" => "mlp.gate_proj.weight",
                "ffn_up" => "mlp.up_proj.weight",
                "ffn_down" => "mlp.down_proj.weight",
                "ffn_gate_inp" => "router.proj.weight",
                "ffn_gate_up_exps" => "experts.gate_up_proj",
                "ffn_down_exps" => "experts.down_proj",
                "inp_gate" => "per_layer_input_gate.weight",
                "proj" => "per_layer_projection.weight",
                "layer_output_scale" => "layer_scalar",
                _ => return Err(unknown()),
            }
        }
    };
    Ok(Some(format!("model.layers.{layer}.{hf}")))
}

/// `experts.gate_up_proj` of layer `i` from split `ffn_gate_exps` + `ffn_up_exps` stacks
/// [E, I, D]: per expert, the gate rows then the up rows (raw ggml rows when both share a
/// type, else through f32).
fn write_split_gate_up(f: &GgufFile, w: &mut ModelWriter, dtype: OutDType, i: usize, gate: &GgufTensor, up: &GgufTensor) -> Result<()> {
    let name = format!("model.layers.{i}.experts.gate_up_proj");
    let &[e, rows, cols] = gate.shape.as_slice() else {
        bail!("shape_mismatch", "GGUF: {} must be 3-D, got {:?}", gate.name, gate.shape);
    };
    if up.shape != gate.shape {
        bail!("shape_mismatch", "GGUF: {} {:?} vs {} {:?}", gate.name, gate.shape, up.name, up.shape);
    }
    let shape = [e, 2 * rows, cols];
    let fuse = |a: &[u8], b: &[u8], per: usize| -> Vec<u8> {
        let mut v = Vec::with_capacity(a.len() + b.len());
        for x in 0..e {
            v.extend_from_slice(&a[x * per..(x + 1) * per]);
            v.extend_from_slice(&b[x * per..(x + 1) * per]);
        }
        v
    };
    if gate.ggml_type == up.ggml_type {
        let per = gate.ggml_type.nbytes((rows * cols) as u64)? as usize;
        let raw = fuse(f.tensor_data(gate), f.tensor_data(up), per);
        let f32_data = || -> Result<Vec<f32>> {
            let (g, u) = (f.to_f32(gate)?, f.to_f32(up)?);
            let per = rows * cols;
            let mut v = Vec::with_capacity(2 * g.len());
            for x in 0..e {
                v.extend_from_slice(&g[x * per..(x + 1) * per]);
                v.extend_from_slice(&u[x * per..(x + 1) * per]);
            }
            Ok(v)
        };
        return write_ggml(w, dtype, &name, gate, &raw, f32_data, &shape);
    }
    let (g, u) = (f.to_f32(gate)?, f.to_f32(up)?);
    let per = rows * cols;
    let mut v = Vec::with_capacity(2 * g.len());
    for x in 0..e {
        v.extend_from_slice(&g[x * per..(x + 1) * per]);
        v.extend_from_slice(&u[x * per..(x + 1) * per]);
    }
    super::add_f32(w, dtype, &name, &shape, &v)
}

/// Write the vision tower of `mmproj` into a temporary `.lapid` (the writer needs the final
/// metadata up front), returning its path and the `"gemma4"` metadata additions.
fn stage_mmproj(mmproj: &Path, dtype: OutDType, out: &Path) -> Result<(PathBuf, Value)> {
    let mm = GgufFile::open(mmproj)?;
    let tmp = out.with_file_name(format!(".{}.mmproj.lapid", out.file_name().and_then(|s| s.to_str()).unwrap_or("model")));
    let meta = lapid_convert::writer::metadata("mmproj", "gemma4", dtype, super::LICENSE, json!({"repo": "gguf", "file": mmproj.display().to_string()}));
    let mut w = ModelWriter::create(&tmp, meta, dtype, role, mm.tensors.len())?;
    let extra = crate::gemma4vl::convert::add_mmproj(&mm, &mut w, dtype)?;
    w.finish(&[])?;
    Ok((tmp, extra))
}

pub(crate) fn convert(path: &Path, mmproj: Option<&Path>, dtype: OutDType, out: &Path) -> Result<()> {
    let f = GgufFile::open(path)?;
    if f.architecture() != Some(ARCH) {
        bail!("unsupported_model", "{}: GGUF architecture {:?} is not gemma4", path.display(), f.architecture());
    }
    let mut root = config_json(&f)?;
    let staged = match mmproj {
        Some(m) => Some(stage_mmproj(m, dtype, out)?),
        None => None,
    };
    if let Some((_, extra)) = &staged {
        if let (Some(o), Some(e)) = (root.as_object_mut(), extra.as_object()) {
            for (k, v) in e {
                o.insert(k.clone(), v.clone());
            }
        }
        // Image token ids from the tokenizer when the projector does not name them.
        let id = |s: &str| f.get("tokenizer.ggml.tokens")?.as_array()?.iter().position(|t| t.as_str() == Some(s));
        for (k, tok) in [("image_token_id", "<|image|>"), ("boi_token_id", "<|image>"), ("eoi_token_id", "<image|>")] {
            if root.get(k).is_none()
                && let Some(i) = id(tok)
            {
                root[k] = json!(i);
            }
        }
    }
    let result = convert_with(&f, path, root, staged.as_ref().map(|(p, _)| p.as_path()), dtype, out);
    if let Some((p, _)) = &staged {
        std::fs::remove_file(p).ok();
    }
    result
}

fn convert_with(f: &GgufFile, path: &Path, root: Value, vision: Option<&Path>, dtype: OutDType, out: &Path) -> Result<()> {
    let cfg = Gemma4MoeConfig::from_json(&root)?;
    let token_name = |id: Option<u64>| id.and_then(|i| f.get("tokenizer.ggml.tokens")?.as_array()?.get(i as usize)?.as_str().map(str::to_string));
    let mut special = Map::new();
    for (k, id) in [("bos_token", "bos_token_id"), ("eos_token", "eos_token_id"), ("pad_token", "padding_token_id"), ("unk_token", "unknown_token_id")] {
        if let Some(s) = token_name(f.get(&format!("tokenizer.ggml.{id}")).and_then(GgufValue::as_u64)) {
            special.insert(k.to_string(), json!(s));
        }
    }
    let mut generation = json!({});
    if let Some(t) = float(f, "general.sampling.temp") {
        generation = json!({"do_sample": true, "temperature": (t * 1e4).round() / 1e4});
        if let Some(k) = f.get("general.sampling.top_k").and_then(GgufValue::as_u64) {
            generation["top_k"] = json!(k);
        }
        if let Some(p) = float(f, "general.sampling.top_p") {
            generation["top_p"] = json!((p * 1e4).round() / 1e4);
        }
    }
    let assets = Assets {
        tokenizer_json: serde_json::to_string(&tokenizer_json(f)?)?,
        chat_template: f
            .get("tokenizer.chat_template")
            .and_then(GgufValue::as_str)
            .map(str::to_string)
            .ok_or_else(|| Error::new("missing_metadata", "GGUF: no tokenizer.chat_template"))?,
        special_tokens: Value::Object(special),
        generation,
    };

    let mut expected = expected_tensors(&cfg);
    let wanted: BTreeSet<&String> = expected.iter().collect();
    let mut mapped = Vec::new();
    let mut split: Vec<(usize, Option<&GgufTensor>, Option<&GgufTensor>)> = Vec::new();
    for t in &f.tensors {
        if let Some(n) = lapid_name(&t.name)?.filter(|n| wanted.contains(n)) {
            mapped.push((n, t));
        } else if let Some(rest) = t.name.strip_prefix("blk.") {
            let (layer, kind) = rest.split_once('.').unwrap_or(("", ""));
            let Ok(i) = layer.parse::<usize>() else { continue };
            let entry = match split.iter_mut().find(|(l, _, _)| *l == i) {
                Some(e) => e,
                None => {
                    split.push((i, None, None));
                    split.last_mut().ok_or_else(|| Error::new("internal", "split list"))?
                }
            };
            match kind {
                "ffn_gate_exps.weight" => entry.1 = Some(t),
                "ffn_up_exps.weight" => entry.2 = Some(t),
                _ => {}
            }
        }
    }
    split.retain(|(_, g, u)| g.is_some() || u.is_some());
    let mut have: BTreeSet<String> = mapped.iter().map(|(n, _)| n.clone()).collect();
    have.extend(split.iter().map(|(i, _, _)| format!("model.layers.{i}.experts.gate_up_proj")));
    lapid_convert::writer::check_complete(&have, &expected)?;

    let vision_file = match vision {
        Some(p) => Some(lapid::format::Reader::open(p)?),
        None => None,
    };
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("gemma4").to_string();
    let source = json!({
        "repo": f.get("general.name").and_then(GgufValue::as_str).map_or_else(|| name.clone(), |n| format!("gguf:{n}")),
        "file": path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
        "format": "gguf",
        "file_hash": file_hash(path)?,
    });
    let meta = metadata(&name, dtype, source, &cfg, &assets);
    let n_vision = vision_file.as_ref().map_or(0, |r| r.tensors.len());
    let mut w = ModelWriter::create(out, meta, dtype, role, mapped.len() + split.len() + n_vision)?;
    add_blobs(&mut w, &assets)?;
    for (n, t) in &mapped {
        write_gguf_tensor(f, &mut w, dtype, n, t, &t.shape)?;
    }
    for (i, g, u) in &split {
        let (Some(g), Some(u)) = (g, u) else {
            bail!("missing_tensor", "GGUF: layer {i} has only one of ffn_gate_exps / ffn_up_exps");
        };
        write_split_gate_up(f, &mut w, dtype, *i, g, u)?;
    }
    if let Some(r) = &vision_file {
        for t in &r.tensors {
            w.add_raw(&t.name, t.dtype, &t.shape, r.tensor_bytes(t))?;
            expected.push(t.name.clone());
        }
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let g = |s: &str| lapid_name(s).ok().flatten();
        assert_eq!(g("blk.0.ffn_gate_inp.weight").as_deref(), Some("model.layers.0.router.proj.weight"));
        assert_eq!(g("blk.0.ffn_gate_inp.scale").as_deref(), Some("model.layers.0.router.scale"));
        assert_eq!(g("blk.2.ffn_down_exps.scale").as_deref(), Some("model.layers.2.router.per_expert_scale"));
        assert_eq!(g("blk.2.ffn_gate_up_exps.weight").as_deref(), Some("model.layers.2.experts.gate_up_proj"));
        assert_eq!(g("blk.1.post_ffw_norm_2.weight").as_deref(), Some("model.layers.1.post_feedforward_layernorm_2.weight"));
        assert_eq!(g("blk.1.ffn_gate_exps.weight"), None);
        assert_eq!(g("blk.0.layer_output_scale.weight").as_deref(), Some("model.layers.0.layer_scalar"));
        assert_eq!(g("rope_freqs.weight"), None);
        assert_eq!(lapid_name("blk.0.mystery.weight").err().map(|e| e.code), Some("unsupported_model"));
    }
}
