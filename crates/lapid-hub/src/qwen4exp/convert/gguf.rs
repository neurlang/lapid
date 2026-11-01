//! GGUF (llama.cpp `qwen4exp`) → `.lapid`.
//!
//! llama.cpp's converter already applies this crate's value conventions (norms `1 + w`,
//! `A = -exp(A_log)`, tiled v heads), so tensors are renamed, merged weights split by rows
//! (raw ggml rows, quantization kept) and copied:
//!
//! | GGUF | output |
//! |---|---|
//! | `token_embd` / `output` | `model.embed_tokens` / `lm_head` |
//! | `output_hc_{norm,down,up}` | `model.hyper_connection_mixer.{hc_norm,input_mix_weight_down,input_mix_weight_up}` |
//! | `blk.N.hc_{attn,ffn}_{norm,down,up,inject}` (+ merged `_down_inject`) | `{attn,mlp}_hyper_connection.*` |
//! | `attn_q/k/v/output/q_norm/k_norm`, `indexer.{q,k}_{proj,norm}` | `self_attn.*`, `self_attn.indexer.*` |
//! | `attn_qkv` + `attn_gate` (or merged `attn_qkv_gate`), `ssm_beta` + `ssm_alpha` (or `ssm_beta_alpha`) | `linear_attn.in_proj_{qkv,z,b,a}` |
//! | `ssm_out` / `ssm_conv1d` [C, K] / `ssm_dt.bias` / `ssm_a` / `ssm_norm` | `linear_attn.out_proj` / `conv1d` [C, 1, K] / `dt_bias` / `A` / `norm` |
//! | `ple_{key,value,norm_key,norm_query,norm_conv,conv1d}`, `per_layer_token_embd` | `ple.*`, `ple.ngram_embedding` |
//! | `ffn_gate_inp`, `ffn_{gate,up,down}_exps`, `ffn_gate_up_exps`, `*_shexp`, `ffn_gate_inp_shexp` | `mlp.*` (Qwen3.5 MoE names) |
//!
//! MTP (nextn) blocks after the trunk are dropped. The tokenizer (`gpt2` BPE, pre-tokenizer
//! `qwen35`) is rebuilt as a Hugging Face `tokenizer.json`.

use std::collections::BTreeSet;
use std::path::Path;

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::gguf::{GgufFile, GgufTensor, GgufValue};
use lapid_convert::writer::{ModelWriter, file_hash};
use serde_json::{Map, Value, json};

use super::{Assets, add_blobs, expected_tensors, is_grouped_norm, metadata, role, write_ggml};
use crate::qwen4exp::config::Qwen4ExpConfig;
use crate::qwen4exp::model::{EMBED, LM_HEAD, MIXER};

const QWEN35_SPLIT: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const ARCH: &str = "qwen4exp";

pub(crate) fn is_qwen4exp_gguf(f: &GgufFile) -> bool {
    f.architecture() == Some(ARCH)
}

fn key<'a>(f: &'a GgufFile, k: &str) -> Result<&'a GgufValue> {
    f.get(k).ok_or_else(|| Error::new("missing_metadata", format!("GGUF: missing {k:?}")))
}

fn uint(f: &GgufFile, k: &str) -> Result<usize> {
    key(f, k)?.as_u64().map(|v| v as usize).ok_or_else(|| Error::new("metadata", format!("GGUF: {k:?} is not an integer")))
}

fn opt_uint(f: &GgufFile, k: &str) -> Option<usize> {
    f.get(k).and_then(GgufValue::as_u64).map(|v| v as usize)
}

fn uints(f: &GgufFile, k: &str) -> Option<Vec<u64>> {
    f.get(k).and_then(GgufValue::as_array).map(|a| {
        a.iter()
            .map(|v| match v {
                GgufValue::Bool(b) => u64::from(*b),
                o => o.as_u64().unwrap_or(0),
            })
            .collect()
    })
}

fn p(k: &str) -> String {
    format!("{ARCH}.{k}")
}

/// Trunk layers (block_count minus MTP / nextn blocks).
fn trunk_layers(f: &GgufFile) -> Result<usize> {
    let blocks = uint(f, &p("block_count"))?;
    Ok(blocks - opt_uint(f, &p("nextn_predict_layers")).unwrap_or(0).min(blocks))
}

/// HF-style text config JSON from the GGUF metadata (then parsed by `Qwen4ExpConfig::from_json`).
fn config_json(f: &GgufFile) -> Result<Value> {
    let n_layers = trunk_layers(f)?;
    let hidden = uint(f, &p("embedding_length"))?;
    let n_head = uint(f, &p("attention.head_count"))?;
    let head_dim = opt_uint(f, &p("attention.key_length")).unwrap_or(hidden / n_head.max(1));
    let hv = uint(f, &p("ssm.time_step_rank"))?;
    let inner = uint(f, &p("ssm.inner_size"))?;
    let recurrent = uints(f, &p("attention.recurrent_layers"));
    let interval = opt_uint(f, &p("full_attention_interval")).unwrap_or(4).max(1);
    let layer_types: Vec<&str> = (0..n_layers)
        .map(|i| {
            let rec = match &recurrent {
                Some(r) => r.get(i).is_some_and(|&x| x != 0),
                None => (i + 1) % interval != 0,
            };
            if rec { "linear_attention" } else { "full_attention" }
        })
        .collect();
    let rot = opt_uint(f, &p("rope.dimension_count")).unwrap_or(head_dim / 4);
    let vocab = f.tensor("token_embd.weight").map(|t| t.shape[0]).ok_or_else(|| Error::new("missing_tensor", "GGUF: no token_embd.weight"))?;
    let inter = uint(f, &p("expert_feed_forward_length"))?;
    let mut v = json!({
        "model_type": "qwen4_exp_text",
        "vocab_size": vocab,
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_head,
        "num_key_value_heads": uint(f, &p("attention.head_count_kv"))?,
        "head_dim": head_dim,
        "rms_norm_eps": f.get(&p("attention.layer_norm_rms_epsilon")).and_then(GgufValue::as_f64).unwrap_or(1e-6),
        "layer_types": layer_types,
        "rope_parameters": {"rope_type": "default",
            "rope_theta": f.get(&p("rope.freq_base")).and_then(GgufValue::as_f64).unwrap_or(1e7),
            "partial_rotary_factor": rot as f64 / head_dim as f64},
        "linear_conv_kernel_dim": opt_uint(f, &p("ssm.conv_kernel")).unwrap_or(4),
        "linear_num_key_heads": uint(f, &p("ssm.group_count"))?,
        "linear_num_value_heads": hv,
        "linear_key_head_dim": uint(f, &p("ssm.state_size"))?,
        "linear_value_head_dim": inner / hv.max(1),
        "tie_word_embeddings": f.tensor("output.weight").is_none(),
        "max_position_embeddings": opt_uint(f, &p("context_length")).unwrap_or(262_144),
        "eos_token_id": f.get("tokenizer.ggml.eos_token_id").and_then(GgufValue::as_u64),
        "pad_token_id": f.get("tokenizer.ggml.padding_token_id").and_then(GgufValue::as_u64),
        "num_experts": uint(f, &p("expert_count"))?,
        "num_experts_per_tok": uint(f, &p("expert_used_count"))?,
        "moe_intermediate_size": inter,
        "shared_expert_intermediate_size": opt_uint(f, &p("expert_shared_feed_forward_length")).unwrap_or(inter),
        "hc_count": uint(f, &p("hyper_connection.count"))?,
        "hc_lowrank": uint(f, &p("hyper_connection.low_rank"))?,
        // llama.cpp hard-codes the sigmoid output gate of the released model
        "output_gate_type": "sigmoid",
    });
    if let Some(nh) = opt_uint(f, &p("attention.indexer.head_count")) {
        let ratio = uints(f, &p("attention.compress_ratios")).and_then(|r| r.into_iter().find(|&x| x > 0)).unwrap_or(0);
        if ratio > 0 {
            v["indexer_n_heads"] = json!(nh);
            v["indexer_head_dim"] = json!(uint(f, &p("attention.indexer.key_length"))?);
            v["indexer_budget"] = json!(uint(f, &p("attention.indexer.top_k"))?);
            v["indexer_compress_ratio"] = json!(ratio);
        }
    }
    if let Some(layers) = uints(f, &p("ple.layers")).filter(|l| !l.is_empty()) {
        let table = f.tensor("per_layer_token_embd.weight").ok_or_else(|| Error::new("missing_tensor", "GGUF: no per_layer_token_embd.weight"))?;
        let need = |k: &str| uints(f, &p(k)).ok_or_else(|| Error::new("missing_metadata", format!("GGUF: missing {k:?}")));
        v["ple"] = json!({
            "layer": layers[0],
            "ngram_size": uint(f, &p("ple.ngram_size"))?,
            "heads_per_ngram": uint(f, &p("ple.heads_per_ngram"))?,
            "conv_kernel": uint(f, &p("ple.conv_kernel"))?,
            "eos_token_id": uint(f, &p("ple.eos_token_id"))?,
            "multipliers": need("ple.layer_multipliers")?,
            "head_offsets": need("ple.head_offsets")?,
            "head_vocab_sizes": need("ple.head_vocab_sizes")?,
            "row_dim": table.shape[1],
            "rows": table.shape[0],
        });
    }
    Ok(v)
}

fn strings(f: &GgufFile, k: &str) -> Result<Vec<String>> {
    key(f, k)?
        .as_array()
        .ok_or_else(|| Error::new("metadata", format!("GGUF: {k:?} is not an array")))?
        .iter()
        .map(|v| v.as_str().map(str::to_string).ok_or_else(|| Error::new("metadata", format!("GGUF: {k:?} holds a non-string"))))
        .collect()
}

/// HF `tokenizer.json` equivalent of the GGUF byte-level BPE tokenizer (Qwen3.5 split regex,
/// NFC normalizer); normal tokens (type 1) form the vocab, control (3) / user-defined (4)
/// tokens become added tokens (special / not special), unused (5) padding is dropped.
/// (Same as `qwen35::convert::gguf`, which is private to that module.)
fn tokenizer_json(f: &GgufFile) -> Result<Value> {
    let model = f.get("tokenizer.ggml.model").and_then(GgufValue::as_str).unwrap_or("");
    let pre = f.get("tokenizer.ggml.pre").and_then(GgufValue::as_str).unwrap_or("");
    if model != "gpt2" || !matches!(pre, "qwen35" | "qwen2") {
        bail!("unsupported_model", "GGUF tokenizer {model:?} / pre-tokenizer {pre:?} is not supported for qwen4exp");
    }
    let tokens = strings(f, "tokenizer.ggml.tokens")?;
    let merges = strings(f, "tokenizer.ggml.merges")?;
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
        match ty {
            1 => {
                vocab.insert(tok.clone(), json!(id));
            }
            3 | 4 => added.push(json!({
                "id": id, "content": tok, "single_word": false, "lstrip": false, "rstrip": false,
                "normalized": false, "special": *ty == 3,
            })),
            _ => {}
        }
    }
    let byte_level = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
    Ok(json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": QWEN35_SPLIT}, "behavior": "Isolated", "invert": false},
            byte_level.clone(),
        ]},
        "post_processor": byte_level.clone(),
        "decoder": byte_level,
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                  "vocab": vocab, "merges": merges},
    }))
}

/// Where a GGUF tensor goes: one output, or a row split into several.
#[derive(Debug, PartialEq)]
enum Target {
    Drop,
    One(String),
    /// Consecutive row ranges (name, rows; `0` = the rest).
    Rows(Vec<(String, usize)>),
}

/// Output of a GGUF tensor name.
fn target(g: &str, cfg: &Qwen4ExpConfig) -> Result<Target> {
    let one = |s: &str| Ok(Target::One(s.to_string()));
    match g {
        "token_embd.weight" => return one(EMBED),
        "output.weight" => return one(LM_HEAD),
        "output_hc_norm.weight" => return one(&format!("{MIXER}.hc_norm.weight")),
        "output_hc_down.weight" => return one(&format!("{MIXER}.input_mix_weight_down.weight")),
        "output_hc_up.weight" => return one(&format!("{MIXER}.input_mix_weight_up.weight")),
        "rope_freqs.weight" => return Ok(Target::Drop),
        "per_layer_token_embd.weight" => {
            let layer = cfg.ple.as_ref().map(|q| q.layer).ok_or_else(|| Error::new("metadata", "GGUF: n-gram table without PLE keys"))?;
            return one(&format!("model.layers.{layer}.ple.ngram_embedding.weight"));
        }
        _ => {}
    }
    let unknown = || Error::new("unsupported_model", format!("GGUF: unknown qwen4exp tensor {g:?}"));
    let rest = g.strip_prefix("blk.").ok_or_else(unknown)?;
    let (layer, rest) = rest.split_once('.').ok_or_else(unknown)?;
    let li: usize = layer.parse().map_err(|_| unknown())?;
    if li >= cfg.n_layers() {
        return Ok(Target::Drop); // MTP / nextn block
    }
    let l = format!("model.layers.{li}");
    let b = &cfg.base;
    let (lr, hc) = (cfg.hc_lowrank, cfg.hc);
    let hcn = |kind: &str, part: &str| format!("{l}.{kind}_hyper_connection.{part}.weight");
    let split = |parts: Vec<(String, usize)>| Ok(Target::Rows(parts));
    let n = match rest {
        "hc_attn_down_inject.weight" => return split(vec![(hcn("attn", "input_mix_weight_down"), lr), (hcn("attn", "block_inject_weight"), hc)]),
        "hc_ffn_down_inject.weight" => return split(vec![(hcn("mlp", "input_mix_weight_down"), lr), (hcn("mlp", "block_inject_weight"), hc)]),
        "attn_qkv_gate.weight" => {
            let a = format!("{l}.linear_attn");
            return split(vec![(format!("{a}.in_proj_qkv.weight"), b.conv_dim()), (format!("{a}.in_proj_z.weight"), b.linear_v_heads * b.linear_v_dim)]);
        }
        "ssm_beta_alpha.weight" => {
            let a = format!("{l}.linear_attn");
            return split(vec![(format!("{a}.in_proj_b.weight"), b.linear_v_heads), (format!("{a}.in_proj_a.weight"), b.linear_v_heads)]);
        }
        "hc_attn_norm.weight" => hcn("attn", "hc_norm"),
        "hc_attn_down.weight" => hcn("attn", "input_mix_weight_down"),
        "hc_attn_up.weight" => hcn("attn", "input_mix_weight_up"),
        "hc_attn_inject.weight" => hcn("attn", "block_inject_weight"),
        "hc_ffn_norm.weight" => hcn("mlp", "hc_norm"),
        "hc_ffn_down.weight" => hcn("mlp", "input_mix_weight_down"),
        "hc_ffn_up.weight" => hcn("mlp", "input_mix_weight_up"),
        "hc_ffn_inject.weight" => hcn("mlp", "block_inject_weight"),
        other => {
            let s = match other {
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "attn_q_norm.weight" => "self_attn.q_norm.weight",
                "attn_k_norm.weight" => "self_attn.k_norm.weight",
                "indexer.q_proj.weight" => "self_attn.indexer.q_proj.weight",
                "indexer.k_proj.weight" => "self_attn.indexer.k_proj.weight",
                "indexer.q_norm.weight" => "self_attn.indexer.q_norm.weight",
                "indexer.k_norm.weight" => "self_attn.indexer.k_norm.weight",
                "attn_qkv.weight" => "linear_attn.in_proj_qkv.weight",
                "attn_gate.weight" => "linear_attn.in_proj_z.weight",
                "ssm_beta.weight" => "linear_attn.in_proj_b.weight",
                "ssm_alpha.weight" => "linear_attn.in_proj_a.weight",
                "ssm_out.weight" => "linear_attn.out_proj.weight",
                "ssm_conv1d.weight" => "linear_attn.conv1d.weight",
                "ssm_dt.bias" => "linear_attn.dt_bias",
                "ssm_a" => "linear_attn.A",
                "ssm_norm.weight" => "linear_attn.norm.weight",
                "ple_key.weight" => "ple.key_proj.weight",
                "ple_value.weight" => "ple.value_proj.weight",
                "ple_norm_key.weight" => "ple.norm_key.weight",
                "ple_norm_query.weight" => "ple.norm_query.weight",
                "ple_norm_conv.weight" => "ple.norm_conv.weight",
                "ple_conv1d.weight" => "ple.conv1d.weight",
                "ffn_gate_inp.weight" => "mlp.gate.weight",
                "ffn_gate_exps.weight" => "mlp.experts.gate_proj",
                "ffn_up_exps.weight" => "mlp.experts.up_proj",
                "ffn_down_exps.weight" => "mlp.experts.down_proj",
                "ffn_gate_up_exps.weight" => "mlp.experts.gate_up_proj",
                "ffn_gate_shexp.weight" => "mlp.shared_expert.gate_proj.weight",
                "ffn_up_shexp.weight" => "mlp.shared_expert.up_proj.weight",
                "ffn_down_shexp.weight" => "mlp.shared_expert.down_proj.weight",
                "ffn_gate_inp_shexp.weight" => "mlp.shared_expert_gate.weight",
                _ => return Err(unknown()),
            };
            format!("{l}.{s}")
        }
    };
    Ok(Target::One(n))
}

/// Row-major output shape of a renamed whole tensor.
fn out_shape(name: &str, t: &GgufTensor, hc: usize) -> Vec<usize> {
    if name.ends_with("conv1d.weight") && t.shape.len() == 2 {
        vec![t.shape[0], 1, t.shape[1]]
    } else if name.ends_with("shared_expert_gate.weight") && t.shape.len() == 1 {
        vec![1, t.shape[0]]
    } else if is_grouped_norm(name) && t.shape.len() == 1 {
        vec![hc, t.shape[0] / hc]
    } else {
        t.shape.clone()
    }
}

/// Write a merged `[R, C]` tensor as consecutive row ranges (raw ggml rows).
fn write_rows(f: &GgufFile, w: &mut ModelWriter, dtype: OutDType, t: &GgufTensor, parts: &[(String, usize)]) -> Result<()> {
    if t.shape.len() != 2 {
        bail!("shape_mismatch", "GGUF {}: expected a matrix, got {:?}", t.name, t.shape);
    }
    let (rows, cols) = (t.shape[0], t.shape[1]);
    let row = t.ggml_type.nbytes(cols as u64)? as usize;
    let data = f.tensor_data(t);
    let mut start = 0;
    for (name, n) in parts {
        let n = if *n == 0 { rows - start } else { *n };
        if start + n > rows {
            bail!("shape_mismatch", "GGUF {}: {rows} rows, split needs {}", t.name, start + n);
        }
        write_ggml(w, dtype, name, t.ggml_type, &[n, cols], &data[start * row..(start + n) * row])?;
        start += n;
    }
    if start != rows {
        bail!("shape_mismatch", "GGUF {}: split covers {start} of {rows} rows", t.name);
    }
    Ok(())
}

/// Split a fused `[E, 2I, K]` gate_up expert stack (raw rows) into gate / up stacks.
fn write_fused_gate_up(f: &GgufFile, w: &mut ModelWriter, dtype: OutDType, prefix: &str, t: &GgufTensor) -> Result<()> {
    if t.shape.len() != 3 || !t.shape[1].is_multiple_of(2) {
        bail!("shape_mismatch", "GGUF {}: expected [E, 2I, K], got {:?}", t.name, t.shape);
    }
    let (e, i2, k) = (t.shape[0], t.shape[1], t.shape[2]);
    let i = i2 / 2;
    let row = t.ggml_type.nbytes(k as u64)? as usize;
    let data = f.tensor_data(t);
    let (mut gate, mut up) = (Vec::with_capacity(data.len() / 2), Vec::with_capacity(data.len() / 2));
    for x in 0..e {
        let base = x * i2 * row;
        gate.extend_from_slice(&data[base..base + i * row]);
        up.extend_from_slice(&data[base + i * row..base + i2 * row]);
    }
    write_ggml(w, dtype, &format!("{prefix}.mlp.experts.gate_proj"), t.ggml_type, &[e, i, k], &gate)?;
    write_ggml(w, dtype, &format!("{prefix}.mlp.experts.up_proj"), t.ggml_type, &[e, i, k], &up)
}

pub(crate) fn convert(path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    let f = GgufFile::open(path)?;
    if !is_qwen4exp_gguf(&f) {
        bail!("unsupported_model", "{}: GGUF architecture {:?} is not qwen4exp", path.display(), f.architecture());
    }
    if opt_uint(&f, "split.count").is_some_and(|n| n > 1) {
        bail!("unsupported_model", "{}: split GGUF; merge the parts first (llama-gguf-split --merge)", path.display());
    }
    let cfg = Qwen4ExpConfig::from_json(&config_json(&f)?)?;
    let token_name = |id: Option<u64>| id.and_then(|i| f.get("tokenizer.ggml.tokens")?.as_array()?.get(i as usize)?.as_str().map(str::to_string));
    let mut special = Map::new();
    for (k, id) in [("eos_token", "eos_token_id"), ("pad_token", "padding_token_id"), ("bos_token", "bos_token_id")] {
        if let Some(s) = token_name(f.get(&format!("tokenizer.ggml.{id}")).and_then(GgufValue::as_u64)) {
            special.insert(k.to_string(), json!(s));
        }
    }
    let mut generation = Map::new();
    for (k, g) in [("temperature", "general.sampling.temp"), ("top_p", "general.sampling.top_p"), ("top_k", "general.sampling.top_k")] {
        if let Some(v) = f.get(g) {
            generation.insert(k.to_string(), v.to_json());
        }
    }
    let assets = Assets {
        tokenizer_json: serde_json::to_string(&tokenizer_json(&f)?)?,
        chat_template: f
            .get("tokenizer.chat_template")
            .and_then(GgufValue::as_str)
            .map(str::to_string)
            .ok_or_else(|| Error::new("missing_metadata", "GGUF: no tokenizer.chat_template"))?,
        special_tokens: Value::Object(special),
        generation: Value::Object(generation),
    };

    let mut mapped = Vec::new();
    let mut have = BTreeSet::new();
    for t in &f.tensors {
        match target(&t.name, &cfg)? {
            Target::Drop => {}
            Target::One(n) => {
                if let Some(p) = n.strip_suffix(".mlp.experts.gate_up_proj") {
                    have.insert(format!("{p}.mlp.experts.gate_proj"));
                    have.insert(format!("{p}.mlp.experts.up_proj"));
                } else {
                    have.insert(n.clone());
                }
                mapped.push((Target::One(n), t));
            }
            Target::Rows(parts) => {
                have.extend(parts.iter().map(|(n, _)| n.clone()));
                mapped.push((Target::Rows(parts), t));
            }
        }
    }
    let expected = expected_tensors(&cfg);
    lapid_convert::writer::check_complete(&have, &expected)?;

    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("qwen4exp").to_string();
    let source = json!({
        "repo": f.get("general.name").and_then(GgufValue::as_str).map_or_else(|| name.clone(), |n| format!("gguf:{n}")),
        "file": path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
        "format": "gguf",
        "file_hash": file_hash(path)?,
    });
    let meta = metadata(&name, dtype, source, &cfg, &assets);
    let mut w = ModelWriter::create(out, meta, dtype, role, expected.len())?;
    add_blobs(&mut w, &assets)?;
    for (tg, t) in &mapped {
        match tg {
            Target::One(n) => match n.strip_suffix(".mlp.experts.gate_up_proj") {
                Some(p) => write_fused_gate_up(&f, &mut w, dtype, p, t)?,
                None => {
                    let shape = out_shape(n, t, cfg.hc);
                    if shape.iter().product::<usize>() != t.numel() {
                        bail!("shape_mismatch", "GGUF {}: shape {shape:?} does not match {:?}", t.name, t.shape);
                    }
                    write_ggml(&mut w, dtype, n, t.ggml_type, &shape, f.tensor_data(t))?
                }
            },
            Target::Rows(parts) => write_rows(&f, &mut w, dtype, t, parts)?,
            Target::Drop => {}
        }
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let cfg = Qwen4ExpConfig::from_json(&json!({"model_type": "qwen4_exp_text", "vocab_size": 1000, "hidden_size": 64,
            "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
            "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32, "hc_count": 4, "hc_lowrank": 8,
            "eos_token_id": 7, "ple_layer_ids": [2], "ngram_vocab_size_base": 50}))
        .expect("cfg");
        let t = |g: &str| target(g, &cfg).expect("target");
        assert_eq!(t("blk.0.attn_qkv.weight"), Target::One("model.layers.0.linear_attn.in_proj_qkv.weight".into()));
        assert_eq!(t("blk.3.indexer.k_norm.weight"), Target::One("model.layers.3.self_attn.indexer.k_norm.weight".into()));
        assert_eq!(t("per_layer_token_embd.weight"), Target::One("model.layers.1.ple.ngram_embedding.weight".into()));
        assert_eq!(t("blk.4.attn_q.weight"), Target::Drop);
        assert_eq!(
            t("blk.0.ssm_beta_alpha.weight"),
            Target::Rows(vec![("model.layers.0.linear_attn.in_proj_b.weight".into(), 4), ("model.layers.0.linear_attn.in_proj_a.weight".into(), 4)])
        );
        assert!(target("blk.0.mystery.weight", &cfg).is_err());
    }
}
