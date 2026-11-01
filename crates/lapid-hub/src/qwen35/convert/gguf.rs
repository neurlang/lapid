//! GGUF (llama.cpp `qwen35` / `qwen35moe`) → `.lapid`.
//!
//! llama.cpp's converter already applies this crate's value conventions (norm `1 + w`,
//! `A = -exp(A_log)`, tiled v heads), so tensors are renamed and copied:
//!
//! | GGUF | output |
//! |---|---|
//! | `token_embd` / `output_norm` / `output` | `model.embed_tokens` / `model.norm` / `lm_head` |
//! | `blk.N.attn_norm` / `post_attention_norm` | `input_layernorm` / `post_attention_layernorm` |
//! | `attn_q` `attn_k` `attn_v` `attn_output` `attn_q_norm` `attn_k_norm` | `self_attn.{q,k,v,o}_proj`, `q_norm`, `k_norm` |
//! | `attn_qkv` / `attn_gate` / `ssm_beta` / `ssm_alpha` / `ssm_out` | `linear_attn.in_proj_qkv` / `in_proj_z` / `in_proj_b` / `in_proj_a` / `out_proj` |
//! | `ssm_conv1d` [C, K] / `ssm_dt.bias` / `ssm_a` / `ssm_norm` | `linear_attn.conv1d` [C, 1, K] / `dt_bias` / `A` / `norm` |
//! | `ffn_gate` / `ffn_up` / `ffn_down` | `mlp.{gate,up,down}_proj` |
//! | `ffn_gate_inp` / `ffn_{gate,up,down}_exps` / `ffn_gate_up_exps` | `mlp.gate` / `mlp.experts.{gate,up,down}_proj` (fused split) |
//! | `ffn_{gate,up,down}_shexp` / `ffn_gate_inp_shexp` [D] | `mlp.shared_expert.*` / `mlp.shared_expert_gate` [1, D] |
//!
//! MTP (nextn) blocks after the trunk are dropped. The tokenizer (`gpt2` BPE, pre-tokenizer
//! `qwen35`) is rebuilt as a Hugging Face `tokenizer.json`.

use std::collections::BTreeSet;
use std::path::Path;

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::ggml::GgmlType;
use lapid_convert::gguf::{GgufFile, GgufTensor, GgufValue};
use lapid_convert::writer::{ModelWriter, file_hash};
use serde_json::{Map, Value, json};

use super::{Assets, add_blobs, add_f32, expected_tensors, format_dtype, metadata, native_format, role, write_gguf_tensor};
use crate::qwen35::config::Qwen35Config;

const QWEN35_SPLIT: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

pub(crate) fn is_qwen35_gguf(f: &GgufFile) -> bool {
    matches!(f.architecture(), Some("qwen35" | "qwen35moe"))
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

/// Trunk layers (block_count minus MTP / nextn blocks).
fn trunk_layers(f: &GgufFile, arch: &str) -> Result<usize> {
    let blocks = uint(f, &format!("{arch}.block_count"))?;
    Ok(blocks - opt_uint(f, &format!("{arch}.nextn_predict_layers")).unwrap_or(0).min(blocks))
}

/// HF-style config JSON from the GGUF metadata (then parsed by `Qwen35Config::from_json`).
fn config_json(f: &GgufFile) -> Result<Value> {
    let arch = f.architecture().unwrap_or("qwen35").to_string();
    let p = |k: &str| format!("{arch}.{k}");
    let n_layers = trunk_layers(f, &arch)?;
    let hidden = uint(f, &p("embedding_length"))?;
    let n_head = uint(f, &p("attention.head_count"))?;
    let head_dim = opt_uint(f, &p("attention.key_length")).unwrap_or(hidden / n_head.max(1));
    let hv = uint(f, &p("ssm.time_step_rank"))?;
    let inner = uint(f, &p("ssm.inner_size"))?;
    let recurrent: Option<Vec<bool>> = f.get(&p("attention.recurrent_layers")).and_then(GgufValue::as_array).map(|a| {
        a.iter()
            .map(|v| match v {
                GgufValue::Bool(b) => *b,
                o => o.as_u64().is_some_and(|x| x != 0),
            })
            .collect()
    });
    let interval = opt_uint(f, &p("full_attention_interval")).unwrap_or(4).max(1);
    let layer_types: Vec<&str> = (0..n_layers)
        .map(|i| {
            let rec = match &recurrent {
                Some(r) => r.get(i).copied().unwrap_or(false),
                None => (i + 1) % interval != 0,
            };
            if rec { "linear_attention" } else { "full_attention" }
        })
        .collect();
    let rot = opt_uint(f, &p("rope.dimension_count")).unwrap_or(head_dim / 4);
    let vocab = f.tensor("token_embd.weight").map(|t| t.shape[0]).ok_or_else(|| Error::new("missing_tensor", "GGUF: no token_embd.weight"))?;
    let tok_id = |k: &str| f.get(&format!("tokenizer.ggml.{k}")).and_then(GgufValue::as_u64);
    let mut v = json!({
        "model_type": if arch == "qwen35moe" { "qwen3_5_moe" } else { "qwen3_5" },
        "vocab_size": vocab,
        "hidden_size": hidden,
        "intermediate_size": opt_uint(f, &p("feed_forward_length")).unwrap_or(0),
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
        "eos_token_id": tok_id("eos_token_id"),
        "pad_token_id": tok_id("padding_token_id"),
    });
    if arch == "qwen35moe" {
        let inter = uint(f, &p("expert_feed_forward_length"))?;
        v["num_experts"] = json!(uint(f, &p("expert_count"))?);
        v["num_experts_per_tok"] = json!(uint(f, &p("expert_used_count"))?);
        v["moe_intermediate_size"] = json!(inter);
        v["shared_expert_intermediate_size"] = json!(opt_uint(f, &p("expert_shared_feed_forward_length")).unwrap_or(inter));
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
fn tokenizer_json(f: &GgufFile) -> Result<Value> {
    let model = f.get("tokenizer.ggml.model").and_then(GgufValue::as_str).unwrap_or("");
    let pre = f.get("tokenizer.ggml.pre").and_then(GgufValue::as_str).unwrap_or("");
    if model != "gpt2" || !matches!(pre, "qwen35" | "qwen2") {
        bail!("unsupported_model", "GGUF tokenizer {model:?} / pre-tokenizer {pre:?} is not supported for qwen35");
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

/// Output name of a GGUF tensor (`None` = dropped).
fn out_name(g: &str, n_layers: usize) -> Result<Option<String>> {
    match g {
        "token_embd.weight" => return Ok(Some("model.embed_tokens.weight".into())),
        "output_norm.weight" => return Ok(Some("model.norm.weight".into())),
        "output.weight" => return Ok(Some("lm_head.weight".into())),
        "rope_freqs.weight" => return Ok(None),
        _ => {}
    }
    let unknown = || Error::new("unsupported_model", format!("GGUF: unknown qwen35 tensor {g:?}"));
    let rest = g.strip_prefix("blk.").ok_or_else(unknown)?;
    let (layer, rest) = rest.split_once('.').ok_or_else(unknown)?;
    let li: usize = layer.parse().map_err(|_| unknown())?;
    if li >= n_layers {
        return Ok(None); // MTP / nextn block
    }
    let p = format!("model.layers.{li}");
    let n = match rest {
        "attn_norm.weight" => "input_layernorm.weight",
        "post_attention_norm.weight" => "post_attention_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "attn_qkv.weight" => "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight" => "linear_attn.in_proj_z.weight",
        "ssm_beta.weight" => "linear_attn.in_proj_b.weight",
        "ssm_alpha.weight" => "linear_attn.in_proj_a.weight",
        "ssm_out.weight" => "linear_attn.out_proj.weight",
        "ssm_conv1d.weight" => "linear_attn.conv1d.weight",
        "ssm_dt.bias" => "linear_attn.dt_bias",
        "ssm_a" => "linear_attn.A",
        "ssm_norm.weight" => "linear_attn.norm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
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
    Ok(Some(format!("{p}.{n}")))
}

/// Row-major output shape of a renamed tensor.
fn out_shape(name: &str, t: &GgufTensor) -> Vec<usize> {
    if name.ends_with("conv1d.weight") && t.shape.len() == 2 {
        vec![t.shape[0], 1, t.shape[1]]
    } else if name.ends_with("shared_expert_gate.weight") && t.shape.len() == 1 {
        vec![1, t.shape[0]]
    } else {
        t.shape.clone()
    }
}

/// Split a fused `[E, 2I, K]` gate_up expert stack into gate / up (raw ggml rows, so the
/// quantization is kept) and write both.
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
    for (part, bytes) in [("gate_proj", gate), ("up_proj", up)] {
        let name = format!("{prefix}.mlp.experts.{part}");
        write_raw(w, dtype, &name, t.ggml_type, &[e, i, k], &bytes)?;
    }
    Ok(())
}

/// Write raw ggml bytes of an `[E, N, K]` expert stack (e.g. a split): native formats kept
/// when a quantized dtype is requested, else dequantized through the policy.
fn write_raw(w: &mut ModelWriter, dtype: OutDType, name: &str, ty: GgmlType, shape: &[usize], bytes: &[u8]) -> Result<()> {
    let quant = !matches!(dtype, OutDType::F32 | OutDType::F16);
    if let (true, Some(fm)) = (quant, native_format(ty)) {
        let (e, n, k) = (shape[0], shape[1], shape[2]);
        let per = bytes.len() / e.max(1);
        let mut out = Vec::new();
        for x in 0..e {
            out.extend_from_slice(&lapid::quant::from_ggml_blocks(fm, &bytes[x * per..(x + 1) * per], n, k));
        }
        return w.add_raw(name, format_dtype(fm), shape, &out);
    }
    let data = lapid_convert::ggml::dequantize(ty, bytes, shape.iter().product())?;
    add_f32(w, dtype, name, shape, &data)
}

pub(crate) fn convert(path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    let f = GgufFile::open(path)?;
    if !is_qwen35_gguf(&f) {
        bail!("unsupported_model", "{}: GGUF architecture {:?} is not qwen35 / qwen35moe", path.display(), f.architecture());
    }
    let cfg = Qwen35Config::from_json(&config_json(&f)?)?;
    let token_name = |id: Option<u32>| id.and_then(|i| f.get("tokenizer.ggml.tokens")?.as_array()?.get(i as usize)?.as_str().map(str::to_string));
    let mut special = Map::new();
    for (k, id) in [("eos_token", cfg.eos.first().copied()), ("pad_token", cfg.pad)] {
        if let Some(s) = token_name(id) {
            special.insert(k.to_string(), json!(s));
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
        generation: json!({}),
    };

    let mut mapped = Vec::new();
    let mut have = BTreeSet::new();
    for t in &f.tensors {
        if let Some(n) = out_name(&t.name, cfg.n_layers)? {
            if let Some(p) = n.strip_suffix(".mlp.experts.gate_up_proj") {
                have.insert(format!("{p}.mlp.experts.gate_proj"));
                have.insert(format!("{p}.mlp.experts.up_proj"));
            } else {
                have.insert(n.clone());
            }
            mapped.push((n, t));
        }
    }
    let expected = expected_tensors(&cfg);
    lapid_convert::writer::check_complete(&have, &expected)?;

    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("qwen35").to_string();
    let source = json!({
        "repo": f.get("general.name").and_then(GgufValue::as_str).map_or_else(|| name.clone(), |n| format!("gguf:{n}")),
        "file": path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
        "format": "gguf",
        "file_hash": file_hash(path)?,
    });
    let meta = metadata(&name, dtype, source, &cfg, &assets);
    let mut w = ModelWriter::create(out, meta, dtype, role, expected.len())?;
    add_blobs(&mut w, &assets)?;
    for (n, t) in &mapped {
        if let Some(p) = n.strip_suffix(".mlp.experts.gate_up_proj") {
            write_fused_gate_up(&f, &mut w, dtype, p, t)?;
        } else {
            write_gguf_tensor(&f, &mut w, dtype, n, t, &out_shape(n, t))?;
        }
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let n = |g: &str| out_name(g, 24).ok().flatten();
        assert_eq!(n("blk.0.attn_qkv.weight").as_deref(), Some("model.layers.0.linear_attn.in_proj_qkv.weight"));
        assert_eq!(n("blk.3.attn_q.weight").as_deref(), Some("model.layers.3.self_attn.q_proj.weight"));
        assert_eq!(n("blk.0.ssm_a").as_deref(), Some("model.layers.0.linear_attn.A"));
        assert_eq!(n("blk.1.ffn_down_exps.weight").as_deref(), Some("model.layers.1.mlp.experts.down_proj"));
        assert_eq!(n("blk.24.attn_q.weight"), None);
        assert_eq!(out_name("blk.0.mystery.weight", 24).err().map(|e| e.code), Some("unsupported_model"));
    }
}
