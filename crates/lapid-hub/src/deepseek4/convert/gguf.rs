//! GGUF (llama.cpp `deepseek4`) → `.lapid`.
//!
//! llama.cpp's converter stores upstream values unchanged (routed experts stacked, the real
//! FP4 experts as MXFP4, hash tables as i32), so tensors are renamed and copied:
//!
//! | GGUF | output |
//! |---|---|
//! | `token_embd` / `output_norm` / `output` | `embed` / `norm` / `head` |
//! | `output_hc_{fn,base,scale}` | `hc_head_{fn,base,scale}` |
//! | `blk.N.hc_{attn,ffn}_{fn,base,scale}`, `attn_norm`, `ffn_norm` | same names under `layers.N` |
//! | `attn_q_a` `attn_q_a_norm` `attn_q_b` `attn_kv` `attn_kv_a_norm` `attn_sinks` | `attn.{wq_a, q_norm, wq_b, wkv, kv_norm, attn_sink}` |
//! | `attn_output_a` [G·o_lora, in] / `attn_output_b` | `attn.wo_a` [G, o_lora, in] / `attn.wo_b` |
//! | `attn_compressor_{kv,gate,ape,norm}` | `attn.compressor.{wkv, wgate, ape, norm}` |
//! | `indexer.attn_q_b` / `indexer.proj` / `indexer_compressor_*` | `attn.indexer.{wq_b, weights_proj, compressor.*}` |
//! | `ffn_gate_inp` / `exp_probs_b` / `ffn_gate_tid2eid` | `ffn.gate.{weight, bias, tid2eid}` |
//! | `ffn_{gate,up,down}_exps` / `_shexp` | `ffn.experts.{w1,w3,w2}` / `ffn.shared_experts.*` |
//!
//! MTP (nextn) blocks and the vision routing bias are dropped. The tokenizer (`gpt2` BPE,
//! DeepSeek V3 pre-tokenizer) is rebuilt as a Hugging Face `tokenizer.json`.

use std::collections::BTreeSet;
use std::path::Path;

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::ggml::GgmlType;
use lapid_convert::gguf::{GgufFile, GgufTensor, GgufValue};
use lapid_convert::writer::{ModelWriter, file_hash};
use serde_json::{Map, Value, json};

use super::{Assets, add_blobs, add_i32, expected_tensors, metadata, role, write_gguf_tensor};
use crate::deepseek4::config::Deepseek4Config;

const ARCH: &str = "deepseek4";
const MXFP4: u32 = 39;
const I32: u32 = 26;
/// DeepSeek V3 / V4 pre-tokenizer splits (llama.cpp `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM`).
const SPLITS: [&str; 3] = [
    "\\p{N}{1,3}",
    "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+",
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
];

pub(crate) fn is_deepseek4_gguf(f: &GgufFile) -> bool {
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

fn opt_f64(f: &GgufFile, k: &str) -> Option<f64> {
    f.get(k).and_then(GgufValue::as_f64)
}

fn first_f64(f: &GgufFile, k: &str) -> Option<f64> {
    match f.get(k)? {
        GgufValue::Array(a) => a.first().and_then(GgufValue::as_f64),
        v => v.as_f64(),
    }
}

/// Upstream-style config JSON from the GGUF metadata (parsed by `Deepseek4Config::from_json`).
fn config_json(f: &GgufFile) -> Result<Value> {
    let p = |k: &str| format!("{ARCH}.{k}");
    let blocks = uint(f, &p("block_count"))?;
    let n_layers = blocks - opt_uint(f, &p("nextn_predict_layers")).unwrap_or(0).min(blocks);
    let ratios: Vec<u64> = key(f, &p("attention.compress_ratios"))?
        .as_array()
        .ok_or_else(|| Error::new("metadata", "GGUF: compress_ratios is not an array"))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0))
        .collect();
    let vocab = f.tensor("token_embd.weight").map(|t| t.shape[0]).ok_or_else(|| Error::new("missing_tensor", "GGUF: no token_embd.weight"))?;
    let gating = opt_uint(f, &p("expert_gating_func")).unwrap_or(4);
    if gating != 4 {
        bail!("unsupported_model", "GGUF: deepseek4 expert gating {gating} (only sqrt-softplus = 4)");
    }
    let tok = |k: &str| f.get(&format!("tokenizer.ggml.{k}")).and_then(GgufValue::as_u64);
    let mut v = json!({
        "model_type": "deepseek_v4",
        "vocab_size": vocab,
        "hidden_size": uint(f, &p("embedding_length"))?,
        "num_hidden_layers": n_layers,
        "num_attention_heads": uint(f, &p("attention.head_count"))?,
        "head_dim": uint(f, &p("attention.key_length"))?,
        "qk_rope_head_dim": uint(f, &p("rope.dimension_count"))?,
        "q_lora_rank": uint(f, &p("attention.q_lora_rank"))?,
        "o_groups": uint(f, &p("attention.output_group_count"))?,
        "o_lora_rank": uint(f, &p("attention.output_lora_rank"))?,
        "n_routed_experts": uint(f, &p("expert_count"))?,
        "n_shared_experts": opt_uint(f, &p("expert_shared_count")).unwrap_or(1),
        "num_experts_per_tok": uint(f, &p("expert_used_count"))?,
        "moe_intermediate_size": uint(f, &p("expert_feed_forward_length"))?,
        "routed_scaling_factor": opt_f64(f, &p("expert_weights_scale")).unwrap_or(1.0),
        "swiglu_limit": first_f64(f, &p("swiglu_clamp_exp")).unwrap_or(0.0),
        "sliding_window": uint(f, &p("attention.sliding_window"))?,
        "compress_ratios": ratios,
        "num_hash_layers": opt_uint(f, &p("hash_layer_count")).unwrap_or(0),
        "rope_theta": opt_f64(f, &p("rope.freq_base")).unwrap_or(10000.0),
        "compress_rope_theta": opt_f64(f, &p("attention.compress_rope_freq_base")).unwrap_or(160000.0),
        "hc_mult": opt_uint(f, &p("hyper_connection.count")).unwrap_or(4),
        "hc_sinkhorn_iters": opt_uint(f, &p("hyper_connection.sinkhorn_iterations")).unwrap_or(20),
        "hc_eps": opt_f64(f, &p("hyper_connection.epsilon")).unwrap_or(1e-6),
        "index_n_heads": uint(f, &p("attention.indexer.head_count"))?,
        "index_head_dim": uint(f, &p("attention.indexer.key_length"))?,
        "index_topk": uint(f, &p("attention.indexer.top_k"))?,
        "rms_norm_eps": opt_f64(f, &p("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6),
        "max_position_embeddings": opt_uint(f, &p("context_length")).unwrap_or(1 << 20),
        "bos_token_id": tok("bos_token_id"),
        "eos_token_id": tok("eos_token_id").unwrap_or(1),
    });
    if f.get(&p("rope.scaling.type")).and_then(GgufValue::as_str) == Some("yarn") {
        v["rope_scaling"] = json!({
            "type": "yarn",
            "factor": opt_f64(f, &p("rope.scaling.factor")).unwrap_or(1.0),
            "original_max_position_embeddings": opt_uint(f, &p("rope.scaling.original_context_length")).unwrap_or(65536),
            "beta_fast": opt_f64(f, &p("rope.scaling.yarn_beta_fast")).unwrap_or(32.0),
            "beta_slow": opt_f64(f, &p("rope.scaling.yarn_beta_slow")).unwrap_or(1.0),
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

/// HF `tokenizer.json` equivalent of the GGUF byte-level BPE tokenizer (DeepSeek V3 splits);
/// every token is in the vocab, control (3) / user-defined (4) tokens are also added tokens
/// (special / not special).
fn tokenizer_json(f: &GgufFile) -> Result<Value> {
    let model = f.get("tokenizer.ggml.model").and_then(GgufValue::as_str).unwrap_or("");
    let pre = f.get("tokenizer.ggml.pre").and_then(GgufValue::as_str).unwrap_or("");
    if model != "gpt2" || !matches!(pre, "deepseek-v3" | "joyai-llm" | "deepseek4" | "deepseek-v4") {
        bail!("unsupported_model", "GGUF tokenizer {model:?} / pre-tokenizer {pre:?} is not supported for deepseek4");
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
        // Every token keeps its id in the BPE vocab (the added tokens below the normal range,
        // e.g. BOS = 0, would otherwise be renumbered by `tokenizers`).
        vocab.insert(tok.clone(), json!(id));
        if matches!(ty, 3 | 4) {
            added.push(json!({
                "id": id, "content": tok, "single_word": false, "lstrip": false, "rstrip": false,
                "normalized": false, "special": *ty == 3,
            }));
        }
    }
    let splits: Vec<Value> = SPLITS
        .iter()
        .map(|r| json!({"type": "Split", "pattern": {"Regex": r}, "behavior": "Isolated", "invert": false}))
        .chain(std::iter::once(json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false})))
        .collect();
    let byte_level = json!({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true});
    Ok(json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": {"type": "Sequence", "normalizers": []},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": splits},
        "post_processor": byte_level.clone(),
        "decoder": byte_level,
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                  "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false,
                  "vocab": vocab, "merges": merges},
    }))
}

/// Output name of a GGUF tensor (`None` = dropped).
fn out_name(g: &str, n_layers: usize) -> Result<Option<String>> {
    let global = match g {
        "token_embd.weight" => Some("embed.weight"),
        "output_norm.weight" => Some("norm.weight"),
        "output.weight" => Some("head.weight"),
        "output_hc_fn.weight" => Some("hc_head_fn"),
        "output_hc_base.weight" => Some("hc_head_base"),
        "output_hc_scale.weight" => Some("hc_head_scale"),
        _ => None,
    };
    if let Some(n) = global {
        return Ok(Some(n.to_string()));
    }
    let unknown = || Error::new("unsupported_model", format!("GGUF: unknown deepseek4 tensor {g:?}"));
    let rest = g.strip_prefix("blk.").ok_or_else(unknown)?;
    let (layer, rest) = rest.split_once('.').ok_or_else(unknown)?;
    let li: usize = layer.parse().map_err(|_| unknown())?;
    if li >= n_layers || rest.starts_with("nextn.") || rest == "exp_probs_b_vl.bias" {
        return Ok(None);
    }
    let n = match rest {
        "attn_norm.weight" => "attn_norm.weight",
        "ffn_norm.weight" => "ffn_norm.weight",
        "hc_attn_fn.weight" => "hc_attn_fn",
        "hc_attn_base.weight" => "hc_attn_base",
        "hc_attn_scale.weight" => "hc_attn_scale",
        "hc_ffn_fn.weight" => "hc_ffn_fn",
        "hc_ffn_base.weight" => "hc_ffn_base",
        "hc_ffn_scale.weight" => "hc_ffn_scale",
        "attn_sinks.weight" => "attn.attn_sink",
        "attn_q_a.weight" => "attn.wq_a.weight",
        "attn_q_a_norm.weight" => "attn.q_norm.weight",
        "attn_q_b.weight" => "attn.wq_b.weight",
        "attn_kv.weight" => "attn.wkv.weight",
        "attn_kv_a_norm.weight" => "attn.kv_norm.weight",
        "attn_output_a.weight" => "attn.wo_a.weight",
        "attn_output_b.weight" => "attn.wo_b.weight",
        "attn_compressor_kv.weight" => "attn.compressor.wkv.weight",
        "attn_compressor_gate.weight" => "attn.compressor.wgate.weight",
        "attn_compressor_ape.weight" => "attn.compressor.ape",
        "attn_compressor_norm.weight" => "attn.compressor.norm.weight",
        "indexer.attn_q_b.weight" => "attn.indexer.wq_b.weight",
        "indexer.proj.weight" => "attn.indexer.weights_proj.weight",
        "indexer_compressor_kv.weight" => "attn.indexer.compressor.wkv.weight",
        "indexer_compressor_gate.weight" => "attn.indexer.compressor.wgate.weight",
        "indexer_compressor_ape.weight" => "attn.indexer.compressor.ape",
        "indexer_compressor_norm.weight" => "attn.indexer.compressor.norm.weight",
        "ffn_gate_inp.weight" => "ffn.gate.weight",
        "exp_probs_b.bias" => "ffn.gate.bias",
        "ffn_gate_tid2eid.weight" => "ffn.gate.tid2eid",
        "ffn_gate_exps.weight" => "ffn.experts.w1",
        "ffn_up_exps.weight" => "ffn.experts.w3",
        "ffn_down_exps.weight" => "ffn.experts.w2",
        "ffn_gate_shexp.weight" => "ffn.shared_experts.w1.weight",
        "ffn_up_shexp.weight" => "ffn.shared_experts.w3.weight",
        "ffn_down_shexp.weight" => "ffn.shared_experts.w2.weight",
        _ => return Err(unknown()),
    };
    Ok(Some(format!("layers.{li}.{n}")))
}

/// MXFP4 (ggml type 39): blocks of 32 = e8m0 exponent + 16 bytes of e2m1 nibbles.
pub(crate) fn dequant_mxfp4(data: &[u8], n: usize) -> Result<Vec<f32>> {
    const K: [f32; 16] = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0];
    if !n.is_multiple_of(32) || data.len() != n / 32 * 17 {
        bail!("format", "mxfp4: {} bytes for {n} values", data.len());
    }
    let mut out = vec![0.0f32; n];
    for (b, blk) in data.as_chunks::<17>().0.iter().enumerate() {
        let d = 2f32.powi(i32::from(blk[0]) - 128);
        for j in 0..16 {
            out[b * 32 + j] = K[(blk[1 + j] & 0x0F) as usize] * d;
            out[b * 32 + j + 16] = K[(blk[1 + j] >> 4) as usize] * d;
        }
    }
    Ok(out)
}

/// Raw bytes of `t`, including the types `lapid_convert` lists without a size (i32, mxfp4).
fn raw<'a>(f: &'a GgufFile, t: &GgufTensor) -> Result<&'a [u8]> {
    match t.ggml_type {
        GgmlType::Other(I32) | GgmlType::Other(MXFP4) if t.nbytes == 0 => {
            let per = if t.ggml_type == GgmlType::Other(I32) { 4 * t.numel() } else { t.numel() / 32 * 17 };
            let mut c = t.clone();
            c.nbytes = per as u64;
            let d = f.tensor_data(&c);
            if d.len() != per {
                bail!("format", "GGUF {}: truncated data", t.name);
            }
            Ok(d)
        }
        _ => Ok(f.tensor_data(t)),
    }
}

fn to_f32(f: &GgufFile, t: &GgufTensor) -> Result<Vec<f32>> {
    if t.ggml_type == GgmlType::Other(MXFP4) {
        return dequant_mxfp4(raw(f, t)?, t.numel());
    }
    f.to_f32(t)
}

pub(crate) fn convert(path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
    let f = GgufFile::open(path)?;
    if !is_deepseek4_gguf(&f) {
        bail!("unsupported_model", "{}: GGUF architecture {:?} is not deepseek4", path.display(), f.architecture());
    }
    let cfg = Deepseek4Config::from_json(&config_json(&f)?)?;
    let assets = Assets { tokenizer_json: serde_json::to_string(&tokenizer_json(&f)?)?, generation: json!({}) };

    let mut mapped = Vec::new();
    let mut have = BTreeSet::new();
    for t in &f.tensors {
        if let Some(n) = out_name(&t.name, cfg.n_layers)? {
            if n.ends_with(".ffn.gate.bias")
                && n.strip_prefix("layers.").and_then(|r| r.split('.').next()).and_then(|x| x.parse::<usize>().ok()).is_some_and(|l| l < cfg.n_hash_layers)
            {
                continue;
            }
            have.insert(n.clone());
            mapped.push((n, t));
        }
    }
    let expected = expected_tensors(&cfg);
    lapid_convert::writer::check_complete(&have, &expected)?;

    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("deepseek4").to_string();
    let source = json!({
        "repo": f.get("general.name").and_then(GgufValue::as_str).map_or_else(|| name.clone(), |n| format!("gguf:{n}")),
        "file": path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
        "format": "gguf",
        "file_hash": file_hash(path)?,
    });
    let meta = metadata(&name, dtype, source, &cfg, &assets);
    let mut w = ModelWriter::create(out, meta, dtype, role, expected.len())?;
    w.set_layer_count(Some(cfg.n_layers));
    add_blobs(&mut w, &assets)?;
    let (h, hd, g, ol) = (cfg.n_head, cfg.head_dim, cfg.o_groups, cfg.o_lora_rank);
    for (n, t) in &mapped {
        if n.ends_with(".ffn.gate.tid2eid") {
            let b = raw(&f, t)?;
            if t.ggml_type != GgmlType::Other(I32) {
                bail!("unsupported_dtype", "GGUF {}: hash table type {} (expected i32)", t.name, t.ggml_type.name());
            }
            let data: Vec<i32> = b.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect();
            add_i32(&mut w, n, &t.shape, &data)?;
            continue;
        }
        let shape = if n.ends_with(".attn.wo_a.weight") {
            if t.shape != [g * ol, h / g * hd] {
                bail!("shape_mismatch", "GGUF {}: expected [{}, {}], got {:?}", t.name, g * ol, h / g * hd, t.shape);
            }
            vec![g, ol, h / g * hd]
        } else {
            t.shape.clone()
        };
        write_gguf_tensor(&mut w, dtype, n, t, raw(&f, t)?, || to_f32(&f, t), &shape)?;
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let n = |g: &str| out_name(g, 43).ok().flatten();
        assert_eq!(n("blk.3.attn_output_a.weight").as_deref(), Some("layers.3.attn.wo_a.weight"));
        assert_eq!(n("blk.2.indexer_compressor_ape.weight").as_deref(), Some("layers.2.attn.indexer.compressor.ape"));
        assert_eq!(n("output_hc_fn.weight").as_deref(), Some("hc_head_fn"));
        assert_eq!(n("blk.43.attn_q_a.weight"), None);
        assert_eq!(n("blk.0.exp_probs_b_vl.bias"), None);
        assert_eq!(out_name("blk.0.mystery.weight", 43).err().map(|e| e.code), Some("unsupported_model"));
    }

    #[test]
    fn mxfp4_block() {
        let mut blk = vec![128u8]; // d = 1
        blk.extend((0..16u8).map(|j| j | (((j + 1) % 16) << 4)));
        let v = dequant_mxfp4(&blk, 32).expect("mxfp4");
        assert_eq!(&v[..8], &[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]);
        assert_eq!(v[16], 1.0);
        assert_eq!(v[31], 0.0);
    }
}
