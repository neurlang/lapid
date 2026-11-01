//! LFM3 checkpoints -> `.lapid`: HF safetensors (names kept, `lm_head` tied) and GGUF
//! (llama.cpp `lfm3` naming, no q/k permutation: LFM3 uses rotate-half RoPE). Precision:
//! Linear weights quantized, the tied embedding at most int8 (`LinearInt8`), depthwise conv
//! weights f16, norms f32.

use lapid::{Error, Result, bail};
use lapid_convert::engine::{ConvertHooks, ConvertSpec, SourceKind};
use lapid_convert::gguf::{GgufFile, GgufValue};
use lapid_convert::mapping::{RoleRule, Rule};
use lapid_convert::policy::Role;
use serde_json::{Value, json};

use super::config::{LayerType, Lfm3Config};
use super::model::EMBED;

const ATTN: [&str; 7] = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm", "self_attn.gate_proj"];
macro_rules! blk {
    ($g:literal, $h:literal) => {
        Rule::new(concat!("blk.{i}.", $g, ".{w}"), concat!("model.layers.{i}.", $h, ".{w}"))
    };
}

pub static CONVERT: ConvertSpec = ConvertSpec {
    architecture: "lfm3",
    license: "lfm1.0",
    hf_model_types: &["lfm3"],
    gguf_architectures: &["lfm3"],
    hf_names: &[Rule::drop("lm_head.weight")],
    gguf_names: &[
        Rule::new("token_embd.weight", EMBED),
        Rule::new("output_norm.weight", "model.embedding_norm.weight"),
        Rule::drop("rope_freqs.weight"),
        Rule::drop("output.weight"),
        blk!("attn_norm", "operator_norm"),
        blk!("ffn_norm", "ffn_norm"),
        blk!("post_attention_norm", "post_operator_norm"),
        blk!("post_ffw_norm", "post_ffn_norm"),
        blk!("ffn_gate", "feed_forward.w1"),
        blk!("ffn_up", "feed_forward.w3"),
        blk!("ffn_down", "feed_forward.w2"),
        blk!("shortconv.conv", "conv.conv"),
        blk!("shortconv.in_proj", "conv.in_proj"),
        blk!("shortconv.out_proj", "conv.out_proj"),
        blk!("attn_q", "self_attn.q_proj"),
        blk!("attn_k", "self_attn.k_proj"),
        blk!("attn_v", "self_attn.v_proj"),
        blk!("attn_output", "self_attn.o_proj"),
        blk!("attn_q_norm", "self_attn.q_norm"),
        blk!("attn_k_norm", "self_attn.k_norm"),
        blk!("attn_gate", "self_attn.gate_proj"),
    ],
    roles: &[RoleRule::new(EMBED, Role::LinearInt8)],
    text_assets: true,
    hooks: &Hooks,
};

struct Hooks;

impl ConvertHooks for Hooks {
    fn hf_config(&self, c: &Value) -> Result<Value> {
        Lfm3Config::from_json(c).map(|_| c.clone())
    }

    fn gguf_config(&self, f: &GgufFile) -> Result<Value> {
        let c = gguf_config(f)?;
        Lfm3Config::from_json(&c).map(|_| c)
    }

    fn expected(&self, c: &Value) -> Result<Vec<String>> {
        let cfg = Lfm3Config::from_json(c)?;
        let mut v = vec![EMBED.to_string(), "model.embedding_norm.weight".to_string()];
        for (i, t) in cfg.layer_types.iter().enumerate() {
            let mut names = vec!["operator_norm", "ffn_norm", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"];
            if cfg.sandwich_norm {
                names.extend(["post_operator_norm", "post_ffn_norm"]);
            }
            let conv = ["conv.conv", "conv.in_proj", "conv.out_proj"];
            if *t == LayerType::Conv {
                names.extend(conv);
                if cfg.conv_bias {
                    v.extend(conv.map(|n| format!("model.layers.{i}.{n}.bias")));
                }
            } else {
                names.extend(ATTN);
            }
            v.extend(names.iter().map(|n| format!("model.layers.{i}.{n}.weight")));
        }
        Ok(v)
    }

    /// GGUF stores the depthwise conv as [D, K]; ours is [D, 1, K].
    fn reshape(&self, name: &str, shape: &[usize], _cfg: &Value, src: SourceKind) -> Option<Vec<usize>> {
        (src == SourceKind::Gguf && name.ends_with("conv.conv.weight") && shape.len() == 2).then(|| vec![shape[0], 1, shape[1]])
    }
}

/// HF-style config from `lfm3.*` keys: per-layer `head_count_kv` (0 = conv layer),
/// `sliding_window_pattern`, `rope.freq_base(_swa)`; `rope_freqs` (1 on rotated pairs,
/// huge on the zero tail) gives the partial rotary factor.
fn gguf_config(f: &GgufFile) -> Result<Value> {
    let get = |k: &str| f.get(&format!("lfm3.{k}"));
    let uint = |k: &str| get(k).and_then(GgufValue::as_u64).map(|v| v as usize).ok_or_else(|| Error::new("missing_metadata", format!("GGUF: missing lfm3.{k}")));
    let float = |k: &str, d: f64| get(k).and_then(GgufValue::as_f64).unwrap_or(d);
    let (n_layers, hidden, n_head) = (uint("block_count")?, uint("embedding_length")?, uint("attention.head_count")?);
    let kv = get("attention.head_count_kv").ok_or_else(|| Error::new("missing_metadata", "GGUF: no head_count_kv"))?;
    let kv: Vec<usize> = match kv.as_array() {
        Some(a) => a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect(),
        None => vec![kv.as_u64().unwrap_or(0) as usize; n_layers],
    };
    let swa: Vec<bool> = get("attention.sliding_window_pattern")
        .and_then(GgufValue::as_array)
        .map(|a| a.iter().map(|v| matches!(v, GgufValue::Bool(true)) || v.as_u64().is_some_and(|x| x != 0)).collect())
        .unwrap_or_default();
    if kv.len() != n_layers {
        bail!("metadata", "GGUF: head_count_kv has {} entries for {n_layers} layers", kv.len());
    }
    let types: Vec<&str> = (0..n_layers)
        .map(|i| match (kv[i], swa.get(i).copied().unwrap_or(false)) {
            (0, _) => "conv",
            (_, true) => "sliding_attention",
            _ => "full_attention",
        })
        .collect();
    let head_dim = get("attention.key_length").and_then(GgufValue::as_u64).map_or(hidden / n_head.max(1), |v| v as usize);
    let partial = match f.tensor("rope_freqs.weight") {
        Some(t) => (2 * f.to_f32(t)?.iter().filter(|&&x| x < 1e3).count()) as f64 / head_dim as f64,
        None => 1.0,
    };
    let vocab = f.tensor("token_embd.weight").map(|t| t.shape[0]).ok_or_else(|| Error::new("missing_tensor", "GGUF: no token_embd.weight"))?;
    let tok = |k: &str| f.get(&format!("tokenizer.ggml.{k}")).and_then(GgufValue::as_u64);
    Ok(json!({
        "model_type": "lfm3", "vocab_size": vocab, "hidden_size": hidden, "intermediate_size": uint("feed_forward_length")?,
        "num_hidden_layers": n_layers, "num_attention_heads": n_head, "num_key_value_heads": kv.iter().copied().find(|&k| k > 0).unwrap_or(0),
        "head_dim": head_dim, "norm_eps": float("attention.layer_norm_rms_epsilon", 1e-5),
        "conv_L_cache": get("shortconv.l_cache").and_then(GgufValue::as_u64).unwrap_or(7),
        "conv_bias": f.tensor("blk.0.shortconv.in_proj.bias").is_some(), "layer_types": types,
        "sliding_window": get("attention.sliding_window").and_then(GgufValue::as_u64).unwrap_or(256),
        "rope_parameters": {
            "full_attention": {"rope_type": "default", "rope_theta": float("rope.freq_base", 1e6), "partial_rotary_factor": partial},
            "sliding_attention": {"rope_type": "default", "rope_theta": float("rope.freq_base_swa", 1e4)},
        },
        "use_sandwich_norm": f.tensor("blk.0.post_attention_norm.weight").is_some(),
        "max_position_embeddings": get("context_length").and_then(GgufValue::as_u64).unwrap_or(128_000),
        "tie_word_embeddings": true, "bos_token_id": tok("bos_token_id"), "eos_token_id": tok("eos_token_id"), "pad_token_id": tok("padding_token_id"),
    }))
}
