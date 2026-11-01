//! Gemma 4 checkpoints -> `.lapid`: HF safetensors (`model.language_model.X` stored as
//! `model.X`; the vision tower keeps its names; audio towers and the unused k/v weights of
//! kv-shared layers dropped) and GGUF (llama.cpp `gemma4`, no q/k permutation) with the
//! vision tower of a sibling `mmproj` (llama.cpp `gemma4v` projector, mapped to HF names).
//! GGUF has no `use_bidirectional_attention`: like llama.cpp, models without per-layer
//! embeddings get it. Precision: Linear weights and expert stacks quantized; the tied
//! embedding, the PLE table and vision linears at most int8; norms, router, scalars and the
//! position table float.

use std::path::{Path, PathBuf};

use lapid::{Error, Result, bail};
use lapid_convert::assets::HfFiles;
use lapid_convert::engine::{ConvertHooks, ConvertSpec, Emit, HfExtras, SourceKind, TensorSource, sibling_mmproj};
use lapid_convert::gguf::{GgufFile, GgufValue};
use lapid_convert::mapping::{RoleRule, Rule};
use lapid_convert::policy::Role;
use lapid_convert::writer::read_json;
use serde_json::{Value, json};

use super::config::{Gemma4Config, VisionConfig};
use super::model::{EMBED, PLE_EMBED};
use super::vision::{EMBED_PROJ, TOWER};

macro_rules! blk {
    ($g:literal, $h:literal) => {
        Rule::new(concat!("blk.{i}.", $g), concat!("model.layers.{i}.", $h))
    };
}
macro_rules! vblk {
    ($g:literal, $h:literal) => {
        Rule::new(concat!("v.blk.{i}.", $g), concat!("model.vision_tower.encoder.layers.{i}.", $h))
    };
}
const POS_TABLE: &str = "model.vision_tower.patch_embedder.position_embedding_table";
const PATCH: &str = "model.vision_tower.patch_embedder.input_proj.weight";

pub static CONVERT: ConvertSpec = ConvertSpec {
    architecture: "gemma4",
    license: "apache-2.0",
    hf_model_types: &["gemma4", "gemma4_text"],
    gguf_architectures: &["gemma4"],
    hf_names: &[
        Rule::new("model.language_model.{a}.{b}.{c}.{d}.{e}", "model.{a}.{b}.{c}.{d}.{e}"),
        Rule::new("model.language_model.{a}.{b}.{c}.{d}", "model.{a}.{b}.{c}.{d}"),
        Rule::new("model.language_model.{a}.{b}.{c}", "model.{a}.{b}.{c}"),
        Rule::new("model.language_model.{a}.{b}", "model.{a}.{b}"),
        Rule::drop("model.audio_tower.*"),
        Rule::drop("model.embed_audio.*"),
        Rule::drop("lm_head.weight"),
    ],
    gguf_names: &[
        Rule::new("token_embd.weight", EMBED),
        Rule::new("per_layer_token_embd.weight", PLE_EMBED),
        Rule::new("per_layer_model_proj.weight", "model.per_layer_model_projection.weight"),
        Rule::new("per_layer_proj_norm.weight", "model.per_layer_projection_norm.weight"),
        Rule::new("output_norm.weight", "model.norm.weight"),
        Rule::drop("rope_freqs.weight"),
        Rule::drop("output.weight"),
        blk!("attn_norm.weight", "input_layernorm.weight"),
        blk!("post_attention_norm.weight", "post_attention_layernorm.weight"),
        blk!("ffn_norm.weight", "pre_feedforward_layernorm.weight"),
        blk!("post_ffw_norm.weight", "post_feedforward_layernorm.weight"),
        blk!("pre_ffw_norm_2.weight", "pre_feedforward_layernorm_2.weight"),
        blk!("post_ffw_norm_1.weight", "post_feedforward_layernorm_1.weight"),
        blk!("post_ffw_norm_2.weight", "post_feedforward_layernorm_2.weight"),
        blk!("post_norm.weight", "post_per_layer_input_norm.weight"),
        blk!("attn_q.weight", "self_attn.q_proj.weight"),
        blk!("attn_k.weight", "self_attn.k_proj.weight"),
        blk!("attn_v.weight", "self_attn.v_proj.weight"),
        blk!("attn_output.weight", "self_attn.o_proj.weight"),
        blk!("attn_q_norm.weight", "self_attn.q_norm.weight"),
        blk!("attn_k_norm.weight", "self_attn.k_norm.weight"),
        blk!("ffn_gate.weight", "mlp.gate_proj.weight"),
        blk!("ffn_up.weight", "mlp.up_proj.weight"),
        blk!("ffn_down.weight", "mlp.down_proj.weight"),
        blk!("ffn_gate_inp.weight", "router.proj.weight"),
        blk!("ffn_gate_inp.scale", "router.scale"),
        blk!("ffn_down_exps.scale", "router.per_expert_scale"),
        blk!("ffn_gate_up_exps.weight", "experts.gate_up_proj"),
        // Older exports split the fused stack; `transform` joins them.
        blk!("ffn_gate_exps.weight", "experts.gate_proj"),
        blk!("ffn_up_exps.weight", "experts.up_proj"),
        blk!("ffn_down_exps.weight", "experts.down_proj"),
        blk!("inp_gate.weight", "per_layer_input_gate.weight"),
        blk!("proj.weight", "per_layer_projection.weight"),
        blk!("layer_output_scale.weight", "layer_scalar"),
    ],
    roles: &[
        RoleRule::new("*.router.*", Role::Param),
        RoleRule::new("*.experts.*", Role::Linear),
        RoleRule::new(EMBED, Role::LinearInt8),
        RoleRule::new(PLE_EMBED, Role::LinearInt8),
        RoleRule::new(POS_TABLE, Role::Dense),
        RoleRule::new("model.vision_tower.*", Role::LinearInt8),
        RoleRule::new("model.embed_vision.*", Role::LinearInt8),
    ],
    text_assets: true,
    hooks: &Hooks,
};

/// llama.cpp `gemma4v` mmproj names (`conversion/gemma.py`, reversed).
static MMPROJ: &[Rule] = &[
    Rule::new("v.patch_embd.weight", PATCH),
    Rule::new("v.position_embd.weight", POS_TABLE),
    Rule::new("v.std_bias", "model.vision_tower.std_bias"),
    Rule::new("v.std_scale", "model.vision_tower.std_scale"),
    Rule::new("mm.input_projection.weight", EMBED_PROJ),
    vblk!("ln1.weight", "input_layernorm.weight"),
    vblk!("attn_post_norm.weight", "post_attention_layernorm.weight"),
    vblk!("ln2.weight", "pre_feedforward_layernorm.weight"),
    vblk!("ffn_post_norm.weight", "post_feedforward_layernorm.weight"),
    vblk!("attn_q_norm.weight", "self_attn.q_norm.weight"),
    vblk!("attn_k_norm.weight", "self_attn.k_norm.weight"),
    vblk!("attn_q.weight", "self_attn.q_proj.linear.weight"),
    vblk!("attn_k.weight", "self_attn.k_proj.linear.weight"),
    vblk!("attn_v.weight", "self_attn.v_proj.linear.weight"),
    vblk!("attn_out.weight", "self_attn.o_proj.linear.weight"),
    vblk!("ffn_gate.weight", "mlp.gate_proj.linear.weight"),
    vblk!("ffn_up.weight", "mlp.up_proj.linear.weight"),
    vblk!("ffn_down.weight", "mlp.down_proj.linear.weight"),
    vblk!("attn_q.{s}", "self_attn.q_proj.{s}"),
    vblk!("attn_k.{s}", "self_attn.k_proj.{s}"),
    vblk!("attn_v.{s}", "self_attn.v_proj.{s}"),
    vblk!("attn_out.{s}", "self_attn.o_proj.{s}"),
    vblk!("ffn_gate.{s}", "mlp.gate_proj.{s}"),
    vblk!("ffn_up.{s}", "mlp.up_proj.{s}"),
    vblk!("ffn_down.{s}", "mlp.down_proj.{s}"),
];

struct Hooks;

/// The config as given, once it parses.
fn checked(c: Value) -> Result<Value> {
    Gemma4Config::from_json(&c).map(|_| c)
}

/// Layer index and module of `model.layers.{i}.{rest}`.
fn layer_of(name: &str) -> Option<(usize, &str)> {
    let (i, rest) = name.strip_prefix("model.layers.")?.split_once('.')?;
    Some((i.parse().ok()?, rest))
}

impl ConvertHooks for Hooks {
    fn hf_config(&self, c: &Value) -> Result<Value> {
        checked(c.clone())
    }

    /// `image_processor` of `processor_config.json`.
    fn hf_extras(&self, files: &HfFiles, cfg: &mut Value) -> Result<HfExtras> {
        if let Some(p) = files.optional("processor_config.json")?
            && let Some(ip) = read_json(&p)?.get("image_processor").filter(|v| v.is_object())
            && cfg.get("vision_config").is_some_and(Value::is_object)
        {
            cfg["image_processor"] = ip.clone();
        }
        Ok(HfExtras::default())
    }

    fn gguf_config(&self, f: &GgufFile) -> Result<Value> {
        checked(gguf_config(f)?)
    }

    fn gguf_companion(&self, model: &Path) -> Result<Option<PathBuf>> {
        let Some(p) = sibling_mmproj(model) else { return Ok(None) };
        match GgufFile::open(&p).and_then(|m| vision_config(&m).map(|_| ())) {
            Ok(()) => Ok(Some(p)),
            Err(e) => {
                eprintln!("[lapid-hub] gemma4: {} not used ({}: {}); converting the text model only", p.display(), e.code, e.message);
                Ok(None)
            }
        }
    }

    fn companion_names(&self) -> &'static [Rule] {
        MMPROJ
    }

    fn companion_config(&self, mm: &GgufFile, model: &GgufFile, cfg: &mut Value) -> Result<()> {
        let vc = vision_config(mm)?;
        cfg["image_processor"] = json!({"patch_size": vc["patch_size"], "pooling_kernel_size": vc["pooling_kernel_size"]});
        cfg["vision_config"] = vc;
        let tokens = model.get("tokenizer.ggml.tokens").and_then(GgufValue::as_array);
        for (k, tok) in [("image_token_id", "<|image|>"), ("boi_token_id", "<|image>"), ("eoi_token_id", "<image|>")] {
            if cfg.get(k).is_none_or(Value::is_null)
                && let Some(i) = tokens.and_then(|t| t.iter().position(|v| v.as_str() == Some(tok)))
            {
                cfg[k] = json!(i);
            }
        }
        Ok(())
    }

    fn expected(&self, c: &Value) -> Result<Vec<String>> {
        let cfg = Gemma4Config::from_json(c)?;
        let mut v = vec![EMBED.to_string(), "model.norm.weight".to_string()];
        if cfg.ple_dim > 0 {
            v.extend([PLE_EMBED, "model.per_layer_model_projection.weight", "model.per_layer_projection_norm.weight"].map(String::from));
        }
        for i in 0..cfg.n_layers {
            let mut n = vec!["input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"];
            n.extend(["layer_scalar", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "self_attn.q_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight"]);
            if i < cfg.first_shared() {
                n.extend(["self_attn.k_proj.weight", "self_attn.k_norm.weight"]);
                if !cfg.k_eq_v(i) {
                    n.push("self_attn.v_proj.weight");
                }
            }
            if cfg.ple_dim > 0 {
                n.extend(["per_layer_input_gate.weight", "per_layer_projection.weight", "post_per_layer_input_norm.weight"]);
            }
            if cfg.moe.is_some() {
                n.extend(["router.proj.weight", "router.scale", "router.per_expert_scale", "experts.gate_up_proj", "experts.down_proj"]);
                n.extend(["pre_feedforward_layernorm_2.weight", "post_feedforward_layernorm_1.weight", "post_feedforward_layernorm_2.weight"]);
            }
            v.extend(n.iter().map(|n| format!("model.layers.{i}.{n}")));
        }
        Ok(v)
    }

    /// mmproj: position table [2, P, D]; clamp scalars as [1].
    fn reshape(&self, name: &str, shape: &[usize], _cfg: &Value, src: SourceKind) -> Option<Vec<usize>> {
        let n: usize = shape.iter().product();
        if src != SourceKind::Gguf || !name.starts_with(TOWER) {
            return None;
        }
        if name == POS_TABLE && shape.len() == 2 {
            return Some(vec![2, shape[0] / 2, shape[1]]);
        }
        (n == 1 && shape.len() != 1).then(|| vec![1])
    }

    /// Matrices a quantized GGUF (or its projector) stores as floats stay floats.
    fn gguf_keep_float(&self, name: &str) -> bool {
        !name.contains(".experts.")
    }

    fn transform(&self, name: &str, src: &dyn TensorSource, c: &Value) -> Result<Emit> {
        // kv-shared layers attend to an earlier layer's K/V: their own k / v are unused.
        if let Some((i, rest)) = layer_of(name)
            && matches!(rest, "self_attn.k_proj.weight" | "self_attn.v_proj.weight" | "self_attn.k_norm.weight")
        {
            let cfg = Gemma4Config::from_json(c)?;
            return Ok(if i >= cfg.first_shared() { Emit::Drop } else { Emit::Keep });
        }
        // GGUF patch kernel [D, 3, p, p] -> HF [D, p * p * 3] (row, col, channel).
        if name == PATCH && src.kind() == SourceKind::Gguf {
            let shape = src.shape(name).unwrap_or_default();
            let &[d, 3, p, _] = shape.as_slice() else { bail!("shape_mismatch", "mmproj: patch kernel {shape:?}") };
            let data = src.f32(name)?;
            let mut out = vec![0.0f32; data.len()];
            for (o, ch, y, x) in (0..d).flat_map(|o| (0..3).flat_map(move |ch| (0..p).flat_map(move |y| (0..p).map(move |x| (o, ch, y, x))))) {
                out[o * 3 * p * p + (y * p + x) * 3 + ch] = data[((o * 3 + ch) * p + y) * p + x];
            }
            return Ok(Emit::Replace(vec![(name.to_string(), vec![d, 3 * p * p], out)]));
        }
        // Split expert stacks [E, I, D]: per expert the gate rows, then the up rows.
        if let Some((i, rest)) = layer_of(name) {
            match rest {
                "experts.up_proj" => return Ok(Emit::Drop),
                "experts.gate_proj" => {
                    let up = format!("model.layers.{i}.experts.up_proj");
                    let (g, u, s) = (src.f32(name)?, src.f32(&up)?, src.shape(name).unwrap_or_default());
                    let per = g.len() / s.first().copied().unwrap_or(1).max(1);
                    let fused: Vec<f32> = g.chunks(per).zip(u.chunks(per)).flat_map(|(a, b)| a.iter().chain(b).copied()).collect();
                    return Ok(Emit::Replace(vec![(format!("model.layers.{i}.experts.gate_up_proj"), vec![s[0], 2 * s[1], s[2]], fused)]));
                }
                _ => {}
            }
        }
        Ok(Emit::Keep)
    }
}

fn need(f: &GgufFile, k: &str) -> Result<usize> {
    f.get(k).and_then(GgufValue::as_u64).map(|v| v as usize).ok_or_else(|| Error::new("missing_metadata", format!("GGUF: missing {k:?}")))
}

/// HF `vision_config` of a llama.cpp `gemma4v` projector.
fn vision_config(mm: &GgufFile) -> Result<Value> {
    let kind = mm.get("clip.vision.projector_type").or_else(|| mm.get("clip.projector_type")).and_then(GgufValue::as_str).unwrap_or("");
    if kind != "gemma4v" {
        bail!("unsupported_model", "mmproj: projector type {kind:?} is not Gemma 4 vision (\"gemma4v\")");
    }
    let (d, n_head) = (need(mm, "clip.vision.embedding_length")?, need(mm, "clip.vision.attention.head_count")?);
    let pos = mm.tensor("v.position_embd.weight").ok_or_else(|| Error::new("missing_tensor", "mmproj: no v.position_embd.weight"))?;
    let v = json!({
        "hidden_size": d, "intermediate_size": need(mm, "clip.vision.feed_forward_length")?, "num_hidden_layers": need(mm, "clip.vision.block_count")?,
        "num_attention_heads": n_head, "num_key_value_heads": mm.get("clip.vision.attention.head_count_kv").and_then(GgufValue::as_u64).unwrap_or(n_head as u64),
        "head_dim": mm.tensor("v.blk.0.attn_q_norm.weight").map_or(d / n_head.max(1), |t| t.numel()),
        "rms_norm_eps": mm.get("clip.vision.attention.layer_norm_epsilon").and_then(GgufValue::as_f64).unwrap_or(1e-6),
        "patch_size": need(mm, "clip.vision.patch_size")?, "pooling_kernel_size": mm.get("clip.vision.projector.scale_factor").and_then(GgufValue::as_u64).unwrap_or(3),
        "position_embedding_size": pos.numel() / (2 * d).max(1), "rope_parameters": {"rope_type": "default", "rope_theta": 100.0},
        "standardize": mm.tensor("v.std_bias").is_some(),
    });
    VisionConfig::from_json(&v).map(|_| v)
}

/// HF-style text config from `gemma4.*` GGUF keys (per-layer arrays where llama.cpp has
/// them; `rope_freqs` gives the partial rotary factor of full layers).
fn gguf_config(f: &GgufFile) -> Result<Value> {
    let get = |k: &str| f.get(&format!("gemma4.{k}"));
    let uint = |k: &str| need(f, &format!("gemma4.{k}"));
    let float = |k: &str| get(k).and_then(GgufValue::as_f64);
    let n_layers = uint("block_count")?;
    let per_layer = |k: &str| -> Result<Vec<usize>> {
        let v = get(k).ok_or_else(|| Error::new("missing_metadata", format!("GGUF: missing gemma4.{k}")))?;
        Ok(match v.as_array() {
            Some(a) => a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect(),
            None => vec![v.as_u64().unwrap_or(0) as usize; n_layers],
        })
    };
    let swa: Vec<bool> = get("attention.sliding_window_pattern")
        .and_then(GgufValue::as_array)
        .map(|a| a.iter().map(|v| matches!(v, GgufValue::Bool(true)) || v.as_u64().is_some_and(|x| x != 0)).collect())
        .unwrap_or_default();
    if swa.len() != n_layers {
        bail!("metadata", "GGUF: sliding_window_pattern has {} entries for {n_layers} layers", swa.len());
    }
    let (ff, kv) = (per_layer("feed_forward_length")?, per_layer("attention.head_count_kv")?);
    let ff_base = ff.iter().copied().min().unwrap_or(0);
    let kv_swa = (0..n_layers).find(|&i| swa[i]).map_or(kv[0], |i| kv[i]);
    let global_hd = uint("attention.key_length")?;
    let partial = match f.tensor("rope_freqs.weight") {
        Some(t) => (2 * f.to_f32(t)?.iter().filter(|&&x| x < 1e3).count()) as f64 / global_hd as f64,
        None => 1.0,
    };
    let shared = get("attention.shared_kv_layers").and_then(GgufValue::as_u64).unwrap_or(0) as usize;
    let first_full = (0..n_layers.saturating_sub(shared)).find(|&i| !swa[i]);
    let rows = |name: &str| f.tensor(name).map(|t| t.shape[0]);
    let tok = |k: &str| f.get(&format!("tokenizer.ggml.{k}")).and_then(GgufValue::as_u64);
    let ple = get("embedding_length_per_layer_input").and_then(GgufValue::as_u64).unwrap_or(0);
    let mut cfg = json!({
        "model_type": "gemma4_text", "vocab_size": rows("token_embd.weight").ok_or_else(|| Error::new("missing_tensor", "GGUF: no token_embd.weight"))?,
        "hidden_size": uint("embedding_length")?, "intermediate_size": ff_base, "use_double_wide_mlp": ff.iter().any(|&x| x > ff_base),
        "num_hidden_layers": n_layers, "num_attention_heads": uint("attention.head_count")?, "num_key_value_heads": kv_swa,
        "num_global_key_value_heads": (0..n_layers).find(|&i| !swa[i]).map_or(kv_swa, |i| kv[i]),
        "attention_k_eq_v": first_full.is_some_and(|i| f.tensor(&format!("blk.{i}.attn_v.weight")).is_none()),
        "head_dim": get("attention.key_length_swa").and_then(GgufValue::as_u64).map_or(global_hd, |v| v as usize), "global_head_dim": global_hd,
        "rms_norm_eps": float("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
        "layer_types": swa.iter().map(|&s| if s { "sliding_attention" } else { "full_attention" }).collect::<Vec<_>>(),
        "sliding_window": get("attention.sliding_window").and_then(GgufValue::as_u64).unwrap_or(512),
        "rope_parameters": {
            "full_attention": {"rope_type": "proportional", "rope_theta": float("rope.freq_base").unwrap_or(1e6), "partial_rotary_factor": partial},
            "sliding_attention": {"rope_type": "default", "rope_theta": float("rope.freq_base_swa").unwrap_or(1e4)},
        },
        "num_kv_shared_layers": shared, "hidden_size_per_layer_input": ple, "vocab_size_per_layer_input": rows("per_layer_token_embd.weight").unwrap_or(0),
        "final_logit_softcapping": float("final_logit_softcapping"), "max_position_embeddings": get("context_length").and_then(GgufValue::as_u64).unwrap_or(131_072),
        "tie_word_embeddings": true, "bos_token_id": tok("bos_token_id"), "eos_token_id": tok("eos_token_id"), "pad_token_id": tok("padding_token_id"),
    });
    if let Some(e) = get("expert_count").and_then(GgufValue::as_u64).filter(|&e| e > 0) {
        let moe_ff = get("expert_feed_forward_length").and_then(|v| v.as_u64().or_else(|| v.as_array().and_then(|a| a.iter().find_map(GgufValue::as_u64))));
        cfg["enable_moe_block"] = json!(true);
        cfg["num_experts"] = json!(e);
        cfg["top_k_experts"] = json!(get("expert_used_count").and_then(GgufValue::as_u64).unwrap_or(0));
        cfg["moe_intermediate_size"] = json!(moe_ff.unwrap_or(0));
    }
    if ple == 0 {
        cfg["use_bidirectional_attention"] = json!("vision");
    }
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use lapid_convert::mapping::{Renamed, rename};

    use super::*;

    #[test]
    fn names_and_roles() {
        let r = |rules, n| match rename(rules, n) {
            Renamed::To(s) => s,
            other => format!("{other:?}"),
        };
        assert_eq!(r(CONVERT.hf_names, "model.language_model.layers.3.mlp.up_proj.weight"), "model.layers.3.mlp.up_proj.weight");
        assert_eq!(r(CONVERT.hf_names, "model.language_model.layers.0.layer_scalar"), "model.layers.0.layer_scalar");
        assert_eq!(r(CONVERT.hf_names, "model.audio_tower.x.weight"), "Drop");
        assert_eq!(r(CONVERT.gguf_names, "blk.2.ffn_down_exps.scale"), "model.layers.2.router.per_expert_scale");
        assert_eq!(r(CONVERT.gguf_names, "blk.0.layer_output_scale.weight"), "model.layers.0.layer_scalar");
        assert_eq!(r(MMPROJ, "v.blk.0.ffn_down.output_min"), "model.vision_tower.encoder.layers.0.mlp.down_proj.output_min");
        assert_eq!(r(MMPROJ, "v.blk.3.attn_q.weight"), "model.vision_tower.encoder.layers.3.self_attn.q_proj.linear.weight");
        assert_eq!(r(MMPROJ, "a.blk.0.ln1.weight"), "NoMatch");
        assert_eq!(CONVERT.role("model.layers.0.experts.gate_up_proj", &[128, 1408, 2816]), Role::Linear);
        assert_eq!(CONVERT.role("model.layers.0.router.proj.weight", &[128, 2816]), Role::Param);
        assert_eq!(CONVERT.role(EMBED, &[262144, 2816]), Role::LinearInt8);
        assert_eq!(CONVERT.role("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", &[768, 768]), Role::LinearInt8);
        assert_eq!(CONVERT.role(POS_TABLE, &[2, 10240, 768]), Role::Dense);
        assert_eq!(CONVERT.role("model.layers.3.self_attn.q_proj.weight", &[4096, 2816]), Role::Linear);
    }
}
