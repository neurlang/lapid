//! Qwen4-Exp checkpoints -> `.lapid`: HF safetensors (`Qwen4ExpForConditionalGeneration` /
//! `...ForCausalLM`; vision tower and MTP head dropped) and GGUF (`qwen4exp`, llama.cpp
//! naming; merge split GGUFs first with `llama-gguf-split --merge`). Names follow HF; the
//! Qwen3.5 parts convert as in `qwen35::convert`.
//!
//! On top of Qwen3.5: hyper-connection mixers (`[hc, D]` grouped norms; GGUF merged
//! `*_down_inject` split by raw rows), the QSA indexer (HF `index_qk_proj` split into q / k,
//! `{q,k}_layernorm` renamed), the PLE block with its n-gram table (HF shards concatenated,
//! hash-constant buffers dropped: recomputed from the config; GGUF `per_layer_token_embd`),
//! GGUF merged `attn_qkv_gate` / `ssm_beta_alpha` split by raw rows. GGUF float tensors stay
//! floats (the released Q4_0 file keeps hc / indexer weights in BF16).

use lapid::{Error, Result, bail};
use lapid_convert::engine::{ConvertHooks, ConvertSpec, Emit, SourceKind, TensorSource};
use lapid_convert::gguf::{GgufFile, GgufValue};
use lapid_convert::mapping::Rule;
use lapid_convert::policy::layer_index;
use serde_json::{Value, json};

use super::config::Qwen4ExpConfig;
use super::model::MIXER;
use crate::llama::model::{EMBED, LM_HEAD};
use crate::qwen35::config::LayerType;
use crate::qwen35::convert::{GGUF_LAYERS, HF_LM, ROLES, concat, expected_layers, gguf_config as qwen35_gguf_config};

macro_rules! blk {
    ($g:literal, $h:literal) => {
        Rule::new(concat!("blk.{i}.", $g), concat!("model.layers.{i}.", $h))
    };
}

/// HF rule for `model.language_model.<from>` and `model.<from>`.
macro_rules! hf {
    ($f:literal, $t:literal) => {
        [Rule::new(concat!("model.language_model.", $f), concat!("model.", $t)), Rule::new(concat!("model.", $f), concat!("model.", $t))]
    };
}

const HF_NAMES: [Rule; 19] = {
    let a = hf!("layers.{i}.linear_attn.A_log", "layers.{i}.linear_attn.A");
    let q = hf!("layers.{i}.self_attn.indexer.q_layernorm.weight", "layers.{i}.self_attn.indexer.q_norm.weight");
    let k = hf!("layers.{i}.self_attn.indexer.k_layernorm.weight", "layers.{i}.self_attn.indexer.k_norm.weight");
    let s = hf!("layers.{i}.ple.ple_embedding.ngram_embedding.shard_{k}.weight", "layers.{i}.ple.ngram_shard.{k}");
    let t = hf!("layers.{i}.ple.ple_embedding.ngram_embedding.weight", "layers.{i}.ple.ngram_embedding.weight");
    let own = [Rule::drop("mtp.*"), Rule::drop("*visual.*"), a[0], a[1], q[0], q[1], k[0], k[1], s[0], s[1], t[0], t[1], Rule::drop("*.ple.ple_embedding.*")];
    concat(own, HF_LM)
};

const GGUF_NAMES: [Rule; 59] = concat(
    [
        Rule::new("token_embd.weight", EMBED),
        Rule::new("output.weight", LM_HEAD),
        Rule::new("output_hc_norm.weight", "model.hyper_connection_mixer.hc_norm.weight"),
        Rule::new("output_hc_down.weight", "model.hyper_connection_mixer.input_mix_weight_down.weight"),
        Rule::new("output_hc_up.weight", "model.hyper_connection_mixer.input_mix_weight_up.weight"),
        Rule::new("per_layer_token_embd.weight", "ple.ngram_table"),
        Rule::drop("rope_freqs.weight"),
        Rule::drop("blk.{i}.nextn.*"),
        blk!("hc_attn_down_inject.weight", "attn_hyper_connection.down_inject"),
        blk!("hc_ffn_down_inject.weight", "mlp_hyper_connection.down_inject"),
        blk!("attn_qkv_gate.weight", "linear_attn.qkv_gate"),
        blk!("ssm_beta_alpha.weight", "linear_attn.beta_alpha"),
        blk!("hc_attn_norm.weight", "attn_hyper_connection.hc_norm.weight"),
        blk!("hc_attn_down.weight", "attn_hyper_connection.input_mix_weight_down.weight"),
        blk!("hc_attn_up.weight", "attn_hyper_connection.input_mix_weight_up.weight"),
        blk!("hc_attn_inject.weight", "attn_hyper_connection.block_inject_weight.weight"),
        blk!("hc_ffn_norm.weight", "mlp_hyper_connection.hc_norm.weight"),
        blk!("hc_ffn_down.weight", "mlp_hyper_connection.input_mix_weight_down.weight"),
        blk!("hc_ffn_up.weight", "mlp_hyper_connection.input_mix_weight_up.weight"),
        blk!("hc_ffn_inject.weight", "mlp_hyper_connection.block_inject_weight.weight"),
        blk!("indexer.q_proj.weight", "self_attn.indexer.q_proj.weight"),
        blk!("indexer.k_proj.weight", "self_attn.indexer.k_proj.weight"),
        blk!("indexer.q_norm.weight", "self_attn.indexer.q_norm.weight"),
        blk!("indexer.k_norm.weight", "self_attn.indexer.k_norm.weight"),
        blk!("ple_key.weight", "ple.key_proj.weight"),
        blk!("ple_value.weight", "ple.value_proj.weight"),
        blk!("ple_norm_key.weight", "ple.norm_key.weight"),
        blk!("ple_norm_query.weight", "ple.norm_query.weight"),
        blk!("ple_norm_conv.weight", "ple.norm_conv.weight"),
        blk!("ple_conv1d.weight", "ple.conv1d.weight"),
    ],
    GGUF_LAYERS,
);

pub static CONVERT: ConvertSpec = ConvertSpec {
    architecture: "qwen4exp",
    license: "apache-2.0",
    hf_model_types: &["qwen4_exp", "qwen4_exp_text"],
    gguf_architectures: &["qwen4exp"],
    hf_names: &HF_NAMES,
    gguf_names: &GGUF_NAMES,
    roles: ROLES,
    text_assets: true,
    hooks: &Hooks,
};

struct Hooks;

/// Grouped `[hc * D]` norm weights are stored `[hc, D]`.
fn is_grouped_norm(name: &str) -> bool {
    name.ends_with(".hc_norm.weight") || name.contains(".ple.norm_")
}

/// `rows` consecutive row ranges of the given sizes as [`Emit::Rows`] parts of width `cols`.
fn row_ranges(parts: &[(String, usize)], cols: usize) -> Emit {
    let mut start = 0;
    Emit::Rows(
        parts
            .iter()
            .map(|(n, r)| {
                start += r;
                (n.clone(), vec![*r, cols], (start - r..start).collect())
            })
            .collect(),
    )
}

impl ConvertHooks for Hooks {
    fn hf_config(&self, c: &Value) -> Result<Value> {
        Ok(Qwen4ExpConfig::from_json(c)?.to_json())
    }

    fn gguf_config(&self, f: &GgufFile) -> Result<Value> {
        Ok(Qwen4ExpConfig::from_json(&gguf_config(f)?)?.to_json())
    }

    fn expected(&self, c: &Value) -> Result<Vec<String>> {
        let cfg = Qwen4ExpConfig::from_json(c)?;
        let hc_parts = ["hc_norm", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"];
        let mut v = vec![EMBED.to_string()];
        if !cfg.base.tie_embeddings {
            v.push(LM_HEAD.to_string());
        }
        v.extend(hc_parts[..3].iter().map(|n| format!("{MIXER}.{n}.weight")));
        expected_layers(&cfg.base, false, &mut v);
        for (i, t) in cfg.base.layer_types.iter().enumerate() {
            let p = format!("model.layers.{i}");
            for hc in ["attn_hyper_connection", "mlp_hyper_connection"] {
                v.extend(hc_parts.iter().map(|n| format!("{p}.{hc}.{n}.weight")));
            }
            if *t == LayerType::FullAttention && cfg.qsa.is_some() {
                v.extend(["q_proj", "k_proj", "q_norm", "k_norm"].map(|n| format!("{p}.self_attn.indexer.{n}.weight")));
            }
            if cfg.ple.as_ref().is_some_and(|q| q.layer == i) {
                v.extend(["key_proj", "value_proj", "norm_key", "norm_query", "norm_conv", "conv1d", "ngram_embedding"].map(|n| format!("{p}.ple.{n}.weight")));
            }
        }
        Ok(v)
    }

    fn reshape(&self, name: &str, shape: &[usize], cfg: &Value, _src: SourceKind) -> Option<Vec<usize>> {
        let hc = cfg["hc_count"].as_u64()? as usize;
        match shape {
            [n] if is_grouped_norm(name) => Some(vec![hc, n / hc]),
            _ => crate::qwen35::convert::reshape(name, shape),
        }
    }

    fn transform(&self, name: &str, src: &dyn TensorSource, cfg: &Value) -> Result<Emit> {
        let cfg = Qwen4ExpConfig::from_json(cfg)?;
        let b = &cfg.base;
        let shape = src.shape(name).unwrap_or_default();
        let cols = shape.last().copied().unwrap_or(0);
        let l = name.rsplit_once('.').map_or("", |x| x.0);
        if layer_index(name).is_some_and(|i| i >= cfg.n_layers()) {
            return Ok(Emit::Drop); // GGUF MTP / nextn blocks
        }
        if name == "ple.ngram_table" {
            let Some(p) = &cfg.ple else { bail!("metadata", "GGUF: n-gram table without PLE keys") };
            return Ok(Emit::Rename(format!("model.layers.{}.ple.ngram_embedding.weight", p.layer)));
        }
        if let Some(p) = name.strip_suffix("_hyper_connection.down_inject") {
            let kind = &p[p.rfind('.').map_or(0, |i| i + 1)..];
            let p = &p[..p.rfind('.').unwrap_or(0)];
            let n = |part: &str| format!("{p}.{kind}_hyper_connection.{part}.weight");
            return Ok(row_ranges(&[(n("input_mix_weight_down"), cfg.hc_lowrank), (n("block_inject_weight"), cfg.hc)], cols));
        }
        if name.ends_with(".linear_attn.qkv_gate") {
            return Ok(row_ranges(&[(format!("{l}.in_proj_qkv.weight"), b.conv_dim()), (format!("{l}.in_proj_z.weight"), b.linear_v_heads * b.linear_v_dim)], cols));
        }
        if name.ends_with(".linear_attn.beta_alpha") {
            return Ok(row_ranges(&[(format!("{l}.in_proj_b.weight"), b.linear_v_heads), (format!("{l}.in_proj_a.weight"), b.linear_v_heads)], cols));
        }
        if let Some(p) = name.strip_suffix(".indexer.index_qk_proj.weight") {
            let Some(q) = &cfg.qsa else { bail!("metadata", "qwen4exp: indexer weights without indexer config") };
            return Ok(row_ranges(&[(format!("{p}.indexer.q_proj.weight"), q.n_heads * q.head_dim), (format!("{p}.indexer.k_proj.weight"), q.head_dim)], cols));
        }
        if let Some((p, k)) = name.split_once(".ple.ngram_shard.") {
            if k != "0" {
                return Ok(Emit::Drop);
            }
            let (mut data, mut rows) = (Vec::new(), 0);
            for k in 0.. {
                let Some(s) = src.shape(&format!("{p}.ple.ngram_shard.{k}")) else { break };
                if s.len() != 2 || s[1] != cols {
                    bail!("shape_mismatch", "{p}: n-gram shard {k} shape {s:?}");
                }
                rows += s[0];
                data.extend(src.f32(&format!("{p}.ple.ngram_shard.{k}"))?);
            }
            return Ok(Emit::Replace(vec![(format!("{p}.ple.ngram_embedding.weight"), vec![rows, cols], data)]));
        }
        if src.kind() == SourceKind::Hf && name.contains(".ple.norm_") {
            let mut data = src.f32(name)?;
            data.iter_mut().for_each(|x| *x += 1.0);
            return Ok(Emit::Replace(vec![(name.to_string(), shape, data)]));
        }
        crate::qwen35::convert::transform(b, name, src)
    }

    fn gguf_keep_float(&self, _name: &str) -> bool {
        true
    }

    fn metadata(&self, meta: &mut Value) {
        meta["qwen4exp"]["weight_layout"] = json!({"rms_norm": "effective_scale", "deltanet_A": "neg_exp", "v_heads": "tiled", "grouped_norms": "[hc, D]"});
    }
}

/// Qwen3.5 keys plus hyper-connections, the QSA indexer and PLE from `qwen4exp.*`.
fn gguf_config(f: &GgufFile) -> Result<Value> {
    let get = |k: &str| f.get(&format!("qwen4exp.{k}"));
    let uint = |k: &str| get(k).and_then(GgufValue::as_u64).ok_or_else(|| Error::new("missing_metadata", format!("GGUF: missing qwen4exp.{k}")));
    let uints = |k: &str| get(k).and_then(GgufValue::as_array).map(|a| a.iter().map(|v| if let GgufValue::Bool(b) = v { u64::from(*b) } else { v.as_u64().unwrap_or(0) }).collect::<Vec<u64>>());
    let mut v = qwen35_gguf_config(f, "qwen4exp")?;
    v["model_type"] = json!("qwen4_exp_text");
    v["hc_count"] = json!(uint("hyper_connection.count")?);
    v["hc_lowrank"] = json!(uint("hyper_connection.low_rank")?);
    // llama.cpp hard-codes the sigmoid output gate of the released model
    v["output_gate_type"] = json!("sigmoid");
    // no compress ratio (a dense export, e.g. by older llama.cpp converters) -> dense attention
    if let (Some(nh), Some(ratio)) = (get("attention.indexer.head_count").and_then(GgufValue::as_u64), uints("attention.compress_ratios").and_then(|r| r.into_iter().find(|&x| x > 0))) {
        v["indexer_n_heads"] = json!(nh);
        v["indexer_head_dim"] = json!(uint("attention.indexer.key_length")?);
        v["indexer_budget"] = json!(uint("attention.indexer.top_k")?);
        v["indexer_compress_ratio"] = json!(ratio);
    }
    if let Some(layers) = uints("ple.layers").filter(|l| !l.is_empty()) {
        let table = f.tensor("per_layer_token_embd.weight").ok_or_else(|| Error::new("missing_tensor", "GGUF: no per_layer_token_embd.weight"))?;
        let need = |k: &str| uints(k).ok_or_else(|| Error::new("missing_metadata", format!("GGUF: missing qwen4exp.{k}")));
        v["ple"] = json!({
            "layer": layers[0], "ngram_size": uint("ple.ngram_size")?, "heads_per_ngram": uint("ple.heads_per_ngram")?,
            "conv_kernel": uint("ple.conv_kernel")?, "eos_token_id": uint("ple.eos_token_id")?,
            "multipliers": need("ple.layer_multipliers")?, "head_offsets": need("ple.head_offsets")?,
            "head_vocab_sizes": need("ple.head_vocab_sizes")?, "row_dim": table.shape[1], "rows": table.shape[0],
        });
    }
    Ok(v)
}
