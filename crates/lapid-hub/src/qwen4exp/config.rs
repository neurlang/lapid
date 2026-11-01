//! Qwen4-Exp hyper-parameters (`qwen4_exp`, e.g. Qwen3.8-Flash-Next). The text decoder is the
//! Qwen3.5 MoE decoder (`Qwen35Config` parses those keys) plus three additions:
//!
//! * hyper-connections: `hc_count` residual streams mixed through a rank-`hc_lowrank` gate;
//! * QSA sparse attention on the full-attention layers (`indexer_*`);
//! * PLE n-gram hash embeddings on one linear-attention layer (`ple_layer_ids`, 1-based in HF).
//!
//! The `.lapid` metadata object `"qwen4exp"` keeps the HF `text_config` keys plus a resolved
//! `"ple"` object (0-based layer, hash multipliers, head offsets / vocab sizes, row width), so
//! HF and GGUF imports load the same way.

use lapid::{Error, Result, bail};
use serde_json::{Value, json};

use crate::qwen35::config::{LayerType, MoeConfig, Qwen35Config};

/// Activation of the Gated DeltaNet output gate (`output_gate_type`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GateAct {
    Sigmoid,
    Silu,
}

impl GateAct {
    pub fn parse(s: &str) -> Result<GateAct> {
        match s {
            "sigmoid" => Ok(GateAct::Sigmoid),
            "silu" | "swish" => Ok(GateAct::Silu),
            o => bail!("unsupported_model", "qwen4exp: output gate {o:?} (expected sigmoid or silu)"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            GateAct::Sigmoid => "sigmoid",
            GateAct::Silu => "silu",
        }
    }
}

/// QSA indexer of the full-attention layers.
#[derive(Clone, Debug, PartialEq)]
pub struct QsaConfig {
    pub n_heads: usize,
    pub head_dim: usize,
    /// Tokens selected from complete blocks per query.
    pub budget: usize,
    /// Tokens per pooled block.
    pub ratio: usize,
}

impl QsaConfig {
    /// Blocks selected per query.
    pub fn top_blocks(&self) -> usize {
        self.budget / self.ratio
    }

    /// Longest key range (tokens) for which every block is selected, i.e. QSA = dense causal.
    pub fn dense_limit(&self) -> usize {
        self.budget + self.ratio - 1
    }
}

/// PLE n-gram hash embedding of one layer.
#[derive(Clone, Debug, PartialEq)]
pub struct PleConfig {
    /// 0-based decoder layer.
    pub layer: usize,
    pub ngram: usize,
    pub heads_per_ngram: usize,
    pub conv_kernel: usize,
    pub eos: u32,
    pub multipliers: Vec<u64>,
    pub offsets: Vec<u64>,
    pub vocab_sizes: Vec<u64>,
    /// Width of one table row (embed_dim / heads).
    pub row_dim: usize,
    /// Rows of the (padded) table.
    pub rows: usize,
}

impl PleConfig {
    pub fn n_heads(&self) -> usize {
        (self.ngram - 1) * self.heads_per_ngram
    }

    pub fn embed_dim(&self) -> usize {
        self.n_heads() * self.row_dim
    }

    /// Conv history length: (kernel - 1) * dilation, dilation = n-gram size.
    pub fn conv_state(&self) -> usize {
        (self.conv_kernel - 1) * self.ngram
    }

    fn to_json(&self) -> Value {
        json!({"layer": self.layer, "ngram_size": self.ngram, "heads_per_ngram": self.heads_per_ngram,
               "conv_kernel": self.conv_kernel, "eos_token_id": self.eos, "multipliers": self.multipliers,
               "head_offsets": self.offsets, "head_vocab_sizes": self.vocab_sizes, "row_dim": self.row_dim, "rows": self.rows})
    }

    fn from_json(v: &Value) -> Result<PleConfig> {
        let u = |k: &str| v.get(k).and_then(Value::as_u64).ok_or_else(|| Error::new("metadata", format!("qwen4exp ple: missing {k:?}")));
        let arr = |k: &str| -> Result<Vec<u64>> {
            v.get(k)
                .and_then(Value::as_array)
                .ok_or_else(|| Error::new("metadata", format!("qwen4exp ple: missing array {k:?}")))?
                .iter()
                .map(|x| x.as_u64().ok_or_else(|| Error::new("metadata", format!("qwen4exp ple: {k:?} holds a non-integer"))))
                .collect()
        };
        let p = PleConfig {
            layer: u("layer")? as usize,
            ngram: u("ngram_size")? as usize,
            heads_per_ngram: u("heads_per_ngram")? as usize,
            conv_kernel: u("conv_kernel")? as usize,
            eos: u("eos_token_id")? as u32,
            multipliers: arr("multipliers")?,
            offsets: arr("head_offsets")?,
            vocab_sizes: arr("head_vocab_sizes")?,
            row_dim: u("row_dim")? as usize,
            rows: u("rows")? as usize,
        };
        p.validate()?;
        Ok(p)
    }

    fn validate(&self) -> Result<()> {
        let h = self.n_heads();
        if self.ngram < 2 || h == 0 || self.conv_kernel == 0 || self.row_dim == 0 {
            bail!("metadata", "qwen4exp ple: ngram {} / heads {h} / kernel {} / row {}", self.ngram, self.conv_kernel, self.row_dim);
        }
        if self.multipliers.len() < self.ngram || self.offsets.len() < h || self.vocab_sizes.len() < h {
            bail!("metadata", "qwen4exp ple: hash constants shorter than ngram {} / heads {h}", self.ngram);
        }
        for i in 0..h {
            if self.vocab_sizes[i] == 0 || self.offsets[i] + self.vocab_sizes[i] > self.rows as u64 {
                bail!("metadata", "qwen4exp ple: head {i} range exceeds the {} table rows", self.rows);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen4ExpConfig {
    /// Decoder dims, layer types, MoE, RoPE, DeltaNet (Qwen3.5 MoE keys).
    pub base: Qwen35Config,
    pub hc: usize,
    pub hc_lowrank: usize,
    pub output_gate: GateAct,
    pub qsa: Option<QsaConfig>,
    pub ple: Option<PleConfig>,
}

fn opt_usize(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(Value::as_u64).map(|x| x as usize)
}

// ---- HF n-gram hash constants (transformers `modeling_qwen4_exp.py`)

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

fn splitmix64(mut v: u64) -> u64 {
    v = v.wrapping_add(SPLITMIX_GAMMA);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}

/// `_build_layer_multipliers`: odd multipliers below `i64::MAX / vocab`.
pub fn layer_multipliers(vocab: u64, ngram: usize, ple_index: u64, seed: u64) -> Vec<u64> {
    let half = ((i64::MAX as u64) / vocab.max(1) / 2).max(1);
    let base = seed.wrapping_add(10007u64.wrapping_mul(ple_index));
    (0..ngram as u64).map(|i| 2 * (splitmix64(base.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(i + 1))) % half) + 1).collect()
}

fn is_prime(v: u64) -> bool {
    if v < 2 {
        return false;
    }
    if v.is_multiple_of(2) {
        return v == 2;
    }
    let mut d = 3;
    while d * d <= v {
        if v.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// Per-head vocab sizes (the `(global head + 1)`-th prime >= `base`) and row offsets.
pub fn head_ranges(base: u64, heads: usize, ple_index: usize) -> (Vec<u64>, Vec<u64>) {
    let (mut sizes, mut offsets) = (Vec::with_capacity(heads), Vec::with_capacity(heads));
    let mut total = 0u64;
    let mut p = base.saturating_sub(1);
    // primes are consecutive across layers: skip the earlier PLE layers' heads
    for i in 0..(ple_index + 1) * heads {
        p += 1;
        while !is_prime(p) {
            p += 1;
        }
        if i >= ple_index * heads {
            sizes.push(p);
            offsets.push(total);
            total += p;
        }
    }
    (sizes, offsets)
}

impl Qwen4ExpConfig {
    /// Parse an HF config (top level with `text_config`, or the text config itself) or the
    /// `.lapid` `"qwen4exp"` metadata object.
    pub fn from_json(root: &Value) -> Result<Qwen4ExpConfig> {
        let v = root.get("text_config").filter(|t| t.is_object()).unwrap_or(root);
        let mt = root.get("model_type").and_then(Value::as_str).unwrap_or("qwen4_exp");
        if !mt.starts_with("qwen4_exp") {
            bail!("unsupported_model", "qwen4exp: model_type {mt:?} is not qwen4_exp");
        }
        // The Qwen3.5 MoE parser reads the shared keys; HF writes the QSA layers as "indexed_attention".
        let mut base_json = v.clone();
        base_json["model_type"] = json!("qwen3_5_moe");
        if let Some(lt) = base_json.get_mut("layer_types").and_then(Value::as_array_mut) {
            for t in lt.iter_mut() {
                if t.as_str() == Some("indexed_attention") {
                    *t = json!("full_attention");
                }
            }
        }
        if base_json.get("tie_word_embeddings").is_none() {
            base_json["tie_word_embeddings"] = root.get("tie_word_embeddings").cloned().unwrap_or(json!(false));
        }
        if base_json.get("eos_token_id").is_none() {
            base_json["eos_token_id"] = root.get("eos_token_id").cloned().unwrap_or(Value::Null);
        }
        let base = Qwen35Config::from_json(&base_json)?;
        let hc = opt_usize(v, "hc_count").unwrap_or(4);
        let hc_lowrank = opt_usize(v, "hc_lowrank").unwrap_or(320);
        let gate = v.get("output_gate_type").and_then(Value::as_str).or_else(|| v.get("hidden_act").and_then(Value::as_str)).unwrap_or("silu");
        let qsa = match (opt_usize(v, "indexer_n_heads"), opt_usize(v, "indexer_head_dim"), opt_usize(v, "indexer_budget"), opt_usize(v, "indexer_compress_ratio")) {
            (Some(n_heads), Some(head_dim), Some(budget), Some(ratio)) => Some(QsaConfig { n_heads, head_dim, budget, ratio }),
            (None, None, None, None) => None,
            _ => bail!("metadata", "qwen4exp: incomplete QSA indexer config"),
        };
        let ple = match v.get("ple").filter(|p| p.is_object()) {
            Some(p) => Some(PleConfig::from_json(p)?),
            None => Self::hf_ple(v, &base)?,
        };
        let cfg = Qwen4ExpConfig { base, hc, hc_lowrank, output_gate: GateAct::parse(gate)?, qsa, ple };
        cfg.validate()?;
        Ok(cfg)
    }

    /// PLE from the HF keys (hash constants recomputed as transformers does).
    fn hf_ple(v: &Value, base: &Qwen35Config) -> Result<Option<PleConfig>> {
        let ids: Vec<usize> = v
            .get("ple_layer_ids")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect())
            .unwrap_or_default();
        let Some(&first) = ids.first() else { return Ok(None) };
        if ids.len() != 1 || first == 0 {
            bail!("unsupported_model", "qwen4exp: exactly one PLE layer is supported, got ple_layer_ids {ids:?}");
        }
        let ngram = opt_usize(v, "ngram_size").unwrap_or(3);
        let per = opt_usize(v, "heads_per_ngram").unwrap_or(8);
        let heads = (ngram.max(2) - 1) * per;
        let embed = opt_usize(v, "ple_embed_dim").unwrap_or(base.hidden_size);
        if heads == 0 || !embed.is_multiple_of(heads) {
            bail!("metadata", "qwen4exp: ple_embed_dim {embed} not divisible by {heads} n-gram heads");
        }
        let (sizes, offsets) = head_ranges(opt_usize(v, "ngram_vocab_size_base").unwrap_or(20_000_000) as u64, heads, 0);
        let total: u64 = sizes.iter().sum();
        let div = opt_usize(v, "make_ngram_vocab_size_divisible_by").unwrap_or(128).max(1) as u64;
        let eos = base.eos.first().copied().ok_or_else(|| Error::new("metadata", "qwen4exp: PLE needs eos_token_id"))?;
        let p = PleConfig {
            layer: first - 1,
            ngram,
            heads_per_ngram: per,
            conv_kernel: opt_usize(v, "ple_conv_kernel_size").unwrap_or(4),
            eos,
            multipliers: layer_multipliers(base.vocab_size as u64, ngram, 0, v.get("seed").and_then(Value::as_u64).unwrap_or(1234)),
            offsets,
            vocab_sizes: sizes,
            row_dim: embed / heads,
            rows: total.div_ceil(div) as usize * div as usize,
        };
        p.validate()?;
        Ok(Some(p))
    }

    fn validate(&self) -> Result<()> {
        let b = &self.base;
        if self.hc < 2 || self.hc_lowrank == 0 {
            bail!("metadata", "qwen4exp: hc_count {} / hc_lowrank {}", self.hc, self.hc_lowrank);
        }
        if b.moe.is_none() {
            bail!("metadata", "qwen4exp: num_experts missing (every layer is MoE)");
        }
        if let Some(q) = &self.qsa
            && (q.n_heads == 0 || q.head_dim == 0 || q.ratio < 2 || q.budget == 0 || !q.budget.is_multiple_of(q.ratio) || b.rotary_dim() > q.head_dim)
        {
            bail!("metadata", "qwen4exp: invalid QSA config {q:?}");
        }
        if let Some(p) = &self.ple
            && b.layer_types.get(p.layer) != Some(&LayerType::LinearAttention)
        {
            bail!("metadata", "qwen4exp: PLE layer {} is not a linear-attention layer", p.layer);
        }
        Ok(())
    }

    pub fn hidden_size(&self) -> usize {
        self.base.hidden_size
    }

    pub fn n_layers(&self) -> usize {
        self.base.n_layers
    }

    pub fn vocab_size(&self) -> usize {
        self.base.vocab_size
    }

    pub fn moe(&self) -> Result<&MoeConfig> {
        self.base.moe.as_ref().ok_or_else(|| Error::new("metadata", "qwen4exp: no MoE config"))
    }

    /// HF `text_config`-style JSON (stored in the `.lapid` metadata).
    pub fn to_json(&self) -> Value {
        let mut v = self.base.to_json();
        v["model_type"] = json!("qwen4_exp_text");
        v["hc_count"] = json!(self.hc);
        v["hc_lowrank"] = json!(self.hc_lowrank);
        v["output_gate_type"] = json!(self.output_gate.name());
        if let Some(q) = &self.qsa {
            v["indexer_n_heads"] = json!(q.n_heads);
            v["indexer_kv_heads"] = json!(1);
            v["indexer_head_dim"] = json!(q.head_dim);
            v["indexer_budget"] = json!(q.budget);
            v["indexer_compress_ratio"] = json!(q.ratio);
        }
        if let Some(p) = &self.ple {
            v["ple"] = p.to_json();
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_constants_match_the_released_gguf() {
        // Qwen3.8-Flash-Next GGUF metadata (qwen4exp.ple.*)
        assert_eq!(layer_multipliers(248_320, 3, 0, 1234), vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]);
        let (sizes, offsets) = head_ranges(20_000_000, 16, 0);
        assert_eq!(&sizes[..4], &[20_000_003, 20_000_023, 20_000_033, 20_000_047]);
        assert_eq!(sizes[15], 20_000_171);
        assert_eq!(offsets[15], 300_001_275);
    }

    #[test]
    fn parse_hf_and_roundtrip() {
        let v = json!({"model_type": "qwen4_exp", "text_config": {
            "model_type": "qwen4_exp_text", "vocab_size": 248320, "hidden_size": 256, "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "indexed_attention"],
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 256,
            "linear_num_key_heads": 2, "linear_num_value_heads": 6, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 256, "shared_expert_intermediate_size": 256,
            "hc_count": 4, "hc_lowrank": 32, "output_gate_type": "sigmoid", "eos_token_id": 248044,
            "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128, "indexer_budget": 16, "indexer_compress_ratio": 4,
            "ple_layer_ids": [2], "ngram_size": 3, "heads_per_ngram": 8, "ngram_vocab_size_base": 1000, "ple_embed_dim": 256,
            "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}}});
        let c = Qwen4ExpConfig::from_json(&v).expect("cfg");
        assert_eq!(c.base.layer_types[3], LayerType::FullAttention);
        let p = c.ple.as_ref().expect("ple");
        assert_eq!((p.layer, p.n_heads(), p.row_dim, p.conv_state()), (1, 16, 16, 9));
        assert_eq!(c.qsa.as_ref().map(QsaConfig::dense_limit), Some(19));
        let again = Qwen4ExpConfig::from_json(&json!({"model_type": "qwen4_exp", "text_config": c.to_json()})).expect("again");
        assert_eq!(c, again);
    }
}
