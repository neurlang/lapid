//! A generic ViT / SigLIP-style vision encoder as lazy lapid graphs, configured by
//! [`VisionConfig`] and loaded from llama.cpp `mmproj` tensor names (the de-facto common
//! naming of `clip` GGUF files, used inside `.lapid` files too):
//!
//! | tensor | shape | |
//! |---|---|---|
//! | `v.patch_embd.weight` / `.bias` | [D, C*P*P] / [D] | patch embedding (conv as matmul) |
//! | `v.position_embd.weight` | [S*S, D] | learned positions (bilinearly resampled to the grid) |
//! | `v.blk.{i}.ln1` / `ln2` | [D] | pre-norms (LayerNorm `.weight` + `.bias`, or RMSNorm) |
//! | `v.blk.{i}.attn_qkv` or `attn_q` / `attn_k` / `attn_v`, `attn_out` | | attention |
//! | `v.blk.{i}.ffn_up` / `ffn_gate` / `ffn_down` | | MLP (gated when `ffn_gate` exists) |
//! | `v.post_ln` | [D] | merger norm (per patch, before the spatial merge) |
//! | `mm.0` / `mm.2` | [O', D*m*m] / [O, O'] | merger MLP (GELU) |
//! | `v.deepstack.{layer}.{norm,fc1,fc2}` | | DeepStack mergers (norm after the merge) |
//!
//! Forward for one image: patches [N, C*P*P] (merge-block order) → `patch_embd` (+ bias) →
//! `+ pos` → blocks (`x += attn(ln1 x)`, `x += mlp(ln2 x)`, bidirectional, 2-D RoPE over
//! (row, col) when configured) → merger → [N / m², O]; DeepStack layers emit their own
//! merged features. Taps: `vision.blocks.{i}`, `vision.merged`, `vision.deepstack.{j}`.

use lapid::{DType, Error, Result, Tensor};
use lapid_nn::attention::{Mask, merge_heads, scaled_dot_product_attention};
pub use lapid_nn::vision::{apply_rope_tables, rope_tables};
use lapid_nn::vision::{block_order_positions, grid_rope_tables, interpolate_pos_table};
use lapid_nn::{Activation, LayerNorm, Linear, Loader, RmsNorm};
use serde::{Deserialize, Serialize};

use super::processor::Patches;
use crate::runtime::Taps;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormKind {
    LayerNorm,
    RmsNorm,
}

/// Encoder hyper-parameters (stored as the `"vision"` metadata object).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub depth: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    #[serde(default = "three")]
    pub in_channels: usize,
    pub norm: NormKind,
    pub eps: f32,
    /// MLP activation, HF name (`gelu_pytorch_tanh`, `gelu`, `silu`, ...).
    pub hidden_act: String,
    /// Side of the learned square position grid (0 = none).
    pub position_grid: usize,
    /// 2-D RoPE theta over (row, col) (None = no RoPE).
    pub rope_theta: Option<f32>,
    pub spatial_merge_size: usize,
    /// Merger output width (the text model's hidden size).
    pub out_hidden_size: usize,
    /// Encoder layers whose output feeds a DeepStack merger (in order).
    #[serde(default)]
    pub deepstack_layers: Vec<usize>,
}

fn three() -> usize {
    3
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads.max(1)
    }

    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.patch_size * self.patch_size
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(Error::new("metadata", format!("vision config: {m}")));
        if self.num_heads == 0 || !self.hidden_size.is_multiple_of(self.num_heads) {
            return bad(format!("hidden {} not divisible by {} heads", self.hidden_size, self.num_heads));
        }
        if self.rope_theta.is_some() && !self.head_dim().is_multiple_of(4) {
            return bad(format!("2-D RoPE needs head dim % 4 == 0, got {}", self.head_dim()));
        }
        if Activation::from_name(&self.hidden_act).is_none() {
            return bad(format!("unknown activation {:?}", self.hidden_act));
        }
        if let Some(&l) = self.deepstack_layers.iter().find(|&&l| l >= self.depth) {
            return bad(format!("deepstack layer {l} >= depth {}", self.depth));
        }
        if self.spatial_merge_size == 0 {
            return bad("spatial_merge_size 0".into());
        }
        Ok(())
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    pub fn from_json(v: &serde_json::Value) -> Result<VisionConfig> {
        let c: VisionConfig = serde_json::from_value(v.clone()).map_err(|e| Error::new("metadata", format!("vision config: {e}")))?;
        c.validate()?;
        Ok(c)
    }

    /// Every tensor the encoder reads (`gated` MLP and `split_qkv` attention variants).
    pub fn expected_tensors(&self, gated: bool, split_qkv: bool) -> Vec<String> {
        let mut v = vec!["v.patch_embd.weight".to_string(), "v.patch_embd.bias".to_string()];
        if self.position_grid > 0 {
            v.push("v.position_embd.weight".into());
        }
        let lnb = self.norm == NormKind::LayerNorm;
        let push_norm = |v: &mut Vec<String>, p: &str| {
            v.push(format!("{p}.weight"));
            if lnb {
                v.push(format!("{p}.bias"));
            }
        };
        for i in 0..self.depth {
            let b = format!("v.blk.{i}");
            push_norm(&mut v, &format!("{b}.ln1"));
            push_norm(&mut v, &format!("{b}.ln2"));
            let mut lin: Vec<&str> = if split_qkv { vec!["attn_q", "attn_k", "attn_v"] } else { vec!["attn_qkv"] };
            lin.extend(["attn_out", "ffn_up", "ffn_down"]);
            if gated {
                lin.push("ffn_gate");
            }
            for l in lin {
                v.push(format!("{b}.{l}.weight"));
                v.push(format!("{b}.{l}.bias"));
            }
        }
        push_norm(&mut v, "v.post_ln");
        for m in ["mm.0", "mm.2"] {
            v.push(format!("{m}.weight"));
            v.push(format!("{m}.bias"));
        }
        for l in &self.deepstack_layers {
            push_norm(&mut v, &format!("v.deepstack.{l}.norm"));
            for f in ["fc1", "fc2"] {
                v.push(format!("v.deepstack.{l}.{f}.weight"));
                v.push(format!("v.deepstack.{l}.{f}.bias"));
            }
        }
        v
    }
}

enum Norm {
    Layer(LayerNorm),
    Rms(RmsNorm),
}

impl Norm {
    fn load(l: &Loader<'_>, p: &str, kind: NormKind, eps: f32, d: usize) -> Result<Norm> {
        let n = match kind {
            NormKind::LayerNorm => Norm::Layer(l.layer_norm(p, eps)?),
            NormKind::RmsNorm => Norm::Rms(l.rms_norm(p, eps)?),
        };
        let w = match &n {
            Norm::Layer(x) => &x.weight,
            Norm::Rms(x) => &x.weight,
        };
        l.expect_shape(w, &format!("{p}.weight"), &[d])?;
        Ok(n)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Norm::Layer(n) => n.forward(x),
            Norm::Rms(n) => n.forward(x),
        }
    }
}

fn linear(l: &Loader<'_>, p: &str, out: usize, inp: usize) -> Result<Linear> {
    let mut lin = l.linear(p, l.contains(&format!("{p}.bias")))?;
    l.expect_shape(&lin.weight, &format!("{p}.weight"), &[out, inp])?;
    // Vision activations carry large outliers: f16 weights on GPUs (where matmuls with f16
    // weights round activations to f16) are widened to f32 once at load.
    if l.device.name() != "cpu" && lin.weight.dtype() == DType::F16 {
        let w = l.device.eval(&[&lin.weight.cast(DType::F32)?])?;
        lin.weight = w.into_iter().next().ok_or_else(|| Error::new("backend", "cast"))?;
    }
    Ok(lin)
}

enum Qkv {
    Fused(Linear),
    Split(Linear, Linear, Linear),
}

struct Block {
    ln1: Norm,
    ln2: Norm,
    qkv: Qkv,
    out: Linear,
    up: Linear,
    gate: Option<Linear>,
    down: Linear,
}

/// Spatial merge + 2-layer GELU MLP (`norm_after_merge`: DeepStack layout).
struct Merger {
    norm: Norm,
    norm_after_merge: bool,
    fc1: Linear,
    fc2: Linear,
}

impl Merger {
    fn forward(&self, x: &Tensor, merge: usize) -> Result<Tensor> {
        let (n, d) = (x.shape()[0], x.shape()[1]);
        let k = merge * merge;
        let h = if self.norm_after_merge {
            self.norm.forward(&x.reshape(&[n / k, d * k])?)?
        } else {
            self.norm.forward(x)?.reshape(&[n / k, d * k])?
        };
        self.fc2.forward(&self.fc1.forward(&h)?.gelu()?)
    }
}

pub struct VisionEncoder {
    pub cfg: VisionConfig,
    patch: Linear,
    pos: Option<Tensor>,
    blocks: Vec<Block>,
    merger: Merger,
    deepstack: Vec<Merger>,
    act: Activation,
}

impl VisionEncoder {
    pub fn load(l: &Loader<'_>, cfg: &VisionConfig) -> Result<VisionEncoder> {
        cfg.validate()?;
        let (d, f, eps, kind) = (cfg.hidden_size, cfg.intermediate_size, cfg.eps, cfg.norm);
        let act = Activation::from_name(&cfg.hidden_act).ok_or_else(|| Error::new("metadata", "vision: unknown activation"))?;
        let patch = linear(l, "v.patch_embd", d, cfg.patch_dim())?;
        let pos = if cfg.position_grid > 0 {
            let t = l.tensor("v.position_embd.weight")?;
            l.expect_shape(&t, "v.position_embd.weight", &[cfg.position_grid * cfg.position_grid, d])?;
            Some(t)
        } else {
            None
        };
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let b = format!("v.blk.{i}");
            let qkv = if l.contains(&format!("{b}.attn_qkv.weight")) {
                Qkv::Fused(linear(l, &format!("{b}.attn_qkv"), 3 * d, d)?)
            } else {
                Qkv::Split(linear(l, &format!("{b}.attn_q"), d, d)?, linear(l, &format!("{b}.attn_k"), d, d)?, linear(l, &format!("{b}.attn_v"), d, d)?)
            };
            let gate = if l.contains(&format!("{b}.ffn_gate.weight")) { Some(linear(l, &format!("{b}.ffn_gate"), f, d)?) } else { None };
            blocks.push(Block {
                ln1: Norm::load(l, &format!("{b}.ln1"), kind, eps, d)?,
                ln2: Norm::load(l, &format!("{b}.ln2"), kind, eps, d)?,
                qkv,
                out: linear(l, &format!("{b}.attn_out"), d, d)?,
                up: linear(l, &format!("{b}.ffn_up"), f, d)?,
                gate,
                down: linear(l, &format!("{b}.ffn_down"), d, f)?,
            });
        }
        let k = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let merger_of = |norm: &str, fc1: &str, fc2: &str, after: bool| -> Result<Merger> {
            let nd = if after { d * k } else { d };
            Ok(Merger {
                norm: Norm::load(l, norm, kind, eps, nd)?,
                norm_after_merge: after,
                fc1: linear(l, fc1, d * k, d * k)?,
                fc2: linear(l, fc2, cfg.out_hidden_size, d * k)?,
            })
        };
        let merger = merger_of("v.post_ln", "mm.0", "mm.2", false)?;
        let mut deepstack = Vec::new();
        for li in &cfg.deepstack_layers {
            let p = format!("v.deepstack.{li}");
            deepstack.push(merger_of(&format!("{p}.norm"), &format!("{p}.fc1"), &format!("{p}.fc2"), true)?);
        }
        Ok(VisionEncoder { cfg: cfg.clone(), patch, pos, blocks, merger, deepstack, act })
    }

    /// 2-D RoPE tables [N, Dh/2] over (row, col) in merge-block order.
    fn rope(&self, gh: usize, gw: usize) -> Result<Option<(Tensor, Tensor)>> {
        let Some(theta) = self.cfg.rope_theta else { return Ok(None) };
        let pos = block_order_positions(gh, gw, self.cfg.spatial_merge_size);
        Ok(Some(grid_rope_tables(&pos, self.cfg.head_dim(), theta)?))
    }

    fn attention(&self, b: &Block, x: &Tensor, rope: Option<&(Tensor, Tensor)>) -> Result<Tensor> {
        let n = x.shape()[0];
        let (h, dh, d) = (self.cfg.num_heads, self.cfg.head_dim(), self.cfg.hidden_size);
        let heads = |t: Tensor| -> Result<Tensor> { t.reshape(&[1, n, h, dh])?.permute(&[0, 2, 1, 3]) };
        let (q, k, v) = match &b.qkv {
            Qkv::Fused(l) => {
                let y = l.forward(x)?;
                (heads(y.slice(1, 0, d)?)?, heads(y.slice(1, d, 2 * d)?)?, heads(y.slice(1, 2 * d, 3 * d)?)?)
            }
            Qkv::Split(lq, lk, lv) => (heads(lq.forward(x)?)?, heads(lk.forward(x)?)?, heads(lv.forward(x)?)?),
        };
        let (q, k) = match rope {
            Some((c, s)) => (apply_rope_tables(&q, c, s)?, apply_rope_tables(&k, c, s)?),
            None => (q, k),
        };
        let a = scaled_dot_product_attention(&q, &k, &v, 1.0 / (dh as f32).sqrt(), &Mask::None)?;
        b.out.forward(&merge_heads(&a)?.reshape(&[n, d])?)
    }

    /// Graph of one image: the merged tokens [N / m^2, O], then each DeepStack feature.
    pub fn build(&self, p: &Patches, taps: &mut Taps) -> Result<Vec<Tensor>> {
        let n = p.len();
        let m = self.cfg.spatial_merge_size;
        if p.dim != self.cfg.patch_dim() || p.data.len() != n * p.dim {
            return Err(Error::new("shape_mismatch", format!("vision: patches of {} values, model takes {}", p.dim, self.cfg.patch_dim())));
        }
        if n == 0 || !p.grid_h.is_multiple_of(m) || !p.grid_w.is_multiple_of(m) {
            return Err(Error::new("shape_mismatch", format!("vision: grid {}x{} not a multiple of merge {m}", p.grid_h, p.grid_w)));
        }
        let mut x = self.patch.forward(&Tensor::from_f32(&p.data, &[n, p.dim])?)?.cast(DType::F32)?;
        if let Some(table) = &self.pos {
            x = x.add(&interpolate_pos_table(table, p.grid_h, p.grid_w, self.cfg.position_grid, m)?)?;
        }
        let rope = self.rope(p.grid_h, p.grid_w)?;
        let mut deep = Vec::new();
        for (i, b) in self.blocks.iter().enumerate() {
            x = x.add(&self.attention(b, &b.ln1.forward(&x)?, rope.as_ref())?)?;
            let h = b.ln2.forward(&x)?;
            let up = match &b.gate {
                Some(g) => self.act.apply(&g.forward(&h)?)?.mul(&b.up.forward(&h)?)?,
                None => self.act.apply(&b.up.forward(&h)?)?,
            };
            x = x.add(&b.down.forward(&up)?)?;
            taps.tap(format_args!("vision.blocks.{i}"), &x);
            if let Some(j) = self.cfg.deepstack_layers.iter().position(|&l| l == i) {
                deep.push(self.deepstack[j].forward(&x, m)?);
            }
        }
        let merged = self.merger.forward(&x, m)?;
        taps.tap("vision.merged", &merged);
        for (j, t) in deep.iter().enumerate() {
            taps.tap(format_args!("vision.deepstack.{j}"), t);
        }
        let mut out = vec![merged];
        out.extend(deep);
        Ok(out)
    }
}
