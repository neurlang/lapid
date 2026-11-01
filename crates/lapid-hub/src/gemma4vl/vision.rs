//! Gemma 4 vision tower (HF `Gemma4VisionModel` + `Gemma4MultimodalEmbedder`) as lazy lapid
//! graphs.
//!
//! Patch embedding: `input_proj(2 * (pixels - 0.5))` over flattened 16x16x3 patches plus the
//! 2-D position table (`table[0][x] + table[1][y]`). Encoder layers (sandwich RMSNorms
//! `x * w`): `x += post_attn_norm(attn(input_norm(x)))`, `x += post_ff_norm(down(gelu_tanh(gate(h))
//! * up(h)))`; attention with q/k RMSNorm, weight-free v norm, scale 1.0, bidirectional, axial
//! 2-D RoPE (first half of each head rotated by the patch column, second half by the row,
//! half-split pairs inside each half, theta 100). Linears are "clippable": input and output
//! clamped to stored scalars. Pooler: average of each k x k patch block (row-major blocks),
//! times `sqrt(hidden)`, optional standardize `(x - bias) * scale`; then the embedder:
//! weight-free RMSNorm and a projection to the text width.

use lapid::{BinaryOp, DType, Error, Result, Tensor, UnaryOp};
use lapid_nn::attention::{merge_heads, scaled_dot_product_attention};
use lapid_nn::{Embedding, Linear, Loader, Mask, RmsNorm, Rope, ops};

use super::config::VisionConfig;
use super::preprocess::PreparedImage;

pub(crate) const TOWER: &str = "model.vision_tower";
pub(crate) const EMBED_PROJ: &str = "model.embed_vision.embedding_projection";

/// Gemma4ClippableLinear: clamp(lin(clamp(x, in_min, in_max)), out_min, out_max).
struct Clip {
    lin: Linear,
    input: (f32, f32),
    output: (f32, f32),
}

fn clamp(x: &Tensor, (lo, hi): (f32, f32)) -> Result<Tensor> {
    let mut x = x.clone();
    if lo.is_finite() {
        x = x.binary_scalar(BinaryOp::Max, f64::from(lo))?;
    }
    if hi.is_finite() {
        x = x.binary_scalar(BinaryOp::Min, f64::from(hi))?;
    }
    Ok(x)
}

impl Clip {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        clamp(&self.lin.forward(&clamp(x, self.input)?)?, self.output)
    }
}

/// Host value of a scalar weight, `default` when absent.
fn scalar(l: &Loader<'_>, name: &str, default: f32) -> Result<f32> {
    if !l.contains(name) {
        return Ok(default);
    }
    let v = l.device.to_vec_f32(&l.tensor_f32(name)?)?;
    match v.as_slice() {
        [x] => Ok(*x),
        _ => Err(Error::new("shape_mismatch", format!("{name}: expected one value, got {}", v.len()))),
    }
}

fn clip(l: &Loader<'_>, prefix: &str, out: usize, inp: usize) -> Result<Clip> {
    let name = format!("{prefix}.linear.weight");
    let lin = Linear::new(l.tensor(&name)?, None).with_act_quant(l.act_quant(&name)?);
    l.expect_shape(&lin.weight, &name, &[out, inp])?;
    let inf = f32::INFINITY;
    Ok(Clip {
        lin,
        input: (scalar(l, &format!("{prefix}.input_min"), -inf)?, scalar(l, &format!("{prefix}.input_max"), inf)?),
        output: (scalar(l, &format!("{prefix}.output_min"), -inf)?, scalar(l, &format!("{prefix}.output_max"), inf)?),
    })
}

fn norm(l: &Loader<'_>, name: &str, d: usize, eps: f32) -> Result<RmsNorm> {
    let n = l.rms_norm(name, eps)?;
    l.expect_shape(&n.weight, &format!("{name}.weight"), &[d])?;
    Ok(n)
}

/// Weight-free RMSNorm over the last axis, in f32.
pub(crate) fn rms_plain(x: &Tensor, eps: f32) -> Result<Tensor> {
    let xf = x.cast(DType::F32)?;
    let inv = xf.mul(&xf)?.mean(-1, true)?.add_scalar(f64::from(eps))?.unary(UnaryOp::Rsqrt)?;
    xf.mul(&inv)
}

struct Layer {
    q: Clip,
    k: Clip,
    v: Clip,
    o: Clip,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    gate: Clip,
    up: Clip,
    down: Clip,
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    pre_ff_norm: RmsNorm,
    post_ff_norm: RmsNorm,
}

pub(crate) struct VisionTower {
    pub cfg: VisionConfig,
    input_proj: Linear,
    /// Position table as [2 * P, D] (x rows, then y rows).
    pos_table: Embedding,
    layers: Vec<Layer>,
    std: Option<(Tensor, Tensor)>,
    proj: Linear,
    rope: Rope,
}

impl VisionTower {
    pub fn load(l: &Loader<'_>, cfg: &VisionConfig, text_hidden: usize) -> Result<VisionTower> {
        let (d, eps, hd) = (cfg.hidden_size, cfg.eps, cfg.head_dim);
        let pp = 3 * cfg.patch_size * cfg.patch_size;
        let ip = format!("{TOWER}.patch_embedder.input_proj.weight");
        let input_proj = Linear::new(l.tensor(&ip)?, None).with_act_quant(l.act_quant(&ip)?);
        l.expect_shape(&input_proj.weight, &ip, &[d, pp])?;
        let tn = format!("{TOWER}.patch_embedder.position_embedding_table");
        let table = l.tensor(&tn)?;
        let ps = cfg.position_embedding_size;
        if table.numel() != 2 * ps * d {
            return Err(Error::new("shape_mismatch", format!("{tn}: expected [2, {ps}, {d}], file has {:?}", table.shape())));
        }
        let table = if table.dtype().is_quant() { table } else { table.reshape(&[2 * ps, d])? };
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("{TOWER}.encoder.layers.{i}");
            let a = format!("{p}.self_attn");
            let m = format!("{p}.mlp");
            let ff = cfg.intermediate_size;
            layers.push(Layer {
                q: clip(l, &format!("{a}.q_proj"), cfg.n_head * hd, d)?,
                k: clip(l, &format!("{a}.k_proj"), cfg.n_kv_head * hd, d)?,
                v: clip(l, &format!("{a}.v_proj"), cfg.n_kv_head * hd, d)?,
                o: clip(l, &format!("{a}.o_proj"), d, cfg.n_head * hd)?,
                q_norm: norm(l, &format!("{a}.q_norm"), hd, eps)?,
                k_norm: norm(l, &format!("{a}.k_norm"), hd, eps)?,
                gate: clip(l, &format!("{m}.gate_proj"), ff, d)?,
                up: clip(l, &format!("{m}.up_proj"), ff, d)?,
                down: clip(l, &format!("{m}.down_proj"), d, ff)?,
                input_norm: norm(l, &format!("{p}.input_layernorm"), d, eps)?,
                post_attn_norm: norm(l, &format!("{p}.post_attention_layernorm"), d, eps)?,
                pre_ff_norm: norm(l, &format!("{p}.pre_feedforward_layernorm"), d, eps)?,
                post_ff_norm: norm(l, &format!("{p}.post_feedforward_layernorm"), d, eps)?,
            });
        }
        let std = if cfg.standardize {
            let (b, s) = (l.tensor_f32(&format!("{TOWER}.std_bias"))?, l.tensor_f32(&format!("{TOWER}.std_scale"))?);
            l.expect_shape(&b, "std_bias", &[d])?;
            l.expect_shape(&s, "std_scale", &[d])?;
            Some((b, s))
        } else {
            None
        };
        let pn = format!("{EMBED_PROJ}.weight");
        let proj = Linear::new(l.tensor(&pn)?, None).with_act_quant(l.act_quant(&pn)?);
        l.expect_shape(&proj.weight, &pn, &[text_hidden, d])?;
        Ok(VisionTower {
            cfg: cfg.clone(),
            input_proj,
            pos_table: Embedding::new(table),
            layers,
            std,
            proj,
            rope: Rope::new(hd / 2, cfg.rope_theta),
        })
    }

    /// Axial RoPE on t [1, H, N, hd]: first half by x (column), second half by y (row).
    fn rope2d(&self, t: &Tensor, xs: &[usize], ys: &[usize]) -> Result<Tensor> {
        let hd = self.cfg.head_dim;
        let a = self.rope.apply_positions(&t.slice(3, 0, hd / 2)?, xs)?;
        let b = self.rope.apply_positions(&t.slice(3, hd / 2, hd)?, ys)?;
        Tensor::cat(&[&a, &b], 3)
    }

    fn layer(&self, ly: &Layer, x: &Tensor, xs: &[usize], ys: &[usize]) -> Result<Tensor> {
        let (n, hd, eps) = (x.shape()[1], self.cfg.head_dim, self.cfg.eps);
        let h = ly.input_norm.forward(x)?;
        let heads = |t: Tensor, nh: usize| t.reshape(&[1, n, nh, hd]);
        let q = ly.q_norm.forward(&heads(ly.q.forward(&h)?, self.cfg.n_head)?)?.permute(&[0, 2, 1, 3])?;
        let k = ly.k_norm.forward(&heads(ly.k.forward(&h)?, self.cfg.n_kv_head)?)?.permute(&[0, 2, 1, 3])?;
        let v = rms_plain(&heads(ly.v.forward(&h)?, self.cfg.n_kv_head)?, eps)?.permute(&[0, 2, 1, 3])?;
        let (q, k) = (self.rope2d(&q, xs, ys)?, self.rope2d(&k, xs, ys)?);
        let a = scaled_dot_product_attention(&q, &k, &v, 1.0, &Mask::None)?;
        let a = ly.o.forward(&merge_heads(&a)?)?;
        let x = x.add(&ly.post_attn_norm.forward(&a)?)?;
        let h = ly.pre_ff_norm.forward(&x)?;
        let f = ly.down.forward(&ops::gelu_tanh(&ly.gate.forward(&h)?)?.mul(&ly.up.forward(&h)?)?)?;
        x.add(&ly.post_ff_norm.forward(&f)?)
    }

    /// Host [M, N] block-average matrix of the pooler (blocks row-major, 1/k^2 weights).
    fn pool_matrix(&self, img: &PreparedImage) -> Result<Tensor> {
        let k = self.cfg.pooling_kernel_size;
        let (_, cols) = img.grid;
        let (n, m) = (img.n_patches(), img.n_soft);
        let bw = cols / k;
        let mut w = vec![0.0f32; m * n];
        let inv = 1.0 / (k * k) as f32;
        for (j, p) in img.positions.iter().enumerate() {
            let block = p[0] / k + bw * (p[1] / k);
            if block >= m {
                return Err(Error::new("shape_mismatch", format!("gemma4 vision: patch {j} pools to block {block} of {m}")));
            }
            w[block * n + j] = inv;
        }
        Tensor::from_f32(&w, &[m, n])
    }

    /// Soft-token embeddings [M, D_text] (f32) of one image. With `trace`, intermediates are
    /// collected as ("vision.embed" | "vision.layers.{i}" | "vision.pooled" | "vision.projected", t).
    pub fn build(&self, img: &PreparedImage, mut trace: Option<&mut Vec<(String, Tensor)>>) -> Result<Tensor> {
        let n = img.n_patches();
        let pp = 3 * self.cfg.patch_size * self.cfg.patch_size;
        if img.patch_size != self.cfg.patch_size || img.patches.len() != n * pp {
            return Err(Error::new("shape_mismatch", format!("gemma4 vision: image patches of {} px, tower {}", img.patch_size, self.cfg.patch_size)));
        }
        let mut push = |name: String, t: &Tensor| {
            if let Some(tr) = trace.as_deref_mut() {
                tr.push((name, t.clone()));
            }
        };
        let ps = self.cfg.position_embedding_size;
        let xs: Vec<usize> = img.positions.iter().map(|p| p[0]).collect();
        let ys: Vec<usize> = img.positions.iter().map(|p| p[1]).collect();
        if xs.iter().chain(&ys).any(|&v| v >= ps) {
            return Err(Error::new("invalid_argument", format!("gemma4 vision: patch grid {:?} exceeds the position table ({ps})", img.grid)));
        }
        let pix = Tensor::from_f32(&img.patches, &[1, n, pp])?.mul_scalar(2.0)?.add_scalar(-1.0)?;
        let mut x = self.input_proj.forward(&pix)?.cast(DType::F32)?;
        let ids: Vec<u32> = xs.iter().map(|&v| v as u32).chain(ys.iter().map(|&v| (ps + v) as u32)).collect();
        let pos = self.pos_table.lookup(&ids)?.cast(DType::F32)?;
        x = x.add(&pos.slice(1, 0, n)?.add(&pos.slice(1, n, 2 * n)?)?)?;
        push("vision.embed".into(), &x);
        for (i, ly) in self.layers.iter().enumerate() {
            x = self.layer(ly, &x, &xs, &ys)?;
            push(format!("vision.layers.{i}"), &x);
        }
        let h = x.reshape(&[n, self.cfg.hidden_size])?;
        let mut pooled = self.pool_matrix(img)?.matmul(&h)?.mul_scalar((self.cfg.hidden_size as f64).sqrt())?;
        if let Some((b, s)) = &self.std {
            pooled = pooled.sub(b)?.mul(s)?;
        }
        push("vision.pooled".into(), &pooled);
        let out = self.proj.forward(&rms_plain(&pooled, self.cfg.eps)?)?.cast(DType::F32)?;
        push("vision.projected".into(), &out);
        Ok(out)
    }
}
