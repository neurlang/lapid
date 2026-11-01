//! Gemma 4 vision tower (HF `Gemma4VisionModel` + `Gemma4MultimodalEmbedder`) and image
//! preprocessing (`Gemma4ImageProcessor`).
//!
//! Preprocessing: aspect-preserving resize to at most `max_soft_tokens * k^2` patches with
//! sides multiple of `k * patch`, rescale to [0, 1], row-major patches flattened
//! `(row, col, channel)`, positions (x, y). Tower: `input_proj(2 * pixels - 1)` plus the
//! position table (`table[0][x] + table[1][y]`); layers with sandwich RMSNorms, q/k norm,
//! weight-free v norm, scale 1, bidirectional attention with axial 2-D RoPE (first half of a
//! head by column, second half by row, theta 100), GeGLU MLP; "clippable" linears clamp
//! input and output to stored scalars. Pooler: mean of each k x k block * sqrt(D), optional
//! standardize `(x - bias) * scale`; embedder: weight-free RMSNorm and a projection to the
//! text width.

use lapid::{DType, Result, Tensor, bail};
use lapid_nn::attention::{merge_heads, scaled_dot_product_attention};
use lapid_nn::{Embedding, Linear, Loader, Mask, RmsNorm, Rope, ops};

use super::config::{ImageProcessorConfig, VisionConfig};
use super::model::{clamp, rms_plain, scalar};
use crate::runtime::Taps;
use crate::vision::Image;

pub const TOWER: &str = "model.vision_tower";
pub const EMBED_PROJ: &str = "model.embed_vision.embedding_projection.weight";

/// One image ready for the tower.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedImage {
    /// [n_patches, 3 * p * p] (HF `pixel_values`, real patches only).
    pub patches: Vec<f32>,
    /// (x, y) = (column, row) of every patch.
    pub positions: Vec<[usize; 2]>,
    /// Patch grid (rows, columns).
    pub grid: (usize, usize),
    /// Soft tokens after pooling.
    pub n_soft: usize,
}

/// HF `get_aspect_ratio_preserving_size` (height, width).
pub fn target_size(height: usize, width: usize, patch: usize, max_patches: usize, k: usize) -> Result<(usize, usize)> {
    let target_px = (max_patches * patch * patch) as f64;
    let factor = (target_px / (height * width) as f64).sqrt();
    let side = k * patch;
    let mut th = ((factor * height as f64) / side as f64).floor() as usize * side;
    let mut tw = ((factor * width as f64) / side as f64).floor() as usize * side;
    let max_side = (max_patches / (k * k)) * side;
    match (th, tw) {
        (0, 0) => bail!("invalid_argument", "gemma4 vision: image {height}x{width} resizes to 0x0"),
        (0, _) => (th, tw) = (side, ((width as f64 / height as f64).floor() as usize * side).min(max_side)),
        (_, 0) => (th, tw) = (((height as f64 / width as f64).floor() as usize * side).min(max_side), side),
        _ => {}
    }
    if (th * tw) as f64 > target_px {
        bail!("invalid_argument", "gemma4 vision: {height}x{width} -> {th}x{tw} exceeds {max_patches} patches");
    }
    Ok((th, tw))
}

/// Resize, rescale and patchify `img`.
pub fn prepare(img: &Image, pc: &ImageProcessorConfig) -> Result<PreparedImage> {
    let (p, k) = (pc.patch_size, pc.pooling_kernel_size);
    let img = if pc.do_resize {
        let (th, tw) = target_size(img.height, img.width, p, pc.max_soft_tokens * k * k, k)?;
        img.resize(tw, th, pc.resample)?
    } else {
        img.clone()
    };
    let (h, w) = (img.height, img.width);
    if p == 0 || k == 0 || h % (p * k) != 0 || w % (p * k) != 0 {
        bail!("invalid_argument", "gemma4 vision: {h}x{w} is not a multiple of {}", p * k);
    }
    let chw = img.to_chw(pc.rescale_factor, pc.mean, pc.std);
    let (rows, cols, pp) = (h / p, w / p, 3 * p * p);
    let mut patches = vec![0.0f32; rows * cols * pp];
    let mut positions = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let base = (r * cols + c) * pp;
            for (iy, ix, ch) in (0..p).flat_map(|y| (0..p).flat_map(move |x| (0..3).map(move |ch| (y, x, ch)))) {
                patches[base + (iy * p + ix) * 3 + ch] = chw[ch * h * w + (r * p + iy) * w + c * p + ix];
            }
            positions.push([c, r]);
        }
    }
    Ok(PreparedImage { patches, positions, grid: (rows, cols), n_soft: rows * cols / (k * k) })
}

/// Gemma4ClippableLinear: clamp(lin(clamp(x, input)), output).
struct Clip {
    lin: Linear,
    input: (f32, f32),
    output: (f32, f32),
}

impl Clip {
    fn load(l: &Loader<'_>, prefix: &str, out: usize, inp: usize) -> Result<Clip> {
        let s = |n: &str, d: f32| scalar(l, &format!("{prefix}.{n}"), d);
        let inf = f32::INFINITY;
        Ok(Clip {
            lin: l.linear_checked(&format!("{prefix}.linear"), out, inp, false)?,
            input: (s("input_min", -inf)?, s("input_max", inf)?),
            output: (s("output_min", -inf)?, s("output_max", inf)?),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        clamp(&self.lin.forward(&clamp(x, self.input)?)?, self.output)
    }
}

struct Layer {
    /// q, k, v, o, gate, up, down.
    lin: [Clip; 7],
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    /// input, post attention, pre feed-forward, post feed-forward.
    norms: [RmsNorm; 4],
}

pub struct VisionTower {
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
        let (d, eps, hd, ff) = (cfg.hidden_size, cfg.eps, cfg.head_dim, cfg.intermediate_size);
        let tn = format!("{TOWER}.patch_embedder.position_embedding_table");
        let ps = cfg.position_embedding_size;
        let table = l.tensor_checked(&tn, &[2, ps, d])?;
        let table = if table.dtype().is_quant() { table } else { table.reshape(&[2 * ps, d])? };
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("{TOWER}.encoder.layers.{i}");
            let (qd, kd) = (cfg.n_head * hd, cfg.n_kv_head * hd);
            let c = |n: &str, o: usize, inp: usize| Clip::load(l, &format!("{p}.{n}"), o, inp);
            let n = |s: &str, dim: usize| l.rms_norm_checked(&format!("{p}.{s}"), dim, eps);
            layers.push(Layer {
                lin: [
                    c("self_attn.q_proj", qd, d)?,
                    c("self_attn.k_proj", kd, d)?,
                    c("self_attn.v_proj", kd, d)?,
                    c("self_attn.o_proj", d, qd)?,
                    c("mlp.gate_proj", ff, d)?,
                    c("mlp.up_proj", ff, d)?,
                    c("mlp.down_proj", d, ff)?,
                ],
                q_norm: n("self_attn.q_norm", hd)?,
                k_norm: n("self_attn.k_norm", hd)?,
                norms: [n("input_layernorm", d)?, n("post_attention_layernorm", d)?, n("pre_feedforward_layernorm", d)?, n("post_feedforward_layernorm", d)?],
            });
        }
        let std = if cfg.standardize {
            Some((l.tensor_f32(&format!("{TOWER}.std_bias"))?, l.tensor_f32(&format!("{TOWER}.std_scale"))?))
        } else {
            None
        };
        Ok(VisionTower {
            input_proj: l.linear_checked(&format!("{TOWER}.patch_embedder.input_proj"), d, 3 * cfg.patch_size * cfg.patch_size, false)?,
            pos_table: Embedding::new(table),
            layers,
            std,
            proj: Linear::new(l.tensor_checked(EMBED_PROJ, &[text_hidden, d])?, None).with_act_quant(l.act_quant(EMBED_PROJ)?),
            rope: Rope::new(hd / 2, cfg.rope_theta),
            cfg: cfg.clone(),
        })
    }

    /// Axial RoPE on t [1, H, N, hd]: first half by x (column), second half by y (row).
    fn rope2d(&self, t: &Tensor, xs: &[usize], ys: &[usize]) -> Result<Tensor> {
        let hd = self.cfg.head_dim;
        let a = self.rope.apply_positions(&t.slice(3, 0, hd / 2)?, xs)?;
        Tensor::cat(&[&a, &self.rope.apply_positions(&t.slice(3, hd / 2, hd)?, ys)?], 3)
    }

    fn layer(&self, ly: &Layer, x: &Tensor, xs: &[usize], ys: &[usize]) -> Result<Tensor> {
        let (n, hd, c) = (x.shape()[1], self.cfg.head_dim, &self.cfg);
        let [q, k, v, o, gate, up, down] = &ly.lin;
        let h = ly.norms[0].forward(x)?;
        let heads = |t: Tensor, nh: usize| t.reshape(&[1, n, nh, hd]);
        let q = self.rope2d(&ly.q_norm.forward(&heads(q.forward(&h)?, c.n_head)?)?.permute(&[0, 2, 1, 3])?, xs, ys)?;
        let k = self.rope2d(&ly.k_norm.forward(&heads(k.forward(&h)?, c.n_kv_head)?)?.permute(&[0, 2, 1, 3])?, xs, ys)?;
        let v = rms_plain(&heads(v.forward(&h)?, c.n_kv_head)?, c.eps)?.permute(&[0, 2, 1, 3])?;
        let a = o.forward(&merge_heads(&scaled_dot_product_attention(&q, &k, &v, 1.0, &Mask::None)?)?)?;
        let x = x.add(&ly.norms[1].forward(&a)?)?;
        let h = ly.norms[2].forward(&x)?;
        let f = down.forward(&ops::gelu_tanh(&gate.forward(&h)?)?.mul(&up.forward(&h)?)?)?;
        x.add(&ly.norms[3].forward(&f)?)
    }

    /// Soft-token embeddings [M, D_text] (f32). Taps: `vision.embed`, `vision.layers.{i}`,
    /// `vision.pooled`, `vision.projected`.
    pub fn build(&self, img: &PreparedImage, taps: &mut Taps) -> Result<Tensor> {
        let (n, c) = (img.positions.len(), &self.cfg);
        let (pp, ps, k) = (3 * c.patch_size * c.patch_size, c.position_embedding_size, c.pooling_kernel_size);
        if img.patches.len() != n * pp || img.grid.1 % k != 0 {
            bail!("shape_mismatch", "gemma4 vision: {} values for {n} patches of {}x{}", img.patches.len(), c.patch_size, c.patch_size);
        }
        let xs: Vec<usize> = img.positions.iter().map(|p| p[0]).collect();
        let ys: Vec<usize> = img.positions.iter().map(|p| p[1]).collect();
        if xs.iter().chain(&ys).any(|&v| v >= ps) {
            bail!("invalid_argument", "gemma4 vision: patch grid {:?} exceeds the position table ({ps})", img.grid);
        }
        let pix = Tensor::from_f32(&img.patches, &[1, n, pp])?.mul_scalar(2.0)?.add_scalar(-1.0)?;
        let ids: Vec<u32> = xs.iter().map(|&v| v as u32).chain(ys.iter().map(|&v| (ps + v) as u32)).collect();
        let pos = self.pos_table.lookup(&ids)?.cast(DType::F32)?;
        let mut x = self.input_proj.forward(&pix)?.cast(DType::F32)?.add(&pos.slice(1, 0, n)?.add(&pos.slice(1, n, 2 * n)?)?)?;
        taps.tap("vision.embed", &x);
        for (i, ly) in self.layers.iter().enumerate() {
            x = self.layer(ly, &x, &xs, &ys)?;
            taps.tap(format_args!("vision.layers.{i}"), &x);
        }
        // Block-average pooling as a [M, N] matmul (blocks row-major, weights 1/k^2).
        let (m, bw) = (img.n_soft, img.grid.1 / k);
        let mut w = vec![0.0f32; m * n];
        for (j, p) in img.positions.iter().enumerate() {
            let block = p[0] / k + bw * (p[1] / k);
            if block >= m {
                bail!("shape_mismatch", "gemma4 vision: patch {j} pools to block {block} of {m}");
            }
            w[block * n + j] = 1.0 / (k * k) as f32;
        }
        let h = x.reshape(&[n, c.hidden_size])?;
        let mut pooled = Tensor::from_f32(&w, &[m, n])?.matmul(&h)?.mul_scalar((c.hidden_size as f64).sqrt())?;
        if let Some((b, s)) = &self.std {
            pooled = pooled.sub(b)?.mul(s)?;
        }
        taps.tap("vision.pooled", &pooled);
        let out = self.proj.forward(&rms_plain(&pooled, c.eps)?)?.cast(DType::F32)?;
        taps.tap("vision.projected", &out);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_and_patches() {
        // 640x488 (w x h) at 280 soft tokens -> 672 x 912 (HF).
        assert_eq!(target_size(488, 640, 16, 2520, 3).expect("size"), (672, 912));
        assert_eq!(target_size(110, 150, 16, 630, 3).expect("size"), (336, 432));
        let rgb: Vec<u8> = (0..96 * 48 * 3).map(|i| (i % 251) as u8).collect();
        let pc = ImageProcessorConfig { do_resize: false, ..ImageProcessorConfig::from_json(None) };
        let p = prepare(&Image::new(96, 48, rgb.clone()).expect("img"), &pc).expect("prep");
        assert_eq!((p.grid, p.n_soft, p.positions[7]), ((3, 6), 2, [1, 1]));
        // Patch 7 (row 1, col 1), pixel (2, 5), channel 1.
        assert_eq!(p.patches[7 * 768 + (2 * 16 + 5) * 3 + 1], f32::from(rgb[((16 + 2) * 96 + 16 + 5) * 3 + 1]) * (1.0 / 255.0));
    }
}
