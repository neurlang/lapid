//! The streaming transformer shared by the flow LM backbone and the Mimi decoder:
//! pre-norm LayerNorm blocks, MHA with interleaved RoPE (the checkpoint's native layout),
//! GELU (erf) MLP without biases, optional layer scale, KV cache (unbounded or windowed).

use lapid::{Result, Tensor};
use lapid_nn::{Activation, Attention, LayerNorm, Loader, Mask, Mlp, Rope, RopeLayout};

use lapid_nn::KvCache;

struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: Mlp,
    /// (layer_scale_1, layer_scale_2) [d]
    scale: Option<(Tensor, Tensor)>,
}

pub(crate) struct Transformer {
    blocks: Vec<Block>,
    rope: Rope,
    /// `Some(w)`: keys with `0 <= pos_q - pos_k < w` are visible; `None`: causal.
    window: Option<usize>,
}

pub(crate) struct TransformerSpec {
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_model: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub window: Option<usize>,
    pub layer_scale: bool,
}

impl Transformer {
    /// Weights under `{prefix}.layers.{i}.` (`self_attn.{q,k,v,out}_proj`, `linear1/2`,
    /// `norm1/2`, `layer_scale_{1,2}.scale`).
    pub fn load(loader: &Loader, prefix: &str, spec: &TransformerSpec) -> Result<Transformer> {
        let mut blocks = Vec::with_capacity(spec.n_layers);
        for i in 0..spec.n_layers {
            let p = format!("{prefix}.layers.{i}");
            let lin = |n: &str| loader.linear(&format!("{p}.{n}"), false);
            let attn = Attention::new(
                lin("self_attn.q_proj")?,
                lin("self_attn.k_proj")?,
                lin("self_attn.v_proj")?,
                lin("self_attn.out_proj")?,
                spec.n_heads,
                spec.n_heads,
            )?;
            let scale = if spec.layer_scale {
                Some((
                    loader.tensor_f32(&format!("{p}.layer_scale_1.scale"))?,
                    loader.tensor_f32(&format!("{p}.layer_scale_2.scale"))?,
                ))
            } else {
                None
            };
            blocks.push(Block {
                norm1: loader.layer_norm(&format!("{p}.norm1"), spec.norm_eps)?,
                attn,
                norm2: loader.layer_norm(&format!("{p}.norm2"), spec.norm_eps)?,
                mlp: Mlp::new(lin("linear1")?, lin("linear2")?, Activation::Gelu),
                scale,
            });
        }
        let rope = Rope::new(spec.d_model / spec.n_heads, spec.rope_theta).with_layout(RopeLayout::Interleaved);
        Ok(Transformer { blocks, rope, window: spec.window })
    }

    pub fn new_cache(&self) -> KvCache {
        KvCache::with_windows(vec![self.window; self.blocks.len()])
    }

    /// x [B, S, D] at positions `cache.offset()..`; stages the new K/V in `cache`. When
    /// `trace` is given, every block output is pushed (named `{name}.{i}`).
    pub fn forward(
        &self,
        x: &Tensor,
        cache: &mut KvCache,
        mut trace: Option<(&str, &mut Vec<Tensor>)>,
    ) -> Result<Tensor> {
        let mask = match self.window {
            Some(w) => Mask::Sliding(w),
            None => Mask::Causal,
        };
        let mut x = x.clone();
        for (i, b) in self.blocks.iter().enumerate() {
            let h = b.norm1.forward(&x)?;
            let offset = cache.offset();
            let q = self.rope.apply(&b.attn.project_q(&h)?, offset)?;
            let (k, v) = b.attn.project_kv(&h)?;
            let (k, v) = cache.append(i, &self.rope.apply(&k, offset)?, &v)?;
            let a = b.attn.attend_heads(&q, &k, &v, &mask)?;
            let a = match &b.scale {
                Some((s1, _)) => a.mul(s1)?,
                None => a,
            };
            x = x.add(&a)?;
            let m = b.mlp.forward(&b.norm2.forward(&x)?)?;
            let m = match &b.scale {
                Some((_, s2)) => m.mul(s2)?,
                None => m,
            };
            x = x.add(&m)?;
            if let Some((name, out)) = trace.as_mut() {
                out.push(x.named(format!("{name}.{i}")));
            }
        }
        Ok(x)
    }
}
