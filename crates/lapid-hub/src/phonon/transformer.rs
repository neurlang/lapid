//! Loading the streaming transformer shared by the flow LM backbone and the Mimi decoder
//! ([`lapid_nn::speech::streaming_transformer`]): pre-norm LayerNorm blocks, MHA with
//! interleaved RoPE (the checkpoint's native layout), GELU (erf) MLP without biases, optional
//! layer scale, KV cache (unbounded or windowed).

use lapid::Result;
use lapid_nn::speech::streaming_transformer::{StreamingTransformer, TransformerLayer};
use lapid_nn::{Activation, Attention, Loader, Mlp, Rope, RopeLayout};

pub(crate) struct TransformerSpec {
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_model: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub window: Option<usize>,
    pub layer_scale: bool,
}

/// Weights under `{prefix}.layers.{i}.` (`self_attn.{q,k,v,out}_proj`, `linear1/2`,
/// `norm1/2`, `layer_scale_{1,2}.scale`).
pub(crate) fn load(loader: &Loader, prefix: &str, spec: &TransformerSpec) -> Result<StreamingTransformer> {
    let mut layers = Vec::with_capacity(spec.n_layers);
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
        let layer_scale = if spec.layer_scale {
            Some((
                loader.tensor_f32(&format!("{p}.layer_scale_1.scale"))?,
                loader.tensor_f32(&format!("{p}.layer_scale_2.scale"))?,
            ))
        } else {
            None
        };
        layers.push(TransformerLayer {
            norm1: loader.layer_norm(&format!("{p}.norm1"), spec.norm_eps)?,
            attn,
            norm2: loader.layer_norm(&format!("{p}.norm2"), spec.norm_eps)?,
            mlp: Mlp::new(lin("linear1")?, lin("linear2")?, Activation::Gelu),
            layer_scale,
        });
    }
    let rope = Rope::new(spec.d_model / spec.n_heads, spec.rope_theta).with_layout(RopeLayout::Interleaved);
    Ok(StreamingTransformer::new(layers, rope, spec.window))
}
