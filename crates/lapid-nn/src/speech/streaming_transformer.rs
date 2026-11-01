//! Streaming pre-norm transformer (Kyutai Moshi / Mimi style, also Pocket TTS's flow LM):
//! LayerNorm blocks, multi-head attention with RoPE, an MLP, optional per-channel layer
//! scale on both residual branches, and a KV cache that is unbounded (causal) or windowed.
//! Weight names stay with the models: they build the layers from loaded parts.

use lapid::{Result, Tensor};

use crate::{Attention, KvCache, LayerNorm, Mask, Mlp, Rope};

/// One block: `x += ls1 * attn(norm1(x))`, then `x += ls2 * mlp(norm2(x))`.
pub struct TransformerLayer {
    pub norm1: LayerNorm,
    pub attn: Attention,
    pub norm2: LayerNorm,
    pub mlp: Mlp,
    /// Layer scales (attention branch, MLP branch), each [d]; `None` = 1.
    pub layer_scale: Option<(Tensor, Tensor)>,
}

pub struct StreamingTransformer {
    layers: Vec<TransformerLayer>,
    rope: Rope,
    /// `Some(w)`: keys with `0 <= pos_q - pos_k < w` are visible; `None`: causal.
    window: Option<usize>,
}

impl StreamingTransformer {
    pub fn new(layers: Vec<TransformerLayer>, rope: Rope, window: Option<usize>) -> StreamingTransformer {
        StreamingTransformer { layers, rope, window }
    }

    /// An empty cache (windowed like the attention).
    pub fn new_cache(&self) -> KvCache {
        KvCache::with_windows(vec![self.window; self.layers.len()])
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
        for (i, b) in self.layers.iter().enumerate() {
            let h = b.norm1.forward(&x)?;
            let offset = cache.offset();
            let q = self.rope.apply(&b.attn.project_q(&h)?, offset)?;
            let (k, v) = b.attn.project_kv(&h)?;
            let (k, v) = cache.append(i, &self.rope.apply(&k, offset)?, &v)?;
            let a = b.attn.attend_heads(&q, &k, &v, &mask)?;
            let a = match &b.layer_scale {
                Some((s1, _)) => a.mul(s1)?,
                None => a,
            };
            x = x.add(&a)?;
            let m = b.mlp.forward(&b.norm2.forward(&x)?)?;
            let m = match &b.layer_scale {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{assert_close, rand_tensor};
    use crate::{Activation, Linear, RopeLayout};
    use lapid::{DType, Device};

    fn layer(d: usize, seed: u64, scaled: bool) -> TransformerLayer {
        let lin = |o: usize, i: usize, s: u64| {
            Linear::new(rand_tensor(&[o, i], seed * 16 + s).0.mul_scalar(0.3).unwrap(), None)
        };
        let fill = |v: f64| Tensor::full(v, &[d], DType::F32);
        TransformerLayer {
            norm1: LayerNorm::new(fill(1.0), fill(0.0), 1e-5),
            attn: Attention::new(lin(d, d, 1), lin(d, d, 2), lin(d, d, 3), lin(d, d, 4), 2, 2).unwrap(),
            norm2: LayerNorm::new(fill(1.0), fill(0.0), 1e-5),
            mlp: Mlp::new(lin(2 * d, d, 5), lin(d, 2 * d, 6), Activation::Gelu),
            layer_scale: scaled.then(|| (fill(0.5), fill(0.25))),
        }
    }

    /// Streaming the sequence in pieces (cache realized between calls) equals one call,
    /// for the causal and the windowed attention.
    #[test]
    fn streamed_equals_full() {
        let (d, s) = (8, 6);
        let dev = Device::cpu();
        for (window, scaled) in [(None, false), (Some(3), true)] {
            let rope = Rope::new(d / 2, 10000.0).with_layout(RopeLayout::Interleaved);
            let t = StreamingTransformer::new(vec![layer(d, 1, scaled), layer(d, 2, scaled)], rope, window);
            let (x, _) = rand_tensor(&[1, s, d], 7);
            let mut cache = t.new_cache();
            let y = t.forward(&x, &mut cache, None).unwrap();
            let full = dev.to_vec_f32(&dev.eval(&[&y]).unwrap()[0]).unwrap();
            let mut cache = t.new_cache();
            let mut got = Vec::new();
            for (a, b) in [(0, 2), (2, 3), (3, 6)] {
                let y = t.forward(&x.slice(1, a, b).unwrap(), &mut cache, None).unwrap();
                let mut outs = vec![&y];
                outs.extend(cache.staged());
                let mut r = dev.eval(&outs).unwrap();
                cache.commit(r.split_off(1)).unwrap();
                got.extend(dev.to_vec_f32(&r[0]).unwrap());
            }
            assert_close(&got, &full, 1e-4);
        }
    }
}
