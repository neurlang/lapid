//! Token embedding and logits projection of a decoder LLM: `lm_head`, or the tied embedding
//! (with int8 activations in the fake-quant reference when the embedding is stored quantized).

use lapid::{DType, Result, Tensor, bail};

use crate::layers::{Embedding, Linear};
use crate::weights::{Loader, fake_quant_act};

pub struct LmHead {
    /// `[vocab, D]` token embedding.
    pub embed: Tensor,
    head: Option<Linear>,
    act_quant: bool,
}

impl LmHead {
    /// Embedding `embed` `[vocab, d]`; logits through `{head}.weight` when `head` is given and
    /// the file has it, else through the embedding.
    pub fn load(l: &Loader<'_>, embed: &str, head: Option<&str>, vocab: usize, d: usize) -> Result<LmHead> {
        let head = head.filter(|h| l.contains(&format!("{h}.weight")));
        Ok(LmHead {
            embed: l.tensor_checked(embed, &[vocab, d])?,
            head: head.map(|h| l.linear_checked(h, vocab, d, false)).transpose()?,
            act_quant: head.is_none() && l.act_quant(embed)?,
        })
    }

    /// f32 embeddings of `ids` `[1, n, D]`.
    pub fn embed(&self, ids: &[u32]) -> Result<Tensor> {
        if ids.is_empty() {
            bail!("invalid_argument", "forward of zero tokens");
        }
        Embedding::new(self.embed.clone()).lookup(ids)?.cast(DType::F32)
    }

    /// Logits of the (normed) hidden states `h [.., D]`.
    pub fn project(&self, h: &Tensor) -> Result<Tensor> {
        if let Some(head) = &self.head {
            return head.forward(h);
        }
        let h = if self.act_quant { fake_quant_act(h)? } else { h.clone() };
        h.matmul_t(&self.embed)
    }
}
