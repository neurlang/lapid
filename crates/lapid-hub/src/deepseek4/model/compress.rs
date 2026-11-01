//! KV compressor (HF `DeepseekV4{CSA,HCA}Compressor`, the indexer's own compressor; llama.cpp
//! `build_{overlap,hca}_compressed_kv_from_state`) as a streaming lapid graph.
//!
//! Every `ratio` source tokens close one window and emit one compressed entry:
//!
//! ```text
//! C = x Wkv^T, Z = x Wgate^T + ape[pos % ratio]                       per token
//! entry_w = rmsnorm(sum_j softmax_j(Z_j) * C_j)   rope'd at position w * ratio
//! ```
//! With `overlap` (CSA, ratio 4) every token carries two series `[Ca | Cb]` (2 x head dim) and
//! window w pools Cb of its own tokens with Ca of the previous window's tokens (width 2 ratio,
//! stride ratio); window 0 has no previous window (zero kv, -inf gate).
//!
//! State between steps ([`CompState`]): the emitted entries, the projected rows of the open
//! window and (overlap only) the rows of the last closed window. All sizes follow from the
//! absolute position, so the graph shapes of a step are known on the host.

use lapid::{DType, Result, Tensor};
use lapid_nn::{Linear, RmsNorm};

use lapid_nn::llm::rope::TailRope;

/// Additive mask value for "not visible" (finite so masks can be summed and multiplied).
pub(crate) const NEG: f32 = -1e30;

#[derive(Clone, Default)]
pub(crate) struct CompState {
    /// Emitted entries `[pos / ratio, head_dim]`.
    pub entries: Option<Tensor>,
    /// Open window rows `[pos % ratio, coff * head_dim]` (kv, gate + ape).
    pub pend_kv: Option<Tensor>,
    pub pend_score: Option<Tensor>,
    /// Last closed window `[ratio, coff * head_dim]` (overlap only).
    pub prev_kv: Option<Tensor>,
    pub prev_score: Option<Tensor>,
}

impl CompState {
    pub fn slots_mut(&mut self) -> [&mut Option<Tensor>; 5] {
        [&mut self.entries, &mut self.pend_kv, &mut self.pend_score, &mut self.prev_kv, &mut self.prev_score]
    }

    pub fn slots(&self) -> [&Option<Tensor>; 5] {
        [&self.entries, &self.pend_kv, &self.pend_score, &self.prev_kv, &self.prev_score]
    }
}

pub(crate) struct Compressor {
    pub wkv: Linear,
    pub wgate: Linear,
    /// `[ratio, coff * head_dim]` f32.
    pub ape: Tensor,
    pub norm: RmsNorm,
    pub ratio: usize,
    pub head_dim: usize,
    pub overlap: bool,
}

impl Compressor {
    /// Feed `x [n, D]` at absolute positions `pos..pos + n`. Returns every entry emitted so far
    /// (old + new; `None` while there are none) and the next state.
    pub fn step(&self, x: &Tensor, pos: usize, rope: &TailRope, st: &CompState) -> Result<(Option<Tensor>, CompState)> {
        let n = x.shape()[0];
        let (r, hd) = (self.ratio, self.head_dim);
        let c = if self.overlap { 2 * hd } else { hd };
        let kv = self.wkv.forward(x)?.cast(DType::F32)?;
        let slots: Vec<i32> = (pos..pos + n).map(|p| (p % r) as i32).collect();
        let ape = self.ape.gather(&Tensor::from_i32(&slots, &[n])?)?; // [n, c]
        let score = self.wgate.forward(x)?.cast(DType::F32)?.add(&ape)?;
        let (all_kv, all_score) = match (&st.pend_kv, &st.pend_score) {
            (Some(pk), Some(ps)) => (Tensor::cat(&[pk, &kv], 0)?, Tensor::cat(&[ps, &score], 0)?),
            _ => (kv, score),
        };
        let total = all_kv.shape()[0];
        let start = pos - pos % r;
        let nw = total / r;
        let mut next = st.clone();
        let rest = |t: &Tensor| -> Result<Option<Tensor>> { if total > nw * r { Ok(Some(t.slice(0, nw * r, total)?)) } else { Ok(None) } };
        next.pend_kv = rest(&all_kv)?;
        next.pend_score = rest(&all_score)?;
        if nw == 0 {
            return Ok((st.entries.clone(), next));
        }
        let chunk_kv = all_kv.slice(0, 0, nw * r)?.reshape(&[nw, r, c])?;
        let chunk_score = all_score.slice(0, 0, nw * r)?.reshape(&[nw, r, c])?;
        let (values, scores) = if self.overlap {
            let prev_ca = |p: &Option<Tensor>, fill: f64| -> Result<Tensor> {
                match p {
                    Some(t) => t.slice(1, 0, hd)?.reshape(&[1, r, hd]),
                    None => Ok(Tensor::full(fill, &[1, r, hd], DType::F32)),
                }
            };
            let ca = |chunk: &Tensor, prev: Tensor| -> Result<Tensor> {
                if nw > 1 {
                    Tensor::cat(&[&prev, &chunk.slice(0, 0, nw - 1)?.slice(2, 0, hd)?], 0)
                } else {
                    Ok(prev)
                }
            };
            let ca_kv = ca(&chunk_kv, prev_ca(&st.prev_kv, 0.0)?)?;
            let ca_s = ca(&chunk_score, prev_ca(&st.prev_score, f64::from(NEG))?)?;
            let v = Tensor::cat(&[&ca_kv, &chunk_kv.slice(2, hd, 2 * hd)?], 1)?; // [nw, 2r, hd]
            let s = Tensor::cat(&[&ca_s, &chunk_score.slice(2, hd, 2 * hd)?], 1)?;
            next.prev_kv = Some(chunk_kv.slice(0, nw - 1, nw)?.reshape(&[r, c])?);
            next.prev_score = Some(chunk_score.slice(0, nw - 1, nw)?.reshape(&[r, c])?);
            (v, s)
        } else {
            (chunk_kv, chunk_score)
        };
        let w = scores.permute(&[0, 2, 1])?.softmax()?; // [nw, hd, L]
        let pooled = values.permute(&[0, 2, 1])?.mul(&w)?.sum(2, false)?; // [nw, hd]
        let normed = self.norm.forward(&pooled)?.cast(DType::F32)?;
        let positions: Vec<usize> = (0..nw).map(|j| start + j * r).collect();
        let (cos, sin) = rope.tables(&positions)?;
        let new = rope.apply(&normed.reshape(&[nw, 1, hd])?, &cos, &sin, false)?.reshape(&[nw, hd])?;
        let entries = match &st.entries {
            Some(old) => Tensor::cat(&[old, &new], 0)?,
            None => new,
        };
        next.entries = Some(entries.clone());
        Ok((Some(entries), next))
    }
}
