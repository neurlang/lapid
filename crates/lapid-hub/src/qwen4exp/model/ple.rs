//! PLE n-gram hash embeddings (HF `Qwen4ExpTextPLELayer` / `Qwen4ExpTextNGramEmbedding`,
//! llama.cpp qwen4exp `build_ple`).
//!
//! Each token hashes its 2..=n-gram contexts (`mixed = t0*m0 ^ t1*m1 ^ ..`, an EOS predecessor
//! or the sequence start cuts the context to EOS) into `(n - 1) * heads_per_ngram` rows of
//! one shared table; the concatenated rows `e [1, n, E]` drive a gated injection into every
//! hyper-connection stream:
//!
//! ```text
//! key   = rmsnorm_s(key_proj(e))          [hc, D] per token     query = rmsnorm_s(x)
//! g     = sigmoid(ssqrt(sum(key * query) / sqrt(D)))              (ssqrt = sign * sqrt(max(|.|, 1e-6)))
//! v     = g * value_proj(e)                (broadcast over the streams)
//! x'    = x + v + silu(dwconv(rmsnorm_s(v)))   (causal depthwise, kernel K, dilation n)
//! ```
//!
//! The table (tens of GB in the released checkpoint) never goes to the device: hashing and
//! the row gather run on the host from the mmap'd `.lapid` tensor, only the gathered rows are
//! uploaded.

use lapid::{DType, Error, Op, Result, Tensor, UnaryOp, bail};
use lapid_nn::{Conv1d, Linear, ops};

use super::hc::grouped_rms_norm;
use crate::qwen4exp::config::PleConfig;

/// The host n-gram table: a `.lapid` host tensor [rows, row_dim] (float or quantized).
pub(crate) struct NgramTable {
    pub table: Tensor,
    pub rows: usize,
    pub cols: usize,
}

fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f32;
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        31 => {
            if frac == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        e => sign * (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
    }
}

impl NgramTable {
    fn bytes(&self) -> Result<&[u8]> {
        match self.table.op() {
            Op::Host(h) => Ok(h.bytes()),
            _ => bail!("internal", "qwen4exp: n-gram table is not a host tensor"),
        }
    }

    /// Row `r` as f32 into `out` (len `cols`).
    fn row(&self, r: usize, out: &mut [f32]) -> Result<()> {
        if r >= self.rows {
            bail!("index_out_of_range", "qwen4exp: n-gram row {r} of {}", self.rows);
        }
        let b = self.bytes()?;
        let c = self.cols;
        let dt = self.table.dtype();
        if let Some(fmt) = dt.quant_format() {
            lapid::quant::dequantize_row(fmt, b, self.rows, c, r, out);
            return Ok(());
        }
        let width = match dt {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            other => bail!("unsupported_dtype", "qwen4exp: n-gram table dtype {other:?}"),
        };
        let row = b.get(r * c * width..(r + 1) * c * width).ok_or_else(|| Error::new("format", "qwen4exp: n-gram table is truncated"))?;
        for (i, o) in out.iter_mut().enumerate() {
            *o = match dt {
                DType::F32 => f32::from_le_bytes([row[4 * i], row[4 * i + 1], row[4 * i + 2], row[4 * i + 3]]),
                DType::F16 => half_to_f32(u16::from_le_bytes([row[2 * i], row[2 * i + 1]])),
                _ => f32::from_bits((u16::from_le_bytes([row[2 * i], row[2 * i + 1]]) as u32) << 16),
            };
        }
        Ok(())
    }
}

/// Table row of every head for each token of `ids`; `hist` are the tokens before `ids`
/// (oldest first, at most `ngram - 1` used).
pub(crate) fn ngram_rows(cfg: &PleConfig, hist: &[u32], ids: &[u32]) -> Vec<usize> {
    let all: Vec<u32> = hist.iter().chain(ids).copied().collect();
    let base = hist.len();
    let heads = cfg.n_heads();
    let eos = cfg.eos as u64;
    let mut rows = Vec::with_capacity(ids.len() * heads);
    for i in 0..ids.len() {
        let p = base + i;
        let mut ctx = vec![all[p] as u64; cfg.ngram];
        let mut cut = false;
        for (s, c) in ctx.iter_mut().enumerate().skip(1) {
            let t = if cut || p < s { None } else { Some(all[p - s] as u64) };
            cut = cut || t.is_none_or(|t| t == eos);
            *c = if cut { eos } else { t.unwrap_or(eos) };
        }
        for n in 2..=cfg.ngram {
            let mut mixed = ctx[0].wrapping_mul(cfg.multipliers[0]);
            for (j, &c) in ctx.iter().enumerate().take(n).skip(1) {
                mixed ^= c.wrapping_mul(cfg.multipliers[j]);
            }
            for g in 0..cfg.heads_per_ngram {
                let h = (n - 2) * cfg.heads_per_ngram + g;
                rows.push((mixed % cfg.vocab_sizes[h] + cfg.offsets[h]) as usize);
            }
        }
    }
    rows
}

pub(crate) struct Ple {
    pub cfg: PleConfig,
    pub table: NgramTable,
    pub key: Linear,
    pub value: Linear,
    /// [hc, D] effective scales.
    pub norm_key: Tensor,
    pub norm_query: Tensor,
    pub norm_conv: Tensor,
    /// Depthwise [hc*D, 1, K], dilation = n-gram size.
    pub conv: Conv1d,
    pub hc: usize,
    pub eps: f32,
}

impl Ple {
    /// Gathered embeddings e [1, n, E] of `ids` (host).
    pub fn embed(&self, hist: &[u32], ids: &[u32]) -> Result<Tensor> {
        let rows = ngram_rows(&self.cfg, hist, ids);
        let c = self.table.cols;
        let mut data = vec![0.0f32; rows.len() * c];
        for (k, &r) in rows.iter().enumerate() {
            self.table.row(r, &mut data[k * c..(k + 1) * c])?;
        }
        Tensor::from_f32(&data, &[1, ids.len(), self.cfg.n_heads() * c])
    }

    /// The injection [1, n, hc, D] added to x [1, n, hc, D]; `conv` is the committed conv
    /// history (None = zeros), replaced by the new lazy one.
    pub fn forward(&self, x: &Tensor, e: &Tensor, conv: &mut Option<Tensor>) -> Result<Tensor> {
        let (n, hc, d) = (x.shape()[1], self.hc, x.shape()[3]);
        let key = grouped_rms_norm(&self.key.forward(e)?.cast(DType::F32)?.reshape(&[1, n, hc, d])?, &self.norm_key, self.eps)?;
        let query = grouped_rms_norm(x, &self.norm_query, self.eps)?;
        let g = key.mul(&query)?.sum(3, true)?.mul_scalar(1.0 / (d as f64).sqrt())?; // [1, n, hc, 1]
        let zero = Tensor::full(0.0, &[1], DType::F32);
        let sign = zero.binary(lapid::BinaryOp::CmpLt, &g)?.sub(&g.binary(lapid::BinaryOp::CmpLt, &zero)?)?;
        let mag = g.unary(UnaryOp::Abs)?.maximum(&Tensor::full(1e-6, &[1], DType::F32))?.sqrt()?;
        let gate = ops::sigmoid(&sign.mul(&mag)?)?;
        let v = self.value.forward(e)?.cast(DType::F32)?.unsqueeze(2)?.mul(&gate)?; // [1, n, hc, D]
        let vn = grouped_rms_norm(&v, &self.norm_conv, self.eps)?.reshape(&[1, n, hc * d])?;
        let y = self.conv.forward_stream(&vn.transpose(1, 2)?, conv)?; // [1, hc*D, n]
        let y = ops::silu(&y.cast(DType::F32)?)?.transpose(1, 2)?.reshape(&[1, n, hc, d])?;
        v.add(&y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PleConfig {
        PleConfig {
            layer: 1,
            ngram: 3,
            heads_per_ngram: 1,
            conv_kernel: 4,
            eos: 9,
            multipliers: vec![3, 5, 7],
            offsets: vec![0, 100],
            vocab_sizes: vec![97, 89],
            row_dim: 4,
            rows: 256,
        }
    }

    #[test]
    fn ngram_hash_cuts_at_eos() {
        let c = cfg();
        let r = |t0: u64, t1: u64, t2: u64| vec![((t0 * 3) ^ (t1 * 5)) % 97, (((t0 * 3) ^ (t1 * 5) ^ (t2 * 7)) % 89) + 100];
        let rows = ngram_rows(&c, &[], &[4, 6, 9, 2]);
        let want: Vec<usize> = [r(4, 9, 9), r(6, 4, 9), r(9, 6, 4), r(2, 9, 9)].concat().into_iter().map(|x| x as usize).collect();
        assert_eq!(rows, want);
        // the history continues a sequence exactly like one call
        assert_eq!(ngram_rows(&c, &[4, 6], &[9, 2]), want[4..].to_vec());
    }

    #[test]
    fn half() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
    }
}
