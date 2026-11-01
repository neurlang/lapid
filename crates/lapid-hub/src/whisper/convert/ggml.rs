//! whisper.cpp legacy ggml model file (`ggml-*.bin`) parser (tensor types and dequantizers:
//! `lapid_convert::ggml`).
//!
//! ```text
//! u32 magic 0x67676d6c ("ggml" as LE u32)
//! i32 x 11 hparams: n_vocab n_audio_ctx n_audio_state n_audio_head n_audio_layer
//!                   n_text_ctx n_text_state n_text_head n_text_layer n_mels ftype
//! i32 n_mel, i32 n_fft, f32[n_mel*n_fft]        mel filters
//! i32 n_tokens, n_tokens x (u32 len, len bytes) vocab (raw token bytes)
//! repeated until EOF:
//!   i32 n_dims, i32 name_len, i32 ggml_type, i32 ne[n_dims], name bytes, data
//! ```
//! `ne[0]` is the innermost (contiguous) dimension, so the row-major shape is `ne` reversed.

use std::path::Path;

use lapid::{Error, Result, bail};
pub(crate) use lapid_convert::ggml::{GgmlType, dequantize};

pub(crate) const GGML_MAGIC: u32 = 0x6767_6d6c;

/// Whether `path` starts with the ggml magic (short files are simply "no").
pub(crate) fn is_ggml_file(path: &Path) -> Result<bool> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| Error::new("io", format!("{}: {e}", path.display())))?;
    let mut m = [0u8; 4];
    match f.read_exact(&mut m) {
        Ok(()) => Ok(u32::from_le_bytes(m) == GGML_MAGIC),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(Error::new("io", format!("{}: {e}", path.display()))),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Hparams {
    pub n_vocab: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_layer: usize,
    pub n_mels: usize,
    pub ftype: i32,
}

pub(crate) struct GgmlTensor<'a> {
    pub name: String,
    /// ggml order: ne[0] innermost.
    pub ne: Vec<usize>,
    pub ttype: GgmlType,
    pub data: &'a [u8],
}

impl GgmlTensor<'_> {
    pub(crate) fn numel(&self) -> usize {
        self.ne.iter().product()
    }

    /// Row-major shape (ne reversed).
    pub(crate) fn shape(&self) -> Vec<usize> {
        self.ne.iter().rev().copied().collect()
    }
}

pub(crate) struct GgmlModel {
    mmap: memmap2::Mmap,
    pub hparams: Hparams,
    pub vocab: Vec<Vec<u8>>,
    tensors_start: usize,
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.b.len()).ok_or_else(|| {
            Error::new("format", format!("ggml file truncated at byte {} (need {n} more)", self.pos))
        })?;
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn i32(&mut self) -> Result<i32> {
        let s = self.take(4)?;
        Ok(i32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    /// Non-negative i32 within `max`.
    fn count(&mut self, what: &str, max: usize) -> Result<usize> {
        let v = self.i32()?;
        if v < 0 || v as usize > max {
            bail!("format", "ggml: {what} = {v} out of range 0..={max}");
        }
        Ok(v as usize)
    }
}

impl GgmlModel {
    pub(crate) fn open(path: &Path) -> Result<GgmlModel> {
        let file = std::fs::File::open(path).map_err(|e| Error::new("io", format!("{}: {e}", path.display())))?;
        // SAFETY: read-only mapping; the file must not be truncated while open.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| Error::new("io", format!("mmap {}: {e}", path.display())))?;
        let mut c = Cursor { b: &mmap, pos: 0 };
        let magic = c.i32()? as u32;
        if magic != GGML_MAGIC {
            bail!("format", "{}: not a whisper.cpp ggml file (magic {magic:#010x})", path.display());
        }
        const MAX: usize = 1 << 20;
        let mut h = [0usize; 10];
        let names = [
            "n_vocab", "n_audio_ctx", "n_audio_state", "n_audio_head", "n_audio_layer", "n_text_ctx", "n_text_state",
            "n_text_head", "n_text_layer", "n_mels",
        ];
        for (v, n) in h.iter_mut().zip(names) {
            *v = c.count(n, MAX)?;
            if *v == 0 {
                bail!("format", "ggml: hparam {n} is 0");
            }
        }
        let ftype = c.i32()?;
        let hparams = Hparams {
            n_vocab: h[0],
            n_audio_ctx: h[1],
            n_audio_state: h[2],
            n_audio_head: h[3],
            n_audio_layer: h[4],
            n_text_ctx: h[5],
            n_text_state: h[6],
            n_text_head: h[7],
            n_text_layer: h[8],
            n_mels: h[9],
            ftype,
        };
        // Mel filters (unused: the hub computes its own filterbank).
        let n_mel = c.count("filters.n_mel", 4096)?;
        let n_fft = c.count("filters.n_fft", 1 << 16)?;
        c.take(n_mel * n_fft * 4)?;
        let n_tokens = c.count("vocab size", MAX)?;
        let mut vocab = Vec::with_capacity(n_tokens);
        for _ in 0..n_tokens {
            let len = c.count("token length", 1 << 16)?;
            vocab.push(c.take(len)?.to_vec());
        }
        let tensors_start = c.pos;
        Ok(GgmlModel { mmap, hparams, vocab, tensors_start })
    }

    /// Parses all tensor records (headers + data slices) after the vocab.
    pub(crate) fn tensors(&self) -> Result<Vec<GgmlTensor<'_>>> {
        let mut c = Cursor { b: &self.mmap, pos: self.tensors_start };
        let mut out = Vec::new();
        while c.pos < c.b.len() {
            let n_dims = c.count("n_dims", 4)?;
            let name_len = c.count("name length", 4096)?;
            let ttype = GgmlType::from_id(c.i32()? as u32);
            if n_dims == 0 || name_len == 0 {
                bail!("format", "ggml: tensor record at byte {} has n_dims={n_dims} name_len={name_len}", c.pos);
            }
            let mut ne = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                ne.push(c.count("ne", 1 << 30)?);
            }
            let name = String::from_utf8(c.take(name_len)?.to_vec())
                .map_err(|_| Error::new("format", "ggml: tensor name is not UTF-8"))?;
            let numel = ne.iter().try_fold(1usize, |a, &d| a.checked_mul(d)).ok_or_else(|| Error::new("format", "ggml: tensor too large"))?;
            let (blk, bsz) = ttype.block().ok_or_else(|| Error::new("unsupported_dtype", format!("ggml: tensor {name}: unsupported type {}", ttype.name())))?;
            if ne[0] % blk != 0 {
                bail!("format", "ggml: tensor {name}: ne[0]={} not a multiple of block size {blk}", ne[0]);
            }
            let data = c.take(numel / blk * bsz)?;
            out.push(GgmlTensor { name, ne, ttype, data });
        }
        Ok(out)
    }
}
