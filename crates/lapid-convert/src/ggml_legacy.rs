//! Legacy (unversioned) ggml model container, `ggml-*.bin`, as written by whisper.cpp and the
//! first ggml examples (tensor types and dequantizers: [`crate::ggml`]).
//!
//! ```text
//! u32 magic 0x67676d6c ("ggml" as LE u32)
//! model header: hyper-parameters, extra tables, vocabulary (layout defined by each model;
//!               read with [`Reader`])
//! repeated until EOF:
//!   i32 n_dims, i32 name_len, i32 ggml_type, i32 ne[n_dims], name bytes, data
//! ```
//! `ne[0]` is the innermost (contiguous) dimension, so the row-major shape is `ne` reversed.
//! Model-specific header parsing and tensor naming live with each model in `lapid-hub`.

use std::path::Path;

use lapid_tensor::{Error, Result, bail};

use crate::ggml::GgmlType;

pub const MAGIC: u32 = 0x6767_6d6c;

/// Whether `path` starts with the legacy ggml magic (short files are simply "no").
pub fn is_legacy_ggml(path: &Path) -> Result<bool> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| Error::new("io", format!("{}: {e}", path.display())))?;
    let mut m = [0u8; 4];
    match f.read_exact(&mut m) {
        Ok(()) => Ok(u32::from_le_bytes(m) == MAGIC),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(Error::new("io", format!("{}: {e}", path.display()))),
    }
}

/// One tensor record; `data` borrows the mapped file.
pub struct GgmlTensor<'a> {
    pub name: String,
    /// ggml order: ne[0] innermost.
    pub ne: Vec<usize>,
    pub ttype: GgmlType,
    pub data: &'a [u8],
}

impl GgmlTensor<'_> {
    pub fn numel(&self) -> usize {
        self.ne.iter().product()
    }

    /// Row-major shape (ne reversed).
    pub fn shape(&self) -> Vec<usize> {
        self.ne.iter().rev().copied().collect()
    }
}

/// Little-endian reader over the mapped file with bounds-checked reads.
pub struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Byte offset of the next read.
    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.b.len()).ok_or_else(|| {
            Error::new("format", format!("ggml file truncated at byte {} (need {n} more)", self.pos))
        })?;
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    pub fn i32(&mut self) -> Result<i32> {
        let s = self.take(4)?;
        Ok(i32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    /// Non-negative i32 within `max`.
    pub fn count(&mut self, what: &str, max: usize) -> Result<usize> {
        let v = self.i32()?;
        if v < 0 || v as usize > max {
            bail!("format", "ggml: {what} = {v} out of range 0..={max}");
        }
        Ok(v as usize)
    }
}

/// A memory-mapped legacy ggml file.
pub struct GgmlFile {
    mmap: memmap2::Mmap,
}

impl GgmlFile {
    /// Map `path` and check the magic.
    pub fn open(path: &Path) -> Result<GgmlFile> {
        let file = std::fs::File::open(path).map_err(|e| Error::new("io", format!("{}: {e}", path.display())))?;
        // SAFETY: read-only mapping; the file must not be truncated while open.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| Error::new("io", format!("mmap {}: {e}", path.display())))?;
        let ggml = GgmlFile { mmap };
        let magic = Reader { b: &ggml.mmap, pos: 0 }.i32()? as u32;
        if magic != MAGIC {
            bail!("format", "{}: not a legacy ggml file (magic {magic:#010x})", path.display());
        }
        Ok(ggml)
    }

    /// Reader positioned at the model header (after the magic).
    pub fn header(&self) -> Reader<'_> {
        Reader { b: &self.mmap, pos: 4 }
    }

    /// Parses all tensor records (headers + data slices) from byte `start` (the end of the
    /// model header) to the end of the file.
    pub fn tensors(&self, start: usize) -> Result<Vec<GgmlTensor<'_>>> {
        let mut c = Reader { b: &self.mmap, pos: start };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_and_tensor_records() {
        let mut b: Vec<u8> = MAGIC.to_le_bytes().to_vec();
        b.extend(7i32.to_le_bytes()); // a one-value model header
        for v in [2i32, 1, 0, 3, 2] {
            b.extend(v.to_le_bytes()); // n_dims, name_len, F32, ne = [3, 2]
        }
        b.push(b'w');
        b.extend((0..6).flat_map(|i| (i as f32).to_le_bytes()));
        let path = std::env::temp_dir().join(format!("lapid-ggml-legacy-{}", std::process::id()));
        std::fs::write(&path, &b).unwrap();
        assert!(is_legacy_ggml(&path).unwrap());
        let f = GgmlFile::open(&path).unwrap();
        let mut h = f.header();
        assert_eq!(h.count("x", 10).unwrap(), 7);
        let t = f.tensors(h.pos()).unwrap();
        assert_eq!((t.len(), t[0].name.as_str(), t[0].shape(), t[0].ttype), (1, "w", vec![2, 3], GgmlType::F32));
        assert_eq!(t[0].data.len(), 24);
        std::fs::remove_file(&path).ok();
    }
}
