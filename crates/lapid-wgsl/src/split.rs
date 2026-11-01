//! Launches whose inputs exceed the device's maximum storage binding (Vulkan
//! `maxStorageBufferRange`, WebGPU `maxStorageBufferBindingSize`) are split into several
//! launches that each bind a sub-range of the large buffer:
//!
//! * MatMul with a transposed weight (float or K-quant, no batch stride): chunks of output
//!   columns; each chunk binds the weight rows it needs and writes its slice of the output.
//! * Gather from a float or K-quant table: chunks of table rows; each chunk writes the
//!   output rows whose index falls inside it (`base` / `all_rows` params of the kernel).
//!
//! Int8/Int4 (SoA planes: codes then scales) cannot be cut into one contiguous range and
//! stay "too_large".

use lapid_compiler::{GatherDims, Kernel, MatMulDims};
use lapid_tensor::{DType, Error, Result};

use crate::{Caps, Plan};

/// One piece of a split launch: its plan and, per input, the byte range to bind
/// (`None` = the whole buffer).
#[derive(Clone, Debug)]
pub struct SubLaunch {
    pub plan: Plan,
    pub ranges: Vec<Option<(u64, u64)>>,
}

/// Per-input bound byte range (`None` = whole buffer).
pub type Ranges = Vec<Option<(u64, u64)>>;

fn too_large(what: &str, size: u64, limit: u64) -> Error {
    Error::new("too_large", format!("{what}: buffer of {size} bytes exceeds the max storage binding ({limit} bytes) and cannot be split"))
}

/// Binding sizes must be multiples of 4 (WebGPU); never past the buffer end.
fn round4(len: u64, avail: u64) -> u64 {
    len.div_ceil(4).saturating_mul(4).min(avail)
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Bytes per row of a `[rows, cols]` table/weight that can be cut by rows, or None.
fn row_bytes(dt: DType, cols: usize) -> Option<u64> {
    match dt.quant_format() {
        Some(f) if f.is_aos() => Some((cols / f.block_elems() * f.block_bytes()) as u64),
        Some(_) => None,
        None => Some((cols * dt.size()) as u64),
    }
}

/// Row chunks `[r0, r1)` of a matrix with `row_bytes` per row starting at byte `base`, so
/// that each chunk's range (from its aligned start) fits `limit` and starts on `align`.
/// Returns (r0, r1, bind_offset, bind_len, elements skipped from bind_offset to row r0).
type Chunk = (usize, usize, u64, u64, u64);

fn row_chunks(rows: usize, rb: u64, base: u64, limit: u64, align: u64, elem: u64) -> Option<Vec<Chunk>> {
    if rb == 0 || rb + align > limit {
        return None;
    }
    // rows per chunk: a multiple of the step that keeps chunk starts aligned when the start
    // offset itself cannot be shifted (elem == 0: quantized, no element offset available)
    let step = if elem == 0 { align / gcd(rb, align) } else { 1 };
    let per = ((limit - align) / rb) as usize / step as usize * step as usize;
    if per == 0 {
        return None;
    }
    let mut out = Vec::new();
    let mut r0 = 0usize;
    while r0 < rows {
        let r1 = (r0 + per).min(rows);
        let start = base + r0 as u64 * rb;
        let aligned = start / align * align;
        let skip = start - aligned;
        if elem == 0 && skip != 0 {
            return None;
        }
        let skip_elems = match skip.checked_div(elem) {
            None => 0,
            Some(e) if skip.is_multiple_of(elem) => e,
            Some(_) => return None,
        };
        let len = (r1 - r0) as u64 * rb + skip;
        out.push((r0, r1, aligned, len, skip_elems));
        r0 = r1;
    }
    Some(out)
}

/// Plan `kernel`, split when an input exceeds `limit` bytes. `in_sizes` are the bound
/// buffers' byte sizes (inputs in order), `align` the minimum storage offset alignment.
/// Returns `Ok(None)` when nothing needs splitting (plan the kernel normally).
pub fn plan_split(kernel: &Kernel, in_sizes: &[u64], limit: u64, align: u64, caps: &Caps) -> Result<Option<Vec<SubLaunch>>> {
    let Some(big) = in_sizes.iter().position(|&s| s > limit) else {
        return Ok(None);
    };
    let size = in_sizes[big];
    let align = align.max(4);
    let n_in = in_sizes.len();
    let others_fit = in_sizes.iter().enumerate().all(|(i, &s)| i == big || s <= limit);
    match kernel {
        Kernel::MatMul(spec, d) if big == 1 && others_fit && spec.b_transposed && (d.batch <= 1 || d.b.batch_stride == 0) => {
            let quant = spec.b_dtype.is_quant();
            let rb = if quant { row_bytes(spec.b_dtype, d.k) } else { Some((d.b.row_stride * spec.b_dtype.size()) as u64) };
            let base = if quant { 0 } else { (d.b.offset * spec.b_dtype.size()) as u64 };
            // aligned weights get aligned chunks (no element offset: vector loads stay on)
            let elem = if base.is_multiple_of(align) { 0 } else { spec.b_dtype.size() as u64 };
            let chunks = rb
                .and_then(|rb| row_chunks(d.n, rb, base, limit, align, elem))
                .ok_or_else(|| too_large("matmul weight", size, limit))?;
            let mut subs = Vec::with_capacity(chunks.len());
            for (n0, n1, off, len, skip) in chunks {
                let mut sd: MatMulDims = d.clone();
                sd.n = n1 - n0;
                sd.b.offset = skip as usize;
                sd.out.offset += n0;
                for v in &mut sd.extra_views {
                    let s = v.strides.get(2).copied().unwrap_or(0);
                    v.offset = (v.offset as isize + n0 as isize * s) as usize;
                }
                let mut ranges = vec![None; n_in];
                ranges[1] = Some((off, round4(len, size - off)));
                subs.push(SubLaunch { plan: crate::plan(&Kernel::MatMul(spec.clone(), sd), caps)?, ranges });
            }
            Ok(Some(subs))
        }
        Kernel::Gather(spec, d) if big == 0 && others_fit => {
            let chunks = row_bytes(spec.table_dtype, d.cols)
                .and_then(|rb| row_chunks(d.rows, rb, 0, limit, align, 0))
                .ok_or_else(|| too_large("gather table", size, limit))?;
            let mut subs = Vec::with_capacity(chunks.len());
            for (r0, r1, off, len, _) in chunks {
                let sd = GatherDims { rows: r1 - r0, cols: d.cols, count: d.count };
                let plan = crate::gather::plan_part(spec, &sd, r0, d.rows, caps)?;
                subs.push(SubLaunch { plan, ranges: vec![Some((off, round4(len, size - off))), None] });
            }
            Ok(Some(subs))
        }
        _ => Err(too_large(kernel.kind(), size, limit)),
    }
}
