//! Prompt preparation: image placeholder expansion (`<|image_pad|>` → one token per merged
//! vision token) and M-RoPE positions (HF `get_rope_index`): text tokens advance (t, h, w)
//! together; an image of `H x W` merged tokens placed at cursor `c` gets
//! (c, c + row, c + col) and moves the cursor to `c + max(H, W)`.

use lapid::{Error, Result};

use super::config::Qwen3VlConfig;
use super::model::Pos3;
use crate::vision::{Image, Patches};

/// One image of a prepared prompt.
pub struct PreparedImage {
    /// Index of its first placeholder token.
    pub start: usize,
    /// Placeholder tokens (merged vision tokens).
    pub tokens: usize,
    pub patches: Patches,
    /// Fingerprint of the source picture (session reuse).
    pub hash: u64,
}

/// Expanded token ids, their M-RoPE positions and the images to encode.
pub struct Prepared {
    pub ids: Vec<u32>,
    pub positions: Vec<Pos3>,
    pub images: Vec<PreparedImage>,
}

/// FNV-1a over size + pixels.
pub fn image_hash(img: &Image) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: u8| {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    };
    for b in (img.width as u64).to_le_bytes().into_iter().chain((img.height as u64).to_le_bytes()) {
        eat(b);
    }
    for &b in &img.rgb {
        eat(b);
    }
    h
}

impl Prepared {
    /// From rendered-and-encoded ids. Each `<|image_pad|>` run (one token as rendered, or
    /// already expanded to the merged token count) takes the next image of `images`.
    pub fn new(cfg: &Qwen3VlConfig, ids: &[u32], images: Vec<(Patches, u64)>) -> Result<Prepared> {
        let m = cfg.vision.spatial_merge_size;
        let pad = cfg.image_token_id;
        let mut out = Vec::with_capacity(ids.len());
        let mut positions = Vec::with_capacity(ids.len());
        let mut prepared = Vec::new();
        let mut it = images.into_iter();
        let mut cursor = 0usize;
        let mut i = 0;
        while i < ids.len() {
            if ids[i] != pad {
                out.push(ids[i]);
                positions.push([cursor; 3]);
                cursor += 1;
                i += 1;
                continue;
            }
            let run = ids[i..].iter().take_while(|&&t| t == pad).count();
            let (patches, hash) = it
                .next()
                .ok_or_else(|| Error::new("invalid_argument", "qwen3vl: more image placeholders than images"))?;
            let (gh, gw) = (patches.grid_h / m, patches.grid_w / m);
            let n = gh * gw;
            if run != 1 && run != n {
                return Err(Error::new("invalid_argument", format!("qwen3vl: image placeholder run of {run} tokens for an image of {n}")));
            }
            prepared.push(PreparedImage { start: out.len(), tokens: n, patches, hash });
            for r in 0..gh {
                for c in 0..gw {
                    out.push(pad);
                    positions.push([cursor, cursor + r, cursor + c]);
                }
            }
            cursor += gh.max(gw);
            i += run;
        }
        if it.next().is_some() {
            return Err(Error::new("invalid_argument", "qwen3vl: more images than image placeholders in the prompt"));
        }
        Ok(Prepared { ids: out, positions, images: prepared })
    }
}
