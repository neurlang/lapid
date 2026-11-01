//! Attention mask of a decoder whose image rows attend to each other bidirectionally (Gemma 3
//! / Gemma 4 `use_bidirectional_attention = "vision"`, PaliGemma-style prefix blocks), with
//! an optional sliding window.

use lapid::{Result, Tensor};

/// Additive [n, sk] mask for the last `n` of `sk` positions: key j visible to query i iff in
/// the window and (`j <= i` or both rows are in the same image block). `blocks[i]` is the
/// block of chunk row i (-1 = text); `None` = plain causal.
pub fn image_block_mask(blocks: Option<&[i32]>, n: usize, sk: usize, window: Option<usize>) -> Result<Tensor> {
    let off = sk - n;
    let mut m = vec![0.0f32; n * sk];
    for i in 0..n {
        let qpos = i + off;
        for j in 0..sk {
            let same = blocks.is_some_and(|b| j >= off && b[i] >= 0 && b[j - off] == b[i]);
            let in_window = window.is_none_or(|w| j > qpos || qpos - j < w);
            if !((j <= qpos || same) && in_window) {
                m[i * sk + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_f32(&m, &[n, sk])
}

#[cfg(test)]
mod tests {
    #[test]
    fn image_block_mask() {
        let dev = lapid::Device::cpu();
        let m = super::image_block_mask(Some(&[-1, 0, 0]), 3, 4, Some(8)).expect("mask");
        let v = dev.to_vec_f32(&dev.eval(&[&m]).expect("eval")[0]).expect("host");
        // query row 1 (image) sees key 3 (its image, later); row 0 (text) does not.
        assert_eq!((v[4 + 3], v[3]), (0.0, f32::NEG_INFINITY));
    }
}
