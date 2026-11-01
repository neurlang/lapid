//! Weight transforms converters need, as plain functions on row-major data: row orders
//! (applied to raw rows, so quantized blocks stay intact), BatchNorm folding and splitting
//! fused tensors.

use lapid_tensor::{Result, bail};

/// Source row of every output row that undoes llama.cpp's q/k permutation
/// (`LlamaModel.permute`): GGUF row `h*Dh + 2i + s` holds HF row `h*Dh + s*Dh/2 + i`.
pub fn llama_qk_unpermute(rows: usize, head_dim: usize) -> Vec<usize> {
    let half = (head_dim / 2).max(1);
    (0..rows)
        .map(|r| {
            let (h, j) = (r / head_dim, r % head_dim);
            let (s, i) = (j / half, j % half);
            h * head_dim + 2 * i + s
        })
        .collect()
}

/// Source row of every output row when groups of `group` rows are reordered:
/// output group `g` takes source group `order[g]` (e.g. reordering value heads).
pub fn group_order(order: &[usize], group: usize) -> Vec<usize> {
    order.iter().flat_map(|&g| (g * group)..(g * group + group)).collect()
}

/// Gather rows: output row `r` = input row `order[r]` (rows of `data.len() / n_rows` elements
/// or bytes).
pub fn gather_rows<T: Copy>(data: &[T], order: &[usize]) -> Result<Vec<T>> {
    let n = order.len();
    if n == 0 || !data.len().is_multiple_of(n) {
        bail!("shape_mismatch", "gather_rows: {} elements for {n} rows", data.len());
    }
    let row = data.len() / n;
    let mut out = Vec::with_capacity(data.len());
    for &src in order {
        if src >= n {
            bail!("shape_mismatch", "gather_rows: source row {src} of {n}");
        }
        out.extend_from_slice(&data[src * row..(src + 1) * row]);
    }
    Ok(out)
}

/// Fold an inference BatchNorm that follows a convolution / linear layer into its weight
/// `w` (`[out, ...]`) and bias: `w' = w * g / sqrt(var + eps)`, `b' = (b - mean) * g / sqrt(var + eps) + beta`
/// (computed in f64).
pub fn fold_batchnorm(
    w: &[f32],
    b: Option<&[f32]>,
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let c = gamma.len();
    if c == 0 || beta.len() != c || mean.len() != c || var.len() != c || !w.len().is_multiple_of(c) || b.is_some_and(|b| b.len() != c) {
        bail!("shape_mismatch", "fold_batchnorm: inconsistent channel counts");
    }
    let per = w.len() / c;
    let mut w2 = Vec::with_capacity(w.len());
    let mut b2 = Vec::with_capacity(c);
    for o in 0..c {
        let s = f64::from(gamma[o]) / (f64::from(var[o]) + eps).sqrt();
        w2.extend(w[o * per..(o + 1) * per].iter().map(|&x| (f64::from(x) * s) as f32));
        b2.push(((b.map_or(0.0, |b| f64::from(b[o])) - f64::from(mean[o])) * s + f64::from(beta[o])) as f32);
    }
    Ok((w2, b2))
}

/// Split a fused tensor along its first axis into parts of `sizes` rows.
pub fn split_rows(data: &[f32], shape: &[usize], sizes: &[usize]) -> Result<Vec<(Vec<usize>, Vec<f32>)>> {
    let Some(&rows) = shape.first() else { bail!("shape_mismatch", "split_rows: scalar") };
    if sizes.iter().sum::<usize>() != rows || data.len() != shape.iter().product::<usize>() {
        bail!("shape_mismatch", "split_rows: {sizes:?} rows of {shape:?}");
    }
    let per: usize = shape[1..].iter().product();
    let mut out = Vec::with_capacity(sizes.len());
    let mut at = 0;
    for &n in sizes {
        let mut s = shape.to_vec();
        s[0] = n;
        out.push((s, data[at * per..(at + n) * per].to_vec()));
        at += n;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpermute() {
        // head_dim 4: GGUF rows [0, 2, 1, 3] interleave the halves.
        assert_eq!(llama_qk_unpermute(8, 4), vec![0, 2, 1, 3, 4, 6, 5, 7]);
        assert_eq!(gather_rows(&[10, 11, 12, 13], &[0, 2, 1, 3]).expect("rows"), vec![10, 12, 11, 13]);
        assert_eq!(group_order(&[1, 0], 2), vec![2, 3, 0, 1]);
    }

    #[test]
    fn batchnorm_and_split() {
        let (w, b) = fold_batchnorm(&[1.0, 2.0, 3.0, 4.0], Some(&[1.0, 0.0]), &[2.0, 1.0], &[0.5, 0.0], &[1.0, 0.0], &[3.0, 0.0], 1.0)
            .expect("fold");
        assert_eq!(w, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(b, vec![0.5, 0.0]);
        let parts = split_rows(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &[1, 2]).expect("split");
        assert_eq!(parts[1], (vec![2, 2], vec![3.0, 4.0, 5.0, 6.0]));
    }
}
