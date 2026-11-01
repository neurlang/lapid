//! Shape/stride utilities: view validation, dimension collapsing and nd offset iteration.

use lapid_compiler::View;
use lapid_tensor::{Error, Result};

/// Validate that every element of `view` over `shape` lies inside a buffer of `len` elements.
pub(crate) fn check_view(what: &str, shape: &[usize], view: &View, len: usize) -> Result<()> {
    if view.strides.len() != shape.len() {
        return Err(Error::new(
            "invalid_launch",
            format!("cpu: {what} view has rank {} but iteration space has rank {}", view.strides.len(), shape.len()),
        ));
    }
    check_strided(what, shape, view.offset, &view.strides, len)
}

pub(crate) fn check_strided(what: &str, shape: &[usize], offset: usize, strides: &[isize], len: usize) -> Result<()> {
    if shape.contains(&0) {
        return Ok(());
    }
    let mut lo = offset as i128;
    let mut hi = offset as i128;
    for (&n, &s) in shape.iter().zip(strides) {
        let ext = (n as i128 - 1) * s as i128;
        if ext < 0 {
            lo += ext;
        } else {
            hi += ext;
        }
    }
    if lo < 0 || hi >= len as i128 {
        return Err(Error::new(
            "out_of_bounds",
            format!("cpu: {what} view (offset {offset}, strides {strides:?}, shape {shape:?}) exceeds buffer of {len} elements"),
        ));
    }
    Ok(())
}

/// Collapsed iteration space: size-1 dims removed and adjacent dims merged when every view
/// is linear across them. Always rank >= 1.
#[derive(Clone, Debug)]
pub(crate) struct Collapsed {
    pub shape: Vec<usize>,
    /// strides[v][d] for view v.
    pub strides: Vec<Vec<isize>>,
}

pub(crate) fn collapse(shape: &[usize], views: &[&[isize]]) -> Collapsed {
    // Build from the innermost dimension outwards.
    let mut dims: Vec<(usize, Vec<isize>)> = Vec::with_capacity(shape.len());
    for d in (0..shape.len()).rev() {
        let n = shape[d];
        if n == 1 {
            continue;
        }
        if let Some((inner_n, inner_s)) = dims.last_mut() {
            let mergeable = views
                .iter()
                .zip(inner_s.iter())
                .all(|(v, &is)| v[d] == is * *inner_n as isize);
            if mergeable {
                *inner_n *= n;
                continue;
            }
        }
        dims.push((n, views.iter().map(|v| v[d]).collect()));
    }
    if dims.is_empty() {
        dims.push((1, vec![0; views.len()]));
    }
    dims.reverse();
    let shape: Vec<usize> = dims.iter().map(|(n, _)| *n).collect();
    let strides = (0..views.len()).map(|v| dims.iter().map(|(_, s)| s[v]).collect()).collect();
    Collapsed { shape, strides }
}

/// Offset contribution of linear index `lin` over `shape` with `strides`.
#[inline]
pub(crate) fn offset_of(mut lin: usize, shape: &[usize], strides: &[isize]) -> isize {
    let mut off = 0isize;
    for d in (0..shape.len()).rev() {
        let n = shape[d];
        let i = lin % n;
        lin /= n;
        off += i as isize * strides[d];
    }
    off
}

/// Fill `out[v][t]` with the element offset of view `v` at linear index `start + t`.
pub(crate) fn fill_offsets(
    start: usize,
    len: usize,
    shape: &[usize],
    bases: &[usize],
    strides: &[Vec<isize>],
    idx: &mut Vec<usize>,
    cur: &mut Vec<isize>,
    out: &mut [Vec<usize>],
) {
    let rank = shape.len();
    idx.clear();
    idx.resize(rank, 0);
    let mut lin = start;
    for d in (0..rank).rev() {
        idx[d] = lin % shape[d];
        lin /= shape[d];
    }
    cur.clear();
    for (v, st) in strides.iter().enumerate() {
        let mut o = bases[v] as isize;
        for d in 0..rank {
            o += idx[d] as isize * st[d];
        }
        cur.push(o);
    }
    for o in out.iter_mut() {
        o.clear();
    }
    for _ in 0..len {
        for (v, o) in out.iter_mut().enumerate() {
            o.push(cur[v] as usize);
        }
        // Increment the nd counter, updating every view's offset incrementally.
        let mut d = rank;
        while d > 0 {
            d -= 1;
            idx[d] += 1;
            for (v, st) in strides.iter().enumerate() {
                cur[v] += st[d];
            }
            if idx[d] < shape[d] {
                break;
            }
            for (v, st) in strides.iter().enumerate() {
                cur[v] -= st[d] * shape[d] as isize;
            }
            idx[d] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_merges_contiguous() {
        let c = collapse(&[2, 3, 4], &[&[12, 4, 1], &[0, 4, 1]]);
        assert_eq!(c.shape, vec![2, 12]);
        assert_eq!(c.strides, vec![vec![12, 1], vec![0, 1]]);
        let c = collapse(&[1, 1], &[&[5, 7]]);
        assert_eq!(c.shape, vec![1]);
        let c = collapse(&[3, 1, 4], &[&[4, 99, 1]]);
        assert_eq!(c.shape, vec![12]);
        assert_eq!(c.strides, vec![vec![1]]);
    }

    #[test]
    fn bounds() {
        let v = View { offset: 3, strides: vec![-1, 4] };
        assert!(check_view("x", &[4, 2], &v, 8).is_ok());
        assert!(check_view("x", &[5, 2], &v, 8).is_err());
        assert!(check_view("x", &[4, 3], &v, 8).is_err());
    }

    #[test]
    fn offsets_iterate_nd() {
        let shape = [2, 3];
        let strides = vec![vec![1isize, 2]];
        let mut out = vec![Vec::new()];
        let (mut idx, mut cur) = (Vec::new(), Vec::new());
        fill_offsets(1, 4, &shape, &[10], &strides, &mut idx, &mut cur, &mut out);
        // linear 1..5 -> (0,1),(0,2),(1,0),(1,1) -> 12, 14, 11, 13
        assert_eq!(out[0], vec![12, 14, 11, 13]);
        assert_eq!(offset_of(4, &shape, &strides[0]), 1 + 2);
    }
}
