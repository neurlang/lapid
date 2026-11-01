//! Strided-view algebra (pure functions: no buffers, no launches).
//!
//! A `View` maps an index of some logical shape to `offset + sum(idx[d] * strides[d])`.
//! Strides produced by the lowering are never negative (there is no flip op).

use crate::ir::View;

pub(crate) fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Dense row-major over `shape` (ignoring the offset). Size-1 dims may have any stride.
pub(crate) fn is_dense(view: &View, shape: &[usize]) -> bool {
    if view.strides.len() != shape.len() {
        return false;
    }
    if numel(shape) == 0 {
        return true;
    }
    let mut expect: isize = 1;
    for d in (0..shape.len()).rev() {
        if shape[d] != 1 && view.strides[d] != expect {
            return false;
        }
        expect *= shape[d] as isize;
    }
    true
}

/// Dense row-major starting at element 0 of the buffer.
pub(crate) fn is_contiguous(view: &View, shape: &[usize]) -> bool {
    view.offset == 0 && is_dense(view, shape)
}

/// Output dim `i` reads input dim `perm[i]`.
pub(crate) fn permute(view: &View, perm: &[usize]) -> Option<View> {
    if perm.len() != view.strides.len() {
        return None;
    }
    let strides = perm.iter().map(|&p| view.strides.get(p).copied()).collect::<Option<Vec<_>>>()?;
    Some(View { offset: view.offset, strides })
}

/// Numpy broadcasting of a view over `src` to `dst` (right-aligned; new and size-1 dims
/// get stride 0). Also implements `Op::Expand` (equal ranks).
pub(crate) fn broadcast_to(view: &View, src: &[usize], dst: &[usize]) -> Option<View> {
    if src.len() > dst.len() || view.strides.len() != src.len() {
        return None;
    }
    let lead = dst.len() - src.len();
    let mut strides = vec![0isize; dst.len()];
    for (i, &s) in src.iter().enumerate() {
        if s == dst[lead + i] {
            strides[lead + i] = view.strides[i];
        } else if s != 1 {
            return None;
        }
    }
    Some(View { offset: view.offset, strides })
}

/// `[start, ..)` along `axis`.
pub(crate) fn slice(view: &View, axis: usize, start: usize) -> Option<View> {
    let st = *view.strides.get(axis)?;
    let off = view.offset as isize + st * start as isize;
    Some(View { offset: usize::try_from(off).ok()?, strides: view.strides.clone() })
}

/// Re-express a view over `src` as a view over `dst` (same numel) without copying, if
/// the memory order allows it (numpy's no-copy reshape rule).
pub(crate) fn reshape(view: &View, src: &[usize], dst: &[usize]) -> Option<View> {
    if numel(src) != numel(dst) || view.strides.len() != src.len() {
        return None;
    }
    if numel(dst) == 0 {
        return Some(View { offset: view.offset, strides: View::contiguous(dst).strides });
    }
    // Non-1 source dims with their strides.
    let old: Vec<(usize, isize)> = src
        .iter()
        .zip(&view.strides)
        .filter(|(s, _)| **s != 1)
        .map(|(s, st)| (*s, *st))
        .collect();
    let mut strides = vec![0isize; dst.len()];
    let mut oi = 0usize;
    let mut ni = 0usize;
    while ni < dst.len() {
        if dst[ni] == 1 {
            ni += 1;
            continue;
        }
        // Smallest groups dst[ni..nj] / old[oi..oj] with equal products.
        let mut np = dst[ni];
        let mut nj = ni + 1;
        let mut op = old.get(oi)?.0;
        let mut oj = oi + 1;
        while np != op {
            if np < op {
                np *= *dst.get(nj)?;
                nj += 1;
            } else {
                op *= old.get(oj)?.0;
                oj += 1;
            }
        }
        // The source dims of the group must be mutually contiguous.
        for k in oi..oj - 1 {
            if old[k].1 != old[k + 1].1 * old[k + 1].0 as isize {
                return None;
            }
        }
        let mut st = old[oj - 1].1;
        for k in (ni..nj).rev() {
            strides[k] = st;
            st *= dst[k] as isize;
        }
        ni = nj;
        oi = oj;
    }
    if oi != old.len() {
        return None;
    }
    Some(View { offset: view.offset, strides })
}

/// A permutation `perm` such that `permute(view, perm)` is contiguous over the permuted
/// shape, i.e. the view covers its buffer densely from offset 0 in some dim order.
pub(crate) fn dense_perm(view: &View, shape: &[usize]) -> Option<Vec<usize>> {
    if view.offset != 0 || view.strides.len() != shape.len() {
        return None;
    }
    let identity: Vec<usize> = (0..shape.len()).collect();
    if is_dense(view, shape) {
        return Some(identity);
    }
    let mut perm = identity;
    perm.sort_by(|&a, &b| view.strides[b].cmp(&view.strides[a]));
    let pv = permute(view, &perm)?;
    let ps: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    if is_dense(&pv, &ps) { Some(perm) } else { None }
}

/// Drop size-1 dims and merge adjacent dims that are mutually contiguous in every stride
/// list. Returns the new shape (rank >= 1) and the matching stride lists.
pub(crate) fn coalesce(shape: &[usize], strides: &[Vec<isize>]) -> (Vec<usize>, Vec<Vec<isize>>) {
    let mut cs: Vec<usize> = Vec::with_capacity(shape.len());
    let mut cst: Vec<Vec<isize>> = vec![Vec::with_capacity(shape.len()); strides.len()];
    for (d, &size) in shape.iter().enumerate() {
        if size == 1 {
            continue;
        }
        let merge = !cs.is_empty()
            && strides
                .iter()
                .zip(&cst)
                .all(|(s, c)| c.last().is_some_and(|&prev| prev == s[d] * size as isize));
        if merge {
            if let Some(last) = cs.last_mut() {
                *last *= size;
            }
            for (s, c) in strides.iter().zip(cst.iter_mut()) {
                if let Some(l) = c.last_mut() {
                    *l = s[d];
                }
            }
        } else {
            cs.push(size);
            for (s, c) in strides.iter().zip(cst.iter_mut()) {
                c.push(s[d]);
            }
        }
    }
    if cs.is_empty() {
        cs.push(1);
        for c in cst.iter_mut() {
            c.push(0);
        }
    }
    (cs, cst)
}

/// One stride that walks the row-major flattening of the batch dims, if one exists
/// (size-1 dims are ignored; all-broadcast gives 0).
pub(crate) fn flatten_batch(shape: &[usize], strides: &[isize]) -> Option<usize> {
    if shape.len() != strides.len() {
        return None;
    }
    let dims: Vec<(usize, isize)> =
        shape.iter().zip(strides).filter(|(s, _)| **s != 1).map(|(s, st)| (*s, *st)).collect();
    for w in dims.windows(2) {
        if w[0].1 != w[1].1 * w[1].0 as isize {
            return None;
        }
    }
    match dims.last() {
        Some(&(_, st)) => usize::try_from(st).ok(),
        None => Some(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(offset: usize, strides: &[isize]) -> View {
        View { offset, strides: strides.to_vec() }
    }

    #[test]
    fn reshape_contiguous_merges_and_splits() {
        let c = View::contiguous(&[2, 3, 4]);
        assert_eq!(reshape(&c, &[2, 3, 4], &[6, 4]), Some(v(0, &[4, 1])));
        assert_eq!(reshape(&c, &[2, 3, 4], &[2, 12]), Some(v(0, &[12, 1])));
        assert_eq!(reshape(&c, &[2, 3, 4], &[1, 24, 1]).map(|x| x.strides[1]), Some(1));
    }

    #[test]
    fn reshape_of_transpose_needs_copy() {
        // [3, 2] transposed view of a contiguous [2, 3].
        assert_eq!(reshape(&v(0, &[1, 3]), &[3, 2], &[6]), None);
        // Splitting a dim of a transposed view is fine.
        assert_eq!(reshape(&v(0, &[1, 3]), &[4, 2], &[2, 2, 2]), Some(v(0, &[2, 1, 3])));
    }

    #[test]
    fn reshape_keeps_broadcast_dims() {
        assert_eq!(reshape(&v(5, &[1, 0]), &[4, 5], &[2, 2, 5]), Some(v(5, &[2, 1, 0])));
    }

    #[test]
    fn broadcast_and_slice() {
        let b = broadcast_to(&v(0, &[1]), &[8], &[4, 8]);
        assert_eq!(b, Some(v(0, &[0, 1])));
        assert_eq!(broadcast_to(&v(0, &[1]), &[3], &[4, 8]), None);
        assert_eq!(slice(&v(2, &[8, 1]), 0, 3), Some(v(26, &[8, 1])));
    }

    #[test]
    fn coalesce_merges_dense_dims() {
        let (s, st) = coalesce(&[2, 3, 4], &[vec![12, 4, 1], vec![0, 4, 1]]);
        assert_eq!(s, vec![2, 12]);
        assert_eq!(st, vec![vec![12, 1], vec![0, 1]]);
        let (s, st) = coalesce(&[1, 1], &[vec![5, 7]]);
        assert_eq!(s, vec![1]);
        assert_eq!(st, vec![vec![0]]);
    }

    #[test]
    fn dense_perm_finds_storage_order() {
        // [B=1, H=2, S=3, D=4] view of contiguous [1, 3, 2, 4].
        let q = v(0, &[24, 4, 8, 1]);
        let p = dense_perm(&q, &[1, 2, 3, 4]);
        assert_eq!(p, Some(vec![0, 2, 1, 3]));
        assert_eq!(dense_perm(&v(0, &[0, 1]), &[4, 8]), None);
    }

    #[test]
    fn batch_flattening() {
        assert_eq!(flatten_batch(&[2, 3], &[12, 4]), Some(4));
        assert_eq!(flatten_batch(&[2, 3], &[0, 0]), Some(0));
        assert_eq!(flatten_batch(&[1, 3], &[99, 4]), Some(4));
        assert_eq!(flatten_batch(&[2, 3], &[4, 8]), None);
        assert_eq!(flatten_batch(&[], &[]), Some(0));
    }
}
