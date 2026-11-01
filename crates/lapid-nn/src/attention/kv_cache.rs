//! Self-attention KV cache.
//!
//! One decoding step is *stage → eval → commit*:
//!
//! ```text
//! let (k, v) = cache.append(layer, &k_new, &v_new)?;   // per layer, while building the graph
//! let mut outs = vec![&logits];
//! outs.extend(cache.staged());                          // updated cache buffers ride along
//! let mut realized = device.eval(&outs)?;
//! cache.commit(realized.split_off(1))?;                 // store realized K/V, advance offset
//! ```
//!
//! `append` returns the K/V the new tokens attend to (cache + new, `[B, Hkv, Sk, Dh]`).
//! Storage: per layer one preallocated buffer `[B, capacity, Hkv, Dh]` (sequence-major, so a
//! K/V projection writes its rows straight into the cache slice). New rows are written with
//! [`Tensor::write_slice`], which updates the realized buffer in place — no copy of the
//! cache per step — and attention reads the valid range `[start, end)` through a view. The
//! buffer grows (doubling, one copy) when full. Layers with a sliding window keep only their
//! last `window` positions; their buffer is compacted when it runs out of room.

use lapid::{Result, Tensor, bail};

/// Default first capacity (positions) of a layer buffer.
const MIN_CAPACITY: usize = 256;

#[derive(Clone)]
struct Layer {
    /// Realized `[B, cap, H, D]` buffers.
    k: Tensor,
    v: Tensor,
    /// Valid positions `[start, end)`.
    start: usize,
    end: usize,
    /// Committed K/V as `[B, H, end - start, D]` views (for `get`).
    view_k: Tensor,
    view_v: Tensor,
}

struct Staged {
    k: Tensor,
    v: Tensor,
    start: usize,
    end: usize,
}

pub struct KvCache {
    layers: Vec<Option<Layer>>,
    windows: Vec<Option<usize>>,
    staged: Vec<Option<Staged>>,
    staged_len: Option<usize>,
    offset: usize,
    /// First capacity of a layer buffer (0 = automatic).
    capacity: usize,
}

/// `[B, cap, H, D]` positions `[start, end)` as `[B, H, n, D]`.
fn heads_view(buf: &Tensor, start: usize, end: usize) -> Result<Tensor> {
    buf.slice(1, start, end)?.permute(&[0, 2, 1, 3])
}

impl KvCache {
    /// Unbounded cache for `n_layers` layers.
    pub fn new(n_layers: usize) -> KvCache {
        KvCache::with_windows(vec![None; n_layers])
    }

    /// Per-layer window: `Some(w)` keeps the last `w` positions (sliding-window layers),
    /// `None` keeps everything.
    pub fn with_windows(windows: Vec<Option<usize>>) -> KvCache {
        let n = windows.len();
        KvCache {
            layers: vec![None; n],
            windows,
            staged: (0..n).map(|_| None).collect(),
            staged_len: None,
            offset: 0,
            capacity: 0,
        }
    }

    /// Like [`with_windows`](Self::with_windows), with layer buffers preallocated for
    /// `capacity` positions (e.g. the model's max context), so they never grow.
    pub fn with_capacity(windows: Vec<Option<usize>>, capacity: usize) -> KvCache {
        let mut c = KvCache::with_windows(windows);
        c.capacity = capacity;
        c
    }

    pub fn n_layers(&self) -> usize {
        self.windows.len()
    }

    /// Absolute position of the next token (tokens committed so far).
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Committed positions held for `layer` (<= window).
    pub fn len(&self, layer: usize) -> usize {
        self.layers.get(layer).and_then(|l| l.as_ref()).map_or(0, |l| l.end - l.start)
    }

    pub fn is_empty(&self) -> bool {
        self.offset == 0
    }

    /// Committed K and V of `layer` as `[B, Hkv, len, Dh]` views, if any.
    pub fn get(&self, layer: usize) -> Option<(&Tensor, &Tensor)> {
        self.layers.get(layer)?.as_ref().map(|l| (&l.view_k, &l.view_v))
    }

    fn grown_capacity(&self, need: usize) -> usize {
        if self.capacity >= need {
            return self.capacity;
        }
        (2 * need).next_power_of_two().max(MIN_CAPACITY)
    }

    /// Stage new K/V [B, Hkv, n, Dh] for `layer`; returns the full K/V to attend to.
    pub fn append(&mut self, layer: usize, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let n_layers = self.n_layers();
        if layer >= n_layers {
            bail!("invalid_argument", "kv cache: layer {layer} out of range ({n_layers} layers)");
        }
        if k.rank() != 4 || k.shape() != v.shape() {
            bail!("shape_mismatch", "kv cache: k {:?} and v {:?} must both be [B, Hkv, n, Dh]", k.shape(), v.shape());
        }
        if self.staged[layer].is_some() {
            bail!("invalid_argument", "kv cache: layer {layer} already staged in this step");
        }
        let n = k.shape()[2];
        match self.staged_len {
            Some(m) if m != n => bail!("shape_mismatch", "kv cache: layer {layer} stages {n} positions, others {m}"),
            _ => self.staged_len = Some(n),
        }
        let (b, h, d) = (k.shape()[0], k.shape()[1], k.shape()[3]);
        // Existing buffers (dtype / head layout must match) or fresh ones.
        let old = self.layers[layer].clone().filter(|l| {
            let s = l.k.shape();
            s[0] == b && s[2] == h && s[3] == d
        });
        let dtype = old.as_ref().map_or(k.dtype(), |l| l.k.dtype());
        let to_cache = |t: &Tensor| -> Result<Tensor> { t.cast(dtype)?.permute(&[0, 2, 1, 3]) };
        let (kc, vc) = (to_cache(k)?, to_cache(v)?);

        let (bk, bv, start, end) = match old {
            Some(l) if l.end + n <= l.k.shape()[1] => (l.k, l.v, l.start, l.end),
            Some(l) => {
                // Full: copy the valid range to the front of a (possibly larger) buffer.
                let len = l.end - l.start;
                let cap = self.grown_capacity(len + n).max(l.k.shape()[1]);
                let fresh = |buf: &Tensor| -> Result<Tensor> {
                    Tensor::zeros(&[b, cap, h, d], dtype).write_slice(1, 0, &buf.slice(1, l.start, l.end)?)
                };
                (fresh(&l.k)?, fresh(&l.v)?, 0, len)
            }
            None => {
                let cap = self.grown_capacity(n);
                let z = Tensor::zeros(&[b, cap, h, d], dtype);
                (z.clone(), z, 0, 0)
            }
        };
        let nk = bk.write_slice(1, end, &kc)?;
        let nv = bv.write_slice(1, end, &vc)?;
        let full_k = heads_view(&nk, start, end + n)?;
        let full_v = heads_view(&nv, start, end + n)?;
        let keep_from = match self.windows[layer] {
            Some(w) => start.max((end + n).saturating_sub(w)),
            None => start,
        };
        self.staged[layer] = Some(Staged { k: nk, v: nv, start: keep_from, end: end + n });
        Ok((full_k, full_v))
    }

    /// Tensors to evaluate with the step's outputs: for every staged layer in order, the
    /// updated K then V buffers (`[B, capacity, Hkv, Dh]`; updated in place on the device).
    pub fn staged(&self) -> Vec<&Tensor> {
        self.staged.iter().flatten().flat_map(|s| [&s.k, &s.v]).collect()
    }

    /// Store the realized tensors (same order as [`staged`](Self::staged)) and advance the
    /// offset by the staged token count.
    pub fn commit(&mut self, realized: Vec<Tensor>) -> Result<()> {
        let n_staged = self.staged.iter().flatten().count();
        if realized.len() != 2 * n_staged {
            bail!("invalid_argument", "kv cache commit: {} tensors for {n_staged} staged layers", realized.len());
        }
        let mut it = realized.into_iter();
        for layer in 0..self.n_layers() {
            let Some(s) = self.staged[layer].take() else { continue };
            let (Some(k), Some(v)) = (it.next(), it.next()) else {
                bail!("internal", "kv cache commit: missing tensors");
            };
            if k.shape() != s.k.shape() || v.shape() != s.v.shape() {
                bail!("shape_mismatch", "kv cache commit: realized {:?} for staged {:?}", k.shape(), s.k.shape());
            }
            let view_k = heads_view(&k, s.start, s.end)?;
            let view_v = heads_view(&v, s.start, s.end)?;
            self.layers[layer] = Some(Layer { k, v, start: s.start, end: s.end, view_k, view_v });
        }
        self.offset += self.staged_len.take().unwrap_or(0);
        Ok(())
    }

    /// Drop staged tensors (e.g. after a failed eval); the committed cache is unchanged.
    pub fn discard(&mut self) {
        self.staged.iter_mut().for_each(|s| *s = None);
        self.staged_len = None;
    }

    /// Forget everything.
    pub fn clear(&mut self) {
        let cap = self.capacity;
        *self = KvCache::with_windows(std::mem::take(&mut self.windows));
        self.capacity = cap;
    }
}

#[cfg(test)]
mod tests {
    use super::KvCache;
    use lapid::{Device, Op, Tensor};

    fn kv(n: usize, seed: f32) -> Tensor {
        let v: Vec<f32> = (0..2 * n * 4).map(|i| seed + i as f32).collect();
        Tensor::from_f32(&v, &[1, 2, n, 4]).unwrap()
    }

    fn buf_ptr(t: &Tensor) -> usize {
        match t.op() {
            Op::Device(d) => std::sync::Arc::as_ptr(d) as *const () as usize,
            _ => 0,
        }
    }

    /// Steps update one device buffer in place; growth copies once; values stay in order.
    #[test]
    fn in_place_steps_and_growth() {
        let dev = Device::cpu();
        let mut c = KvCache::with_capacity(vec![None], 4);
        let mut all: Vec<Tensor> = Vec::new();
        let mut ptrs = Vec::new();
        for (step, n) in [2usize, 1, 1, 3, 1].into_iter().enumerate() {
            let k = kv(n, 100.0 * step as f32);
            all.push(k.clone());
            let (fk, _) = c.append(0, &k, &k).unwrap();
            let mut outs = vec![&fk];
            outs.extend(c.staged());
            let mut r = dev.eval(&outs).unwrap();
            let rest = r.split_off(1);
            ptrs.push(buf_ptr(&rest[0]));
            c.commit(rest).unwrap();
            let want = dev.to_vec_f32(&Tensor::cat(&all.iter().collect::<Vec<_>>(), 2).unwrap()).unwrap();
            assert_eq!(dev.to_vec_f32(&r[0]).unwrap(), want, "step {step}");
            assert_eq!(dev.to_vec_f32(c.get(0).unwrap().0).unwrap(), want);
        }
        // Steps 1 and 2 write into the step-0 buffer (capacity 4); step 3 grows.
        assert_eq!(ptrs[0], ptrs[1]);
        assert_eq!(ptrs[1], ptrs[2]);
        assert_ne!(ptrs[2], ptrs[3]);
        assert_eq!(ptrs[3], ptrs[4]);
        assert_eq!(c.len(0), 8);
    }
}

#[cfg(test)]
mod growth_tests {
    use super::KvCache;
    use lapid::{Device, Tensor};

    /// Several layers growing their buffers in the same step (each growth is a fresh copy plus
    /// an in-place write): the realized buffers must be outputs, not pooled temps.
    #[test]
    fn multi_layer_growth_stays_correct() {
        let dev = Device::cpu();
        let layers = 3;
        let mut c = KvCache::with_capacity(vec![None; layers], 4);
        let mut hist: Vec<Vec<Tensor>> = vec![Vec::new(); layers];
        for step in 0..9 {
            let n = 1 + step % 3;
            let mut fulls = Vec::new();
            for (l, h) in hist.iter_mut().enumerate() {
                let v: Vec<f32> = (0..2 * n * 4).map(|i| (step * 1000 + l * 100 + i) as f32).collect();
                let k = Tensor::from_f32(&v, &[1, 2, n, 4]).unwrap();
                h.push(k.clone());
                let (fk, fv) = c.append(l, &k, &k.neg().unwrap()).unwrap();
                fulls.push(fk);
                fulls.push(fv);
            }
            let mut outs: Vec<&Tensor> = fulls.iter().collect();
            outs.extend(c.staged());
            let mut r = dev.eval(&outs).unwrap();
            let rest = r.split_off(fulls.len());
            c.commit(rest).unwrap();
            // A throwaway eval that recycles pooled temps.
            let junk = Tensor::from_f32(&vec![7.0; 4096], &[4096]).unwrap().exp().unwrap().sqrt().unwrap();
            dev.eval(&[&junk.mul_scalar(2.0).unwrap()]).unwrap();
            for (l, h) in hist.iter().enumerate() {
                let want = dev.to_vec_f32(&Tensor::cat(&h.iter().collect::<Vec<_>>(), 2).unwrap()).unwrap();
                assert_eq!(dev.to_vec_f32(&r[2 * l]).unwrap(), want, "step {step} layer {l} (returned)");
                let (ck, cv) = c.get(l).unwrap();
                assert_eq!(dev.to_vec_f32(ck).unwrap(), want, "step {step} layer {l} (committed k)");
                let neg: Vec<f32> = want.iter().map(|x| -x).collect();
                assert_eq!(dev.to_vec_f32(cv).unwrap(), neg, "step {step} layer {l} (committed v)");
            }
        }
    }
}
