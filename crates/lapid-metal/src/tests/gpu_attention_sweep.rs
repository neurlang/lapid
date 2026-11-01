//! Attention regression sweep against the naive f64 reference (`gpu_attention::check`):
//! causal / not, mask / none, sq = 1, sq = sk, sq < sk, sk not a multiple of the tile sizes,
//! d in {32, 64, 80, 128}, 1..12 heads, f32 and f16 storage, and KV-cache layouts (k / v
//! read from a [B, cap, H, D] buffer as [B, H, len, D], len < cap).

use lapid_tensor::DType;

use super::gpu_attention::{Case, check};
use super::util::backend;

#[test]
fn attention_shape_sweep() {
    let Some(be) = backend() else { return };
    let shapes: &[(usize, usize)] = &[(1, 1), (1, 77), (1, 300), (3, 50), (5, 5), (64, 64), (33, 97), (40, 200), (130, 130)];
    let mut n = 0;
    for &(sq, sk) in shapes {
        for d in [32usize, 64, 80, 128] {
            for (ci, &(causal, mask)) in [(false, false), (true, false), (false, true), (true, true)].iter().enumerate() {
                let h = [1usize, 3, 12, 2][(ci + d / 16) % 4];
                for dtype in [DType::F32, DType::F16] {
                    let bshd = (sq + d) % 3 == 0;
                    check(&be, &Case { b: 1 + (sk % 2), h, sq, sk, d, causal, mask, dtype, bshd, cap: 0 });
                    n += 1;
                }
            }
        }
    }
    assert!(n > 200);
}

#[test]
fn attention_causal_single_head_square() {
    // h = 1, sq = sk = 64, d = 64 causal (a CPU regression shape).
    let Some(be) = backend() else { return };
    for dtype in [DType::F32, DType::F16] {
        for mask in [false, true] {
            check(&be, &Case { b: 1, h: 1, sq: 64, sk: 64, d: 64, causal: true, mask, dtype, bshd: false, cap: 0 });
        }
    }
}

#[test]
fn attention_kv_cache_layout() {
    let Some(be) = backend() else { return };
    // (sq, len, cap): decode steps, a prompt prefix, len not a multiple of any tile size.
    for &(sq, len, cap) in &[(1usize, 1usize, 448usize), (1, 37, 448), (1, 129, 448), (1, 447, 448), (4, 200, 448), (12, 12, 64), (70, 70, 100), (33, 97, 128)] {
        for d in [64usize, 80] {
            for causal in [false, true] {
                for dtype in [DType::F32, DType::F16] {
                    check(&be, &Case { b: 2, h: 6, sq, sk: len, d, causal, mask: false, dtype, bshd: false, cap });
                }
            }
        }
    }
}
