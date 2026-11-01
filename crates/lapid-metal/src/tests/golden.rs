//! Shader text tests (no GPU): every key renders a complete module, and representative
//! kernels match the checked-in golden sources in `src/tests/golden/` byte for byte.
//! Regenerate after an intentional template change with `LAPID_BLESS=1 cargo test -p lapid-metal golden`.

use std::path::PathBuf;
use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::DType;

use super::keys::{all_keys, gelu_expr, gelu_plus, int_all_ops, load, mat_spec};
use crate::codegen::{self, Idx, Key};

#[test]
fn every_key_renders_completely() {
    for key in all_keys() {
        let src = codegen::source(&key);
        assert!(!src.contains("{{"), "{key:?}: unfilled slot\n{src}");
        assert!(!src.contains("// #if") && !src.contains("// #endif"), "{key:?}: unresolved conditional");
        assert!(src.contains("kernel void lapid_kernel("), "{key:?}: no entry point");
        assert!(src.starts_with("// lapid Metal prelude"), "{key:?}: no prelude");
        let opens = src.matches('{').count();
        let closes = src.matches('}').count();
        assert_eq!(opens, closes, "{key:?}: unbalanced braces\n{src}");
    }
}

fn golden_keys() -> Vec<(&'static str, Key)> {
    let bias_gelu = Some(Expr::Binary(lapid_tensor::BinaryOp::Add, Box::new(gelu_expr(0)), load(1)));
    vec![
        (
            "elementwise_strided",
            Key::Ew(
                Arc::new(EwSpec { rank: 3, in_dtypes: vec![DType::F16, DType::F32], out_dtype: DType::BF16, expr: gelu_plus(0, 1) }),
                vec![Idx::Strided, Idx::Scalar],
                Idx::Flat,
            ),
        ),
        (
            "reduce_row_max",
            Key::Reduce(
                Arc::new(ReduceSpec { op: lapid_tensor::ReduceOp::Max, rank: 2, in_dtypes: vec![DType::F32], out_dtype: DType::F32, expr: Expr::Load(0) }),
                true,
            ),
        ),
        ("gemm_int8_bias_gelu", Key::Gemm(mat_spec(DType::F32, DType::Int8, true, vec![DType::F32], bias_gelu.clone(), DType::F32))),
        ("gemv_int4_m1", Key::Gemv(mat_spec(DType::F32, DType::Int4, true, vec![DType::F32], bias_gelu, DType::F32), 1)),
        ("gemv_f16_m2", Key::Gemv(mat_spec(DType::F32, DType::F16, true, vec![], None, DType::F32), 2)),
        (
            "attention_tile_d64_causal",
            Key::AttnTile(Arc::new(AttentionSpec { dtype: DType::F32, mask_dtype: None, causal: true, out_dtype: DType::F32 }), 64),
        ),
        (
            "attention_vec_d64_split_mask",
            Key::AttnVec(Arc::new(AttentionSpec { dtype: DType::F32, mask_dtype: Some(DType::F32), causal: false, out_dtype: DType::F32 }), 64, false),
        ),
        (
            "elementwise_int",
            Key::Ew(
                Arc::new(EwSpec { rank: 2, in_dtypes: vec![DType::I32, DType::F32], out_dtype: DType::I32, expr: int_all_ops() }),
                vec![Idx::Strided, Idx::Flat],
                Idx::Flat,
            ),
        ),
        ("gather_axis_f16", Key::GatherAxis(Arc::new(GatherAxisSpec { dtype: DType::F16 }))),
        ("argmax_f32", Key::TopK(Arc::new(TopKSpec { in_dtype: DType::F32 }), true)),
        ("topk_f16", Key::TopK(Arc::new(TopKSpec { in_dtype: DType::F16 }), false)),
        ("gemv_q4k_m1", Key::Gemv(mat_spec(DType::F32, DType::Q4K, true, vec![], None, DType::F32), 1)),
        ("gemm_q6k", Key::Gemm(mat_spec(DType::F32, DType::Q6K, true, vec![], None, DType::F32))),
        (
            "gemm_id_int8",
            Key::GemmId(crate::codegen::matmul_id::mat_spec(&MatMulIdSpec { x_dtype: DType::F32, w_dtype: DType::Int8, out_dtype: DType::F32 })),
        ),
        ("moe_route", Key::MoeRoute),
        ("gated_delta_net_dk128", Key::DeltaNet(Arc::new(GatedDeltaNetSpec { out_dtype: DType::F32 }), 4)),
        ("layer_norm", Key::LayerNorm(Arc::new(LayerNormSpec { in_dtype: DType::F32, param_dtype: DType::F32, out_dtype: DType::F32 }))),
        ("rms_norm", Key::RmsNorm(Arc::new(LayerNormSpec { in_dtype: DType::F32, param_dtype: DType::F32, out_dtype: DType::F32 }))),
    ]
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/tests/golden")
}

#[test]
fn golden_sources() {
    let bless = std::env::var("LAPID_BLESS").is_ok_and(|v| v == "1");
    let dir = golden_dir();
    let mut failures = Vec::new();
    for (name, key) in golden_keys() {
        let src = codegen::source(&key);
        let path = dir.join(format!("{name}.metal"));
        if bless {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(&path, &src).unwrap();
            continue;
        }
        match std::fs::read_to_string(&path) {
            Ok(want) if want == src => {}
            Ok(_) => failures.push(format!("{name}: differs from {} (LAPID_BLESS=1 to update)", path.display())),
            Err(e) => failures.push(format!("{name}: cannot read {}: {e} (LAPID_BLESS=1 to create)", path.display())),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn plans_are_shape_independent() {
    // Same spec, different sizes → same key (one pipeline serves every shape).
    let s = mat_spec(DType::F32, DType::Int8, true, vec![], None, DType::F32);
    let dims = |m: usize, n: usize| MatMulDims {
        batch: 1,
        m,
        n,
        k: 64,
        a: MatView { offset: 0, batch_stride: 0, row_stride: 64 },
        b: MatView { offset: 0, batch_stride: 0, row_stride: 64 },
        extra_views: vec![],
        out: MatView::contiguous(m, n),
    };
    let p1 = codegen::plan(&Kernel::MatMul(s.clone(), dims(100, 64))).unwrap();
    let p2 = codegen::plan(&Kernel::MatMul(s.clone(), dims(1500, 128))).unwrap();
    assert_eq!(p1.stages[0].key, p2.stages[0].key);
    assert_ne!(p1.stages[0].params, p2.stages[0].params);
    let g = codegen::plan(&Kernel::MatMul(s, dims(1, 128))).unwrap();
    assert!(matches!(g.stages[0].key, Key::Gemv(_, 1)));
}

#[test]
fn attention_plans() {
    let s = Arc::new(AttentionSpec { dtype: DType::F32, mask_dtype: None, causal: false, out_dtype: DType::F32 });
    let v = |s0: usize, d: usize| View::contiguous(&[1, 12, s0, d]);
    let dims = |sq: usize, sk: usize, d: usize| AttentionDims {
        batch: 1,
        heads: 12,
        sq,
        sk,
        d,
        scale: 0.125f32.to_bits(),
        q: v(sq, d),
        k: v(sk, d),
        v: v(sk, d),
        mask: None,
    };
    let enc = codegen::plan(&Kernel::Attention(s.clone(), dims(1500, 1500, 64))).unwrap();
    assert!(matches!(enc.stages[0].key, Key::AttnTile(_, 64)));
    assert_eq!(enc.stages[0].groups, [24, 12, 1]);
    let dec = codegen::plan(&Kernel::Attention(s.clone(), dims(1, 1500, 64))).unwrap();
    assert_eq!(dec.stages.len(), 2);
    assert!(matches!(dec.stages[0].key, Key::AttnVec(_, 64, false)));
    assert_eq!(dec.stages[0].groups, [12, 1, 12]);
    assert_eq!(dec.scratch, 12 * 12 * 66 * 4);
    let small = codegen::plan(&Kernel::Attention(s, dims(1, 100, 64))).unwrap();
    assert_eq!(small.stages.len(), 1);
    assert!(matches!(small.stages[0].key, Key::AttnVec(_, 64, true)));
}
