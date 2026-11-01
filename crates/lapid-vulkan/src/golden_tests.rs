//! Golden files: the final WGSL of a representative set of kernel keys, checked into
//! `tests/golden/`. Any change to a template or slot filler shows up as a diff there.
//! Regenerate with `LAPID_BLESS=1 cargo test -p lapid-vulkan golden`.

use std::path::PathBuf;
use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::{BinaryOp, DType, ReduceOp, UnaryOp};

use crate::codegen::{self, IdxMode, Key};
use crate::context::Caps;

fn caps(full: bool) -> Caps {
    Caps {
        f16: full,
        int8: true,
        int16: false,
        int_dot: full,
        subgroup: full,
        subgroup_ballot: full,
        subgroup_shuffle: full,
        subgroup_vote: full,
        subgroup_size: 32,
        full_subgroups: full,
        max_groups: [65535; 3],
        max_shared: 49152,
        max_invocations: 1024,
        storage_align: 64,
        max_storage_range: u32::MAX as u64,
        push_descriptor: full,
        max_push_descriptors: 32,
        coopmat: full,
        timestamp_period: 1.0,
    }
}

fn ld(i: usize) -> Box<Expr> {
    Box::new(Expr::Load(i))
}

fn gelu_bias() -> Expr {
    let x = || Box::new(Expr::Binary(BinaryOp::Add, ld(0), ld(1)));
    Expr::Binary(
        BinaryOp::Mul,
        x(),
        Box::new(Expr::Binary(
            BinaryOp::Mul,
            Box::new(Expr::Binary(
                BinaryOp::Add,
                Box::new(Expr::Unary(UnaryOp::Erf, Box::new(Expr::Binary(BinaryOp::Mul, x(), Box::new(Expr::konst(0.70710677)))))),
                Box::new(Expr::konst(1.0)),
            )),
            Box::new(Expr::konst(0.5)),
        )),
    )
}

fn mm(b: DType, epi: bool) -> Arc<MatMulSpec> {
    Arc::new(MatMulSpec {
        a_dtype: DType::F32,
        b_dtype: b,
        b_transposed: true,
        extra_dtypes: if epi { vec![DType::F32] } else { vec![] },
        out_dtype: DType::F32,
        epilogue: epi.then(gelu_bias),
    })
}

fn attn(mask: bool, causal: bool) -> Arc<AttentionSpec> {
    Arc::new(AttentionSpec { dtype: DType::F32, mask_dtype: mask.then_some(DType::F32), causal, out_dtype: DType::F32 })
}

/// i32 out = (a / b) ^ (i32(f32(b) * 0.5) >> 3) with a, b i32 words.
fn int_ew() -> Arc<EwSpec> {
    let w = |i| Box::new(Expr::LoadBits(i));
    let half = Expr::Convert(
        Conv::F32ToI32,
        Box::new(Expr::Binary(BinaryOp::Mul, Box::new(Expr::Convert(Conv::I32ToF32, w(1))), Box::new(Expr::konst(0.5)))),
    );
    let shr = Expr::IntBinary(IntOp::ShrS, Box::new(half), Box::new(Expr::konst_int(3)));
    let expr = Expr::IntBinary(IntOp::Xor, Box::new(Expr::IntBinary(IntOp::DivS, w(0), w(1))), Box::new(shr));
    Arc::new(EwSpec { rank: 2, in_dtypes: vec![DType::I32, DType::I32], out_dtype: DType::I32, expr })
}

/// (file stem, key, full-feature caps?)
fn cases() -> Vec<(&'static str, Key, bool)> {
    let ew = Arc::new(EwSpec { rank: 2, in_dtypes: vec![DType::F32, DType::F32], out_dtype: DType::F16, expr: gelu_bias() });
    let red = Arc::new(ReduceSpec {
        op: ReduceOp::Sum,
        rank: 2,
        in_dtypes: vec![DType::F32],
        out_dtype: DType::F32,
        expr: Expr::Binary(BinaryOp::Mul, ld(0), ld(0)),
    });
    let red_max = Arc::new(ReduceSpec { op: ReduceOp::Max, rank: 3, in_dtypes: vec![DType::F16], out_dtype: DType::F32, expr: Expr::Load(0) });
    vec![
        ("elementwise_flat_scalar", Key::Ew(ew.clone(), vec![IdxMode::Flat, IdxMode::Scalar], IdxMode::Flat), true),
        ("elementwise_strided_nof16", Key::Ew(ew, vec![IdxMode::Strided, IdxMode::Flat], IdxMode::Strided), false),
        ("reduce_row_sum", Key::Reduce(red, true), true),
        ("reduce_thread_max", Key::Reduce(red_max, false), true),
        ("gather_int8", Key::Gather(Arc::new(GatherSpec { table_dtype: DType::Int8, out_dtype: DType::F32 })), true),
        ("softmax", Key::Softmax(Arc::new(SoftmaxSpec { in_dtype: DType::F32, out_dtype: DType::F32 })), true),
        (
            "layer_norm_nosubgroup",
            Key::LayerNorm(Arc::new(LayerNormSpec { in_dtype: DType::F32, param_dtype: DType::F32, out_dtype: DType::F32 })),
            false,
        ),
        ("gemm_f32", Key::Gemm(mm(DType::F32, true)), true),
        ("gemm_coop_int8_gelu", Key::GemmCoop(mm(DType::Int8, true)), true),
        ("gemm_coop_f16", Key::GemmCoop(mm(DType::F16, false)), true),
        ("gemm_coop_int4", Key::GemmCoop(mm(DType::Int4, false)), true),
        ("gemv_int8_gelu", Key::Gemv(mm(DType::Int8, true), true, false), true),
        ("gemv_int4", Key::Gemv(mm(DType::Int4, false), true, false), true),
        ("gemv_f16_vec", Key::Gemv(mm(DType::F16, false), true, true), true),
        ("gemv_f32_nosubgroup", Key::Gemv(mm(DType::F32, false), false, false), false),
        ("attention_split_direct_causal", Key::AttnSplit(attn(false, true), true, 64), true),
        ("attention_split_mask", Key::AttnSplit(attn(true, false), false, 64), true),
        ("attention_merge", Key::AttnMerge(attn(false, false)), true),
        ("attention_coop", Key::AttnCoop(attn(false, false)), true),
        ("attention_tile_d64_causal", Key::AttnTile(attn(true, true), 64), true),
        ("elementwise_int", Key::Ew(int_ew(), vec![IdxMode::Flat, IdxMode::Strided], IdxMode::Flat), true),
        ("gather_axis_f16", Key::GatherAxis(Arc::new(GatherAxisSpec { dtype: DType::F16 })), true),
        ("gather_axis_f32", Key::GatherAxis(Arc::new(GatherAxisSpec { dtype: DType::F32 })), false),
        ("topk_argmax", Key::TopK(Arc::new(TopKSpec { in_dtype: DType::F32 }), true), true),
        ("topk_sort", Key::TopK(Arc::new(TopKSpec { in_dtype: DType::F16 }), false), true),
        ("gemv_q4k", Key::Gemv(mm(DType::Q4K, false), true, false), true),
        ("gemv_q5k_nosubgroup", Key::Gemv(mm(DType::Q5K, false), false, false), false),
        ("gemv_q6k_gelu", Key::Gemv(mm(DType::Q6K, true), true, false), true),
        ("gemm_coop_q4k", Key::GemmCoop(mm(DType::Q4K, false)), true),
        ("gemm_q6k", Key::Gemm(mm(DType::Q6K, false)), false),
        ("gather_q5k", Key::Gather(Arc::new(GatherSpec { table_dtype: DType::Q5K, out_dtype: DType::F32 })), true),
        (
            "matmul_id_q4k",
            Key::MatMulId(Arc::new(MatMulIdSpec { x_dtype: DType::F32, w_dtype: DType::Q4K, out_dtype: DType::F32 })),
            true,
        ),
        (
            "matmul_id_int8_nosubgroup",
            Key::MatMulId(Arc::new(MatMulIdSpec { x_dtype: DType::F32, w_dtype: DType::Int8, out_dtype: DType::F32 })),
            false,
        ),
        ("gated_delta_net", Key::GatedDeltaNet(Arc::new(GatedDeltaNetSpec { out_dtype: DType::F32 })), true),
    ]
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("golden")
}

#[test]
fn golden_kernels() {
    let bless = std::env::var("LAPID_BLESS").is_ok_and(|v| !v.is_empty() && v != "0");
    let dir = golden_dir();
    let mut failures = Vec::new();
    for (name, key, full) in cases() {
        let src = codegen::source(&key, &caps(full));
        assert!(!src.contains("{{"), "{name}: unfilled template slot");
        let path = dir.join(format!("{name}.wgsl"));
        if bless {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(&path, &src).unwrap();
            continue;
        }
        match std::fs::read_to_string(&path) {
            Ok(want) if want == src => {}
            Ok(_) => failures.push(format!("{name}: differs from {}", path.display())),
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    assert!(failures.is_empty(), "golden kernels out of date (rerun with LAPID_BLESS=1 after reviewing):\n{}", failures.join("\n"));
}

/// Every golden kernel must also compile with naga for its caps.
#[test]
fn golden_kernels_compile() {
    for (name, key, full) in cases() {
        let c = caps(full);
        let src = codegen::source(&key, &c);
        if let Err(e) = crate::pipeline::compile_wgsl(name, &src, &c) {
            panic!("{name}: {}", e.message);
        }
    }
}
