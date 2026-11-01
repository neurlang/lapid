//! Representative kernel keys covering every template and code variant.

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::{BinaryOp, DType, ReduceOp, UnaryOp};

use crate::codegen::{Idx, Key};

pub(crate) fn load(i: usize) -> Box<Expr> {
    Box::new(Expr::Load(i))
}

/// gelu(x_a) (erf form)
pub(crate) fn gelu_expr(a: usize) -> Expr {
    let x = || load(a);
    Expr::Binary(
        BinaryOp::Mul,
        Box::new(Expr::Binary(BinaryOp::Mul, x(), Box::new(Expr::konst(0.5)))),
        Box::new(Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::konst(1.0)),
            Box::new(Expr::Unary(UnaryOp::Erf, Box::new(Expr::Binary(BinaryOp::Mul, x(), Box::new(Expr::konst(std::f32::consts::FRAC_1_SQRT_2)))))),
        )),
    )
}

/// gelu(x_a) + 2 * x_b
pub(crate) fn gelu_plus(a: usize, b: usize) -> Expr {
    Expr::Binary(
        BinaryOp::Add,
        Box::new(gelu_expr(a)),
        Box::new(Expr::Binary(BinaryOp::Mul, load(b), Box::new(Expr::konst(2.0)))),
    )
}

pub(crate) fn mat_spec(a: DType, b: DType, bt: bool, extras: Vec<DType>, epi: Option<Expr>, out: DType) -> Arc<MatMulSpec> {
    Arc::new(MatMulSpec { a_dtype: a, b_dtype: b, b_transposed: bt, extra_dtypes: extras, out_dtype: out, epilogue: epi })
}

fn ib(op: IntOp, a: Expr, b: Expr) -> Expr {
    Expr::IntBinary(op, Box::new(a), Box::new(b))
}

/// An i32 expression touching every integer node, conversion and an integer `Where`
/// (inputs: 0 = i32, 1 = f32). Integer root.
pub(crate) fn int_all_ops() -> Expr {
    let a = || Expr::LoadBits(0);
    let f = || Expr::Load(1);
    let mut e = Expr::Convert(Conv::F32ToI32, Box::new(f()));
    for (i, op) in [
        IntOp::Add,
        IntOp::Sub,
        IntOp::Mul,
        IntOp::DivS,
        IntOp::DivU,
        IntOp::RemS,
        IntOp::RemU,
        IntOp::MinS,
        IntOp::MinU,
        IntOp::MaxS,
        IntOp::MaxU,
        IntOp::And,
        IntOp::Or,
        IntOp::Xor,
        IntOp::Shl,
        IntOp::ShrS,
        IntOp::ShrU,
        IntOp::CmpLtS,
        IntOp::CmpLtU,
        IntOp::CmpEq,
    ]
    .into_iter()
    .enumerate()
    {
        let rhs = if i % 2 == 0 { a() } else { Expr::ConstBits(i as u32 * 3 + 1) };
        e = ib(IntOp::Xor, e, ib(op, a(), rhs));
    }
    let un = Expr::IntUnary(IntUnaryOp::Neg, Box::new(Expr::IntUnary(IntUnaryOp::Abs, Box::new(Expr::IntUnary(IntUnaryOp::Not, Box::new(a()))))));
    let conv = Expr::Convert(
        Conv::F32Bits,
        Box::new(Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::Convert(Conv::I32ToF32, Box::new(a()))),
            Box::new(Expr::Binary(
                BinaryOp::Add,
                Box::new(Expr::Convert(Conv::U32ToF32, Box::new(a()))),
                Box::new(Expr::Convert(Conv::BitsF32, Box::new(Expr::ConstBits(0x3f80_0000)))),
            )),
        )),
    );
    let u = Expr::Convert(Conv::F32ToU32, Box::new(f()));
    Expr::Where(Box::new(ib(IntOp::And, a(), Expr::ConstBits(1))), Box::new(ib(IntOp::Add, e, un)), Box::new(ib(IntOp::Sub, conv, u)))
}

/// Every structural variant over a representative dtype mix.
pub(crate) fn all_keys() -> Vec<Key> {
    let mut keys = Vec::new();
    for &(ina, inb, out) in &[
        (DType::F32, DType::F32, DType::F32),
        (DType::F16, DType::BF16, DType::F16),
        (DType::I32, DType::F32, DType::I32),
        (DType::F32, DType::F16, DType::BF16),
    ] {
        let spec = Arc::new(EwSpec { rank: 3, in_dtypes: vec![ina, inb], out_dtype: out, expr: gelu_plus(0, 1) });
        keys.push(Key::Ew(spec.clone(), vec![Idx::Flat, Idx::Scalar], Idx::Flat));
        keys.push(Key::Ew(spec, vec![Idx::Strided, Idx::Strided], Idx::Strided));
    }
    let all_ops = Expr::Where(
        Box::new(Expr::Binary(BinaryOp::CmpLt, load(0), Box::new(Expr::konst(0.0)))),
        Box::new(Expr::Binary(
            BinaryOp::Pow,
            Box::new(Expr::Unary(UnaryOp::Abs, load(0))),
            Box::new(Expr::Unary(UnaryOp::Tanh, Box::new(Expr::Unary(UnaryOp::Sin, load(0))))),
        )),
        Box::new(Expr::Binary(
            BinaryOp::Max,
            Box::new(Expr::Unary(UnaryOp::Log, Box::new(Expr::Unary(UnaryOp::Exp, load(0))))),
            Box::new(Expr::Binary(
                BinaryOp::Min,
                Box::new(Expr::Unary(UnaryOp::Rsqrt, Box::new(Expr::Unary(UnaryOp::Cos, load(0))))),
                Box::new(Expr::Binary(
                    BinaryOp::Div,
                    Box::new(Expr::Unary(UnaryOp::Recip, Box::new(Expr::Unary(UnaryOp::Sqrt, load(0))))),
                    Box::new(Expr::Binary(BinaryOp::CmpEq, Box::new(Expr::Unary(UnaryOp::Neg, load(0))), Box::new(Expr::konst(f32::INFINITY)))),
                )),
            )),
        )),
    );
    keys.push(Key::Ew(Arc::new(EwSpec { rank: 1, in_dtypes: vec![DType::F32], out_dtype: DType::F32, expr: all_ops }), vec![Idx::Flat], Idx::Flat));
    let fill = Arc::new(EwSpec { rank: 0, in_dtypes: vec![], out_dtype: DType::F32, expr: Expr::konst(f32::NEG_INFINITY) });
    keys.push(Key::Ew(fill, vec![], Idx::Flat));
    for op in [ReduceOp::Sum, ReduceOp::Max] {
        for rank in [1, 3] {
            let spec = Arc::new(ReduceSpec {
                op,
                rank,
                in_dtypes: vec![DType::F16, DType::F32],
                out_dtype: DType::F32,
                expr: Expr::Binary(BinaryOp::Mul, load(0), load(1)),
            });
            keys.push(Key::Reduce(spec.clone(), true));
            keys.push(Key::Reduce(spec, false));
        }
    }
    let bias = Some(Expr::Binary(BinaryOp::Add, load(0), load(1)));
    for b in [DType::F32, DType::F16, DType::BF16, DType::Int8, DType::Int4] {
        for a in [DType::F32, DType::F16, DType::BF16] {
            let s = mat_spec(a, b, true, vec![DType::F32], bias.clone(), DType::F32);
            keys.push(Key::Gemm(s.clone()));
            keys.push(Key::Gemv(s.clone(), 1));
            keys.push(Key::Gemv(s, 4));
        }
    }
    for b in [DType::Q4K, DType::Q5K, DType::Q6K] {
        let s = mat_spec(DType::F32, b, true, vec![DType::F32], bias.clone(), DType::F32);
        keys.push(Key::Gemm(s.clone()));
        keys.push(Key::Gemv(s, 3));
        keys.push(Key::Gather(Arc::new(GatherSpec { table_dtype: b, out_dtype: DType::F32 })));
    }
    for w in [DType::F32, DType::F16, DType::BF16, DType::Int8, DType::Int4, DType::Q4K, DType::Q5K, DType::Q6K] {
        for x in [DType::F32, DType::F16] {
            let s = crate::codegen::matmul_id::mat_spec(&MatMulIdSpec { x_dtype: x, w_dtype: w, out_dtype: DType::F32 });
            keys.push(Key::GemvId(s.clone()));
            keys.push(Key::GemmId(s));
        }
    }
    keys.push(Key::MoeRoute);
    for dkc in [1, 2, 4, 8] {
        keys.push(Key::DeltaNet(Arc::new(GatedDeltaNetSpec { out_dtype: DType::F32 }), dkc));
    }
    keys.push(Key::DeltaNet(Arc::new(GatedDeltaNetSpec { out_dtype: DType::F16 }), 4));
    keys.push(Key::Gemm(mat_spec(DType::F32, DType::F16, false, vec![], None, DType::F16)));
    keys.push(Key::Gemm(mat_spec(DType::F32, DType::F32, false, vec![DType::F16], Some(gelu_plus(0, 1)), DType::BF16)));
    for t in [DType::F32, DType::F16, DType::BF16, DType::Int8, DType::Int4] {
        keys.push(Key::Gather(Arc::new(GatherSpec { table_dtype: t, out_dtype: DType::F32 })));
    }
    keys.push(Key::Softmax(Arc::new(SoftmaxSpec { in_dtype: DType::F16, out_dtype: DType::F32 })));
    keys.push(Key::LayerNorm(Arc::new(LayerNormSpec { in_dtype: DType::F32, param_dtype: DType::F32, out_dtype: DType::F16 })));
    keys.push(Key::RmsNorm(Arc::new(LayerNormSpec { in_dtype: DType::F16, param_dtype: DType::F32, out_dtype: DType::F16 })));
    for causal in [false, true] {
        for mask in [None, Some(DType::F32)] {
            let s = Arc::new(AttentionSpec { dtype: DType::F16, mask_dtype: mask, causal, out_dtype: DType::F32 });
            keys.push(Key::AttnTile(s.clone(), 64));
            keys.push(Key::AttnTile(s.clone(), 128));
            keys.push(Key::AttnTile(s.clone(), 40));
            keys.push(Key::AttnVec(s.clone(), 64, true));
            keys.push(Key::AttnVec(s.clone(), 64, false));
            keys.push(Key::AttnVec(s.clone(), 200, false));
            keys.push(Key::AttnVec(s, 40, true));
        }
    }
    keys.push(Key::AttnMerge(DType::F32));
    keys.push(Key::AttnMerge(DType::F16));
    // Integer expressions: elementwise (raw store), f32 root reading ints, reduce prologue,
    // matmul epilogue.
    let int_ew = Arc::new(EwSpec { rank: 2, in_dtypes: vec![DType::I32, DType::F32], out_dtype: DType::I32, expr: int_all_ops() });
    keys.push(Key::Ew(int_ew.clone(), vec![Idx::Flat, Idx::Flat], Idx::Flat));
    keys.push(Key::Ew(int_ew, vec![Idx::Strided, Idx::Scalar], Idx::Strided));
    let to_f = Expr::Convert(Conv::I32ToF32, Box::new(int_all_ops()));
    keys.push(Key::Ew(
        Arc::new(EwSpec { rank: 1, in_dtypes: vec![DType::I32, DType::F32], out_dtype: DType::F16, expr: to_f.clone() }),
        vec![Idx::Flat, Idx::Flat],
        Idx::Flat,
    ));
    let rs = Arc::new(ReduceSpec { op: ReduceOp::Sum, rank: 2, in_dtypes: vec![DType::I32, DType::F32], out_dtype: DType::F32, expr: to_f });
    keys.push(Key::Reduce(rs.clone(), true));
    keys.push(Key::Reduce(rs, false));
    let epi = Expr::Where(
        Box::new(ib(IntOp::CmpLtS, Expr::LoadBits(1), Expr::ConstBits(0))),
        Box::new(Expr::Convert(Conv::BitsF32, Box::new(ib(IntOp::Xor, Expr::LoadBits(0), Expr::ConstBits(0x8000_0000))))),
        Box::new(Expr::Load(0)),
    );
    let ms = mat_spec(DType::F32, DType::F16, true, vec![DType::I32], Some(epi), DType::F32);
    keys.push(Key::Gemm(ms.clone()));
    keys.push(Key::Gemv(ms, 2));
    for t in [DType::F32, DType::F16, DType::BF16, DType::I32] {
        keys.push(Key::GatherAxis(Arc::new(GatherAxisSpec { dtype: t })));
        keys.push(Key::TopK(Arc::new(TopKSpec { in_dtype: t }), true));
        keys.push(Key::TopK(Arc::new(TopKSpec { in_dtype: t }), false));
    }
    keys
}
