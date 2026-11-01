//! Elementwise, cast, reduction and movement ops: one C function per `Tensor` method.
//!
//! Every op is lazy: it validates shapes/dtypes, builds a graph node and writes a new handle
//! to `*out`; nothing runs until `lapid_eval` / `lapid_to_host_*`. Inputs are borrowed.

use engine::{BinaryOp, ReduceOp, Result, Tensor, UnaryOp};

use crate::error::{LapidStatus, guard, invalid};
use crate::handle::{LapidTensor, axis_arg, put_tensor, slice, tensor, tensors};
use crate::*;

pub(crate) unsafe fn op1(x: *const LapidTensor, out: *mut *mut LapidTensor, f: impl FnOnce(&Tensor) -> Result<Tensor>) -> LapidStatus {
    guard(|| {
        let x = unsafe { tensor(x)? };
        put_tensor(out, f(x)?)
    })
}

pub(crate) unsafe fn op2(
    a: *const LapidTensor,
    b: *const LapidTensor,
    out: *mut *mut LapidTensor,
    f: impl FnOnce(&Tensor, &Tensor) -> Result<Tensor>,
) -> LapidStatus {
    guard(|| {
        let (a, b) = unsafe { (tensor(a)?, tensor(b)?) };
        put_tensor(out, f(a, b)?)
    })
}

fn unary_op(op: LapidUnaryOp) -> Result<UnaryOp> {
    Ok(match op {
        LAPID_UNARY_NEG => UnaryOp::Neg,
        LAPID_UNARY_EXP => UnaryOp::Exp,
        LAPID_UNARY_LOG => UnaryOp::Log,
        LAPID_UNARY_SQRT => UnaryOp::Sqrt,
        LAPID_UNARY_RSQRT => UnaryOp::Rsqrt,
        LAPID_UNARY_RECIP => UnaryOp::Recip,
        LAPID_UNARY_SIN => UnaryOp::Sin,
        LAPID_UNARY_COS => UnaryOp::Cos,
        LAPID_UNARY_TANH => UnaryOp::Tanh,
        LAPID_UNARY_ERF => UnaryOp::Erf,
        LAPID_UNARY_ABS => UnaryOp::Abs,
        other => return Err(invalid(format!("unknown unary op code {other}"))),
    })
}

fn binary_op(op: LapidBinaryOp) -> Result<BinaryOp> {
    Ok(match op {
        LAPID_BINARY_ADD => BinaryOp::Add,
        LAPID_BINARY_SUB => BinaryOp::Sub,
        LAPID_BINARY_MUL => BinaryOp::Mul,
        LAPID_BINARY_DIV => BinaryOp::Div,
        LAPID_BINARY_MAX => BinaryOp::Max,
        LAPID_BINARY_MIN => BinaryOp::Min,
        LAPID_BINARY_POW => BinaryOp::Pow,
        LAPID_BINARY_CMP_LT => BinaryOp::CmpLt,
        LAPID_BINARY_CMP_EQ => BinaryOp::CmpEq,
        LAPID_BINARY_REM => BinaryOp::Rem,
        LAPID_BINARY_BITAND => BinaryOp::BitAnd,
        LAPID_BINARY_BITOR => BinaryOp::BitOr,
        LAPID_BINARY_BITXOR => BinaryOp::BitXor,
        LAPID_BINARY_SHL => BinaryOp::Shl,
        LAPID_BINARY_SHR => BinaryOp::Shr,
        other => return Err(invalid(format!("unknown binary op code {other}"))),
    })
}

// ---------- elementwise: generic ----------

/// Elementwise unary op (`LAPID_UNARY_*`). Float tensors; NEG and ABS also accept i32.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_unary(x: *const LapidTensor, op: LapidUnaryOp, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, |x| x.unary(unary_op(op)?)) }
}

/// Elementwise binary op (`LAPID_BINARY_*`) with numpy broadcasting; same dtypes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_binary(
    a: *const LapidTensor,
    op: LapidBinaryOp,
    b: *const LapidTensor,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    unsafe { op2(a, b, out, |a, b| a.binary(binary_op(op)?, b)) }
}

/// Binary op with a scalar right-hand side (cast to `a`'s dtype).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_binary_scalar(
    a: *const LapidTensor,
    op: LapidBinaryOp,
    value: f64,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    unsafe { op1(a, out, |a| a.binary_scalar(binary_op(op)?, value)) }
}

// ---------- elementwise: unary ----------

/// -x
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_neg(x: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, Tensor::neg) }
}
/// e^x
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_exp(x: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, Tensor::exp) }
}
/// Natural log.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_log(x: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, Tensor::log) }
}
/// Square root.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_sqrt(x: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, Tensor::sqrt) }
}
/// 1 / sqrt(x)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_rsqrt(x: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, |x| x.unary(UnaryOp::Rsqrt)) }
}
/// 1 / x
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_recip(x: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, |x| x.unary(UnaryOp::Recip)) }
}
/// sin(x)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_sin(x: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, |x| x.unary(UnaryOp::Sin)) }
}
/// cos(x)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_cos(x: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, |x| x.unary(UnaryOp::Cos)) }
}
/// tanh(x)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_tanh(x: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, Tensor::tanh) }
}
/// erf(x)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_erf(x: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, Tensor::erf) }
}
/// |x|
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_abs(x: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, |x| x.unary(UnaryOp::Abs)) }
}
/// Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_gelu(x: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, Tensor::gelu) }
}

// ---------- elementwise: binary (numpy broadcasting) ----------

/// a + b
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_add(a: *const LapidTensor, b: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op2(a, b, out, Tensor::add) }
}
/// a - b
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_sub(a: *const LapidTensor, b: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op2(a, b, out, Tensor::sub) }
}
/// a * b
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_mul(a: *const LapidTensor, b: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op2(a, b, out, Tensor::mul) }
}
/// a / b
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_div(a: *const LapidTensor, b: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op2(a, b, out, Tensor::div) }
}
/// max(a, b)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_maximum(a: *const LapidTensor, b: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op2(a, b, out, Tensor::maximum) }
}
/// min(a, b)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_minimum(a: *const LapidTensor, b: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op2(a, b, out, Tensor::minimum) }
}
/// a ^ b (float)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_pow(a: *const LapidTensor, b: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op2(a, b, out, |a, b| a.binary(BinaryOp::Pow, b)) }
}
/// a < b as 0/1 in the operand dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_cmp_lt(a: *const LapidTensor, b: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op2(a, b, out, |a, b| a.binary(BinaryOp::CmpLt, b)) }
}
/// a == b as 0/1 in the operand dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_cmp_eq(a: *const LapidTensor, b: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op2(a, b, out, |a, b| a.binary(BinaryOp::CmpEq, b)) }
}
/// Truncated remainder (i32; sign of a).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_rem(a: *const LapidTensor, b: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op2(a, b, out, Tensor::rem) }
}
/// a & b (i32)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_bitand(a: *const LapidTensor, b: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op2(a, b, out, Tensor::bitand) }
}
/// a | b (i32)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_bitor(a: *const LapidTensor, b: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op2(a, b, out, Tensor::bitor) }
}
/// a ^ b bitwise (i32)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_bitxor(a: *const LapidTensor, b: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op2(a, b, out, Tensor::bitxor) }
}
/// a << (b & 31) (i32)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_shl(a: *const LapidTensor, b: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op2(a, b, out, Tensor::shl) }
}
/// Arithmetic a >> (b & 31) (i32)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_shr(a: *const LapidTensor, b: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op2(a, b, out, Tensor::shr) }
}
/// x + value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_add_scalar(x: *const LapidTensor, value: f64, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, |x| x.add_scalar(value)) }
}
/// x * value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_mul_scalar(x: *const LapidTensor, value: f64, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, |x| x.mul_scalar(value)) }
}
/// cond != 0 ? a : b, broadcasting all three.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_where(
    cond: *const LapidTensor,
    a: *const LapidTensor,
    b: *const LapidTensor,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    guard(|| {
        let (c, a, b) = unsafe { (tensor(cond)?, tensor(a)?, tensor(b)?) };
        put_tensor(out, Tensor::where_(c, a, b)?)
    })
}
/// Convert to `dtype` (`LAPID_DTYPE_F32/F16/BF16/I32`; float to i32 truncates toward zero).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_cast(x: *const LapidTensor, dtype: LapidDType, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, |x| x.cast(dtype_from_c(dtype)?)) }
}

// ---------- reductions ----------

/// Reduce (`LAPID_REDUCE_*`) over one axis (negative counts from the end); `keepdim` keeps it
/// as size 1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_reduce(
    x: *const LapidTensor,
    op: LapidReduceOp,
    axis: i64,
    keepdim: bool,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    unsafe {
        op1(x, out, |x| {
            let op = match op {
                LAPID_REDUCE_SUM => ReduceOp::Sum,
                LAPID_REDUCE_MAX => ReduceOp::Max,
                other => return Err(invalid(format!("unknown reduce op code {other}"))),
            };
            x.reduce(op, axis_arg(axis)?, keepdim)
        })
    }
}
/// Sum over `axis`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_sum(x: *const LapidTensor, axis: i64, keepdim: bool, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, |x| x.sum(axis_arg(axis)?, keepdim)) }
}
/// Max over `axis`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_max(x: *const LapidTensor, axis: i64, keepdim: bool, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, |x| x.max(axis_arg(axis)?, keepdim)) }
}
/// Mean over `axis` (float).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_mean(x: *const LapidTensor, axis: i64, keepdim: bool, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, |x| x.mean(axis_arg(axis)?, keepdim)) }
}
/// Index (i32) of the largest value along `axis` (first on ties); the axis is removed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_argmax(x: *const LapidTensor, axis: i64, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, |x| x.argmax(axis_arg(axis)?)) }
}
/// `lapid_argmax` over the last axis.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_argmax_last(x: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, Tensor::argmax_last) }
}

// ---------- movement (views) ----------

/// New shape with the same element count; one dim may be `LAPID_DIM_INFER` (SIZE_MAX).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_reshape(
    x: *const LapidTensor,
    shape: *const usize,
    rank: usize,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    unsafe { op1(x, out, |x| x.reshape(slice(shape, rank, "shape")?)) }
}
/// Reorder axes: output dim i is input dim `perm[i]` (`rank` entries).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_permute(
    x: *const LapidTensor,
    perm: *const usize,
    rank: usize,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    unsafe { op1(x, out, |x| x.permute(slice(perm, rank, "perm")?)) }
}
/// Swap two axes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_transpose(x: *const LapidTensor, a: i64, b: i64, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, |x| x.transpose(axis_arg(a)?, axis_arg(b)?)) }
}
/// Numpy-style broadcast to `shape` (missing leading dims added, size-1 dims expanded).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_expand(
    x: *const LapidTensor,
    shape: *const usize,
    rank: usize,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    unsafe { op1(x, out, |x| x.expand(slice(shape, rank, "shape")?)) }
}
/// Insert a size-1 axis at `axis` (of the output).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_unsqueeze(x: *const LapidTensor, axis: i64, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, |x| x.unsqueeze(axis_arg(axis)?)) }
}
/// Remove the size-1 axis `axis`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_squeeze(x: *const LapidTensor, axis: i64, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, |x| x.squeeze(axis_arg(axis)?)) }
}
/// Elements `[start, end)` along `axis`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_slice(
    x: *const LapidTensor,
    axis: i64,
    start: usize,
    end: usize,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    unsafe { op1(x, out, |x| x.slice(axis_arg(axis)?, start, end)) }
}
/// Pad `axis` with `before` / `after` elements of `value`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_pad(
    x: *const LapidTensor,
    axis: i64,
    before: usize,
    after: usize,
    value: f64,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    unsafe { op1(x, out, |x| x.pad(axis_arg(axis)?, before, after, value)) }
}
/// Concatenate `xs[0..n]` along `axis` (same dtype and rank, same dims elsewhere).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_cat(
    xs: *const *const LapidTensor,
    n: usize,
    axis: i64,
    out: *mut *mut LapidTensor,
) -> LapidStatus {
    guard(|| {
        let xs = unsafe { tensors(xs, n, "xs")? };
        put_tensor(out, Tensor::cat(&xs, axis_arg(axis)?)?)
    })
}
/// Materialize as a contiguous row-major tensor.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lapid_contiguous(x: *const LapidTensor, out: *mut *mut LapidTensor) -> LapidStatus {
    unsafe { op1(x, out, Tensor::contiguous) }
}
