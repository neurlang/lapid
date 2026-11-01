use super::*;

fn f32s(shape: &[usize]) -> Tensor {
    Tensor::zeros(shape, DType::F32)
}

fn quant(rows: usize, cols: usize) -> Tensor {
    let bytes = DType::Int8.storage_bytes(&[rows, cols]);
    Tensor::from_host(HostData::from_vec(vec![0; bytes]), &[rows, cols], DType::Int8).unwrap()
}

#[test]
fn ids_are_monotonic() {
    let a = f32s(&[1]);
    let b = a.exp().unwrap();
    assert!(b.id() > a.id());
}

#[test]
fn named_copies_node() {
    let a = f32s(&[2, 3]).exp().unwrap();
    let n = a.named("x");
    assert_eq!(n.name(), Some("x"));
    assert_ne!(n.id(), a.id());
    assert_eq!(n.shape(), a.shape());
    assert!(matches!(n.op(), Op::Unary(UnaryOp::Exp)));
    assert_eq!(n.src()[0].id(), a.src()[0].id());
}

#[test]
fn axis_resolution() {
    let t = f32s(&[2, 3, 4]);
    assert_eq!(t.axis(-1).unwrap(), 2);
    assert_eq!(t.axis(0).unwrap(), 0);
    assert_eq!(t.axis(3).unwrap_err().code, "invalid_argument");
    assert_eq!(t.axis(-4).unwrap_err().code, "invalid_argument");
}

#[test]
fn from_host_validates_length() {
    let err = Tensor::from_host(HostData::from_vec(vec![0; 7]), &[2], DType::F32).unwrap_err();
    assert_eq!(err.code, "shape_mismatch");
    assert!(Tensor::from_f32(&[1.0, 2.0], &[3]).is_err());
    assert!(Tensor::from_i32(&[1, 2, 3], &[3]).is_ok());
    let err = Tensor::from_host(HostData::from_vec(vec![0; 64]), &[2, 7], DType::Int8).unwrap_err();
    assert_eq!(err.code, "shape_mismatch");
}

#[test]
fn binary_broadcasts_with_reshape_and_expand() {
    let a = f32s(&[2, 3, 4]);
    let b = f32s(&[4]);
    let c = a.add(&b).unwrap();
    assert_eq!(c.shape(), &[2, 3, 4]);
    let srcs = c.src();
    assert_eq!(srcs[0].id(), a.id());
    assert_eq!(srcs[1].shape(), &[2, 3, 4]);
    assert!(matches!(srcs[1].op(), Op::Expand));
    assert!(matches!(srcs[1].src()[0].op(), Op::Reshape));
    assert_eq!(srcs[1].src()[0].shape(), &[1, 1, 4]);
}

#[test]
fn binary_two_sided_broadcast() {
    let c = f32s(&[3, 1]).mul(&f32s(&[1, 5])).unwrap();
    assert_eq!(c.shape(), &[3, 5]);
    assert!(c.src().iter().all(|s| s.shape() == [3, 5]));
}

#[test]
fn binary_rejects_bad_shapes_and_dtypes() {
    assert_eq!(f32s(&[2, 3]).add(&f32s(&[4])).unwrap_err().code, "shape_mismatch");
    let h = Tensor::zeros(&[2], DType::F16);
    assert_eq!(f32s(&[2]).add(&h).unwrap_err().code, "unsupported_dtype");
    assert_eq!(quant(1, 32).add(&quant(1, 32)).unwrap_err().code, "unsupported_dtype");
}

#[test]
fn scalar_becomes_const_of_output_shape() {
    let t = Tensor::zeros(&[2, 2], DType::F16);
    let y = t.mul_scalar(3.0).unwrap();
    let c = &y.src()[1];
    assert!(matches!(c.op(), Op::Const(v) if *v == 3.0));
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.dtype(), DType::F16);
    assert_eq!(y.dtype(), DType::F16);
}

#[test]
fn unary_dtype_rules() {
    let i = Tensor::zeros(&[3], DType::I32);
    assert!(i.neg().is_ok());
    assert_eq!(i.exp().unwrap_err().code, "unsupported_dtype");
}

#[test]
fn where_broadcasts_all() {
    let w = Tensor::where_(&f32s(&[3, 1]), &f32s(&[1, 4]), &f32s(&[4])).unwrap();
    assert_eq!(w.shape(), &[3, 4]);
    assert!(w.src().iter().all(|s| s.shape() == [3, 4]));
}

#[test]
fn cast_rules() {
    let t = f32s(&[2]);
    assert_eq!(t.cast(DType::F32).unwrap().id(), t.id());
    assert_eq!(t.cast(DType::F16).unwrap().dtype(), DType::F16);
    assert_eq!(t.cast(DType::Int8).unwrap_err().code, "unsupported_dtype");
}

#[test]
fn reductions() {
    let t = f32s(&[2, 3, 4]);
    assert_eq!(t.sum(1, false).unwrap().shape(), &[2, 4]);
    assert_eq!(t.max(-1, true).unwrap().shape(), &[2, 3, 1]);
    let m = t.mean(-1, true).unwrap();
    assert!(matches!(m.op(), Op::Binary(BinaryOp::Mul)));
    assert!(matches!(m.src()[1].op(), Op::Const(v) if (*v - 0.25).abs() < 1e-12));
    let am = t.argmax_last().unwrap();
    assert_eq!((am.shape(), am.dtype()), (&[2usize, 3][..], DType::I32));
}

#[test]
fn indexing_and_write_slice() {
    let t = f32s(&[2, 5, 4]);
    let (v, i) = t.topk(3, 1).unwrap();
    assert_eq!((v.shape(), i.shape(), i.dtype()), (&[2usize, 3, 4][..], &[2usize, 3, 4][..], DType::I32));
    assert!(matches!(i.op(), Op::TopK { k: 3, axis: 1 }));
    assert!(matches!(v.op(), Op::GatherAxis { axis: 1 }));
    assert_eq!(t.topk(6, 1).unwrap_err().code, "invalid_argument");
    assert_eq!(t.argmax(1).unwrap().shape(), &[2, 4]);
    let bad_idx = f32s(&[2, 3, 4]);
    assert_eq!(t.gather_axis(1, &bad_idx).unwrap_err().code, "unsupported_dtype");
    let dest = f32s(&[2, 8, 4]);
    let w = dest.write_slice(1, 3, &t).unwrap();
    assert!(matches!(w.op(), Op::WriteSlice { axis: 1, start: 3 }));
    assert_eq!(w.shape(), dest.shape());
    assert_eq!(dest.write_slice(1, 4, &t).unwrap_err().code, "invalid_argument");
    assert_eq!(dest.write_slice(0, 0, &t).unwrap_err().code, "shape_mismatch");
    let ints = Tensor::from_i32(&[1, 2, 3], &[3]).unwrap();
    assert!(ints.bitand(&ints).is_ok() && ints.shl(&ints).is_ok());
    assert_eq!(f32s(&[3]).rem(&f32s(&[3])).unwrap_err().code, "unsupported_dtype");
    assert_eq!(f32s(&[3]).binary_scalar(BinaryOp::Shl, 1.0).unwrap_err().code, "unsupported_dtype");
}

#[test]
fn reshape_infer_and_errors() {
    let t = f32s(&[2, 3, 4]);
    assert_eq!(t.reshape(&[6, usize::MAX]).unwrap().shape(), &[6, 4]);
    assert_eq!(t.reshape(&[5, usize::MAX]).unwrap_err().code, "shape_mismatch");
    assert_eq!(t.reshape(&[usize::MAX, usize::MAX]).unwrap_err().code, "invalid_argument");
    assert_eq!(t.reshape(&[2, 3, 4]).unwrap().id(), t.id());
    let q = quant(4, 32);
    assert_eq!(q.reshape(&[4, 32]).unwrap().id(), q.id());
    assert_eq!(q.reshape(&[2, 64]).unwrap_err().code, "unsupported_dtype");
}

#[test]
fn movement_ops() {
    let t = f32s(&[2, 3, 4]);
    assert_eq!(t.permute(&[2, 0, 1]).unwrap().shape(), &[4, 2, 3]);
    assert!(t.permute(&[0, 0, 1]).is_err());
    assert_eq!(t.transpose(-1, -2).unwrap().shape(), &[2, 4, 3]);
    assert_eq!(t.unsqueeze(-1).unwrap().shape(), &[2, 3, 4, 1]);
    assert_eq!(t.unsqueeze(0).unwrap().shape(), &[1, 2, 3, 4]);
    assert_eq!(t.unsqueeze(1).unwrap().squeeze(1).unwrap().shape(), &[2, 3, 4]);
    assert_eq!(t.squeeze(0).unwrap_err().code, "shape_mismatch");
    assert_eq!(t.slice(1, 1, 3).unwrap().shape(), &[2, 2, 4]);
    assert!(t.slice(1, 2, 4).is_err());
    assert_eq!(t.pad(-1, 1, 2, 0.0).unwrap().shape(), &[2, 3, 7]);
    assert_eq!(f32s(&[1, 4]).expand(&[3, 2, 4]).unwrap().shape(), &[3, 2, 4]);
    assert!(f32s(&[2, 4]).expand(&[3, 4]).is_err());
}

#[test]
fn cat_rules() {
    let a = f32s(&[2, 3]);
    let b = f32s(&[2, 5]);
    assert_eq!(Tensor::cat(&[&a], 1).unwrap().id(), a.id());
    let c = Tensor::cat(&[&a, &b], -1).unwrap();
    assert_eq!(c.shape(), &[2, 8]);
    assert_eq!(Tensor::cat(&[&a, &b], 0).unwrap_err().code, "shape_mismatch");
    let empty = f32s(&[2, 0]);
    assert_eq!(Tensor::cat(&[&empty, &a], 1).unwrap().id(), a.id());
    assert!(Tensor::cat(&[], 0).is_err());
}

#[test]
fn matmul_shapes() {
    let a = f32s(&[2, 3, 4]);
    assert_eq!(a.matmul(&f32s(&[4, 5])).unwrap().shape(), &[2, 3, 5]);
    assert_eq!(a.matmul_t(&f32s(&[5, 4])).unwrap().shape(), &[2, 3, 5]);
    let y = f32s(&[3, 4]).matmul(&f32s(&[2, 4, 5])).unwrap();
    assert_eq!(y.shape(), &[2, 3, 5]);
    assert_eq!(y.src()[0].shape(), &[2, 3, 4]);
    let y = f32s(&[2, 1, 3, 4]).matmul(&f32s(&[1, 6, 4, 5])).unwrap();
    assert_eq!(y.shape(), &[2, 6, 3, 5]);
    assert_eq!(a.matmul(&f32s(&[5, 4])).unwrap_err().code, "shape_mismatch");
    let h = Tensor::zeros(&[5, 4], DType::F16);
    assert_eq!(a.matmul_t(&h).unwrap().dtype(), DType::F32);
}

#[test]
fn quantized_matmul_only_transposed_2d() {
    let a = f32s(&[1, 7, 64]);
    let q = quant(16, 64);
    assert_eq!(a.matmul_t(&q).unwrap().shape(), &[1, 7, 16]);
    assert_eq!(f32s(&[7, 16]).matmul(&quant(16, 64)).unwrap_err().code, "unsupported_dtype");
}

#[test]
fn gather_shape_and_dtype() {
    let table = Tensor::zeros(&[10, 8], DType::F16);
    let idx = Tensor::from_i32(&[1, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    let g = table.gather(&idx).unwrap();
    assert_eq!(g.shape(), &[2, 3, 8]);
    assert_eq!(g.dtype(), DType::F32);
    assert!(quant(10, 32).gather(&idx).is_ok());
    assert_eq!(table.gather(&f32s(&[2])).unwrap_err().code, "unsupported_dtype");
}

#[test]
fn composites() {
    let x = f32s(&[2, 3, 8]);
    assert!(matches!(x.softmax().unwrap().op(), Op::Softmax));
    let ln = x.layer_norm(&f32s(&[8]), &f32s(&[8]), 1e-5).unwrap();
    assert!(matches!(ln.op(), Op::LayerNorm { .. }));
    assert_eq!(x.layer_norm(&f32s(&[7]), &f32s(&[8]), 1e-5).unwrap_err().code, "shape_mismatch");

    let q = f32s(&[1, 4, 3, 16]);
    let k = f32s(&[1, 4, 9, 16]);
    let a = Tensor::attention(&q, &k, &k, None, 0.25, true).unwrap();
    assert_eq!(a.shape(), &[1, 4, 3, 16]);
    assert_eq!(a.src().len(), 3);
    let m = f32s(&[3, 9]);
    let a = Tensor::attention(&q, &k, &k, Some(&m), 0.25, false).unwrap();
    assert_eq!(a.src()[3].shape(), &[1, 4, 3, 9]);
    assert!(Tensor::attention(&k, &q, &q, None, 1.0, true).is_err());
    assert!(Tensor::attention(&q, &k, &f32s(&[1, 4, 8, 16]), None, 1.0, false).is_err());
}

#[test]
fn gelu_is_built_from_primitives() {
    let y = f32s(&[4]).gelu().unwrap();
    assert!(matches!(y.op(), Op::Binary(BinaryOp::Mul)));
    assert_eq!(y.shape(), &[4]);
}
