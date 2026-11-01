//! Calls every exported op through its C signature on the CPU device and checks values
//! against the Rust API (same engine, so equality is exact).

use std::ffi::CStr;
use std::ptr::{null, null_mut};

use engine::{Device, Tensor};

use crate::*;

type T = *mut LapidTensor;

fn ok(s: LapidStatus) {
    if s != LAPID_OK {
        let msg = unsafe { CStr::from_ptr(lapid_last_error()) }.to_string_lossy().into_owned();
        panic!("status {s}: {msg}");
    }
}

fn from_f32(v: &[f32], shape: &[usize]) -> T {
    let mut t = null_mut();
    ok(unsafe { lapid_tensor_from_data(v.as_ptr().cast(), shape.as_ptr(), shape.len(), LAPID_DTYPE_F32, &mut t) });
    t
}

fn from_i32(v: &[i32], shape: &[usize]) -> T {
    let mut t = null_mut();
    ok(unsafe { lapid_tensor_from_data(v.as_ptr().cast(), shape.as_ptr(), shape.len(), LAPID_DTYPE_I32, &mut t) });
    t
}

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| ((i * 37) % 11) as f32 * scale - 1.0).collect()
}

/// Evaluate a C handle and a Rust tensor and require identical f32 results.
fn same(t: T, want: &Tensor) {
    let dev = Device::cpu();
    let got = unsafe { &(*t).0 };
    assert_eq!(got.shape(), want.shape());
    assert_eq!(dev.to_vec_f32(got).expect("eval C"), dev.to_vec_f32(want).expect("eval Rust"));
    unsafe { lapid_tensor_free(t) };
}

fn rust(t: T) -> Tensor {
    unsafe { (*t).0.clone() }
}

#[test]
fn every_op_matches_rust() {
    let a = from_f32(&ramp(24, 0.25), &[2, 3, 4]);
    let b = from_f32(&ramp(24, 0.5), &[2, 3, 4]);
    let pos = from_f32(&ramp(24, 0.25).iter().map(|x| x.abs() + 0.5).collect::<Vec<_>>(), &[2, 3, 4]);
    let i = from_i32(&[5, -7, 12, 3], &[4]);
    let j = from_i32(&[2, 3, -5, 1], &[4]);
    let (ra, rb, rp, ri, rj) = (rust(a), rust(b), rust(pos), rust(i), rust(j));
    let mut o: T = null_mut();
    unsafe {
        type U = unsafe extern "C" fn(*const LapidTensor, *mut T) -> LapidStatus;
        let unary: [(U, Tensor); 12] = [
            (lapid_neg, ra.neg().unwrap()),
            (lapid_exp, ra.exp().unwrap()),
            (lapid_log, rp.log().unwrap()),
            (lapid_sqrt, rp.sqrt().unwrap()),
            (lapid_rsqrt, rp.unary(engine::UnaryOp::Rsqrt).unwrap()),
            (lapid_recip, rp.unary(engine::UnaryOp::Recip).unwrap()),
            (lapid_sin, ra.unary(engine::UnaryOp::Sin).unwrap()),
            (lapid_cos, ra.unary(engine::UnaryOp::Cos).unwrap()),
            (lapid_tanh, ra.tanh().unwrap()),
            (lapid_erf, ra.erf().unwrap()),
            (lapid_abs, ra.unary(engine::UnaryOp::Abs).unwrap()),
            (lapid_gelu, ra.gelu().unwrap()),
        ];
        for (k, (f, want)) in unary.iter().enumerate() {
            let x = if matches!(k, 2..=5) { pos } else { a };
            ok(f(x, &mut o));
            same(o, want);
        }
        ok(lapid_unary(a, LAPID_UNARY_EXP, &mut o));
        same(o, &ra.exp().unwrap());

        type B = unsafe extern "C" fn(*const LapidTensor, *const LapidTensor, *mut T) -> LapidStatus;
        let float_bin: [(B, Tensor); 9] = [
            (lapid_add, ra.add(&rb).unwrap()),
            (lapid_sub, ra.sub(&rb).unwrap()),
            (lapid_mul, ra.mul(&rb).unwrap()),
            (lapid_div, ra.div(&rp).unwrap()),
            (lapid_maximum, ra.maximum(&rb).unwrap()),
            (lapid_minimum, ra.minimum(&rb).unwrap()),
            (lapid_pow, rp.binary(engine::BinaryOp::Pow, &rb).unwrap()),
            (lapid_cmp_lt, ra.binary(engine::BinaryOp::CmpLt, &rb).unwrap()),
            (lapid_cmp_eq, ra.binary(engine::BinaryOp::CmpEq, &rb).unwrap()),
        ];
        for (k, (f, want)) in float_bin.iter().enumerate() {
            let (x, y) = match k {
                3 => (a, pos),
                6 => (pos, b),
                _ => (a, b),
            };
            ok(f(x, y, &mut o));
            same(o, want);
        }
        let int_bin: [(B, Tensor); 6] = [
            (lapid_rem, ri.rem(&rj).unwrap()),
            (lapid_bitand, ri.bitand(&rj).unwrap()),
            (lapid_bitor, ri.bitor(&rj).unwrap()),
            (lapid_bitxor, ri.bitxor(&rj).unwrap()),
            (lapid_shl, ri.shl(&rj).unwrap()),
            (lapid_shr, ri.shr(&rj).unwrap()),
        ];
        for (f, want) in &int_bin {
            ok(f(i, j, &mut o));
            same(o, want);
        }
        ok(lapid_binary(a, LAPID_BINARY_SUB, b, &mut o));
        same(o, &ra.sub(&rb).unwrap());
        ok(lapid_binary_scalar(a, LAPID_BINARY_MUL, 3.0, &mut o));
        same(o, &ra.mul_scalar(3.0).unwrap());
        ok(lapid_add_scalar(a, 2.0, &mut o));
        same(o, &ra.add_scalar(2.0).unwrap());
        ok(lapid_mul_scalar(a, -2.0, &mut o));
        same(o, &ra.mul_scalar(-2.0).unwrap());
        let cond = ra.binary(engine::BinaryOp::CmpLt, &rb).unwrap();
        let c = crate::handle::into_raw(LapidTensor(cond.clone()));
        ok(lapid_where(c, a, b, &mut o));
        same(o, &Tensor::where_(&cond, &ra, &rb).unwrap());
        ok(lapid_cast(a, LAPID_DTYPE_F16, &mut o));
        same(o, &ra.cast(engine::DType::F16).unwrap());

        // reductions
        ok(lapid_reduce(a, LAPID_REDUCE_MAX, 1, true, &mut o));
        same(o, &ra.max(1, true).unwrap());
        ok(lapid_sum(a, -1, false, &mut o));
        same(o, &ra.sum(-1, false).unwrap());
        ok(lapid_max(a, 0, false, &mut o));
        same(o, &ra.max(0, false).unwrap());
        ok(lapid_mean(a, 2, true, &mut o));
        same(o, &ra.mean(2, true).unwrap());
        ok(lapid_argmax(a, 1, &mut o));
        same(o, &ra.argmax(1).unwrap());
        ok(lapid_argmax_last(a, &mut o));
        same(o, &ra.argmax_last().unwrap());

        // movement
        let shape = [LAPID_DIM_INFER_RS, 4];
        ok(lapid_reshape(a, shape.as_ptr(), 2, &mut o));
        same(o, &ra.reshape(&[6, 4]).unwrap());
        ok(lapid_permute(a, [2usize, 0, 1].as_ptr(), 3, &mut o));
        same(o, &ra.permute(&[2, 0, 1]).unwrap());
        ok(lapid_transpose(a, 0, -1, &mut o));
        same(o, &ra.transpose(0, -1).unwrap());
        ok(lapid_expand(a, [2usize, 2, 3, 4].as_ptr(), 4, &mut o));
        same(o, &ra.expand(&[2, 2, 3, 4]).unwrap());
        let mut u: T = null_mut();
        ok(lapid_unsqueeze(a, 1, &mut u));
        same(lapid_retained(u), &ra.unsqueeze(1).unwrap());
        ok(lapid_squeeze(u, 1, &mut o));
        same(o, &ra);
        lapid_tensor_free(u);
        ok(lapid_slice(a, 2, 1, 3, &mut o));
        same(o, &ra.slice(2, 1, 3).unwrap());
        ok(lapid_pad(a, 1, 1, 2, 9.0, &mut o));
        same(o, &ra.pad(1, 1, 2, 9.0).unwrap());
        let xs: [*const LapidTensor; 2] = [a, b];
        ok(lapid_cat(xs.as_ptr(), 2, 1, &mut o));
        same(o, &Tensor::cat(&[&ra, &rb], 1).unwrap());
        ok(lapid_contiguous(a, &mut o));
        same(o, &ra);

        // linear algebra / indexing / composites
        let w = from_f32(&ramp(20, 0.1), &[5, 4]);
        let rw = rust(w);
        ok(lapid_matmul_t(a, w, &mut o));
        same(o, &ra.matmul_t(&rw).unwrap());
        let bt = rb.transpose(-1, -2).unwrap();
        let bth = crate::handle::into_raw(LapidTensor(bt.clone()));
        ok(lapid_matmul(a, bth, &mut o));
        same(o, &ra.matmul(&bt).unwrap());
        let ids = from_i32(&[4, 0, 2], &[3]);
        ok(lapid_gather(w, ids, &mut o));
        same(o, &rw.gather(&rust(ids)).unwrap());
        let gi = from_i32(&[3, 0, 1, 2, 2, 1], &[2, 3, 1]);
        ok(lapid_gather_axis(a, -1, gi, &mut o));
        same(o, &ra.gather_axis(-1, &rust(gi)).unwrap());
        let (mut v, mut ix): (T, T) = (null_mut(), null_mut());
        ok(lapid_topk(a, 2, -1, &mut v, &mut ix));
        let (wv, wi) = ra.topk(2, -1).unwrap();
        same(v, &wv);
        same(ix, &wi);
        ok(lapid_topk_indices(a, 2, -1, &mut o));
        same(o, &wi);
        let src = from_f32(&[7.0; 8], &[2, 1, 4]);
        ok(lapid_write_slice(a, 1, 1, src, &mut o));
        same(o, &ra.write_slice(1, 1, &rust(src)).unwrap());
        ok(lapid_softmax(a, &mut o));
        same(o, &ra.softmax().unwrap());
        let g = from_f32(&ramp(4, 0.3), &[4]);
        let bias = from_f32(&ramp(4, 0.2), &[4]);
        ok(lapid_layer_norm(a, g, bias, 1e-5, &mut o));
        same(o, &ra.layer_norm(&rust(g), &rust(bias), 1e-5).unwrap());
        ok(lapid_rms_norm(a, g, 1e-6, &mut o));
        same(o, &ra.rms_norm(&rust(g), 1e-6).unwrap());
        let q = from_f32(&ramp(48, 0.1), &[1, 2, 3, 8]);
        let k = from_f32(&ramp(64, 0.2), &[1, 2, 4, 8]);
        let vv = from_f32(&ramp(64, 0.3), &[1, 2, 4, 8]);
        ok(lapid_attention(q, k, vv, null(), 0.35, true, &mut o));
        same(o, &Tensor::attention(&rust(q), &rust(k), &rust(vv), None, 0.35, true).unwrap());
        let mask = from_f32(&ramp(12, -0.5), &[1, 1, 3, 4]);
        ok(lapid_attention(q, k, vv, mask, 0.35, false, &mut o));
        same(o, &Tensor::attention(&rust(q), &rust(k), &rust(vv), Some(&rust(mask)), 0.35, false).unwrap());
        let x = from_f32(&ramp(12, 0.2), &[3, 4]);
        let experts = from_f32(&ramp(60, 0.1), &[3, 5, 4]);
        let eid = from_i32(&[0, 2, 1, 1, 2, 0], &[3, 2]);
        ok(lapid_matmul_id(x, experts, eid, &mut o));
        same(o, &rust(x).matmul_id(&rust(experts), &rust(eid)).unwrap());
        let (bt_, tt, hh, dk, dv) = (1, 3, 2, 4, 4);
        let gq = from_f32(&ramp(bt_ * tt * hh * dk, 0.1), &[bt_, tt, hh, dk]);
        let gk = from_f32(&ramp(bt_ * tt * hh * dk, 0.05), &[bt_, tt, hh, dk]);
        let gv = from_f32(&ramp(bt_ * tt * hh * dv, 0.2), &[bt_, tt, hh, dv]);
        let gg = from_f32(&ramp(bt_ * tt * hh, -0.1), &[bt_, tt, hh]);
        let gb = from_f32(&ramp(bt_ * tt * hh, 0.05).iter().map(|x| x.abs()).collect::<Vec<_>>(), &[bt_, tt, hh]);
        let gs = from_f32(&ramp(bt_ * hh * dk * dv, 0.01), &[bt_, hh, dk, dv]);
        ok(lapid_gated_delta_net(gq, gk, gv, gg, gb, gs, 0.5, &mut o));
        let want =
            Tensor::gated_delta_net(&rust(gq), &rust(gk), &rust(gv), &rust(gg), &rust(gb), &rust(gs), 0.5).unwrap();
        same(o, &want);

        for h in [a, b, pos, i, j, c, w, bth, ids, gi, src, g, bias, q, k, vv, mask, x, experts, eid] {
            lapid_tensor_free(h);
        }
        for h in [gq, gk, gv, gg, gb, gs] {
            lapid_tensor_free(h);
        }
    }
}

const LAPID_DIM_INFER_RS: usize = usize::MAX;

/// Add a reference and return the same handle (exercises retain).
unsafe fn lapid_retained(t: T) -> T {
    unsafe { lapid_tensor_retain(t) };
    t
}

#[test]
fn errors_and_eval() {
    unsafe {
        let mut o: T = null_mut();
        // NULL input
        assert_eq!(lapid_exp(null(), &mut o), LAPID_ERR_INVALID_ARGUMENT);
        let err = CStr::from_ptr(lapid_last_error()).to_str().unwrap();
        assert!(err.contains("\"invalid_argument\""), "{err}");
        assert!(o.is_null());
        // unknown codes
        let a = from_f32(&[1.0, 2.0], &[2]);
        assert_eq!(lapid_unary(a, 99, &mut o), LAPID_ERR_INVALID_ARGUMENT);
        assert_eq!(lapid_cast(a, 42, &mut o), LAPID_ERR_INVALID_ARGUMENT);
        assert_eq!(lapid_check_abi(LAPID_ABI_VERSION + 1), LAPID_ERR_ABI_MISMATCH);
        ok(lapid_check_abi(LAPID_ABI_VERSION));
        lapid_clear_error();
        assert!(lapid_last_error().is_null());
        // quantized shape validation does not panic
        let mut q: T = null_mut();
        let shape = [3usize, 33];
        let buf = [0u8; 4096];
        let s = lapid_tensor_from_data(buf.as_ptr().cast(), shape.as_ptr(), 2, LAPID_DTYPE_INT8, &mut q);
        assert_eq!(s, LAPID_ERR_SHAPE_MISMATCH);

        // eval + realized + upload + to_host
        let mut dev = null_mut();
        let name = c"cpu";
        ok(lapid_device_open(name.as_ptr(), &mut dev));
        let mut backend = null();
        ok(lapid_device_backend(dev, &mut backend));
        assert_eq!(CStr::from_ptr(backend).to_str().unwrap(), "cpu");
        ok(lapid_exp(a, &mut o));
        let outs = [o as *const LapidTensor];
        let mut res: [T; 1] = [null_mut()];
        ok(lapid_eval(dev, outs.as_ptr(), 1, res.as_mut_ptr()));
        let mut realized = false;
        ok(lapid_tensor_is_realized(res[0], &mut realized));
        assert!(realized);
        let mut host = [0f32; 2];
        assert_eq!(lapid_to_host_f32(dev, res[0], host.as_mut_ptr(), 1), LAPID_ERR_INVALID_ARGUMENT);
        ok(lapid_to_host_f32(dev, res[0], host.as_mut_ptr(), 2));
        assert_eq!(host, [1f32.exp(), 2f32.exp()]);
        let mut up = null_mut();
        ok(lapid_upload(dev, a, &mut up));
        let mut dump = null_mut();
        ok(lapid_schedule_dump(outs.as_ptr(), 1, &mut dump));
        assert!(!CStr::from_ptr(dump).to_bytes().is_empty());
        lapid_free(dump.cast());
        let mut json = null_mut();
        ok(lapid_devices_json(&mut json));
        let parsed: serde_json::Value = serde_json::from_str(CStr::from_ptr(json).to_str().unwrap()).unwrap();
        assert_eq!(parsed[0]["id"], "cpu");
        lapid_free(json.cast());
        for t in [a, o, res[0], up] {
            lapid_tensor_free(t);
        }
        lapid_device_free(dev);
    }
}
