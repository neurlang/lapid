//! Launches over buffers larger than the storage binding limit (forced small here) are
//! split into sub-range launches: results must equal the unsplit run bit for bit.

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::{BinaryOp, DType, DeviceData};

use crate::tests::{backend, f16b, f32b, load, mat_spec, rnd};

fn exec(be: &crate::VulkanBackend, kernel: &Kernel, inputs: &[Vec<u8>], out_bytes: usize, limit: Option<u64>) -> lapid_tensor::Result<Vec<u8>> {
    be.set_binding_limit(limit)?;
    let mut bufs: Vec<DeviceData> = inputs.iter().map(|b| be.upload(b).unwrap()).collect();
    bufs.push(be.alloc(out_bytes).unwrap());
    let n = inputs.len();
    let r = be.execute(&[Launch { kernel: kernel.clone(), inputs: (0..n).collect(), output: n }], &bufs);
    be.set_binding_limit(None)?;
    r?;
    be.download(&bufs[n])
}

fn weight(d: DType, n: usize, k: usize, seed: u32) -> Vec<u8> {
    let w = rnd(n * k, seed);
    match d {
        DType::F32 => f32b(&w),
        DType::F16 => f16b(&w),
        d => lapid_quant::quantize(d.quant_format().unwrap(), &w, n, k),
    }
}

#[test]
fn split_matmul_weights() {
    let Some(be) = backend() else { return };
    let (n, k) = (300usize, 512usize);
    for d in [DType::F32, DType::F16, DType::Q4K, DType::Q6K] {
        for m in [1usize, 3, 40] {
            let w = weight(d, n, k, 5);
            let a = f32b(&rnd(m * k, 6));
            let bias = f32b(&rnd(n, 7));
            let epi = Expr::Binary(BinaryOp::Add, load(0), load(1));
            let spec = mat_spec(DType::F32, d, true, vec![DType::F32], Some(epi));
            let dims = MatMulDims {
                batch: 1,
                m,
                n,
                k,
                a: MatView { offset: 0, batch_stride: m * k, row_stride: k },
                b: MatView { offset: 0, batch_stride: 0, row_stride: k },
                extra_views: vec![View { offset: 0, strides: vec![0, 0, 1] }],
                out: MatView::contiguous(m, n),
            };
            let kern = Kernel::MatMul(spec, dims);
            let ins = [a, w.clone(), bias];
            assert!(w.len() > ins[0].len());
            let want = exec(&be, &kern, &ins, m * n * 4, None).unwrap();
            // ~5 chunks of the weight
            let limit = (w.len() as u64 / 5).max(ins[0].len() as u64).max(4096);
            let got = exec(&be, &kern, &ins, m * n * 4, Some(limit)).unwrap();
            assert!(got == want, "split matmul {} m={m} differs", d.name());
        }
    }
    // int8 SoA cannot be split
    let w = weight(DType::Int8, n, k, 5);
    let dims = MatMulDims {
        batch: 1,
        m: 1,
        n,
        k,
        a: MatView { offset: 0, batch_stride: k, row_stride: k },
        b: MatView { offset: 0, batch_stride: 0, row_stride: k },
        extra_views: vec![],
        out: MatView::contiguous(1, n),
    };
    let kern = Kernel::MatMul(mat_spec(DType::F32, DType::Int8, true, vec![], None), dims);
    let e = exec(&be, &kern, &[f32b(&rnd(k, 1)), w.clone()], n * 4, Some(w.len() as u64 / 4)).unwrap_err();
    assert_eq!(e.code, "too_large");
}

#[test]
fn split_gather_tables() {
    let Some(be) = backend() else { return };
    let (rows, cols) = (1000usize, 256usize);
    for d in [DType::F32, DType::F16, DType::Q5K] {
        let t = weight(d, rows, cols, 9);
        let idx: Vec<i32> = vec![0, 999, 1, 500, 250, 750, 3, 999, 0, 640];
        let ib: Vec<u8> = bytemuck::cast_slice(&idx).to_vec();
        let spec = Arc::new(GatherSpec { table_dtype: d, out_dtype: DType::F32 });
        let kern = Kernel::Gather(spec, GatherDims { rows, cols, count: idx.len() });
        let want = exec(&be, &kern, &[t.clone(), ib.clone()], idx.len() * cols * 4, None).unwrap();
        let got = exec(&be, &kern, &[t.clone(), ib.clone()], idx.len() * cols * 4, Some(t.len() as u64 / 7)).unwrap();
        assert!(got == want, "split gather {} differs", d.name());
        let bad: Vec<u8> = bytemuck::cast_slice(&[3i32, 1000]).to_vec();
        let kern_bad = match kern {
            Kernel::Gather(s, d) => Kernel::Gather(s, GatherDims { count: 2, ..d }),
            _ => unreachable!(),
        };
        let e = exec(&be, &kern_bad, &[t, bad], 2 * cols * 4, Some(4096)).unwrap_err();
        assert_eq!(e.code, "index_out_of_range");
    }
}
