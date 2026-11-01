use super::*;
use lapid_tensor::HostData;

use crate::policy::Role;

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// Host interpreter for the few ops the Gram graphs and the toy model use (2-D only).
fn interp(t: &Tensor) -> Vec<f32> {
    match t.op() {
        Op::Host(h) => h.bytes().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        Op::Cast(_) | Op::Reshape | Op::Contiguous => interp(&t.src()[0]),
        Op::Permute(p) => {
            assert_eq!(p, &[1, 0]);
            let s = t.src()[0].shape().to_vec();
            let x = interp(&t.src()[0]);
            let mut y = vec![0.0; x.len()];
            for i in 0..s[0] {
                for j in 0..s[1] {
                    y[j * s[0] + i] = x[i * s[1] + j];
                }
            }
            y
        }
        Op::MatMul { b_transposed } => {
            let (a, b) = (&t.src()[0], &t.src()[1]);
            let (av, bv) = (interp(a), interp(b));
            let (m, k) = (a.shape()[0], a.shape()[1]);
            let n = t.shape()[1];
            let mut y = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    y[i * n + j] = (0..k)
                        .map(|l| av[i * k + l] * if *b_transposed { bv[j * k + l] } else { bv[l * n + j] })
                        .sum();
                }
            }
            y
        }
        op => panic!("toy interpreter: {op:?}"),
    }
}

/// Toy model: y = relu-free two-layer `x @ W1ᵀ @ W2ᵀ` over random correlated inputs.
struct Toy {
    w1: Tensor,
    w2: Tensor,
    xs: Vec<Tensor>,
}

impl Calibrator for Toy {
    fn weights(&self) -> Result<Vec<(String, Tensor)>> {
        Ok(vec![("l1.weight".into(), self.w1.clone()), ("l2.weight".into(), self.w2.clone())])
    }
    fn len(&self) -> usize {
        self.xs.len()
    }
    fn forward(&mut self, i: usize) -> Result<Vec<Tensor>> {
        Ok(vec![self.xs[i].matmul_t(&self.w1)?.matmul_t(&self.w2)?])
    }
    fn eval(&self, outputs: &[&Tensor]) -> Result<Vec<Vec<f32>>> {
        Ok(outputs.iter().map(|t| interp(t)).collect())
    }
}

fn leaf(v: &[f32], shape: &[usize]) -> Tensor {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    Tensor::from_host(HostData::from_vec(bytes), shape, DType::F32).expect("leaf")
}

fn role(name: &str, shape: &[usize]) -> Role {
    if shape.len() == 1 { Role::Param } else if name == "emb.weight" { Role::LinearInt8 } else { Role::Linear }
}

#[test]
fn finds_matmul_inputs() {
    let w = leaf(&noise(64 * 32, 1), &[64, 32]);
    let other = leaf(&noise(64 * 32, 2), &[64, 32]);
    let x = leaf(&noise(4 * 32, 3), &[4, 32]);
    let y = x.matmul_t(&w).expect("mm").add(&x.matmul_t(&other).expect("mm")).expect("add");
    let found = matmul_inputs(&[y], &[("w".into(), w)]);
    assert_eq!(found.len(), 1);
    assert_eq!((found[0].0.as_str(), found[0].1.id()), ("w", x.id()));
}

#[test]
fn text_chunks() {
    let c = split_text("a b\n\nccc\n\n\n\nd", 6);
    assert_eq!(c, vec!["a b".to_string(), "ccc\n\nd".to_string()]);
    assert_eq!(split_text("x\r\n\r\ny", 100), vec!["x\n\ny".to_string()]);
    assert_eq!(split_text("aaa bbb ccc", 7), vec!["aaa".to_string(), "bbb ccc".to_string()]);
    assert!(CalibrationSource::Builtin.texts(2000, 100).expect("builtin").len() > 4);
    assert_eq!(CalibrationSource::Builtin.texts(200, 3).expect("builtin").len(), 3);
    assert_eq!(CalibrationSource::parse("builtin"), CalibrationSource::Builtin);
}

#[test]
fn calibrated_file_beats_rtn() {
    let (d, h, o, tokens) = (64, 96, 32, 48);
    let w1v = noise(h * d, 4);
    let w2v = noise(o * h, 5);
    let mix = noise(6 * d, 6);
    let xs: Vec<Tensor> = (0..4)
        .map(|i| {
            let base = noise(tokens * 6, 10 + i);
            let x: Vec<f32> = (0..tokens * d)
                .map(|e| {
                    let (t, c) = (e / d, e % d);
                    (0..6).map(|k| base[t * 6 + k] * mix[k * d + c]).sum::<f32>() * if c % 9 == 0 { 6.0 } else { 1.0 }
                })
                .collect();
            leaf(&x, &[tokens, d])
        })
        .collect();
    let dir = std::env::temp_dir().join(format!("lapid-calib-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let (fp, q_cal, q_rtn) = (dir.join("f32.lapid"), dir.join("cal.lapid"), dir.join("rtn.lapid"));
    let meta = crate::writer::metadata("toy", "toy", OutDType::F32, "mit", json!({"repo": "x/y"}));
    let mut w = ModelWriter::create(&fp, meta, OutDType::F32, role, 4).expect("create");
    w.add_blob("b/x.txt", "text/plain", b"hi").expect("blob");
    w.add_f32("l1.weight", &[h, d], &w1v).expect("w1");
    w.add_f32("l2.weight", &[o, h], &w2v).expect("w2");
    w.add_f32("norm.weight", &[d], &noise(d, 7)).expect("norm");
    w.add_f32("emb.weight", &[8, d], &noise(8 * d, 8)).expect("emb");
    let names: Vec<String> = ["l1.weight", "l2.weight", "norm.weight", "emb.weight"].map(String::from).to_vec();
    w.finish(&names).expect("finish");

    let mut toy = Toy { w1: leaf(&w1v, &[h, d]), w2: leaf(&w2v, &[o, h]), xs: xs.clone() };
    // A tiny budget forces one pass per weight.
    let opts = CalibrationOptions { memory_budget: 1, ..Default::default() };
    convert_calibrated(&fp, &q_cal, OutDType::Int4, role, &mut toy, &CalibrationSource::Builtin, &opts).expect("calibrated");
    let r = Reader::open(&fp).expect("open");
    requantize(&r, &q_rtn, OutDType::Int4, role, BTreeMap::new(), json!({"method": "rtn"})).expect("rtn");

    let cal = Reader::open(&q_cal).expect("open cal");
    assert_eq!(cal.metadata["dtype"], "int4");
    assert_eq!(cal.metadata["quantization"]["method"], "gptq");
    assert_eq!(cal.metadata["quantization"]["passes"], 2);
    assert_eq!(cal.blob("b/x.txt"), Some(&b"hi"[..]));
    assert_eq!(cal.tensor("emb.weight").map(|t| t.dtype), Some(DType::Int8));
    assert_eq!(cal.tensor("norm.weight").map(|t| t.dtype), Some(DType::F32));
    let rtn = Reader::open(&q_rtn).expect("open rtn");
    // Output error of the two-layer model on the calibration inputs.
    let out_err = |f: &Reader| -> f64 {
        let w1q = read_f32(f, "l1.weight").expect("w1").1;
        let w2q = read_f32(f, "l2.weight").expect("w2").1;
        let mut e = 0.0;
        for x in &xs {
            let want = interp(&x.matmul_t(&leaf(&w1v, &[h, d])).unwrap().matmul_t(&leaf(&w2v, &[o, h])).unwrap());
            let got = interp(&x.matmul_t(&leaf(&w1q, &[h, d])).unwrap().matmul_t(&leaf(&w2q, &[o, h])).unwrap());
            e += want.iter().zip(&got).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>();
        }
        e
    };
    let (e_cal, e_rtn) = (out_err(&cal), out_err(&rtn));
    assert!(e_cal < 0.7 * e_rtn, "calibrated {e_cal} vs rtn {e_rtn}");
    std::fs::remove_dir_all(&dir).ok();
}
