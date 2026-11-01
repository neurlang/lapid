//! Decoder-step micro benchmark: a Whisper-small-shaped text decoder with random weights,
//! run token by token with a growing self-attention KV cache (same graph shape as
//! lapid-hub's Whisper decoder). No model files, no full transcription.
//!
//! ```text
//! cargo run --release -p lapid --example decode_step -- --dtype int8 --device cpu --steps 32
//! LAPID_PROFILE=1 ...   # per-eval compile / resolve / execute breakdown
//! ```

use std::time::Instant;

use lapid::quant::{Format, quantize};
use lapid::{DType, Device, HostData, Result, Tensor};

/// Counts heap allocations (to keep the compiler's host overhead honest).
struct Counting;

static ALLOCS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: std::alloc::Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: std::alloc::Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::System.realloc(p, l, n) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

struct Ctx {
    dev: Device,
    dtype: DType,
    rng: Rng,
}

impl Ctx {
    fn up(&self, t: Tensor) -> Result<Tensor> {
        self.dev.upload(&t)
    }
    fn f32(&mut self, shape: &[usize], scale: f32) -> Result<Tensor> {
        let n = shape.iter().product();
        let v = self.rng.vec(n, scale);
        self.up(Tensor::from_f32(&v, shape)?)
    }
    /// Weight [rows, cols] in the benchmark dtype.
    fn weight(&mut self, rows: usize, cols: usize) -> Result<Tensor> {
        let v = self.rng.vec(rows * cols, 0.05);
        let t = match self.dtype {
            DType::Int8 | DType::Int4 => {
                let f = if self.dtype == DType::Int8 { Format::Int8 } else { Format::Int4 };
                Tensor::from_host(HostData::from_vec(quantize(f, &v, rows, cols)), &[rows, cols], self.dtype)?
            }
            DType::F16 => {
                let b: Vec<u8> = v.iter().flat_map(|&x| half::f16::from_f32(x).to_le_bytes()).collect();
                Tensor::from_host(HostData::from_vec(b), &[rows, cols], DType::F16)?
            }
            _ => Tensor::from_f32(&v, &[rows, cols])?,
        };
        self.up(t)
    }
    fn linear(&mut self, out: usize, inp: usize, bias: bool) -> Result<Linear> {
        let w = self.weight(out, inp)?;
        let b = if bias { Some(self.f32(&[out], 0.02)?) } else { None };
        Ok(Linear::new(w, b))
    }
    fn ln(&mut self, d: usize) -> Result<LayerNorm> {
        let w = self.up(Tensor::from_f32(&vec![1.0; d], &[d])?)?;
        let b = self.f32(&[d], 0.01)?;
        Ok(LayerNorm::new(w, b, 1e-5))
    }
}

struct Mha {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    scale: f32,
}

impl Mha {
    fn new(c: &mut Ctx, d: usize, heads: usize) -> Result<Mha> {
        Ok(Mha {
            q: c.linear(d, d, true)?,
            k: c.linear(d, d, false)?,
            v: c.linear(d, d, true)?,
            out: c.linear(d, d, true)?,
            heads,
            scale: 1.0 / ((d / heads) as f32).sqrt(),
        })
    }
    fn split(&self, x: &Tensor) -> Result<Tensor> {
        let s = x.shape();
        x.reshape(&[s[0], s[1], self.heads, s[2] / self.heads])?.permute(&[0, 2, 1, 3])
    }
    fn kv(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((self.split(&self.k.forward(x)?)?, self.split(&self.v.forward(x)?)?))
    }
    fn attend(&self, x: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
        let q = self.split(&self.q.forward(x)?)?;
        let a = Tensor::attention(&q, k, v, None, self.scale, causal)?;
        let s = a.shape();
        let (b, h, n, dh) = (s[0], s[1], s[2], s[3]);
        self.out.forward(&a.permute(&[0, 2, 1, 3])?.contiguous()?.reshape(&[b, n, h * dh])?)
    }
}

struct Layer {
    self_ln: LayerNorm,
    self_attn: Mha,
    cross_ln: LayerNorm,
    cross_attn: Mha,
    cross_k: Tensor,
    cross_v: Tensor,
    mlp_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

#[allow(clippy::too_many_arguments)]
fn build_step(
    layers: &[Layer],
    embed: &Tensor,
    positions: &Tensor,
    final_ln: &LayerNorm,
    cache_k: &[Tensor],
    cache_v: &[Tensor],
    token: u32,
    step: usize,
) -> Result<Vec<Tensor>> {
    let n_layers = layers.len();
    let ids = Tensor::from_i32(&[token as i32], &[1, 1])?;
    let tok = Embedding { weight: embed.clone() }.forward(&ids)?;
    let mut x = tok.add(&positions.slice(0, step, step + 1)?)?;
    let mut new_k = Vec::with_capacity(n_layers);
    let mut new_v = Vec::with_capacity(n_layers);
    for (i, l) in layers.iter().enumerate() {
        let h = l.self_ln.forward(&x)?;
        let (k, v) = l.self_attn.kv(&h)?;
        let (k, v) = if step > 0 {
            (Tensor::cat(&[&cache_k[i], &k], 2)?, Tensor::cat(&[&cache_v[i], &v], 2)?)
        } else {
            (k.contiguous()?, v.contiguous()?)
        };
        x = x.add(&l.self_attn.attend(&h, &k, &v, true)?)?;
        new_k.push(k);
        new_v.push(v);
        let h = l.cross_ln.forward(&x)?;
        x = x.add(&l.cross_attn.attend(&h, &l.cross_k, &l.cross_v, false)?)?;
        let h = l.mlp_ln.forward(&x)?;
        x = x.add(&l.fc2.forward(&l.fc1.forward(&h)?.gelu()?)?)?;
    }
    let logits = final_ln.forward(&x)?.matmul_t(&embed)?;
    let mut outs = vec![logits];
    outs.extend(new_k);
    outs.extend(new_v);
    Ok(outs)
}

fn arg(args: &[String], name: &str, default: &str) -> String {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.to_string())
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    if v.is_empty() { 0.0 } else { v[v.len() / 2] }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dtype = match arg(&args, "--dtype", "int8").as_str() {
        "fp32" | "f32" => DType::F32,
        "fp16" | "f16" => DType::F16,
        "int4" => DType::Int4,
        _ => DType::Int8,
    };
    let dev = Device::from_name(&arg(&args, "--device", "cpu"))?;
    let steps: usize = arg(&args, "--steps", "32").parse().unwrap_or(32);
    let n_layers: usize = arg(&args, "--layers", "12").parse().unwrap_or(12);
    let (d, heads, mlp, vocab, ctx, audio) = (768, 12, 3072, 51865, 448, 1500);
    let dh = d / heads;

    let t_setup = Instant::now();
    let mut c = Ctx { dev: dev.clone(), dtype, rng: Rng(0x5eed) };
    let embed = c.weight(vocab, d)?;
    let positions = c.f32(&[ctx, d], 0.1)?;
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        layers.push(Layer {
            self_ln: c.ln(d)?,
            self_attn: Mha::new(&mut c, d, heads)?,
            cross_ln: c.ln(d)?,
            cross_attn: Mha::new(&mut c, d, heads)?,
            cross_k: c.f32(&[1, heads, audio, dh], 1.0)?,
            cross_v: c.f32(&[1, heads, audio, dh], 1.0)?,
            mlp_ln: c.ln(d)?,
            fc1: c.linear(mlp, d, true)?,
            fc2: c.linear(d, mlp, true)?,
        });
    }
    let final_ln = c.ln(d)?;
    eprintln!(
        "decode_step: {} {} layers={n_layers} steps={steps} setup {:.0} ms",
        dev.description(),
        dtype.name(),
        t_setup.elapsed().as_secs_f64() * 1e3
    );

    let passes: usize = arg(&args, "--passes", "2").parse().unwrap_or(2);
    let mut cache_k: Vec<Tensor> = Vec::new();
    let mut cache_v: Vec<Tensor> = Vec::new();
    let mut token = 50258u32;
    for pass in 0..passes {
        cache_k.clear();
        cache_v.clear();
        let (mut build_ms, mut eval_ms, mut total_ms) = (Vec::new(), Vec::new(), Vec::new());
        token = 50258;
        for step in 0..steps {
            let t0 = Instant::now();
            let outs = build_step(&layers, &embed, &positions, &final_ln, &cache_k, &cache_v, token, step)?;
            let outs: Vec<&Tensor> = outs.iter().collect();
            let t1 = Instant::now();
            let mut realized = dev.eval(&outs)?;
            let t2 = Instant::now();
            let nv = realized.split_off(1 + n_layers);
            let nk = realized.split_off(1);
            let lv = dev.to_vec_f32(&realized[0])?;
            cache_k = nk;
            cache_v = nv;
            token = (argmax(&lv) % 50000) as u32;
            let t3 = Instant::now();
            build_ms.push((t1 - t0).as_secs_f64() * 1e3);
            eval_ms.push((t2 - t1).as_secs_f64() * 1e3);
            total_ms.push((t3 - t0).as_secs_f64() * 1e3);
        }
        // Skip the first (cold) step in the medians.
        let skip = usize::from(steps > 2);
        let sum: f64 = total_ms[skip..].iter().sum();
        println!(
            "pass {pass} ({}): decode_step {} {}: per-step median total {:.3} ms (graph build {:.3} ms, eval {:.3} ms), first step {:.1} ms, sum of {} steps {:.1} ms",
            if pass == 0 { "cold compile" } else { "schedule cache warm" },
            dev.name(),
            dtype.name(),
            median(&mut total_ms[skip..].to_vec()),
            median(&mut build_ms[skip..].to_vec()),
            median(&mut eval_ms[skip..].to_vec()),
            total_ms[0],
            steps - skip,
            sum
        );
    }
    if !cache_k.is_empty() {
        let outs = build_step(&layers, &embed, &positions, &final_ln, &cache_k, &cache_v, token, steps)?;
        // Min over batches: robust against other load on the machine.
        let mut best = f64::MAX;
        let mut launches = 0;
        for _ in 0..10 {
            let t = Instant::now();
            for _ in 0..20 {
                launches = lapid::compiler::compile(&outs, &Default::default())?.launches.len();
            }
            best = best.min(t.elapsed().as_secs_f64() * 1e3 / 20.0);
        }
        let a0 = ALLOCS.load(std::sync::atomic::Ordering::Relaxed);
        let _ = lapid::compiler::compile(&outs, &Default::default())?;
        let allocs = ALLOCS.load(std::sync::atomic::Ordering::Relaxed) - a0;
        println!(
            "compile: {best:.3} ms per decode-step graph ({launches} launches, {allocs} allocations, best of 10x20)"
        );
    }
    Ok(())
}

// Minimal layers (the full versions live in lapid-nn, which depends on lapid).

struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    fn new(weight: Tensor, bias: Option<Tensor>) -> Linear {
        Linear { weight, bias }
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.matmul_t(&self.weight)?;
        match &self.bias {
            Some(b) => y.add(&b.cast(y.dtype())?),
            None => Ok(y),
        }
    }
}

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm {
    fn new(weight: Tensor, bias: Tensor, eps: f32) -> LayerNorm {
        LayerNorm { weight, bias, eps }
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.layer_norm(&self.weight, &self.bias, self.eps)
    }
}

struct Embedding {
    weight: Tensor,
}

impl Embedding {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        self.weight.gather(ids)
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}
