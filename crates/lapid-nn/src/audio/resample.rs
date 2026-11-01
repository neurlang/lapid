//! Band-limited resampling with a Kaiser-windowed sinc filter.
//!
//! The rate ratio is reduced to `up / down`; output sample `n` sits at input time
//! `n * down / up`, whose fractional part is one of `up` phases. For moderate `up` the
//! per-phase filters are precomputed (and normalised to unit DC gain); otherwise taps are
//! evaluated on the fly. Cutoff = 0.945 × the lower Nyquist, 32 zero crossings per side,
//! Kaiser β = 8.6 (≈ 90 dB stopband).

const ZERO_CROSSINGS: f64 = 32.0;
const ROLLOFF: f64 = 0.945;
const KAISER_BETA: f64 = 8.6;
const MAX_TABLE_PHASES: u64 = 4096;

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Modified Bessel function of the first kind, order 0 (power series).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let q = x * x / 4.0;
    for k in 1..200 {
        term *= q / (k * k) as f64;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

struct Kernel {
    cutoff: f64,
    half_width: f64,
    i0_beta: f64,
}

impl Kernel {
    /// Filter value at distance `x` (in input samples) from the output time.
    fn eval(&self, x: f64) -> f64 {
        let ax = x.abs();
        if ax >= self.half_width {
            return 0.0;
        }
        let r = ax / self.half_width;
        let window = bessel_i0(KAISER_BETA * (1.0 - r * r).sqrt()) / self.i0_beta;
        let t = std::f64::consts::PI * self.cutoff * x;
        let sinc = if t.abs() < 1e-12 { 1.0 } else { t.sin() / t };
        self.cutoff * sinc * window
    }
}

/// Resample mono `input` from `from` Hz to `to` Hz.
pub fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() || from == 0 || to == 0 {
        return input.to_vec();
    }
    let g = gcd(from as u64, to as u64);
    let up = to as u64 / g;
    let down = from as u64 / g;
    let cutoff = if to < from { to as f64 / from as f64 } else { 1.0 } * ROLLOFF;
    let kernel = Kernel {
        cutoff,
        half_width: ZERO_CROSSINGS / cutoff,
        i0_beta: bessel_i0(KAISER_BETA),
    };
    // taps j in (base - w, base + w]
    let w = kernel.half_width.ceil() as i64 + 1;
    let n_taps = (2 * w) as usize;
    let n_in = input.len() as i64;
    let n_out = ((input.len() as u128 * up as u128).div_ceil(down as u128)) as usize;

    // Taps for phase p: weight of input sample (base - w + 1 + i), output time base + p/up.
    let phase_taps = |p: u64| -> Vec<f64> {
        let frac = p as f64 / up as f64;
        let mut taps: Vec<f64> =
            (0..n_taps).map(|i| kernel.eval(frac - (i as f64 - w as f64 + 1.0))).collect();
        let sum: f64 = taps.iter().sum();
        if sum.abs() > 1e-12 {
            for t in &mut taps {
                *t /= sum;
            }
        }
        taps
    };
    let table: Option<Vec<Vec<f64>>> =
        (up <= MAX_TABLE_PHASES).then(|| (0..up).map(&phase_taps).collect());

    let mut out = Vec::with_capacity(n_out);
    for n in 0..n_out as u64 {
        let num = n as u128 * down as u128;
        let base = (num / up as u128) as i64;
        let phase = (num % up as u128) as u64;
        let computed;
        let taps: &[f64] = match &table {
            Some(t) => &t[phase as usize],
            None => {
                computed = phase_taps(phase);
                &computed
            }
        };
        let first = base - w + 1;
        let mut acc = 0.0f64;
        for (i, &c) in taps.iter().enumerate() {
            let j = first + i as i64;
            if j >= 0 && j < n_in {
                acc += input[j as usize] as f64 * c;
            }
        }
        out.push(acc as f32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f64, rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / rate as f64).sin() as f32)
            .collect()
    }

    #[test]
    fn identity() {
        let x = vec![0.1, 0.2, 0.3];
        assert_eq!(resample(&x, 16000, 16000), x);
    }

    #[test]
    fn downsample_preserves_tone() {
        let x = sine(1000.0, 44100, 44100);
        let y = resample(&x, 44100, 16000);
        assert_eq!(y.len(), 16000);
        let reference = sine(1000.0, 16000, 16000);
        // ignore edges
        let err: f32 = y[500..15500]
            .iter()
            .zip(&reference[500..15500])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(err < 1e-3, "max err {err}");
    }

    #[test]
    fn removes_aliasing_content() {
        // 10 kHz is above the 8 kHz Nyquist of the output: must be attenuated.
        let x = sine(10_000.0, 48000, 48000);
        let y = resample(&x, 48000, 16000);
        let rms = (y[1000..15000].iter().map(|v| v * v).sum::<f32>() / 14000.0).sqrt();
        assert!(rms < 1e-3, "rms {rms}");
    }

    #[test]
    fn upsample() {
        let x = sine(440.0, 8000, 8000);
        let y = resample(&x, 8000, 16000);
        assert_eq!(y.len(), 16000);
        let reference = sine(440.0, 16000, 16000);
        let err = y[400..15600]
            .iter()
            .zip(&reference[400..15600])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-3, "max err {err}");
    }
}
