//! Timed stages and run statistics.

use std::time::Instant;

use serde_json::{Map, Value, json};

/// Milliseconds since `t`.
pub fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Median (NaN for an empty list).
pub fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

pub fn min(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

/// `{ median, min, runs }` of a per-run quantity.
pub fn stats(values: &[f64]) -> Value {
    json!({ "median": median(values.to_vec()), "min": min(values), "runs": values })
}

/// Named stage durations in execution order.
#[derive(Debug, Clone, Default)]
pub struct Stages {
    stages: Vec<(String, f64)>,
}

impl Stages {
    pub fn new() -> Stages {
        Stages::default()
    }

    /// Run `f`, recording its duration under `name`.
    pub fn time<T>(&mut self, name: &str, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let out = f();
        self.record(name, ms(t));
        out
    }

    pub fn record(&mut self, name: &str, ms: f64) {
        self.stages.push((name.to_string(), ms));
    }

    pub fn get(&self, name: &str) -> Option<f64> {
        self.stages.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
    }

    /// `{ "<stage>": ms, ... }`.
    pub fn to_json(&self) -> Value {
        Value::Object(self.stages.iter().map(|(k, v)| (k.clone(), json!(v))).collect::<Map<_, _>>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_odd_even() {
        assert_eq!(median(vec![3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(vec![4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(median(vec![]).is_nan());
    }

    #[test]
    fn min_of() {
        assert_eq!(min(&[3.0, 1.5, 2.0]), 1.5);
    }

    #[test]
    fn stages_recorded() {
        let mut s = Stages::new();
        assert_eq!(s.time("a", || 7), 7);
        s.record("b", 2.0);
        assert_eq!(s.get("b"), Some(2.0));
        assert!(s.to_json()["a"].is_number());
    }
}
