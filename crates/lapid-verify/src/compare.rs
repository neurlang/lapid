//! Collects named comparisons (ours vs a reference) into report rows.

use serde_json::{Value, json};

use crate::report::{LayerReport, diff};

/// A set of compared tensors plus notes about anything that could not be compared.
#[derive(Debug, Clone, Default)]
pub struct Comparison {
    pub layers: Vec<LayerReport>,
    pub notes: Vec<String>,
}

impl Comparison {
    pub fn new() -> Comparison {
        Comparison::default()
    }

    /// Compare one tensor; size mismatches become failed rows (with a note), not errors.
    pub fn compare(&mut self, name: &str, ours: &[f32], reference: &[f32], tolerance: f32) -> &LayerReport {
        let row = match diff(ours, reference) {
            Ok(d) => LayerReport::new(name, d, tolerance),
            Err(e) => {
                self.notes.push(format!("{name}: {}", e.message));
                LayerReport::failed(name)
            }
        };
        self.layers.push(row);
        &self.layers[self.layers.len() - 1]
    }

    /// Compare every `expected` name between two named lists (e.g. traced activations).
    /// Names missing on our side fail; names we produce beyond `expected` are noted.
    pub fn compare_named(
        &mut self,
        expected: &[String],
        ours: &[(String, Vec<f32>)],
        reference: impl Fn(&str) -> lapid::Result<Vec<f32>>,
        tolerance: f32,
    ) -> lapid::Result<()> {
        for name in expected {
            let r = reference(name)?;
            match ours.iter().find(|(n, _)| n == name) {
                Some((_, o)) => {
                    self.compare(name, o, &r, tolerance);
                }
                None => {
                    self.notes.push(format!("{name}: not produced"));
                    self.layers.push(LayerReport::failed(name.as_str()));
                }
            }
        }
        for (name, _) in ours {
            if !expected.contains(name) {
                self.notes.push(format!("{name}: produced but not in the reference (ignored)"));
            }
        }
        Ok(())
    }

    pub fn ok(&self) -> bool {
        self.layers.iter().all(|l| l.ok)
    }

    pub fn first_failing(&self) -> Option<&str> {
        self.layers.iter().find(|l| !l.ok).map(|l| l.name.as_str())
    }

    /// Largest relative error (NaN if any row is NaN).
    pub fn max_rel(&self) -> f32 {
        self.layers.iter().map(|l| l.rel).fold(0.0, |a, r| if r.is_nan() || a.is_nan() { f32::NAN } else { a.max(r) })
    }

    pub fn row(&self, name: &str) -> Option<&LayerReport> {
        self.layers.iter().find(|l| l.name == name)
    }

    /// `{ ok, max_rel, first_failing, layers, notes }`.
    pub fn to_json(&self) -> Value {
        json!({
            "ok": self.ok(),
            "max_rel": self.max_rel(),
            "first_failing": self.first_failing(),
            "layers": self.layers,
            "notes": self.notes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_and_notes() {
        let mut c = Comparison::new();
        assert!(c.compare("a", &[1.0, 2.0], &[1.0, 2.0], 1e-3).ok);
        assert!(!c.compare("b", &[1.0], &[1.0, 2.0], 1e-3).ok);
        assert_eq!(c.first_failing(), Some("b"));
        assert_eq!(c.notes.len(), 1);
        assert!(!c.ok());
        let ours = vec![("x".to_string(), vec![1.0f32]), ("extra".to_string(), vec![0.0])];
        c.compare_named(&["x".to_string(), "y".to_string()], &ours, |_| Ok(vec![1.0]), 1e-3).expect("cmp");
        assert!(c.row("x").is_some_and(|r| r.ok));
        assert!(c.row("y").is_some_and(|r| !r.ok));
        assert!(c.notes.iter().any(|n| n.starts_with("extra")));
    }
}
