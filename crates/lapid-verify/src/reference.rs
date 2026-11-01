//! Reference activations: a safetensors file of named f32 (or f16/bf16) tensors.

use std::path::{Path, PathBuf};

use lapid::{Error, Result};
use lapid_convert::safetensors::SafeTensors;

/// Named reference tensors, read on demand.
pub struct Activations {
    path: PathBuf,
    st: SafeTensors,
    names: Vec<String>,
}

impl Activations {
    pub fn open(path: &Path) -> Result<Activations> {
        let st = SafeTensors::open(path)?;
        let names = st.names();
        Ok(Activations { path: path.to_path_buf(), st, names })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn has(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name)
    }

    /// Values as f32. Error "missing_tensor" when absent.
    pub fn get(&self, name: &str) -> Result<Vec<f32>> {
        self.check(name)?;
        self.st.to_f32(name)
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        self.check(name)?;
        self.st.shape(name)
    }

    /// `{prefix}{i}` for i = 0, 1, … while present (e.g. "encoder.layers.").
    pub fn numbered(&self, prefix: &str) -> Vec<String> {
        (0..).map(|i| format!("{prefix}{i}")).take_while(|n| self.has(n)).collect()
    }

    fn check(&self, name: &str) -> Result<()> {
        if self.has(name) {
            Ok(())
        } else {
            Err(Error::new("missing_tensor", format!("{}: no tensor {name:?}", self.path.display())))
        }
    }
}
