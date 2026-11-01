//! Shared error type for all lapid crates. CONTRACT FILE — owned by the coordinator.

use std::fmt;

/// Machine-readable error. `code` is a stable snake_case id (e.g. "shape_mismatch",
/// "unsupported_dtype", "io", "format", "backend"), `message` is for humans.
/// `to_json()` gives `{"error": code, "message": ...}` for the CLI / agents.
#[derive(Debug, Clone)]
pub struct Error {
    pub code: &'static str,
    pub message: String,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Error { code, message: message.into() }
    }

    pub fn to_json(&self) -> String {
        serde_json::json!({ "error": self.code, "message": self.message }).to_string()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::new("io", e.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::new("json", e.to_string())
    }
}

/// `bail!("code", "fmt {}", x)` — return an Err early.
#[macro_export]
macro_rules! bail {
    ($code:expr, $($arg:tt)*) => {
        return Err($crate::Error::new($code, format!($($arg)*)))
    };
}
