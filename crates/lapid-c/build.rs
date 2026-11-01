//! Generates `include/lapid.h` from this crate's `extern "C"` items with cbindgen on every
//! build, so the header can never drift from the code. The header stays committed for
//! people who only download the release archives; CI checks it is up to date.

use std::path::PathBuf;

fn main() {
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    let config = cbindgen::Config::from_file(dir.join("cbindgen.toml")).expect("read cbindgen.toml");
    let bindings = match cbindgen::Builder::new().with_crate(&dir).with_config(config).generate() {
        Ok(b) => b,
        Err(e) => {
            // Don't break the Rust build over the header; the CI check still catches it.
            println!("cargo:warning=lapid.h not regenerated: {e}");
            return;
        }
    };
    // `write_to_file` only touches the file when the content changed.
    bindings.write_to_file(dir.join("include/lapid.h"));
}
