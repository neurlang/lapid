//! Compiles the C examples with the system C compiler and runs them against the static
//! library (and the shared library on Linux / macOS). Skipped when no C compiler exists.
//! C is only used here; the crate itself never compiles C.

use std::path::{Path, PathBuf};
use std::process::Command;

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// `target/<profile>`: this test binary lives in `target/<profile>/deps`.
fn profile_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("test executable path");
    exe.parent().and_then(Path::parent).expect("target/<profile>/deps").to_path_buf()
}

/// `cargo test` builds only the test harness, not the staticlib / cdylib outputs: build
/// them with the same cargo, target directory and profile. Returns `target/<profile>`.
fn build_libs() -> PathBuf {
    let dir = profile_dir();
    let target = dir.parent().expect("target dir");
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["build", "-p", "lapid-c", "--lib"]).env("CARGO_TARGET_DIR", target);
    if dir.file_name().is_some_and(|n| n == "release") {
        cmd.arg("--release");
    }
    let status = cmd.status().expect("run cargo build");
    assert!(status.success(), "cargo build -p lapid-c failed");
    dir
}

fn c_compiler() -> Option<String> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let ok = Command::new(&cc).arg("--version").output().is_ok_and(|o| o.status.success());
    ok.then_some(cc)
}

/// System libraries the Rust static library needs (`--print native-static-libs`).
fn native_libs() -> &'static [&'static str] {
    if cfg!(target_os = "macos") {
        &["-liconv", "-framework", "Metal", "-framework", "CoreFoundation", "-framework", "Foundation", "-lobjc", "-lc", "-lm"]
    } else {
        &["-ldl", "-lgcc_s", "-lutil", "-lrt", "-lpthread", "-lm", "-lc"]
    }
}

fn out_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lapid-c-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Tiny model file for model.c: embed [5, 4], w [3, 4] (f32), blob "notes.txt".
fn write_tiny_model(path: &Path) {
    let embed: Vec<f32> = (0..20).map(|i| (i as f32) * 0.25 - 2.0).collect();
    let w: Vec<f32> = (0..12).map(|i| ((i * 7) % 5) as f32 * 0.5 - 1.0).collect();
    let meta = serde_json::json!({
        "name": "tiny-c-test", "architecture": "test", "dtype": "fp32", "license": "apache-2.0",
        "source": {"repo": "local/tiny-c-test"}, "converter": "lapid-c test"
    });
    let mut wr = lapid_format::Writer::create(path, meta).expect("create .lapid");
    let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    wr.add("embed", lapid_tensor::DType::F32, &[5, 4], &bytes(&embed)).expect("add embed");
    wr.add("w", lapid_tensor::DType::F32, &[3, 4], &bytes(&w)).expect("add w");
    wr.add_blob("notes.txt", "text/plain", b"hello from lapid").expect("add blob");
    wr.finish().expect("finish .lapid");
}

fn compile(cc: &str, src: &str, exe: &Path, link: &[String]) {
    let c = crate_dir();
    let status = Command::new(cc)
        .arg(c.join("examples/c").join(src))
        .arg("-I")
        .arg(c.join("include"))
        .args(["-std=c99", "-Wall", "-Wextra", "-Werror", "-o"])
        .arg(exe)
        .args(link)
        .status()
        .expect("run C compiler");
    assert!(status.success(), "compiling {src} failed");
}

fn run(exe: &Path, args: &[&str], expect: &str) {
    let out = Command::new(exe).args(args).env_remove("LAPID_DEVICE").output().expect("run example");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    println!("{} {args:?}:\n{stdout}{stderr}", exe.display());
    assert!(out.status.success(), "{} exited with {}", exe.display(), out.status);
    assert!(stdout.contains(expect), "missing '{expect}' in output");
}

#[test]
fn c_examples() {
    if cfg!(windows) {
        println!("skipped: the C example test runs on Linux / macOS");
        return;
    }
    let Some(cc) = c_compiler() else {
        println!("skipped: no C compiler (set CC or install cc)");
        return;
    };
    let lib_dir = build_libs();
    let static_lib = lib_dir.join("liblapid.a");
    assert!(static_lib.exists(), "{} missing after cargo build", static_lib.display());

    let dir = out_dir();
    let model = dir.join("tiny.lapid");
    write_tiny_model(&model);
    let model = model.to_str().expect("utf-8 temp path");

    // Static: link the archive directly so the shared library next to it is not picked.
    let mut link = vec![static_lib.display().to_string()];
    link.extend(native_libs().iter().map(|s| s.to_string()));
    for (src, expect, args) in [("matmul.c", "matmul: OK", vec![]), ("model.c", "model: OK", vec![model])] {
        let exe = dir.join(src.trim_end_matches(".c"));
        compile(&cc, src, &exe, &link);
        run(&exe, &args, expect);
    }

    // Shared: -llapid with an rpath to the build directory.
    let dylib = lib_dir.join(if cfg!(target_os = "macos") { "liblapid.dylib" } else { "liblapid.so" });
    if dylib.exists() {
        let link = vec![
            format!("-L{}", lib_dir.display()),
            "-llapid".into(),
            format!("-Wl,-rpath,{}", lib_dir.display()),
            "-lm".into(),
        ];
        let exe = dir.join("matmul_shared");
        compile(&cc, "matmul.c", &exe, &link);
        run(&exe, &[], "matmul: OK");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
