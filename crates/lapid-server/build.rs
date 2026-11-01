//! The web UI lives in `ui/` (React + Vite, pnpm) and is embedded from `ui/dist`, which is
//! not committed. Release builds run `pnpm build` first. Source builds without Node get a
//! small placeholder page instead, so `cargo build` never needs Node.

use std::path::Path;

const PLACEHOLDER: &str = r#"<!doctype html><html><head><meta charset="utf-8"><title>lapid</title></head>
<body style="font-family:system-ui;max-width:40rem;margin:4rem auto;line-height:1.5">
<h1>lapid server</h1><p>The API is running (<a href="/docs">/docs</a>, <code>/v1</code>).</p>
<p>The web UI was not built into this binary. Use a release binary, or build it with
<code>pnpm -C crates/lapid-server/ui install && pnpm -C crates/lapid-server/ui build</code>.</p>
</body></html>
"#;

fn main() {
    let dist = Path::new("ui/dist");
    println!("cargo:rerun-if-changed=ui/dist");
    if !dist.join("index.html").exists() && !dist.join("index.html.gz").exists() {
        let _ = std::fs::create_dir_all(dist);
        let _ = std::fs::write(dist.join("index.html"), PLACEHOLDER);
    }
}
