//! Kernel templates: real `.wgsl` files under `src/kernels/` with `{{SLOT}}` placeholders.
//!
//! Rust only fills slots (dtype load/store helpers, fused expressions, constants). The one
//! piece of "template language" is line-level conditionals for hardware variants:
//!
//! ```text
//! // #if SUBGROUP          (or `// #if !SUBGROUP`)
//! ...lines kept only when the flag is set...
//! // #endif
//! ```
//!
//! Conditionals nest. Every kernel template defines `fn kernel(wg_lin: u32, wg_total: u32)`;
//! `module` prepends the shared prelude and appends the entry point.

use crate::Caps;

pub const PRELUDE: &str = include_str!("kernels/prelude.wgsl");
pub const ENTRY: &str = include_str!("kernels/entry.wgsl");
pub const WG_REDUCE: &str = include_str!("kernels/wg_reduce.wgsl");
/// Integer-word helpers (`gi_*`), appended to the prelude when a kernel calls one.
pub const INT: &str = include_str!("kernels/int.wgsl");

/// Fill `src`: replace every `{{NAME}}` from `slots`, then resolve `// #if` blocks against
/// `flags` (names that are set) — so shared snippets inserted through a slot (e.g. the
/// workgroup reductions) get their variants resolved too. Unknown slots are left in place
/// (naga then reports them with the numbered source, and the golden tests catch them).
pub fn fill(src: &str, slots: &[(&str, &str)], flags: &[&str]) -> String {
    let mut src = src.to_string();
    for (k, v) in slots {
        src = src.replace(&format!("{{{{{k}}}}}"), v);
    }
    let mut out = String::with_capacity(src.len() + 256);
    // stack of "is this block active"
    let mut stack: Vec<bool> = Vec::new();
    for line in src.lines() {
        let t = line.trim_start();
        if let Some(cond) = t.strip_prefix("// #if ") {
            let cond = cond.trim();
            let (neg, name) = match cond.strip_prefix('!') {
                Some(n) => (true, n.trim()),
                None => (false, cond),
            };
            let on = flags.contains(&name) != neg;
            let parent = stack.last().copied().unwrap_or(true);
            stack.push(parent && on);
            continue;
        }
        if t.starts_with("// #endif") {
            stack.pop();
            continue;
        }
        if stack.last().copied().unwrap_or(true) {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Hardware flags shared by every template.
pub fn cap_flags(caps: &Caps, subgroups: bool) -> Vec<&'static str> {
    let mut f = Vec::new();
    if caps.webgpu {
        f.push("WEBGPU");
    }
    if caps.f16 {
        f.push("F16");
    }
    if subgroups && caps.subgroup {
        f.push("SUBGROUP");
        if caps.subgroup_size == 32 && caps.full_subgroups {
            // 256-thread workgroups are exactly 8 full 32-lane subgroups
            f.push("SG32");
        }
    }
    f
}

/// Complete module: prelude + filled kernel template + entry point with `wg` threads.
/// `subgroups` = the kernel uses subgroup builtins (`g_sgi`, `g_sgid`, ...).
pub fn module(caps: &Caps, kernel_src: &str, slots: &[(&str, &str)], extra_flags: &[&str], wg: u32, subgroups: bool) -> String {
    let mut flags = cap_flags(caps, subgroups);
    flags.extend_from_slice(extra_flags);
    let wg_s = wg.to_string();
    let mut s = fill(PRELUDE, &[], &flags);
    let kernel = fill(kernel_src, slots, &flags);
    if kernel.contains("gi_") {
        s.push_str(INT);
    }
    s.push_str(&kernel);
    s.push_str(&fill(ENTRY, &[("WG", &wg_s)], &flags));
    if caps.webgpu {
        // Tint rejects constant expressions that evaluate to NaN / inf: make every float
        // bit-pattern constant a runtime value (`g_rt0` is a zero-initialized private).
        s = s.replace("bitcast<f32>(0x", "bitcast<f32>(g_rt0 | 0x");
        // -f32::MAX spelled with 8 digits rounds past f32::MAX in Tint's exact check
        s = s.replace("3.4028235e38", "3.4028234e38");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::fill;

    #[test]
    fn conditionals_and_slots() {
        let src = "a {{X}}\n// #if F\nb\n// #if !G\nc\n// #endif\n// #endif\n// #if G\nd\n// #endif\ne\n";
        assert_eq!(fill(src, &[("X", "1")], &["F"]), "a 1\nb\nc\ne\n");
        assert_eq!(fill(src, &[("X", "2")], &["G"]), "a 2\nd\ne\n");
        assert_eq!(fill(src, &[], &["F", "G"]), "a {{X}}\nb\nd\ne\n");
    }
}
