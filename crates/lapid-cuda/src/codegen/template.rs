//! Kernel templates: real `.ptx` files under `src/kernels/` with `{{SLOT}}` placeholders.
//!
//! Rust only fills slots (argument lists, typed loads / stores, fused expressions,
//! unrolled register blocks). The one piece of "template language" is line-level
//! conditionals for structural variants:
//!
//! ```text
//! // #if MASK          (or `// #if !MASK`)
//! ...lines kept only when the flag is set...
//! // #endif
//! ```
//!
//! Conditionals nest. Every template defines `.visible .entry lapid_main`.

/// Fill `src`: replace every `{{NAME}}` from `slots`, then resolve `// #if` blocks against
/// `flags` (names that are set). Unknown slots are left in place (the JIT then reports
/// them with the numbered source; the tests catch them).
pub(crate) fn fill(src: &str, slots: &[(&str, &str)], flags: &[&str]) -> String {
    let mut src = src.to_string();
    for (k, v) in slots {
        src = src.replace(&format!("{{{{{k}}}}}"), v);
    }
    let mut out = String::with_capacity(src.len() + 256);
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

#[cfg(test)]
mod tests {
    use super::fill;

    #[test]
    fn conditionals_and_slots() {
        let src =
            "a {{X}}\n// #if F\nb\n// #if !G\nc\n// #endif\n// #endif\n// #if G\nd\n// #endif\ne\n";
        assert_eq!(fill(src, &[("X", "1")], &["F"]), "a 1\nb\nc\ne\n");
        assert_eq!(fill(src, &[("X", "2")], &["G"]), "a 2\nd\ne\n");
        assert_eq!(fill(src, &[], &["F", "G"]), "a {{X}}\nb\nd\ne\n");
    }
}
