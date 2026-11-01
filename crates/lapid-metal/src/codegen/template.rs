//! Kernel templates: real `.metal` files under `src/kernels/` with `{{SLOT}}` placeholders.
//!
//! Rust only fills slots (buffer bindings, element types, fused expressions, constants).
//! The one piece of "template language" is line-level conditionals for code variants:
//!
//! ```text
//! // #if FLAG          (or `// #if !FLAG`)
//! ...lines kept only when the flag is set...
//! // #endif
//! ```
//!
//! Conditionals nest. Every template defines `kernel void lapid_kernel(...)`; `module`
//! prepends the shared prelude.

pub(crate) const PRELUDE: &str = include_str!("../kernels/prelude.metal");
pub(crate) const ELEMENTWISE: &str = include_str!("../kernels/elementwise.metal");
pub(crate) const REDUCE_ROW: &str = include_str!("../kernels/reduce_row.metal");
pub(crate) const REDUCE_THREAD: &str = include_str!("../kernels/reduce_thread.metal");
pub(crate) const SOFTMAX: &str = include_str!("../kernels/softmax.metal");
pub(crate) const LAYER_NORM: &str = include_str!("../kernels/layer_norm.metal");
pub(crate) const GATHER: &str = include_str!("../kernels/gather.metal");
pub(crate) const GATHER_AXIS: &str = include_str!("../kernels/gather_axis.metal");
pub(crate) const ARGMAX: &str = include_str!("../kernels/argmax.metal");
pub(crate) const TOPK: &str = include_str!("../kernels/topk.metal");
pub(crate) const GEMM: &str = include_str!("../kernels/gemm.metal");
pub(crate) const GEMV: &str = include_str!("../kernels/gemv.metal");
pub(crate) const GEMV_KQ: &str = include_str!("../kernels/gemv_kq.metal");
pub(crate) const ATTN_TILE: &str = include_str!("../kernels/attn_tile.metal");
pub(crate) const ATTN_VEC: &str = include_str!("../kernels/attn_vec.metal");
pub(crate) const ATTN_MERGE: &str = include_str!("../kernels/attn_merge.metal");
pub(crate) const KQUANT: &str = include_str!("../kernels/kquant.metal");
pub(crate) const MOE_ROUTE: &str = include_str!("../kernels/moe_route.metal");
pub(crate) const DELTA_NET: &str = include_str!("../kernels/gated_delta_net.metal");

/// Entry point name of every kernel.
pub(crate) const ENTRY: &str = "lapid_kernel";

/// Fill `src`: replace every `{{NAME}}` from `slots`, then resolve `// #if` blocks against
/// `flags` (the names that are set). Unknown slots are left in place; the Metal compiler
/// then reports them with the numbered source and the golden tests catch them.
pub(crate) fn fill(src: &str, slots: &[(&str, &str)], flags: &[&str]) -> String {
    let mut src = src.to_string();
    for (k, v) in slots {
        src = src.replace(&format!("{{{{{k}}}}}"), v);
    }
    let mut out = String::with_capacity(src.len() + 256);
    // Stack of "is this block active".
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

/// Complete Metal source: prelude + filled kernel template.
pub(crate) fn module(kernel_src: &str, slots: &[(&str, &str)], flags: &[&str]) -> String {
    let mut s = String::with_capacity(PRELUDE.len() + kernel_src.len() + 1024);
    s.push_str(PRELUDE);
    s.push('\n');
    s.push_str(&fill(kernel_src, slots, flags));
    s
}

/// `module` with the K-quant helpers (`kquant.metal`) between the prelude and the kernel.
pub(crate) fn module_kq(kernel_src: &str, slots: &[(&str, &str)], flags: &[&str]) -> String {
    let mut s = String::with_capacity(PRELUDE.len() + KQUANT.len() + kernel_src.len() + 1024);
    s.push_str(PRELUDE);
    s.push('\n');
    s.push_str(KQUANT);
    s.push('\n');
    s.push_str(&fill(kernel_src, slots, flags));
    s
}

/// Source with 1-based line numbers (error messages and `LAPID_DEBUG` dumps).
pub(crate) fn numbered(src: &str) -> String {
    let mut out = String::with_capacity(src.len() + src.len() / 8);
    for (i, line) in src.lines().enumerate() {
        out.push_str(&format!("{:4} | {line}\n", i + 1));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditionals_and_slots() {
        let src = "a {{X}}\n// #if F\nb\n// #if !G\nc\n// #endif\n// #endif\n// #if G\nd\n// #endif\ne\n";
        assert_eq!(fill(src, &[("X", "1")], &["F"]), "a 1\nb\nc\ne\n");
        assert_eq!(fill(src, &[("X", "2")], &["G"]), "a 2\nd\ne\n");
        assert_eq!(fill(src, &[], &["F", "G"]), "a {{X}}\nb\nd\ne\n");
    }

    #[test]
    fn numbering() {
        assert_eq!(numbered("x\ny"), "   1 | x\n   2 | y\n");
    }
}
