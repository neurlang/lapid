//! lapid-tensor: dtypes, ops and the lazy tensor graph.

pub mod buffer;
pub mod decompose;
pub mod dtype;
pub mod error;
pub mod fxhash;
pub mod op;
pub mod tensor;
pub mod walk;

pub use buffer::{DeviceBuffer, DeviceData, HostData};
pub use dtype::DType;
pub use error::{Error, Result};
pub use op::{BinaryOp, Op, ReduceOp, UnaryOp};
pub use tensor::{Node, Shape, Srcs, Tensor};

/// Topologically sorted list of all nodes reachable from `outputs` (srcs before users),
/// deduplicated by id.
pub fn toposort(outputs: &[Tensor]) -> Vec<Tensor> {
    let mut seen = fxhash::FxHashSet::default();
    seen.reserve(256);
    let mut order = Vec::with_capacity(256);
    // Borrowed walk: every node is kept alive by `outputs`; only the result clones.
    let mut stack: Vec<(&Tensor, bool)> = outputs.iter().rev().map(|t| (t, false)).collect();
    while let Some((t, expanded)) = stack.pop() {
        if expanded {
            order.push(t.clone());
            continue;
        }
        if !seen.insert(t.id()) {
            continue;
        }
        stack.push((t, true));
        for s in t.src().iter().rev() {
            if !seen.contains(&s.id()) {
                stack.push((s, false));
            }
        }
    }
    order
}
