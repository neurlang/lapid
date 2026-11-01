//! Per-launch-spec cache of everything an elementwise / reduce launch derives from its spec
//! (compiled program, raw-input mask, bound dtypes, cost estimate).
//!
//! Keyed by the address of the spec's `Arc`: schedules are cached by the runtime and keep
//! their specs alive, so a decode step finds every entry with one pointer hash instead of
//! cloning and hashing the expression tree. Each entry holds a clone of the `Arc`, so an
//! address can never be reused by another spec while its entry exists.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use lapid_compiler::{EwSpec, Expr, ReduceSpec};
use lapid_tensor::{BinaryOp, DType, UnaryOp};

use crate::expr::Program;
use crate::{Ctx, expr_int};

/// What a launch needs from its spec.
pub(crate) struct Compiled {
    pub prog: Arc<Program>,
    /// Dtype each input is bound as (raw inputs as f32 lanes).
    pub dts: Vec<DType>,
    /// Estimated work per element (see `expr_cost`), including loads and the store.
    pub cost: usize,
}

type Entry = (Arc<dyn Any + Send + Sync>, Arc<Compiled>);

const MAX_ENTRIES: usize = 8192;

#[derive(Default)]
pub(crate) struct SpecCache {
    map: Mutex<HashMap<usize, Entry>>,
}

impl SpecCache {
    fn get<S: Send + Sync + 'static>(&self, spec: &Arc<S>, make: impl FnOnce() -> Compiled) -> Arc<Compiled> {
        let key = Arc::as_ptr(spec) as *const () as usize;
        {
            let map = self.map.lock().unwrap_or_else(|p| p.into_inner());
            if let Some((_, c)) = map.get(&key) {
                return c.clone();
            }
        }
        let c = Arc::new(make());
        let mut map = self.map.lock().unwrap_or_else(|p| p.into_inner());
        if map.len() >= MAX_ENTRIES {
            map.clear();
        }
        map.insert(key, (spec.clone() as Arc<dyn Any + Send + Sync>, c.clone()));
        c
    }
}

fn compile(ctx: &Ctx, expr: &Expr, in_dtypes: &[DType]) -> Compiled {
    let raw = expr_int::raw_inputs(expr, in_dtypes);
    let dts = in_dtypes.iter().zip(&raw).map(|(&d, &r)| if r { DType::F32 } else { d }).collect();
    let prog = ctx.program_typed(expr, in_dtypes, &raw);
    Compiled { prog, dts, cost: expr_cost(expr) + 1 }
}

impl Ctx<'_> {
    pub(crate) fn compiled_ew(&self, spec: &Arc<EwSpec>) -> Arc<Compiled> {
        self.programs.specs.get(spec, || compile(self, &spec.expr, &spec.in_dtypes))
    }

    pub(crate) fn compiled_reduce(&self, spec: &Arc<ReduceSpec>) -> Arc<Compiled> {
        self.programs.specs.get(spec, || compile(self, &spec.expr, &spec.in_dtypes))
    }
}

/// Rough per-element cost of evaluating `e` (a load or a simple op = 1, divisions and
/// square roots 3, transcendental functions 8). Used to decide when a launch is worth
/// splitting across threads.
pub(crate) fn expr_cost(e: &Expr) -> usize {
    match e {
        Expr::Load(_) | Expr::LoadBits(_) => 1,
        Expr::Const(_) | Expr::ConstBits(_) => 0,
        Expr::Unary(op, a) => {
            let c = match op {
                UnaryOp::Neg | UnaryOp::Abs => 1,
                UnaryOp::Sqrt | UnaryOp::Rsqrt | UnaryOp::Recip => 3,
                _ => 8,
            };
            c + expr_cost(a)
        }
        Expr::Binary(op, a, b) => {
            let c = match op {
                BinaryOp::Div => 3,
                BinaryOp::Pow => 10,
                _ => 1,
            };
            c + expr_cost(a) + expr_cost(b)
        }
        Expr::IntUnary(_, a) | Expr::Convert(_, a) => 1 + expr_cost(a),
        Expr::IntBinary(_, a, b) => 1 + expr_cost(a) + expr_cost(b),
        Expr::Where(c, a, b) => 1 + expr_cost(c) + expr_cost(a) + expr_cost(b),
    }
}
