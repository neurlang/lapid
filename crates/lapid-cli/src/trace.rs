//! `LAPID_TRACE=1`: sum the wall time of every `tracing` span by name (plus its `kind`
//! field, if any) and print a table at exit. `LAPID_TRACE_PER=<span>` normalises the
//! table by that span's count (default `lm.step`, i.e. per generated or prefilled step).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

type Totals = HashMap<String, (u64, f64)>;

fn totals() -> &'static Mutex<Totals> {
    static T: OnceLock<Mutex<Totals>> = OnceLock::new();
    T.get_or_init(Mutex::default)
}

struct Timing {
    label: String,
    start: Instant,
}

#[derive(Default)]
struct KindVisitor(Option<String>);

impl Visit for KindVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "kind" {
            self.0 = Some(value.to_string());
        }
    }
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "kind" {
            self.0 = Some(format!("{value:?}"));
        }
    }
}

struct SumLayer;

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for SumLayer {
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut kind = KindVisitor::default();
        attrs.record(&mut kind);
        let name = attrs.metadata().name();
        let label = match kind.0 {
            Some(k) => format!("{name}[{k}]"),
            None => name.to_string(),
        };
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(Timing { label, start: Instant::now() });
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        let ext = span.extensions();
        let Some(t) = ext.get::<Timing>() else { return };
        let ms = t.start.elapsed().as_secs_f64() * 1e3;
        let mut g = totals().lock().unwrap_or_else(|e| e.into_inner());
        let e = g.entry(t.label.clone()).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += ms;
    }
}

/// Install the summing subscriber when `LAPID_TRACE` is set; returns whether it did.
pub fn init() -> bool {
    if std::env::var_os("LAPID_TRACE").is_none_or(|v| v.is_empty() || v == "0") {
        return false;
    }
    tracing_subscriber::registry().with(SumLayer).try_init().is_ok()
}

/// The table, sorted by total time (call once at exit).
pub fn report() {
    let per = std::env::var("LAPID_TRACE_PER").unwrap_or_else(|_| "lm.step".into());
    let g = totals().lock().unwrap_or_else(|e| e.into_inner());
    let norm = g.get(&per).map_or(1, |e| e.0.max(1));
    let mut rows: Vec<_> = g.iter().collect();
    rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    let mut out = format!("[lapid-trace] per {per} (n={norm}): count, ms, total ms, avg us\n");
    for (name, (n, ms)) in rows {
        let _ = writeln!(
            out,
            "  {name:40} {:8.1}x {:9.3} ms {:10.1} ms {:9.2} us",
            *n as f64 / norm as f64,
            ms / norm as f64,
            ms,
            ms * 1e3 / *n as f64
        );
    }
    eprint!("{out}");
}
